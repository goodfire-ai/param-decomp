"""Declarative placement: semantic axis names + a typed rules table → derived PartitionSpecs.

The ab-initio sharding design (see PLACEMENT_DESIGN.md). Three vocabularies:

- **Semantic axes** — dimension NAMES declared once by the code that owns each tensor
  (`("stack", "d_in", "C")` for a V stack; `("batch", "position", "feature")` for the waist).
- **Mesh axes** — the logical grid the run config declares (`replicate`, `fsdp`, `tp`).
- **Rules** — the config-owned mapping `semantic axis -> mesh axes`, one `Rule` per
  placement row. The rows are TYPED FIELDS of `PlacementRules`
  (`components.{optimizer_state, compute_weights, faithfulness_weights,
  faithfulness_deltas, operands, ns_compute}`,
  `ci_fn.{attention, ffn, input, output}.{optimizer_state, compute_weights, operands,
  ns_compute}`,
  `activations.{external, component}`, and
  `target.{embedding, normalization, position_encoding, column, row, output,
  intermediate, component}`) — never string keys; future rows
  become future fields.

Both name vocabularies are closed Literals (`axes.MeshAxis` / `axes.SemanticAxis`), so a
misspelled axis name is a type error before it can silently replicate a tensor.

A tensor's PartitionSpec at a row is DERIVED: look up each of its dim names in that
row's rule; unlisted names are replicated. Forward phase transitions are mechanical
reshards between two rows of the table; ordinary autodiff supplies their reverse.
`describe` prints the whole policy as the startup audit.

Construction is per-RUN: `from_config(spec, mesh, sites)` binds the table to the run's
mesh and resolved site set, refusing any semantic group the table cannot place — there
are NO fallback arms; every group takes the table's one set of rows or construction
refuses with the explicit remedies. The group census flows downward as data
(`ComponentsPlacement.group_stack_lens`); consumers validate what they receive
(PLACEMENT_DESIGN.md, "Presets").

The rule language is deliberately WEAK: exact-name lookup (semantic axis name → mesh
axes), no patterns, no conditionals, no expressions. Weird cases get a literal spec
override on a named row, not a smarter language. Load-bearing surfaces only: rules pin
optimizer-state trees, phase entries, and the activation waist; GSPMD propagates between
pins as today.
"""

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from math import prod
from typing import Literal, assert_never

import jax
from jax.sharding import AbstractMesh, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array

from param_decomp.core.axes import Axes, MeshAssignment, MeshAxis, SemanticAxis
from param_decomp.core.components import (
    ComponentStacks,
    SiteSpec,
    activation_axes,
    group_shape,
    vu_groups,
)
from param_decomp.core.configs import (
    CIWeightPlacementConfig,
    PlacementTableConfig,
    RuleConfig,
    TargetActivationRef,
    TargetLinearPlacementConfig,
    TargetWeightPlacementConfig,
)
from param_decomp.core.linear_plan import LinearPlan, placed_linear, spec_axes

Rule = Mapping[SemanticAxis, MeshAssignment]

CIWeightFamily = Literal["attention", "ffn", "input", "output"]
"""The CI transformer's weight families — the closed vocabulary `CIFnPlacement.linear_plan`
dispatches on (one `CIWeightPlacement` each)."""

# Muon's batched Newton-Schulz sees every leaf as a canonical `[g, rows<=cols]` stack;
# an `ns_compute` row declares that stack's split verbatim (`ns_staging_sharding`), so
# the row is orientation-blind by construction. Only `stack` may carry an assignment
# (enforced at `_bind`): the NS Gram contraction needs whole matrices per device — a
# sharded matrix axis is an explicit-mode type error, and staging at a
# matrix-axis-carrying layout (e.g. the persist row verbatim) re-triggers the SPMD
# involuntary-full-rematerialization fallback (task 577).


def _mesh_axes_of(assignment: MeshAssignment) -> tuple[MeshAxis, ...]:
    if assignment is None:
        return ()
    if isinstance(assignment, str):
        return (assignment,)
    return tuple(assignment)


@dataclass(frozen=True)
class PlacedRule:
    """One row of the placement table, bound to the mesh: spec derivation + fail-fast
    validation. `label` is PRINT-ONLY (audit lines, error messages) — never a lookup key.
    Unlisted AXIS NAMES are replicated — the quiet default is per-axis, never per-row
    (a row must be declared, even if `{}` = replicated)."""

    mesh: Mesh | AbstractMesh
    label: str
    rule: Rule

    def __post_init__(self) -> None:
        mesh_axes = set(self.mesh.axis_names)
        for axis, assignment in self.rule.items():
            unknown = set(_mesh_axes_of(assignment)) - mesh_axes
            assert not unknown, (
                f"placement rule {self.label!r}: axis {axis!r} maps to unknown mesh "
                f"axes {sorted(unknown)} (mesh has {sorted(mesh_axes)})"
            )
            per = _mesh_axes_of(assignment)
            assert len(per) == len(set(per)), (
                f"placement rule {self.label!r}: axis {axis!r} repeats a mesh axis ({assignment})"
            )
        # NOTE: one mesh axis MAY appear under several semantic names in a rule
        # (`d_in -> fsdp, d_out -> fsdp`) because no single tensor carries both;
        # uniqueness is a per-TENSOR invariant, checked in `spec_for`.

    def spec_for(self, axes: Axes) -> P:
        """The derived PartitionSpec for a tensor with semantic `axes` at this row."""
        entries: tuple[MeshAssignment, ...] = tuple(self.rule.get(name) for name in axes)
        used: list[MeshAxis] = []
        for e in entries:
            used.extend(_mesh_axes_of(e))
        assert len(used) == len(set(used)), (
            f"{self.label}: tensor axes {axes} derive a spec using a mesh axis twice ({entries})"
        )
        return P(*entries)

    def sharding_for(self, axes: Axes) -> NamedSharding:
        return NamedSharding(self.mesh, self.spec_for(axes))

    def shard_count(self, axis_name: SemanticAxis) -> int:
        """How many ways `axis_name` is split at this row (1 = unsharded)."""
        return prod(self.mesh.shape[a] for a in _mesh_axes_of(self.rule.get(axis_name)))

    def validate_shape(self, axes: Axes, shape: tuple[int, ...]) -> None:
        """Fail-fast divisibility: every dim must tile its assigned mesh-axis product."""
        assert len(axes) == len(shape), (self.label, axes, shape)
        for name, dim in zip(axes, shape, strict=True):
            n = prod(self.mesh.shape[a] for a in _mesh_axes_of(self.rule.get(name)))
            assert dim % n == 0, (
                f"{self.label}: semantic axis {name!r} (dim {dim}) does not tile its mesh "
                f"assignment ({self.rule.get(name)!r} = ÷{n})"
            )


def ns_staging_sharding(row: PlacedRule, mesh: Mesh | AbstractMesh) -> NamedSharding:
    """One semantic kind's muon-NS staging waypoint: the row's stack split verbatim,
    matrices whole per device. Only a stacked-muon optimizer consumes it, so its tiling
    claim fires at that consumer (`assert_stacked_muon_*_staging`), not at rules
    construction — a non-muon run keeps any-stack-length placement."""
    assert set(row.rule) <= {"stack"}, (row.label, sorted(row.rule))
    return NamedSharding(mesh, P(row.rule.get("stack"), None, None))


@dataclass(frozen=True)
class ComponentsPlacement:
    """The declared lifecycle of trainable V/U weights. ONE set of rows places every
    semantic group — a group the rows cannot place refused at construction, so no
    per-group dispatch exists anywhere downstream. `group_stack_lens` is the resolved
    site-set census the consumer boundary re-checks its arrays against
    (`_validate_component_stacks`); `ns_compute` is the muon-NS staging waypoint
    (`ns_staging_sharding`)."""

    optimizer_state: PlacedRule
    compute_weights: PlacedRule
    faithfulness_weights: PlacedRule
    faithfulness_deltas: PlacedRule
    operands: PlacedRule
    ns_compute: PlacedRule
    group_stack_lens: Mapping[str, int]

    def compute_weight_provenance(self, axes: Axes) -> frozenset[str]:
        """The mesh axes the compute residents were gathered over from optimizer
        ownership — the plans' `weight_reduced`."""
        return dropped_mesh_axes(self.optimizer_state, self.compute_weights, axes)


@dataclass(frozen=True)
class CIWeightPlacement:
    optimizer_state: PlacedRule
    compute_weights: PlacedRule
    operands: PlacedRule
    ns_compute: PlacedRule


@dataclass(frozen=True)
class CIFnPlacement:
    attention: CIWeightPlacement
    ffn: CIWeightPlacement
    input: CIWeightPlacement
    output: CIWeightPlacement
    vectors: PlacedRule
    activations: PlacedRule

    def linear_plan(
        self,
        family: CIWeightFamily,
        stored_axes: tuple[SemanticAxis, SemanticAxis],
        activation_ndim: int,
        *,
        transposed: bool,
    ) -> LinearPlan:
        assert isinstance(self.activations.mesh, Mesh), type(self.activations.mesh)
        match family:
            case "attention":
                weights = self.attention
            case "ffn":
                weights = self.ffn
            case "input":
                weights = self.input
            case "output":
                weights = self.output
            case _:
                assert_never(family)

        def spec(row: PlacedRule) -> P:
            stored = row.spec_for(stored_axes)
            return P(*reversed(stored)) if transposed else stored

        operand_axes = tuple(reversed(stored_axes)) if transposed else stored_axes
        input_axes = activation_axes(activation_ndim, operand_axes[0])
        output_axes = activation_axes(activation_ndim, operand_axes[1])
        return LinearPlan(
            mesh=self.activations.mesh,
            input=self.activations.spec_for(input_axes),
            operand_input=self.activations.spec_for(input_axes),
            resident_weight=spec(weights.compute_weights),
            operand=spec(weights.operands),
            output=self.activations.spec_for(output_axes),
            weight_reduced=dropped_mesh_axes(
                weights.optimizer_state, weights.compute_weights, ("stack", *stored_axes)
            ),
        )


@dataclass(frozen=True)
class ActivationsPlacement:
    """The component linear's replicated public waist and C-sharded internal waist."""

    external: PlacedRule
    component: PlacedRule


@dataclass(frozen=True)
class TargetLinearPlacement:
    """One frozen-target linear declaration, fully resolved to placement rows."""

    persist: PlacedRule
    operand: PlacedRule
    input: PlacedRule
    output: PlacedRule


@dataclass(frozen=True)
class TargetComponentLinearPlacement:
    """The component-replaced linear's public activation contract."""

    input: PlacedRule
    output: PlacedRule


@dataclass(frozen=True)
class TargetWeightPlacement:
    """One non-block frozen weight's resting and execution layouts."""

    persist: PlacedRule
    operand: PlacedRule


@dataclass(frozen=True)
class TargetPlacement:
    """Every frozen-target weight role and its execution contract."""

    embedding: TargetWeightPlacement
    normalization: PlacedRule
    position_encoding: PlacedRule
    column: TargetLinearPlacement
    row: TargetLinearPlacement
    output: TargetWeightPlacement
    intermediate: PlacedRule
    component: TargetComponentLinearPlacement


def _collect_placed_rules(value: object, rows: list[PlacedRule], seen: set[int]) -> None:
    """Every distinct `PlacedRule` reachable through dataclass fields and mappings, in
    field-declaration order, first occurrence wins (target linears alias the activation
    rows). Deriving the enumeration from the structure makes a forgotten row impossible:
    a row field that exists is mesh-checked and audited."""
    if isinstance(value, PlacedRule):
        if id(value) not in seen:
            seen.add(id(value))
            rows.append(value)
    elif isinstance(value, Mapping):
        for item in value.values():
            _collect_placed_rules(item, rows, seen)
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _collect_placed_rules(getattr(value, field.name), rows, seen)


@dataclass(frozen=True)
class PlacementRules:
    """The resolved placement policy for one run: a mesh plus one `PlacedRule` per row."""

    mesh: Mesh | AbstractMesh
    components: ComponentsPlacement
    ci_fn: CIFnPlacement
    activations: ActivationsPlacement
    target: TargetPlacement

    def __post_init__(self) -> None:
        for row in self._rows():
            assert row.mesh is self.mesh, (row.label, "row bound to a different mesh")

    def component_linear_plan(
        self,
        weight_axes: tuple[SemanticAxis, SemanticAxis],
        input_axes: Axes,
        output_axes: Axes,
    ) -> LinearPlan:
        assert isinstance(self.mesh, Mesh), type(self.mesh)
        match weight_axes:
            case ("d_in", "C"):
                input_row = self.target.component.input
                output_row = self.activations.component
            case ("C", "d_out"):
                input_row = self.activations.component
                output_row = self.target.component.output
            case _:
                raise AssertionError(weight_axes)
        return LinearPlan(
            mesh=self.mesh,
            input=input_row.spec_for(input_axes),
            operand_input=input_row.spec_for(input_axes),
            resident_weight=self.components.compute_weights.spec_for(weight_axes),
            operand=self.components.operands.spec_for(weight_axes),
            output=output_row.spec_for(output_axes),
            weight_reduced=self.components.compute_weight_provenance(("stack", *weight_axes)),
        )

    def target_native_component_linear_plan(
        self,
        target: TargetLinearPlacement,
        weight_axes: tuple[SemanticAxis, SemanticAxis],
        input_axes: Axes,
        output_axes: Axes,
    ) -> LinearPlan:
        assert isinstance(self.mesh, Mesh), type(self.mesh)
        match weight_axes:
            case ("d_in", "C"):
                input_row = target.input
                operand_input_row = self.target.component.input
                output_row = self.activations.component
            case ("C", "d_out"):
                input_row = self.activations.component
                operand_input_row = input_row
                output_row = target.output
            case _:
                raise AssertionError(weight_axes)
        return LinearPlan(
            mesh=self.mesh,
            input=input_row.spec_for(input_axes),
            operand_input=operand_input_row.spec_for(input_axes),
            resident_weight=self.components.compute_weights.spec_for(weight_axes),
            operand=self.components.operands.spec_for(weight_axes),
            output=output_row.spec_for(output_axes),
            weight_reduced=self.components.compute_weight_provenance(("stack", *weight_axes)),
        )

    def _rows(self) -> tuple[PlacedRule, ...]:
        rows: list[PlacedRule] = []
        _collect_placed_rules(self, rows, set())
        return tuple(rows)

    def describe(
        self,
        tensors: Mapping[str, tuple[PlacedRule, Axes, tuple[int, ...]]] | None = None,
        sharded_tensors: Mapping[str, tuple[NamedSharding, tuple[int, ...]]] | None = None,
        not_audited: tuple[str, ...] = (),
    ) -> str:
        """The policy as one printable table (startup log + documentation). With `tensors`
        (`{label: (row, axes, shape)}`) it also prints each tensor's derived spec and
        per-device share — the placement audit a human or agent reads before a run.

        Any audited tensor larger than `REPLICATION_FLAG_ELEMS` that derives to fully
        replicated is flagged; tensor families absent from the per-tensor audit are
        listed as NOT AUDITED rather than silently omitted."""
        lines = ["placement rules:"]
        mesh_desc = ", ".join(f"{a}={s}" for a, s in self.mesh.shape.items())
        lines.append(f"  mesh: {mesh_desc}")
        for row in self._rows():
            body = (
                ", ".join(f"{k}->{v}" for k, v in row.rule.items()) if row.rule else "(replicated)"
            )
            lines.append(f"  {row.label:<22} {body}")
        lines.append("target declarations:")
        for name, declaration in (("column", self.target.column), ("row", self.target.row)):
            lines.append(
                f"  {name:<9} persist={declaration.persist.label} "
                f"operand={declaration.operand.label} input={declaration.input.label} "
                f"output={declaration.output.label}"
            )
        component = self.target.component
        for name, declaration in (
            ("embedding", self.target.embedding),
            ("output", self.target.output),
        ):
            lines.append(
                f"  {name:<9} persist={declaration.persist.label} "
                f"operand={declaration.operand.label}"
            )
        lines.append(
            f"  {'component':<9} input={component.input.label} output={component.output.label}"
        )
        if tensors:
            lines.append("derived placements:")
            for label, (row, axes, shape) in tensors.items():
                row.validate_shape(axes, shape)
                spec = row.spec_for(axes)
                n_shards = prod(row.shard_count(n) for n in axes)
                share = prod(shape) // n_shards
                flag = (
                    "   ⚠ FULLY REPLICATED (large)"
                    if n_shards == 1 and prod(shape) > REPLICATION_FLAG_ELEMS
                    else ""
                )
                lines.append(
                    f"  {label:<36} {row.label:<22} {str(spec):<40} per-device {share:,} elems{flag}"
                )
        if sharded_tensors:
            for label, (sharding, shape) in sharded_tensors.items():
                used = tuple(
                    axis for assignment in sharding.spec for axis in _mesh_axes_of(assignment)
                )
                assert len(used) == len(set(used)), (label, sharding.spec)
                n_shards = prod(self.mesh.shape[axis] for axis in used)
                assert prod(shape) % n_shards == 0, (label, shape, sharding.spec)
                flag = (
                    "   ⚠ FULLY REPLICATED (large)"
                    if n_shards == 1 and prod(shape) > REPLICATION_FLAG_ELEMS
                    else ""
                )
                lines.append(
                    f"  {label:<36} {'target/derived':<22} {str(sharding.spec):<40} "
                    f"per-device {prod(shape) // n_shards:,} elems{flag}"
                )
        if not_audited:
            lines.append(
                "  NOT AUDITED (absent from the per-tensor derivation audit): "
                + ", ".join(not_audited)
            )
        return "\n".join(lines)


def _assert_ns_row_tiles(row: PlacedRule, kinds: Mapping[str, int]) -> None:
    n = row.shard_count("stack")
    non_tiling = {kind: g for kind, g in kinds.items() if g % n != 0}
    assert not non_tiling, (
        f"stacked muon stages every kind at {row.label} (stack -> "
        f"{row.rule.get('stack')!r}, ÷{n}), which these kinds' stack lengths do not "
        "tile: "
        + ", ".join(f"{kind} (stacks {g})" for kind, g in sorted(non_tiling.items()))
        + ". There is no padding and no alternative split. Use a mesh the stacks tile,"
        " an explicit table whose ns_compute rows they do tile, or `impl: optax`."
    )


def assert_stacked_muon_component_staging(rules: PlacementRules) -> None:
    """The V/U stacked-muon staging claim, fired only for a run whose components
    optimizer is `impl: stacked` — the sole consumer of `components.ns_compute`."""
    _assert_ns_row_tiles(rules.components.ns_compute, rules.components.group_stack_lens)


def assert_stacked_muon_ci_staging(rules: PlacementRules, n_chunks: int) -> None:
    """The chunkwise-CI stacked-muon staging claim (every CI kind stacks `n_chunks`),
    fired only for a run whose CI optimizer is `impl: stacked` on a placed chunkwise
    fn — the sole consumer of the `ci_fn.*.ns_compute` rows."""
    for family in ("attention", "ffn", "input", "output"):
        weights: CIWeightPlacement = getattr(rules.ci_fn, family)
        _assert_ns_row_tiles(weights.ns_compute, {f"ci_fn/{family}": n_chunks})


def dropped_mesh_axes(source: PlacedRule, destination: PlacedRule, axes: Axes) -> frozenset[str]:
    """Mesh axes the destination row drops from the source row's coverage of `axes` —
    the gather axes of the source→destination transition. Size-1 axes are kept: jax's
    dot transpose demands the weight's reduced set equal the batch contraction's spec
    axes AS WRITTEN, size-1 included. Fail-closed: a destination that covers a mesh
    axis the source does not is not a gather and has no chained-reduced typing; it
    must go through a plain `reshard` instead."""
    source_axes = spec_axes(source.spec_for(axes))
    destination_axes = spec_axes(destination.spec_for(axes))
    assert destination_axes <= source_axes, (source.label, destination.label, axes)
    return source_axes - destination_axes


def materialize_reduced_weights(
    value: Array, *, source: PlacedRule, destination: PlacedRule, axes: Axes
) -> Array:
    """The optimizer-state→compute-weights gather as ONE typed reshard: the gathered
    mesh axes are typed `reduced`, so the backward's reduction is DEFERRED to this same
    boundary — cotangents ride the loop unreduced and reduce-scatter once here, never
    inside the scan (the chained-reduced spelling). The optimization_barrier pins any
    upstream compute-dtype cast to materialize BEFORE the gather, so the collective
    moves compute-dtype bytes, not the fp32 master's."""
    source.validate_shape(axes, value.shape)
    reduced = dropped_mesh_axes(source, destination, axes)
    spec = destination.spec_for(axes)
    return jax.sharding.reshard(
        jax.lax.optimization_barrier(value),
        NamedSharding(destination.mesh, P(*spec, reduced=reduced)),
    )


def component_stacks_to_compute_weights(
    components: ComponentStacks, placement: ComponentsPlacement
) -> ComponentStacks:
    """Materialize the compute-weight residents from optimizer ownership (dtype is the
    caller's: the step casts masters to compute dtype first). One typed reshard per
    stack, `reduced` over the gathered axes — see `materialize_reduced_weights`."""
    stacks: dict[str, tuple[Array, Array]] = {}
    for group, (vs, us) in components.stacks.items():
        stacks[group] = (
            materialize_reduced_weights(
                vs,
                source=placement.optimizer_state,
                destination=placement.compute_weights,
                axes=ComponentStacks.V_AXES,
            ),
            materialize_reduced_weights(
                us,
                source=placement.optimizer_state,
                destination=placement.compute_weights,
                axes=ComponentStacks.U_AXES,
            ),
        )
    return ComponentStacks(stacks=stacks, site_slots=components.site_slots)


def component_stacks_to_faithfulness_weights(
    components: ComponentStacks, placement: ComponentsPlacement
) -> ComponentStacks:
    """Materialize the declared faithfulness operands from optimizer ownership."""
    destination = placement.faithfulness_weights
    stacks: dict[str, tuple[Array, Array]] = {
        group: (
            jax.sharding.reshard(vs, destination.sharding_for(ComponentStacks.V_AXES)),
            jax.sharding.reshard(us, destination.sharding_for(ComponentStacks.U_AXES)),
        )
        for group, (vs, us) in components.stacks.items()
    }
    return ComponentStacks(stacks=stacks, site_slots=components.site_slots)


def constrain_faithfulness_deltas(
    deltas: dict[str, Array], placement: ComponentsPlacement
) -> dict[str, Array]:
    axes: Axes = ("stack", "d_out", "d_in")
    row = placement.faithfulness_deltas
    out: dict[str, Array] = {}
    for group, delta in deltas.items():
        row.validate_shape(axes, delta.shape)
        out[group] = jax.sharding.reshard(delta, row.sharding_for(axes))
    return out


# ── presets ──────────────────────────────────────────────────────────────────
# Named rule tables, one per deliberately supported layout. `stack` is the V/U
# semantic-group stack axis (components.ComponentStacks); `d` covers d_in and d_out via the
# per-tensor axes tuples. All presets share the activation waist rule (batch over the
# full data mesh) — that surface is layout-invariant (SPEC §4.1 pins).

# ÷N master rows keep `replicate` MINOR on whichever dim carries it: each compute shard
# is then the contiguous concat of its own replicate-group's ÷N optimizer shards, so
# optimizer-state→compute-weights (and its transpose) partitions as a pure all-gather /
# reduce-scatter over `replicate`. Replicate-major scatters the ÷N shards across the
# wrong compute groups and GSPMD legalizes both directions as a full grid-transpose
# collective-permute. Nested-axis order is semantics (PLACEMENT_DESIGN.md invariant 5);
# this constant is where the CI-fn master rows say so (the V/U matrix layout instead
# parks `replicate` minor on C — `_MATRIX_OWNER` below).
_ZERO1_DATA: tuple[MeshAxis, ...] = ("fsdp", "replicate")

# The activation waist stays replicate-major: it matches the live batch pins (token /
# bsc-source device_puts). Batch never reconstructs to an fsdp-only layout, so the order
# carries consistency weight only, no comms cost.
_BATCH: tuple[MeshAxis, ...] = ("replicate", "fsdp")

# describe() flags any audited tensor this large that ends up fully replicated — the
# design's precondition for the quiet unlisted-axis-replicates default (lesson 2).
REPLICATION_FLAG_ELEMS = 10_000_000


def _validate_group_stacks(
    desc: str,
    optimizer_state: PlacedRule,
    faithfulness_weights: PlacedRule,
    faithfulness_deltas: PlacedRule,
    sites: tuple[SiteSpec, ...],
) -> dict[str, int]:
    """The run's semantic-group census, refused unless every group tiles every component
    row. Placement is TOTAL and fallback-free: one table places all groups, and a group
    the table cannot place dies here, at construction, with the remedies spelled out —
    the user authors an appropriate config; the code never decides for them."""
    resolved = {
        group: (group_shape(specs), len(specs)) for group, specs in vu_groups(sites).items()
    }
    for row in (optimizer_state, faithfulness_weights, faithfulness_deltas):
        n = row.shard_count("stack")
        non_tiling = {group: g for group, (_, g) in resolved.items() if g % n != 0}
        assert not non_tiling, (
            f"{desc}: {row.label} shards the component stack axis "
            f"(stack -> {row.rule.get('stack')!r}, ÷{n}), which these semantic groups' "
            "stack lengths do not tile: "
            + ", ".join(f"{group} (stacks {g})" for group, g in sorted(non_tiling.items()))
            + f". There is no fallback. Use a mesh where every group's stack length "
            f"divides {n}, or a placement whose component rows do not shard the stack "
            f"axis (`sharding: zero1` rests every master intra-matrix)."
        )
    for (d_in, d_out, c), g in resolved.values():
        for row in (optimizer_state, faithfulness_weights):
            row.validate_shape(ComponentStacks.V_AXES, (g, d_in, c))
            row.validate_shape(ComponentStacks.U_AXES, (g, c, d_out))
        faithfulness_deltas.validate_shape(("stack", "d_out", "d_in"), (g, d_out, d_in))
    return {group: g for group, (_, g) in resolved.items()}


# ── consumed-axis vocabularies (the fail-closed rule-key check) ───────────────
# Every rule key must name a semantic axis some tensor actually CONSUMES at that row
# (the row's `spec_for` / `validate_shape` call sites). Lookup is exact-name with a
# quiet unlisted-axis-replicates default, so an unconsumed key — a typo'd axis in an
# explicit table — would otherwise silently replicate the tensor it meant to shard.
# These sets enumerate today's consumers; a new consumed axis is a new entry here,
# never a loosening. (The CI weight rows derive theirs from their declared tensor
# axes in `_bind.ci_placement`.)
_VU_AXES = frozenset(ComponentStacks.V_AXES) | frozenset(ComponentStacks.U_AXES)
_DELTA_AXES: frozenset[SemanticAxis] = frozenset({"stack", "d_out", "d_in"})
# The two activation waists double as the target linears' input/output contract, so they
# also cover FrozenAttn.core's head-split views ("position"/"q_head"/"kv_head"/"head_dim").
_WAIST_AXES: frozenset[SemanticAxis] = frozenset(
    {"batch", "position", "feature", "q_head", "kv_head", "head_dim"}
)
_COMPONENT_WAIST_AXES: frozenset[SemanticAxis] = frozenset({"batch", "position", "C"})
_CI_ACTIVATION_AXES: frozenset[SemanticAxis] = frozenset(
    {"batch", "position", "feature", "input", "q_head", "kv_head", "ffn_hidden", "C", "d_model"}
)
_CI_VECTOR_AXES: frozenset[SemanticAxis] = frozenset({"stack", "ffn_hidden", "d_model", "C"})
_EMBEDDING_AXES: frozenset[SemanticAxis] = frozenset({"vocab", "d_model"})
_NORMALIZATION_AXES: frozenset[SemanticAxis] = frozenset({"d_model", "head_dim"})
_POSITION_ENCODING_AXES: frozenset[SemanticAxis] = frozenset({"rope_frequency"})
_TARGET_LINEAR_PERSIST_AXES: frozenset[SemanticAxis] = frozenset({"layer", "d_out", "d_in"})
_TARGET_LINEAR_OPERAND_AXES: frozenset[SemanticAxis] = frozenset({"d_out", "d_in"})


# ── the placement table as a value ────────────────────────────────────────────
# One record, constructed as data: presets are literal instances, an explicit
# `PlacementTableConfig` parses into one (`_table_from_config`), and `_bind` turns a
# table into `PlacementRules` for a concrete mesh + site set.


@dataclass(frozen=True)
class _ComponentsTable:
    optimizer_state: Rule
    compute_weights: Rule
    faithfulness_weights: Rule
    faithfulness_deltas: Rule
    operands: Rule
    ns_compute: Rule


@dataclass(frozen=True)
class _CIWeightTable:
    optimizer_state: Rule
    compute_weights: Rule
    operands: Rule
    ns_compute: Rule


@dataclass(frozen=True)
class _CIFnTable:
    attention: _CIWeightTable
    ffn: _CIWeightTable
    input: _CIWeightTable
    output: _CIWeightTable
    vectors: Rule
    activations: Rule


@dataclass(frozen=True)
class _ActivationsTable:
    external: Rule
    component: Rule


@dataclass(frozen=True)
class _TargetWeightTable:
    persist: Rule
    operand: Rule


@dataclass(frozen=True)
class _TargetLinearTable:
    persist: Rule
    operand: Rule
    input: TargetActivationRef
    output: TargetActivationRef


@dataclass(frozen=True)
class _TargetComponentTable:
    input: TargetActivationRef
    output: TargetActivationRef


@dataclass(frozen=True)
class _TargetTable:
    embedding: _TargetWeightTable
    normalization: Rule
    position_encoding: Rule
    column: _TargetLinearTable
    row: _TargetLinearTable
    output: _TargetWeightTable
    intermediate: Rule
    component: _TargetComponentTable


@dataclass(frozen=True)
class PlacementTable:
    """The placement policy as a pure VALUE — rules not yet bound to a mesh or site set.
    Mirrors `PlacementRules` row for row and `PlacementTableConfig` field for field."""

    components: _ComponentsTable
    ci_fn: _CIFnTable
    activations: _ActivationsTable
    target: _TargetTable


_REPLICATED: Rule = {}

_STACK_OWNER: Rule = {"stack": "replicate", "d_in": "fsdp", "d_out": "fsdp", "C": "tp"}
# The intra-matrix ÷N layout parks the replicate shard on C (Adam is elementwise, so
# the master layout is free to choose): entry to the ÷fsdp compute weights is a pure
# minor-axis all-gather over `replicate` on C (exit, its reduce-scatter), and the
# matrix faithfulness rows ARE this layout, so that transition is the identity. The
# matrix delta row scatters both C contractions (tp and replicate) onto d_in.
_MATRIX_OWNER: Rule = {"d_in": "fsdp", "d_out": "fsdp", "C": ("tp", "replicate")}
_STACK_FAITHFULNESS_DELTA: Rule = {"stack": "replicate", "d_out": "fsdp"}
_MATRIX_FAITHFULNESS_DELTA: Rule = {"d_out": "fsdp", "d_in": ("tp", "replicate")}

# NS waypoint (`ns_compute`): ONE staging for every preset — each kind's stack split
# over the node axis (`replicate`), matrices whole per device, zero padding ever. The
# intra-node NS redundancy (a kind's shard replicated across the node's fsdp x tp plane)
# is accepted: NS is a sliver of the step, and comms-free beats FLOP-optimal. Under
# owner persistence the ingress is the identity on the stack axis (masters already
# stack-owned); under intra-matrix or replicated persistence it is a shard->shard hop
# chain (`muon_stacked.staging_hops`) — never a whole-fp32-stack-per-rank
# materialization (replicated staging would peak at the SUM of every kind's fp32
# stack, mesh-invariantly).
_NS_STACK_SPLIT: Rule = {"stack": "replicate"}

_OWNER_COMPONENTS = _ComponentsTable(
    optimizer_state=_STACK_OWNER,
    compute_weights={"d_in": "fsdp", "d_out": "fsdp", "C": "tp"},
    faithfulness_weights=_STACK_OWNER,
    faithfulness_deltas=_STACK_FAITHFULNESS_DELTA,
    operands={"C": "tp"},
    ns_compute=_NS_STACK_SPLIT,
)
# zero1 rests every master intra-matrix; its faithfulness rows ARE that master layout
# (the weights transition is the identity), so no persistence row shards the stack axis
# and every semantic group — any stack length — is placeable.
_ZERO1_COMPONENTS = _ComponentsTable(
    optimizer_state=_MATRIX_OWNER,
    compute_weights={"d_in": "fsdp", "d_out": "fsdp", "C": "tp"},
    faithfulness_weights=_MATRIX_OWNER,
    faithfulness_deltas=_MATRIX_FAITHFULNESS_DELTA,
    operands={"C": "tp"},
    ns_compute=_NS_STACK_SPLIT,
)
_DDP_COMPONENTS = _ComponentsTable(
    optimizer_state=_REPLICATED,
    compute_weights=_REPLICATED,
    faithfulness_weights=_REPLICATED,
    faithfulness_deltas=_REPLICATED,
    operands=_REPLICATED,
    ns_compute=_NS_STACK_SPLIT,
)

_ZERO1_CI = _CIFnTable(
    attention=_CIWeightTable(
        optimizer_state={"d_model": _ZERO1_DATA, "q_head": "tp", "kv_head": "tp"},
        compute_weights={"d_model": "fsdp", "q_head": "tp", "kv_head": "tp"},
        operands={"q_head": "tp", "kv_head": "tp"},
        ns_compute=_NS_STACK_SPLIT,
    ),
    ffn=_CIWeightTable(
        optimizer_state={"ffn_hidden": ("tp", "fsdp", "replicate")},
        compute_weights={"ffn_hidden": ("tp", "fsdp")},
        operands={"ffn_hidden": "tp"},
        ns_compute=_NS_STACK_SPLIT,
    ),
    input=_CIWeightTable(
        optimizer_state={"input": "tp", "d_model": _ZERO1_DATA},
        compute_weights={"input": "tp", "d_model": "fsdp"},
        operands={"input": "tp"},
        ns_compute=_NS_STACK_SPLIT,
    ),
    output=_CIWeightTable(
        optimizer_state={"d_model": _ZERO1_DATA, "C": "tp"},
        compute_weights={"d_model": "fsdp", "C": "tp"},
        operands={"C": "tp"},
        ns_compute=_NS_STACK_SPLIT,
    ),
    vectors={"ffn_hidden": "tp", "C": "tp"},
    activations={
        "batch": _BATCH,
        "input": "tp",
        "q_head": "tp",
        "kv_head": "tp",
        "ffn_hidden": "tp",
        "C": "tp",
    },
)
_STACK_OWNER_CI = replace(
    _ZERO1_CI,
    attention=_CIWeightTable(
        optimizer_state={"stack": "replicate", "d_model": "fsdp", "q_head": "tp", "kv_head": "tp"},
        compute_weights={"d_model": "fsdp", "q_head": "tp", "kv_head": "tp"},
        operands={"q_head": "tp", "kv_head": "tp"},
        ns_compute=_NS_STACK_SPLIT,
    ),
    ffn=_CIWeightTable(
        optimizer_state={"stack": "replicate", "ffn_hidden": ("tp", "fsdp")},
        compute_weights={"ffn_hidden": ("tp", "fsdp")},
        operands={"ffn_hidden": "tp"},
        ns_compute=_NS_STACK_SPLIT,
    ),
    input=_CIWeightTable(
        optimizer_state={"stack": "replicate", "input": "tp", "d_model": "fsdp"},
        compute_weights={"input": "tp", "d_model": "fsdp"},
        operands={"input": "tp"},
        ns_compute=_NS_STACK_SPLIT,
    ),
    output=_CIWeightTable(
        optimizer_state={"stack": "replicate", "d_model": "fsdp", "C": "tp"},
        compute_weights={"d_model": "fsdp", "C": "tp"},
        operands={"C": "tp"},
        ns_compute=_NS_STACK_SPLIT,
    ),
)
_DDP_CI_WEIGHTS = _CIWeightTable(
    optimizer_state=_REPLICATED,
    compute_weights=_REPLICATED,
    operands=_REPLICATED,
    ns_compute=_NS_STACK_SPLIT,
)
_DDP_CI = _CIFnTable(
    attention=_DDP_CI_WEIGHTS,
    ffn=_DDP_CI_WEIGHTS,
    input=_DDP_CI_WEIGHTS,
    output=_DDP_CI_WEIGHTS,
    vectors=_REPLICATED,
    activations={"batch": _BATCH},
)

_SHARDED_ACTIVATIONS = _ActivationsTable(
    external={"batch": _BATCH},
    component={"batch": _BATCH, "C": "tp"},
)

_SHARDED_TARGET = _TargetTable(
    embedding=_TargetWeightTable(persist={"d_model": "fsdp"}, operand=_REPLICATED),
    normalization=_REPLICATED,
    position_encoding=_REPLICATED,
    column=_TargetLinearTable(
        persist={"d_in": "fsdp", "d_out": "tp"},
        operand={"d_out": "tp"},
        input="external",
        output="intermediate",
    ),
    row=_TargetLinearTable(
        persist={"d_out": "fsdp", "d_in": "tp"},
        operand={"d_in": "tp"},
        input="intermediate",
        output="external",
    ),
    output=_TargetWeightTable(persist={"d_model": "fsdp"}, operand=_REPLICATED),
    intermediate={"batch": _BATCH, "feature": "tp", "q_head": "tp", "kv_head": "tp"},
    component=_TargetComponentTable(input="external", output="external"),
)
_REPLICATED_TARGET = _TargetTable(
    embedding=_TargetWeightTable(persist=_REPLICATED, operand=_REPLICATED),
    normalization=_REPLICATED,
    position_encoding=_REPLICATED,
    column=_TargetLinearTable(
        persist=_REPLICATED, operand=_REPLICATED, input="external", output="intermediate"
    ),
    row=_TargetLinearTable(
        persist=_REPLICATED, operand=_REPLICATED, input="intermediate", output="external"
    ),
    output=_TargetWeightTable(persist=_REPLICATED, operand=_REPLICATED),
    intermediate={"batch": _BATCH},
    component=_TargetComponentTable(input="external", output="external"),
)

# The built-in tables: `zero1` (intra-matrix ÷N over the full data mesh — no row shards
# the stack axis, so every semantic group is placeable; ~equivalent comms to `owner`
# under elementwise optimizers), `owner` (stack ÷replicate, d ÷fsdp — the muon-motivated
# D4-amended layout, node-local NS; a stack that doesn't tile ÷replicate refuses at
# construction), `ddp` (everything replicated — single-node / small-model runs).
PRESETS: Mapping[str, PlacementTable] = {
    "owner": PlacementTable(
        components=_OWNER_COMPONENTS,
        ci_fn=_STACK_OWNER_CI,
        activations=_SHARDED_ACTIVATIONS,
        target=_SHARDED_TARGET,
    ),
    "zero1": PlacementTable(
        components=_ZERO1_COMPONENTS,
        ci_fn=_ZERO1_CI,
        activations=_SHARDED_ACTIVATIONS,
        target=_SHARDED_TARGET,
    ),
    "ddp": PlacementTable(
        components=_DDP_COMPONENTS,
        ci_fn=_DDP_CI,
        activations=_SHARDED_ACTIVATIONS,
        target=_REPLICATED_TARGET,
    ),
}
PRESET_NAMES = tuple(PRESETS)


def _bind(
    desc: str,
    table: PlacementTable,
    mesh: Mesh | AbstractMesh,
    sites: tuple[SiteSpec, ...],
) -> PlacementRules:
    """The one constructor: bind the table's rules to the mesh, stamp the printed
    labels, fail closed on rule keys no tensor consumes, and refuse any semantic group
    the component rows cannot place."""
    components = table.components
    target = table.target

    assert not target.normalization, (
        "target.normalization is a replicated execution row; sharding it needs an explicit "
        "persist→operand lifecycle"
    )
    assert not target.position_encoding, (
        "target.position_encoding is replicated; no sharded execution is implemented"
    )

    def row(label: str, rule: Rule, consumes: frozenset[SemanticAxis]) -> PlacedRule:
        unconsumed = set(rule) - consumes
        assert not unconsumed, (
            f"{desc}: row {label!r} keys {sorted(unconsumed)} name no semantic axis any "
            f"tensor consumes at this row (consumable: {sorted(consumes)}); an unconsumed "
            f"key silently replicates the tensor it meant to shard"
        )
        return PlacedRule(mesh=mesh, label=label, rule=rule)

    def ns_row(label: str, rule: Rule) -> PlacedRule:
        assert set(rule) <= {"stack"}, (
            f"{desc}: {label} may assign only `stack` (got {sorted(rule)}) — the batched NS"
            f" needs whole matrices per device (`ns_staging_sharding`), and a"
            f" matrix-axis-carrying waypoint re-triggers the SPMD"
            f" involuntary-full-rematerialization fallback"
        )
        return PlacedRule(mesh=mesh, label=label, rule=rule)

    optimizer_state_row = row("components/optimizer_state", components.optimizer_state, _VU_AXES)
    faithfulness_weights_row = row(
        "components/faithfulness_weights", components.faithfulness_weights, _VU_AXES
    )
    faithfulness_deltas_row = row(
        "components/faithfulness_deltas", components.faithfulness_deltas, _DELTA_AXES
    )
    group_stack_lens = _validate_group_stacks(
        desc,
        optimizer_state_row,
        faithfulness_weights_row,
        faithfulness_deltas_row,
        sites,
    )
    ns_compute_row = ns_row("components/ns_compute", components.ns_compute)
    external_row = row("activations/external", table.activations.external, _WAIST_AXES)
    intermediate_row = row("target/intermediate", target.intermediate, _WAIST_AXES)

    def activation(ref: TargetActivationRef) -> PlacedRule:
        match ref:
            case "external":
                return external_row
            case "intermediate":
                return intermediate_row

    def ci_placement(
        label: str, rules: _CIWeightTable, tensor_axes: tuple[Axes, ...]
    ) -> CIWeightPlacement:
        # This family's consumable keys ARE its declared tensor axes (the operand rows
        # apply to the compute residents, whose stack axis the scan has already split).
        stored: frozenset[SemanticAxis] = frozenset(axis for axes in tensor_axes for axis in axes)
        placement = CIWeightPlacement(
            optimizer_state=row(f"ci_fn/{label}.optimizer_state", rules.optimizer_state, stored),
            compute_weights=row(f"ci_fn/{label}.compute_weights", rules.compute_weights, stored),
            operands=row(f"ci_fn/{label}.operands", rules.operands, stored.difference({"stack"})),
            ns_compute=ns_row(f"ci_fn/{label}.ns_compute", rules.ns_compute),
        )
        for axes in tensor_axes:
            placement.optimizer_state.spec_for(axes)
        return placement

    def target_linear(label: str, declaration: _TargetLinearTable) -> TargetLinearPlacement:
        return TargetLinearPlacement(
            persist=row(
                f"target/{label}.persist", declaration.persist, _TARGET_LINEAR_PERSIST_AXES
            ),
            operand=row(
                f"target/{label}.operand", declaration.operand, _TARGET_LINEAR_OPERAND_AXES
            ),
            input=activation(declaration.input),
            output=activation(declaration.output),
        )

    def target_weight(label: str, declaration: _TargetWeightTable) -> TargetWeightPlacement:
        return TargetWeightPlacement(
            persist=row(f"target/{label}.persist", declaration.persist, _EMBEDDING_AXES),
            operand=row(f"target/{label}.operand", declaration.operand, _EMBEDDING_AXES),
        )

    return PlacementRules(
        mesh=mesh,
        components=ComponentsPlacement(
            optimizer_state=optimizer_state_row,
            compute_weights=row("components/compute_weights", components.compute_weights, _VU_AXES),
            faithfulness_weights=faithfulness_weights_row,
            faithfulness_deltas=faithfulness_deltas_row,
            operands=row(
                "components/operands", components.operands, _VU_AXES.difference({"stack"})
            ),
            ns_compute=ns_compute_row,
            group_stack_lens=group_stack_lens,
        ),
        ci_fn=CIFnPlacement(
            attention=ci_placement(
                "attention",
                table.ci_fn.attention,
                (
                    ("stack", "q_head", "d_model"),
                    ("stack", "kv_head", "d_model"),
                    ("stack", "d_model", "q_head"),
                ),
            ),
            ffn=ci_placement(
                "ffn",
                table.ci_fn.ffn,
                (("stack", "d_model", "ffn_hidden"), ("stack", "ffn_hidden", "d_model")),
            ),
            input=ci_placement("input", table.ci_fn.input, (("stack", "input", "d_model"),)),
            output=ci_placement("output", table.ci_fn.output, (("stack", "d_model", "C"),)),
            vectors=row("ci_fn/vectors", table.ci_fn.vectors, _CI_VECTOR_AXES),
            activations=row("ci_fn/activations", table.ci_fn.activations, _CI_ACTIVATION_AXES),
        ),
        activations=ActivationsPlacement(
            external=external_row,
            component=row(
                "activations/component", table.activations.component, _COMPONENT_WAIST_AXES
            ),
        ),
        target=TargetPlacement(
            embedding=target_weight("embedding", target.embedding),
            normalization=row("target/normalization", target.normalization, _NORMALIZATION_AXES),
            position_encoding=row(
                "target/position_encoding", target.position_encoding, _POSITION_ENCODING_AXES
            ),
            column=target_linear("column", target.column),
            row=target_linear("row", target.row),
            output=target_weight("output", target.output),
            intermediate=intermediate_row,
            component=TargetComponentLinearPlacement(
                input=activation(target.component.input),
                output=activation(target.component.output),
            ),
        ),
    )


def _rule(config: RuleConfig) -> Rule:
    """YAML lists become ordered tuples — nested-axis ORDER is semantics
    (PLACEMENT_DESIGN.md invariant 5)."""
    return {axis: tuple(v) if isinstance(v, list) else v for axis, v in config.items()}


def _ci_weight_table(config: CIWeightPlacementConfig) -> _CIWeightTable:
    return _CIWeightTable(
        optimizer_state=_rule(config.optimizer_state),
        compute_weights=_rule(config.compute_weights),
        operands=_rule(config.operands),
        ns_compute=_rule(config.ns_compute),
    )


def _target_linear_table(config: TargetLinearPlacementConfig) -> _TargetLinearTable:
    return _TargetLinearTable(
        persist=_rule(config.persist),
        operand=_rule(config.operand),
        input=config.input,
        output=config.output,
    )


def _target_weight_table(config: TargetWeightPlacementConfig) -> _TargetWeightTable:
    return _TargetWeightTable(persist=_rule(config.persist), operand=_rule(config.operand))


def _table_from_config(config: PlacementTableConfig) -> PlacementTable:
    components = config.components
    return PlacementTable(
        components=_ComponentsTable(
            optimizer_state=_rule(components.optimizer_state),
            compute_weights=_rule(components.compute_weights),
            faithfulness_weights=_rule(components.faithfulness_weights),
            faithfulness_deltas=_rule(components.faithfulness_deltas),
            operands=_rule(components.operands),
            ns_compute=_rule(components.ns_compute),
        ),
        ci_fn=_CIFnTable(
            attention=_ci_weight_table(config.ci_fn.attention),
            ffn=_ci_weight_table(config.ci_fn.ffn),
            input=_ci_weight_table(config.ci_fn.input),
            output=_ci_weight_table(config.ci_fn.output),
            vectors=_rule(config.ci_fn.vectors),
            activations=_rule(config.ci_fn.activations),
        ),
        activations=_ActivationsTable(
            external=_rule(config.activations.external),
            component=_rule(config.activations.component),
        ),
        target=_TargetTable(
            embedding=_target_weight_table(config.target.embedding),
            normalization=_rule(config.target.normalization),
            position_encoding=_rule(config.target.position_encoding),
            column=_target_linear_table(config.target.column),
            row=_target_linear_table(config.target.row),
            output=_target_weight_table(config.target.output),
            intermediate=_rule(config.target.intermediate),
            component=_TargetComponentTable(
                input=config.target.component.input,
                output=config.target.component.output,
            ),
        ),
    )


def from_config(
    spec: str | PlacementTableConfig, mesh: Mesh | AbstractMesh, sites: tuple[SiteSpec, ...]
) -> PlacementRules:
    """The configured placement spec (`runtime.sharding`) + the run's resolved site set →
    the TOTAL placement policy: a preset name, or an explicit table (already
    parse-validated by `PlacementTableConfig` — closed row vocabulary). Construction is
    the decision point: a semantic group the component rows cannot place refuses, with
    the remedies spelled out — there is no fallback arm. The same construction serves
    the run's own topology (config build via `sharding.hsdp_abstract_mesh`,
    pre-submission; the composition roots) and a consumer re-placing a finished run on
    its own mesh."""
    match spec:
        case str():
            assert spec in PRESETS, f"unknown placement preset {spec!r} (have {PRESET_NAMES})"
            table, desc = PRESETS[spec], f"sharding preset {spec!r}"
        case PlacementTableConfig():
            table, desc = _table_from_config(spec), "explicit placement table"
    return _bind(desc, table, mesh, sites)


# ── component-stack placement (lookup + boundary validation) ─────────────────
# `ComponentStacks` is placement-free; this is where its persistence placement is read
# off the rules.


def _validate_component_stacks(stacks: ComponentStacks, rules: PlacementRules) -> None:
    """BOUNDARY VALIDATION of the build-time group census against the stacks actually
    held — validation of received data at a trust boundary. Disagreement between the two
    worlds (different semantic groups, or a different stack length for one) is an
    upstream bug and dies here."""
    lengths = stacks.group_lengths()
    census = rules.components.group_stack_lens
    assert census.keys() == lengths.keys(), (
        f"placement was built for component groups {sorted(census)}; "
        f"these stacks hold {sorted(lengths)}"
    )
    for group, g in lengths.items():
        assert census[group] == g, (
            f"placement expects a {census[group]}-stack for "
            f"component group {group!r}; these stacks hold {g}"
        )


def component_stacks_shardings(
    stacks: ComponentStacks[Array], rules: PlacementRules
) -> ComponentStacks[NamedSharding]:
    """The V/U persistence placement (semantic axes `V (stack, d_in, C)` /
    `U (stack, C, d_out)`; `owner` is the hybrid HSDP layout of the 2026-07-15 SPEC D4
    amendment). Boundary-validated by `_validate_component_stacks`; divisibility was
    validated at rules construction against these same shapes."""
    _validate_component_stacks(stacks, rules)
    lengths = stacks.group_lengths()
    row = rules.components.optimizer_state
    placed: dict[str, tuple[NamedSharding, NamedSharding]] = {}
    for group, (Vs, _) in stacks.stacks.items():
        assert Vs.shape[0] == lengths[group], (
            "stacks disagree with the static site_slots index — the audit would lie"
        )
        placed[group] = (
            row.sharding_for(ComponentStacks.V_AXES),
            row.sharding_for(ComponentStacks.U_AXES),
        )
    return ComponentStacks(stacks=placed, site_slots=stacks.site_slots)


# --------------- applying rows: frozen-target linears and activations ---------------


def constrain_weight(weight: Array, row: PlacedRule | None, axes: Axes) -> Array:
    if row is None:
        return weight
    row.validate_shape(axes, weight.shape)
    return jax.sharding.reshard(weight, row.sharding_for(axes))


def materialize_stored_weight(
    weight: Array,
    persist: PlacedRule,
    operand: PlacedRule,
    *,
    axes: Axes,
) -> Array:
    persist.validate_shape(axes, weight.shape)
    return constrain_weight(weight, operand, axes)


def constrain_activation(x: Array, row: PlacedRule | None) -> Array:
    if row is None:
        return x
    axes = activation_axes(x.ndim, "feature")
    row.validate_shape(axes, x.shape)
    return jax.sharding.reshard(x, row.sharding_for(axes))


def placed_target_linear(x: Array, weight: Array, placement: TargetLinearPlacement | None) -> Array:
    if placement is None:
        return x @ weight.T
    return placed_linear(x, weight.T, target_linear_plan(x, placement))


def target_linear_plan(x: Array, placement: TargetLinearPlacement) -> LinearPlan:
    assert isinstance(placement.input.mesh, jax.sharding.Mesh)
    weight_axes = ("d_out", "d_in")
    axes = activation_axes(x.ndim, "feature")
    operand = placement.operand.spec_for(weight_axes)
    return LinearPlan(
        mesh=placement.input.mesh,
        input=placement.input.spec_for(axes),
        operand_input=placement.input.spec_for(axes),
        resident_weight=P(*reversed(placement.persist.spec_for(weight_axes))),
        operand=P(*reversed(operand)),
        output=placement.output.spec_for(axes),
        weight_reduced=frozenset(),
    )


def component_stacks_audit(
    stacks: ComponentStacks, rules: PlacementRules
) -> dict[str, tuple[PlacedRule, Axes, tuple[int, ...]]]:
    """`{label: (row, axes, shape)}` for `rules.describe(...)` — the startup audit
    (boundary-validated); group lengths from the static `site_slots` (works on
    eval-shape trees)."""
    _validate_component_stacks(stacks, rules)
    row = rules.components.optimizer_state
    out: dict[str, tuple[PlacedRule, Axes, tuple[int, ...]]] = {}
    for group, (Vs, Us) in stacks.stacks.items():
        out[f"V {group}"] = (row, ComponentStacks.V_AXES, Vs.shape)
        out[f"U {group}"] = (row, ComponentStacks.U_AXES, Us.shape)
    return out
