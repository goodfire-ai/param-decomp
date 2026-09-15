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
  `activations.{external, masked_external, component}`, and
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
mesh and resolved site set — there are NO fallback arms; every group takes the table's
one set of rows. A stack length that does not tile the stack-sharding persist rows is
placed by PADDING the persist stack with trailing all-zero slots (an enumerated
`StackCensus.stack_pad`, stripped at the entry — compute never sees pads, and the entry
gather moves only real slots); any
other unplaceable group refuses at construction with the explicit remedies. The census
flows downward as data — the V/U groups' at `ComponentsPlacement.group_census`, the
chunkwise CI fn's chunk stack at `CIFnPlacement.chunks` (resolved where the CI arch meets
the rows, `ci_fn.resolve_ci_placement`); consumers validate what they receive
(PLACEMENT_DESIGN.md, "Presets").

The rule language is deliberately WEAK: exact-name lookup (semantic axis name → mesh
axes), no patterns, no conditionals, no expressions. Weird cases get a literal spec
override on a named row, not a smarter language. Load-bearing surfaces only: rules pin
optimizer-state trees, phase entries, and the activation waist; GSPMD propagates between
pins as today.
"""

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from math import lcm, prod
from typing import Literal, cast, get_args

import jax
from jax.sharding import AbstractMesh, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array

from param_decomp.core.axes import Axes, MeshAssignment, MeshAxis, SemanticAxis
from param_decomp.core.components import (
    EXPERT_U_AXES,
    EXPERT_V_AXES,
    ComponentStacks,
    Dense,
    ExpertBlocked,
    Factorization,
    SiteSpec,
    activation_axes,
    vu_groups,
)
from param_decomp.core.configs import (
    CIWeightPlacementConfig,
    PlacementPresetName,
    PlacementSpec,
    PlacementTableConfig,
    RuleConfig,
    SequenceSharding,
    TargetActivationRef,
    TargetLinearPlacementConfig,
    TargetWeightPlacementConfig,
)
from param_decomp.core.linear_plan import (
    ExpertBlockLinearPlan,
    ExpertContraction,
    LinearPlan,
    placed_linear,
    spec_axes,
)

Rule = Mapping[SemanticAxis, MeshAssignment]

CIWeightFamily = Literal["attention", "ffn", "input", "output"]
"""The CI transformer's weight families — the closed vocabulary `CIFnRows.linear_plan`
dispatches on (one `CIWeightPlacement` each)."""

# Muon's batched Newton-Schulz sees every leaf as a canonical `[g, rows<=cols]` stack;
# an `ns_compute` row declares that stack's split verbatim (`ns_staging_sharding`), so
# the row is orientation-blind by construction. Only `stack` may carry an assignment
# (enforced at `_bind`): the NS Gram contraction needs whole matrices per device — a
# sharded matrix axis is an explicit-mode type error, and staging at a
# matrix-axis-carrying layout (e.g. the persist row verbatim) re-triggers the SPMD
# involuntary-full-rematerialization fallback (task 577).


def batch_axes(mesh: Mesh | AbstractMesh) -> tuple[MeshAxis, ...]:
    """The mesh axes the always-leading batch dimension shards over: every axis but
    `tp`, which holds replicas — `(replicate, fsdp)` on the three-axis mesh, `(data,)`
    on the resident two-axis mesh. Every batch consumer derives its spec here; a
    hardcoded axis tuple would silently mis-place batches on the other mesh shape."""
    return cast(tuple[MeshAxis, ...], tuple(axis for axis in mesh.axis_names if axis != "tp"))


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
            unknown = set(assignment) - mesh_axes
            assert not unknown, (
                f"placement rule {self.label!r}: axis {axis!r} maps to unknown mesh "
                f"axes {sorted(unknown)} (mesh has {sorted(mesh_axes)})"
            )
            assert len(assignment) == len(set(assignment)), (
                f"placement rule {self.label!r}: axis {axis!r} repeats a mesh axis ({assignment})"
            )
        # NOTE: one mesh axis MAY appear under several semantic names in a rule
        # (`d_in -> fsdp, d_out -> fsdp`) because no single tensor carries both;
        # uniqueness is a per-TENSOR invariant, checked in `spec_for`.

    def assignment(self, axis: SemanticAxis) -> MeshAssignment:
        """The mesh axes `axis` shards over at this row — `()` for an unlisted name (the
        quiet replicated default lives here and nowhere else)."""
        return self.rule.get(axis, ())

    def spec_for(self, axes: Axes) -> P:
        """The derived PartitionSpec for a tensor with semantic `axes` at this row."""
        entries = tuple(self.assignment(name) for name in axes)
        used = [axis for entry in entries for axis in entry]
        assert len(used) == len(set(used)), (
            f"{self.label}: tensor axes {axes} derive a spec using a mesh axis twice ({entries})"
        )
        return P(*entries)

    def sharding_for(self, axes: Axes) -> NamedSharding:
        return NamedSharding(self.mesh, self.spec_for(axes))

    def shard_count(self, axis_name: SemanticAxis) -> int:
        """How many ways `axis_name` is split at this row (1 = unsharded)."""
        return prod(self.mesh.shape[a] for a in self.assignment(axis_name))

    def validate_shape(self, axes: Axes, shape: tuple[int, ...]) -> None:
        """Fail-fast divisibility: every dim must tile its assigned mesh-axis product."""
        assert len(axes) == len(shape), (self.label, axes, shape)
        for name, dim in zip(axes, shape, strict=True):
            n = self.shard_count(name)
            assert dim % n == 0, (
                f"{self.label}: semantic axis {name!r} (dim {dim}) does not tile its mesh "
                f"assignment ({self.assignment(name)!r} = ÷{n})"
            )


def ns_staging_sharding(row: PlacedRule, mesh: Mesh | AbstractMesh) -> NamedSharding:
    """One semantic kind's muon-NS staging waypoint: the row's stack split verbatim,
    matrices whole per device. Only a stacked-muon optimizer consumes it, so its tiling
    claim fires at that consumer (`assert_stacked_muon_*_staging`), not at rules
    construction — a non-muon run keeps any-stack-length placement."""
    assert set(row.rule) <= {"stack"}, (row.label, sorted(row.rule))
    return NamedSharding(mesh, P(row.assignment("stack"), None, None))


@dataclass(frozen=True)
class StackCensus:
    """One persist stack as placement resolved it: its REAL length and the persist-layer
    pad — the trailing all-zero stack slots that make the persist stack tile every
    stack-sharding persist row (`stack_pad = 0` wherever the real length already tiles).
    Pads exist between the persist layer and the entry boundary that strips them
    (`strip_stack_pad`); compute never sees them. The V/U semantic groups resolve one
    each (`GroupCensus`); the chunkwise CI fn resolves one for its chunk stack
    (`CIFnPlacement.chunks`)."""

    stack_len: int
    stack_pad: int

    @property
    def padded_stack_len(self) -> int:
        """The persist arrays' stack extent — what masters and moments actually carry."""
        return self.stack_len + self.stack_pad


@dataclass(frozen=True)
class GroupCensus(StackCensus):
    """One semantic V/U group's stack census plus its factorization (the source of its
    leaf axes and shapes)."""

    factorization: Factorization

    @property
    def ns_stack_len(self) -> int:
        """The stack length of the group's CANONICAL `[g, rows, cols]` muon-NS view:
        NS stages the PADDED persist arrays (zero pad matrices are numerically inert
        under Newton-Schulz), and stacked NS folds an expert axis into the batch axis
        (`muon_stacked._canonicalize`), so an expert-blocked group's canonical stack is
        `padded_stack_len * n_experts`."""
        match self.factorization:
            case Dense():
                return self.padded_stack_len
            case ExpertBlocked(n_experts=n_experts):
                return self.padded_stack_len * n_experts


@dataclass(frozen=True)
class ComponentsPlacement:
    """The declared lifecycle of trainable V/U weights. ONE set of rows places every
    semantic group — a group the rows cannot place refused at construction, so no
    per-group dispatch exists anywhere downstream. `group_census` is the resolved
    site-set census: the consumer boundary re-checks its arrays against it
    (`_validate_component_stacks`), and every transition reads each group's leaf axes
    off its factorization. `ns_compute` is the muon-NS staging waypoint
    (`ns_staging_sharding`)."""

    optimizer_state: PlacedRule
    compute_weights: PlacedRule
    faithfulness_weights: PlacedRule
    faithfulness_deltas: PlacedRule
    operands: PlacedRule
    ns_compute: PlacedRule
    group_census: Mapping[str, GroupCensus]

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
class CIMoEPlacement:
    """The MoE CI transformer's expert-carrying weight families: the per-block
    concat-wide expert FFN (leaves `[stack, layer, expert, d_model, ffn_hidden]` /
    `[stack, layer, expert, ffn_hidden, d_model]` — `layer` is the bank's covered
    target layer, replicated) and the per-slot fused expert heads
    (`[stack, expert, ffn_hidden, C_block]`). The expert axis co-locates with the
    target's expert shard, so the CI fn's routed FFN compute and its narrow heads ride
    the same expert-parallel schedule as the target with no rank ever touching another
    shard's expert weights."""

    expert_ffn: CIWeightPlacement
    expert_head: CIWeightPlacement


@dataclass(frozen=True)
class CIFnRows:
    """The CI fn's placement rows as the table binds them — before any arch is known.
    `resolve_ci_placement` pairs them with the arch's chunk-stack census into the
    `CIFnPlacement` every placed CI fn carries."""

    attention: CIWeightPlacement
    ffn: CIWeightPlacement
    input: CIWeightPlacement
    output: CIWeightPlacement
    vectors: PlacedRule
    activations: PlacedRule
    moe: CIMoEPlacement | None
    """The MoE chunkwise arch's extra families. `None` ⟺ the table carries none (every
    current preset); `resolve_ci_placement` refuses pairing an MoE arch with it."""

    @property
    def chunk_persist_rows(self) -> tuple[PlacedRule, ...]:
        """The rows every chunkwise arch's stacked leaves REST at — the four weight
        families' masters and the vector leaves — whose stack cuts the chunk-stack
        census must tile. The MoE families' masters join for the MoE arch."""
        return (
            self.attention.optimizer_state,
            self.ffn.optimizer_state,
            self.input.optimizer_state,
            self.output.optimizer_state,
            self.vectors,
        )

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
class CIFnPlacement(CIFnRows):
    """The chunkwise CI fn's RESOLVED placement: the rows plus the chunk-stack census
    (`ci_fn.resolve_ci_placement` — the one constructor, at run assembly). Every placed
    chunkwise fn carries one (`PlacedCIFn.placement`); the consumer boundaries validate
    the fn they hold against `chunks` and never re-decide it."""

    chunks: StackCensus

    @staticmethod
    def resolved(rows: CIFnRows, chunks: StackCensus) -> "CIFnPlacement":
        return CIFnPlacement(**{f.name: getattr(rows, f.name) for f in fields(rows)}, chunks=chunks)


@dataclass(frozen=True)
class ActivationsPlacement:
    """The component linear's replicated public waist and C-sharded internal waist.

    `masked_external` is the MASKED passes' between-blocks residual row: identical to
    `external` (the same object) unless the run authors `sequence_sharding:
    sequence_parallel`, which adds `position -> tp` — the masked forwards then carry the
    residual position-sharded between blocks, gathering to `external` at each block entry
    and reduce-scattering back at each block exit, while clean forwards keep `external`
    everywhere."""

    external: PlacedRule
    masked_external: PlacedRule
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
    ci_fn: CIFnRows
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

    def expert_component_linear_plan(
        self,
        target: TargetLinearPlacement,
        contraction: ExpertContraction,
        factor: Literal["V", "U"],
        activation_ndim: int,
    ) -> ExpertBlockLinearPlan:
        """One expert-blocked factor linear of a target-native site, planned from the
        rows. Fused-width activations are expert-major, so their expert-blocked view
        `[*lead, expert, block]` is the flat spec with the fused assignment moved whole
        onto the expert axis and the block dim replicated (`_expert_view`); the
        component waist's expert view derives from the component row directly (its
        `expert` key — present in expert-carrying tables)."""
        assert isinstance(self.mesh, Mesh), type(self.mesh)
        external_axes = activation_axes(activation_ndim, "feature")
        component_axes = activation_axes(activation_ndim, "C")
        expert_component_axes: Axes = (*component_axes[:-1], "expert", "C_block")

        def expert_view(spec: P) -> P:
            return P(*spec, None)

        component_spec = self.activations.component.spec_for(expert_component_axes)
        match contraction, factor:
            case ("fused_output", "V"):
                input_spec = target.input.spec_for(external_axes)
                operand_input_spec = self.target.component.input.spec_for(external_axes)
                output_spec = component_spec
                weight_axes = EXPERT_V_AXES[1:]
            case ("fused_output", "U"):
                input_spec = component_spec
                operand_input_spec = component_spec
                output_spec = expert_view(target.output.spec_for(external_axes))
                weight_axes = EXPERT_U_AXES[1:]
            case ("fused_input", "V"):
                input_spec = expert_view(target.input.spec_for(external_axes))
                operand_input_spec = input_spec
                output_spec = component_spec
                weight_axes = EXPERT_V_AXES[1:]
            case ("fused_input", "U"):
                input_spec = component_spec
                operand_input_spec = component_spec
                output_spec = target.output.spec_for(external_axes)
                weight_axes = EXPERT_U_AXES[1:]
        return ExpertBlockLinearPlan(
            mesh=self.mesh,
            contraction=contraction,
            factor=factor,
            input=input_spec,
            operand_input=operand_input_spec,
            resident_weight=self.components.compute_weights.spec_for(weight_axes),
            operand=self.components.operands.spec_for(weight_axes),
            output=output_spec,
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
                ", ".join(
                    f"{axis}->{','.join(assignment) or '()'}"
                    for axis, assignment in row.rule.items()
                )
                if row.rule
                else "(replicated)"
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
                n_shards = prod(self.mesh.shape[axis] for axis in spec_axes(sharding.spec))
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
        f"{row.assignment('stack')!r}, ÷{n}), which these kinds' stack lengths do not "
        "tile: "
        + ", ".join(f"{kind} (stacks {g})" for kind, g in sorted(non_tiling.items()))
        + ". NS stages the persist stacks as they rest (persist pads included); there is"
        " no NS-only padding and no alternative split. Use a mesh the stacks tile, an"
        " explicit table whose ns_compute rows they do tile, or an Adam-family optimizer."
    )


def assert_stacked_muon_component_staging(rules: PlacementRules) -> None:
    """The V/U stacked-muon staging claims, fired only for a run whose components
    optimizer is muon — the sole consumer of `components.ns_compute`.

    Two claims. The NS split must tile each group's CANONICAL stack length
    (`GroupCensus.ns_stack_len`, the expert axis folded in). And the MASTER layout must
    not split a within-block matrix axis across a staging axis: Newton-Schulz needs
    whole matrices per device, so such a layout silently pays a per-step staging
    round-trip of the full master bytes across those axes — muon pairs with the
    owner-flavored layouts (block axes co-resident with the stack cut), while
    Adam-family optimizers stay layout-agnostic. Fail-closed, no warning arm."""
    census = rules.components.group_census
    _assert_ns_row_tiles(
        rules.components.ns_compute,
        {name: entry.ns_stack_len for name, entry in census.items()},
    )
    masters = rules.components.optimizer_state
    staging = rules.components.ns_compute.assignment("stack")
    for name, entry in census.items():
        match entry.factorization:
            case Dense():
                block_axes: tuple[SemanticAxis, ...] = ("d_in", "C", "d_out")
            case ExpertBlocked():
                block_axes = ("d_in", "C_block", "d_out")
        offending = {
            axis: masters.assignment(axis)
            for axis in block_axes
            if set(masters.assignment(axis)) & set(staging)
        }
        assert not offending, (
            f"stacked muon refuses this master layout: group {name!r}'s within-block "
            f"matrix axes ride the NS staging axes "
            f"({', '.join(f'{axis} -> {assignment!r}' for axis, assignment in sorted(offending.items()))}"
            f" vs ns_compute stack -> {staging!r}). "
            f"Newton-Schulz needs whole matrices per device, so staging would "
            f"round-trip the full master bytes across those axes every step. Use an "
            f"owner-flavored placement (`sharding: owner` / `owner-replicated-resident` "
            f"/ `owner-replicated-resident-moe` — masters stack-cut, blocks whole), or "
            f"an Adam-family optimizer (layout-agnostic)."
        )


def assert_stacked_muon_ci_staging(placement: CIFnPlacement) -> None:
    """The chunkwise-CI stacked-muon staging claim (every CI kind stacks the PADDED
    chunk count — NS stages the persist stacks as they rest), fired only for a run whose
    CI optimizer is muon on a placed chunkwise fn — the sole consumer of the
    `ci_fn.*.ns_compute` rows."""
    stacks = placement.chunks.padded_stack_len
    for family in ("attention", "ffn", "input", "output"):
        weights: CIWeightPlacement = getattr(placement, family)
        _assert_ns_row_tiles(weights.ns_compute, {f"ci_fn/{family}": stacks})


def assert_stacked_muon_moe_ci_staging(placement: CIFnPlacement, n_experts: int) -> None:
    """`assert_stacked_muon_ci_staging`'s MoE sibling: the dense families stack the padded
    chunk count, and the expert families' tiled quantity is the CANONICAL NS stack
    length — the expert axis folds into the batch axis (`muon_stacked._canonicalize`),
    so each `[n_chunks, E, a, b]` bank and fused-head leaf stages `n_chunks·E`
    matrices."""
    assert_stacked_muon_ci_staging(placement)
    moe = placement.moe
    assert moe is not None, "the MoE CI staging claim needs the table's moe families"
    folded = placement.chunks.padded_stack_len * n_experts
    _assert_ns_row_tiles(moe.expert_ffn.ns_compute, {"ci_fn/moe.expert_ffn": folded})
    _assert_ns_row_tiles(moe.expert_head.ns_compute, {"ci_fn/moe.expert_head": folded})


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
    value: Array,
    *,
    census: StackCensus,
    source: PlacedRule,
    destination: PlacedRule,
    axes: Axes,
) -> Array:
    """One stacked leaf's persist→compute entry as typed reshards: the gathered mesh
    axes are typed `reduced`, so the backward's reduction is DEFERRED to this same
    boundary — cotangents ride the loop unreduced and reduce-scatter once here, never
    inside the scan (the chained-reduced spelling). The optimization_barrier pins any
    upstream compute-dtype cast to materialize BEFORE the collectives, so they move
    compute-dtype bytes, not the fp32 master's.

    Two routes, enumerated on the census. A pad-free stack gathers along its persist
    layout in one reshard. A padded stack must not gather its pads — zero slots riding
    the step's largest collective and its largest transient — and the typed slice cannot
    drop them while the stack axis is cut, so the stack first hops to its
    `padded_entry_waypoint` (an all-to-all moving one padded slot per device, after which
    the stack rests whole), the pads exit there device-local (`strip_stack_pad`), and the
    gather moves REAL slots only. The transpose runs the route backwards: a real-slot
    reduce-scatter, the slice's transpose writing exact zeros into the pad slots, and the
    all-to-all returning every slot to its owner."""
    source.validate_shape(axes, value.shape)
    assert axes[0] == "stack" and value.shape[0] == census.padded_stack_len, (
        axes,
        value.shape,
        census,
    )
    reduced = dropped_mesh_axes(source, destination, axes)
    resident = NamedSharding(destination.mesh, P(*destination.spec_for(axes), reduced=reduced))
    persisted = jax.lax.optimization_barrier(value)
    if census.stack_pad == 0:
        return jax.sharding.reshard(persisted, resident)
    waypoint = padded_entry_waypoint(source, destination, axes)
    staged = jax.sharding.reshard(persisted, waypoint.sharding_for(axes))
    return jax.sharding.reshard(strip_stack_pad(staged, census), resident)


def padded_entry_waypoint(source: PlacedRule, destination: PlacedRule, axes: Axes) -> PlacedRule:
    """A padded stack's entry waypoint, derived from the compute row: that layout with
    the persist row's stack cut re-parked MINOR on the leaf's last axis, so the stack
    axis rests whole while every device still holds ÷cut of the bytes. Persist→waypoint
    is then a stack→minor-axis all-to-all and waypoint→compute a minor-axis all-gather
    of the real slots — both pure because the cut nests minor (nested major it
    legalizes as a collective-permute). Spelled for the stack-only entry the owner
    layouts declare (the only rows that cut, hence pad, a stack): the compute row rests
    every matrix axis exactly where the persist row does."""
    stack_cut = source.assignment("stack")
    gathered = dropped_mesh_axes(source, destination, axes)
    assert gathered == frozenset(stack_cut), (
        f"{source.label} → {destination.label} for {axes}: the padded entry is spelled for "
        f"a stack-only gather, but this transition gathers {sorted(gathered)} while the "
        f"persist stack cut is {stack_cut!r}"
    )
    minor = axes[-1]
    return PlacedRule(
        mesh=destination.mesh,
        label=f"{destination.label} (padded-entry waypoint)",
        rule={**destination.rule, minor: (*destination.assignment(minor), *stack_cut)},
    )


def validate_stacked_leaf(
    census: StackCensus,
    source: PlacedRule,
    destination: PlacedRule,
    axes: Axes,
    shape: tuple[int, ...],
) -> None:
    """Construction-time tiling of one stacked persist leaf at its PADDED shape: the
    persist row and, for a padded stack, the entry waypoint — so a mesh the padded entry
    cannot spell refuses where the rows are bound, never at trace."""
    assert shape[0] == census.padded_stack_len, (shape, census)
    source.validate_shape(axes, shape)
    if census.stack_pad:
        padded_entry_waypoint(source, destination, axes).validate_shape(axes, shape)


def strip_stack_pad(value: Array, census: StackCensus) -> Array:
    """THE pad exit: drop a persist stack's trailing pad slots where the stack axis rests
    whole on every device — at a padded entry's waypoint (`materialize_reduced_weights`)
    and on the CI vector leaves, whose row `_bind` refuses `stack`. Its autodiff
    transpose scatters the real grads into the leading real slots and writes exact zeros
    into the pad slots."""
    if census.stack_pad == 0:
        return value
    assert value.shape[0] == census.padded_stack_len, (value.shape, census)
    return jax.lax.slice_in_dim(value, 0, census.stack_len, axis=0)


def component_stacks_to_compute_weights(
    components: ComponentStacks, placement: ComponentsPlacement
) -> ComponentStacks:
    """Materialize the compute-weight residents from optimizer ownership (dtype is the
    caller's: the step casts masters to compute dtype first). One entry per stack
    (`materialize_reduced_weights`, `reduced` over the gathered axes): the residents
    carry ONLY real stacks."""
    _validate_component_stacks(components, placement)
    stacks: dict[str, tuple[Array, Array]] = {}
    for group, (vs, us) in components.stacks.items():
        census = placement.group_census[group]
        factorization = census.factorization
        stacks[group] = (
            materialize_reduced_weights(
                vs,
                census=census,
                source=placement.optimizer_state,
                destination=placement.compute_weights,
                axes=factorization.v_axes,
            ),
            materialize_reduced_weights(
                us,
                census=census,
                source=placement.optimizer_state,
                destination=placement.compute_weights,
                axes=factorization.u_axes,
            ),
        )
    return ComponentStacks(stacks=stacks, site_slots=components.site_slots)


def component_stacks_to_faithfulness_weights(
    components: ComponentStacks, placement: ComponentsPlacement
) -> ComponentStacks:
    """Materialize the declared faithfulness operands from optimizer ownership. The
    faithfulness lane RIDES the persist pads (its rows stack-shard exactly like the
    masters, so real-length stacks could not even be placed here): pad slots flow
    through as exact zeros — zero V·U, zero delta, zero loss — and exit at the loss
    reduction (`faithfulness.make_faithfulness_loss`)."""
    _validate_component_stacks(components, placement)
    destination = placement.faithfulness_weights
    stacks: dict[str, tuple[Array, Array]] = {
        group: (
            jax.sharding.reshard(
                vs, destination.sharding_for(placement.group_census[group].factorization.v_axes)
            ),
            jax.sharding.reshard(
                us, destination.sharding_for(placement.group_census[group].factorization.u_axes)
            ),
        )
        for group, (vs, us) in components.stacks.items()
    }
    return ComponentStacks(
        stacks=stacks, site_slots=components.site_slots, stack_pads=components.stack_pads
    )


def constrain_faithfulness_deltas(
    deltas: dict[str, Array], placement: ComponentsPlacement
) -> dict[str, Array]:
    row = placement.faithfulness_deltas
    out: dict[str, Array] = {}
    for group, delta in deltas.items():
        census = placement.group_census[group]
        axes = census.factorization.delta_axes
        expected = census.factorization.delta_leaf_shape(census.padded_stack_len)
        assert delta.shape == expected, (group, delta.shape, expected)
        row.validate_shape(axes, delta.shape)
        out[group] = jax.sharding.reshard(delta, row.sharding_for(axes))
    return out


# ── presets ──────────────────────────────────────────────────────────────────
# Named rule tables, one per deliberately supported layout. `stack` is the V/U
# semantic-group stack axis (components.ComponentStacks); `d` covers d_in and d_out via the
# per-tensor axes tuples. All presets share the activation waist rule (batch over the
# full data mesh) — that surface is layout-invariant.

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


def _stack_cut(persist_rows: tuple[PlacedRule, ...]) -> int:
    """The extent every persist stack resting at `persist_rows` must tile: the least
    common multiple of the rows' stack splits (1 where no row cuts the stack)."""
    return lcm(*(row.shard_count("stack") for row in persist_rows))


def resolve_stack_census(stack_len: int, persist_rows: tuple[PlacedRule, ...]) -> StackCensus:
    """A persist stack's census: the real length padded to the next multiple of its
    persist rows' stack cut — an enumerated fact, never a fallback row."""
    return StackCensus(stack_len=stack_len, stack_pad=-stack_len % _stack_cut(persist_rows))


def _resolve_group_census(
    optimizer_state: PlacedRule,
    compute_weights: PlacedRule,
    faithfulness_weights: PlacedRule,
    faithfulness_deltas: PlacedRule,
    sites: tuple[SiteSpec, ...],
) -> dict[str, GroupCensus]:
    """The run's semantic-group census. Placement is TOTAL and fallback-free: one table
    places all groups. A group whose stack length does not tile the stack-sharding
    component rows is placed by PADDING its persist stack to the next common multiple
    (`GroupCensus.stack_pad` — an enumerated fact, never a fallback row); matrix-axis
    tiling failures — the padded entry's waypoint included (`validate_stacked_leaf`) —
    still die here, at construction."""
    stack_cut = _stack_cut((optimizer_state, faithfulness_weights, faithfulness_deltas))
    census = {
        name: GroupCensus(
            factorization=group.factorization,
            stack_len=len(group.specs),
            stack_pad=-len(group.specs) % stack_cut,
        )
        for name, group in vu_groups(sites).items()
    }
    for entry in census.values():
        factorization, g = entry.factorization, entry.padded_stack_len
        for axes, shape in (
            (factorization.v_axes, factorization.v_leaf_shape(g)),
            (factorization.u_axes, factorization.u_leaf_shape(g)),
        ):
            validate_stacked_leaf(entry, optimizer_state, compute_weights, axes, shape)
            faithfulness_weights.validate_shape(axes, shape)
        faithfulness_deltas.validate_shape(
            factorization.delta_axes, factorization.delta_leaf_shape(g)
        )
    return census


# ── consumed-axis vocabularies (the fail-closed rule-key check) ───────────────
# Every rule key must name a semantic axis some tensor actually CONSUMES at that row
# (the row's `spec_for` / `validate_shape` call sites). Lookup is exact-name with a
# quiet unlisted-axis-replicates default, so an unconsumed key — a typo'd axis in an
# explicit table — would otherwise silently replicate the tensor it meant to shard.
# These sets enumerate today's consumers; a new consumed axis is a new entry here,
# never a loosening. The component rows derive theirs from the RUN's factorizations
# (`_vu_consumable_axes` in `_bind`) — a rule keyed on `expert` refuses unless some
# group actually holds expert-blocked leaves — and the CI weight rows derive theirs
# from their declared tensor axes in `_bind.ci_placement`.
# The two activation waists double as the target linears' input/output contract, so they
# also cover FrozenAttn.core's head-split views ("position"/"q_head"/"kv_head"/"head_dim").
_WAIST_AXES: frozenset[SemanticAxis] = frozenset(
    {"batch", "position", "feature", "q_head", "kv_head", "head_dim"}
)
_COMPONENT_WAIST_AXES: frozenset[SemanticAxis] = frozenset({"batch", "position", "C"})
# The waist's expert-blocked view — consumable only for runs holding expert-blocked
# groups (`_bind` widens the component row with these exactly then).
_EXPERT_WAIST_AXES: frozenset[SemanticAxis] = frozenset({"expert", "C_block"})
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
class _CIMoETable:
    expert_ffn: _CIWeightTable
    expert_head: _CIWeightTable


@dataclass(frozen=True)
class _CIFnTable:
    attention: _CIWeightTable
    ffn: _CIWeightTable
    input: _CIWeightTable
    output: _CIWeightTable
    vectors: Rule
    activations: Rule
    moe: _CIMoETable | None


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


def _collect_rules(value: object, rules: list[Rule]) -> None:
    """Every rule reachable through dataclass fields, in field-declaration order; the
    non-rule leaves (activation refs, an absent `moe`) are skipped."""
    if isinstance(value, Mapping):
        rules.append(cast(Rule, value))
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _collect_rules(getattr(value, field.name), rules)


@dataclass(frozen=True)
class PlacementTable:
    """The placement policy as a pure VALUE — rules not yet bound to a mesh or site set.
    Mirrors `PlacementRules` row for row and `PlacementTableConfig` field for field."""

    components: _ComponentsTable
    ci_fn: _CIFnTable
    activations: _ActivationsTable
    target: _TargetTable

    @property
    def mesh_axes(self) -> frozenset[MeshAxis]:
        """The mesh vocabulary the table is spelled in: every mesh axis some row names.
        A table binds only a mesh whose axes these are (`_bind`)."""
        rules: list[Rule] = []
        _collect_rules(self, rules)
        return frozenset(
            axis for rule in rules for assignment in rule.values() for axis in assignment
        )


_REPLICATED: Rule = {}

_STACK_OWNER: Rule = {
    "stack": ("replicate",),
    "d_in": ("fsdp",),
    "d_out": ("fsdp",),
    "C": ("tp",),
}
# The intra-matrix ÷N layout parks the replicate shard on C (Adam is elementwise, so
# the master layout is free to choose): entry to the ÷fsdp compute weights is a pure
# minor-axis all-gather over `replicate` on C (exit, its reduce-scatter), and the
# matrix faithfulness rows ARE this layout, so that transition is the identity. The
# matrix delta row scatters both C contractions (tp and replicate) onto d_in.
_MATRIX_OWNER: Rule = {"d_in": ("fsdp",), "d_out": ("fsdp",), "C": ("tp", "replicate")}
_STACK_FAITHFULNESS_DELTA: Rule = {"stack": ("replicate",), "d_out": ("fsdp",)}
_MATRIX_FAITHFULNESS_DELTA: Rule = {"d_out": ("fsdp",), "d_in": ("tp", "replicate")}

# NS waypoint (`ns_compute`): ONE staging for every preset — each kind's stack split
# over the node axis (`replicate`), matrices whole per device, no NS-only padding (the
# persist pads ride through as inert zero matrices). The
# intra-node NS redundancy (a kind's shard replicated across the node's fsdp x tp plane)
# is accepted: NS is a sliver of the step, and comms-free beats FLOP-optimal. Under
# owner persistence the ingress is the identity on the stack axis (masters already
# stack-owned); under intra-matrix or replicated persistence it is a shard->shard hop
# chain (`muon_stacked.staging_hops`) — never a whole-fp32-stack-per-rank
# materialization (replicated staging would peak at the SUM of every kind's fp32
# stack, mesh-invariantly).
_NS_STACK_SPLIT: Rule = {"stack": ("replicate",)}

_HSDP_COMPUTE_WEIGHTS: Rule = {"d_in": ("fsdp",), "d_out": ("fsdp",), "C": ("tp",)}
_C_OPERANDS: Rule = {"C": ("tp",)}

_OWNER_COMPONENTS = _ComponentsTable(
    optimizer_state=_STACK_OWNER,
    compute_weights=_HSDP_COMPUTE_WEIGHTS,
    faithfulness_weights=_STACK_OWNER,
    faithfulness_deltas=_STACK_FAITHFULNESS_DELTA,
    operands=_C_OPERANDS,
    ns_compute=_NS_STACK_SPLIT,
)
# zero1 rests every master intra-matrix; its faithfulness rows ARE that master layout
# (the weights transition is the identity), so no persistence row shards the stack axis
# and every semantic group — any stack length — is placeable.
_ZERO1_COMPONENTS = _ComponentsTable(
    optimizer_state=_MATRIX_OWNER,
    compute_weights=_HSDP_COMPUTE_WEIGHTS,
    faithfulness_weights=_MATRIX_OWNER,
    faithfulness_deltas=_MATRIX_FAITHFULNESS_DELTA,
    operands=_C_OPERANDS,
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
        optimizer_state={"d_model": _ZERO1_DATA, "q_head": ("tp",), "kv_head": ("tp",)},
        compute_weights={"d_model": ("fsdp",), "q_head": ("tp",), "kv_head": ("tp",)},
        operands={"q_head": ("tp",), "kv_head": ("tp",)},
        ns_compute=_NS_STACK_SPLIT,
    ),
    ffn=_CIWeightTable(
        optimizer_state={"ffn_hidden": ("tp", "fsdp", "replicate")},
        compute_weights={"ffn_hidden": ("tp", "fsdp")},
        operands={"ffn_hidden": ("tp",)},
        ns_compute=_NS_STACK_SPLIT,
    ),
    input=_CIWeightTable(
        optimizer_state={"input": ("tp",), "d_model": _ZERO1_DATA},
        compute_weights={"input": ("tp",), "d_model": ("fsdp",)},
        operands={"input": ("tp",)},
        ns_compute=_NS_STACK_SPLIT,
    ),
    output=_CIWeightTable(
        optimizer_state={"d_model": _ZERO1_DATA, "C": ("tp",)},
        compute_weights={"d_model": ("fsdp",), "C": ("tp",)},
        operands=_C_OPERANDS,
        ns_compute=_NS_STACK_SPLIT,
    ),
    vectors={"ffn_hidden": ("tp",), "C": ("tp",)},
    activations={
        "batch": _BATCH,
        "input": ("tp",),
        "q_head": ("tp",),
        "kv_head": ("tp",),
        "ffn_hidden": ("tp",),
        "C": ("tp",),
    },
    moe=None,
)
_STACK_OWNER_CI = replace(
    _ZERO1_CI,
    attention=_CIWeightTable(
        optimizer_state={
            "stack": ("replicate",),
            "d_model": ("fsdp",),
            "q_head": ("tp",),
            "kv_head": ("tp",),
        },
        compute_weights={"d_model": ("fsdp",), "q_head": ("tp",), "kv_head": ("tp",)},
        operands={"q_head": ("tp",), "kv_head": ("tp",)},
        ns_compute=_NS_STACK_SPLIT,
    ),
    ffn=_CIWeightTable(
        optimizer_state={"stack": ("replicate",), "ffn_hidden": ("tp", "fsdp")},
        compute_weights={"ffn_hidden": ("tp", "fsdp")},
        operands={"ffn_hidden": ("tp",)},
        ns_compute=_NS_STACK_SPLIT,
    ),
    input=_CIWeightTable(
        optimizer_state={"stack": ("replicate",), "input": ("tp",), "d_model": ("fsdp",)},
        compute_weights={"input": ("tp",), "d_model": ("fsdp",)},
        operands={"input": ("tp",)},
        ns_compute=_NS_STACK_SPLIT,
    ),
    output=_CIWeightTable(
        optimizer_state={"stack": ("replicate",), "d_model": ("fsdp",), "C": ("tp",)},
        compute_weights={"d_model": ("fsdp",), "C": ("tp",)},
        operands=_C_OPERANDS,
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
    moe=None,
)

_SHARDED_ACTIVATIONS = _ActivationsTable(
    external={"batch": _BATCH},
    component={"batch": _BATCH, "C": ("tp",)},
)

_SHARDED_TARGET = _TargetTable(
    embedding=_TargetWeightTable(persist={"d_model": ("fsdp",)}, operand=_REPLICATED),
    normalization=_REPLICATED,
    position_encoding=_REPLICATED,
    column=_TargetLinearTable(
        persist={"d_in": ("fsdp",), "d_out": ("tp",)},
        operand={"d_out": ("tp",)},
        input="external",
        output="intermediate",
    ),
    row=_TargetLinearTable(
        persist={"d_out": ("fsdp",), "d_in": ("tp",)},
        operand={"d_in": ("tp",)},
        input="intermediate",
        output="external",
    ),
    output=_TargetWeightTable(persist={"d_model": ("fsdp",)}, operand=_REPLICATED),
    intermediate={"batch": _BATCH, "feature": ("tp",), "q_head": ("tp",), "kv_head": ("tp",)},
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

# ── the resident respelling ───────────────────────────────────────────────────
# The `*-replicated-resident` presets are the zero1/owner tables re-spelled for the
# two-axis `(data, tp)` mesh with the bf16 working copy RESIDENT whole: every
# compute-weight and target-persist execution row IS its operand row (÷tp only), so the
# once-per-step masters→resident entry gather is the step's ONLY weight collective —
# nothing left to gather inside any loop. Residency makes the fsdp singleton structural:
# `fsdp` assignments disappear and `replicate` IS the whole data axis — the same
# placement the three-axis `(N, 1, tp)` spelling produced, with nothing left to
# pretend-shard.


def _resident_assignment(assignment: MeshAssignment) -> MeshAssignment:
    return tuple("data" if axis == "replicate" else axis for axis in assignment if axis != "fsdp")


def _resident_rule(rule: Rule) -> Rule:
    """One rule re-spelled for the `(data, tp)` mesh: `fsdp` assignments drop,
    `replicate` renames to `data`; entries that lose every mesh axis drop out
    (unlisted = replicated)."""
    respelled: dict[SemanticAxis, MeshAssignment] = {
        axis: _resident_assignment(assignment) for axis, assignment in rule.items()
    }
    return {axis: assignment for axis, assignment in respelled.items() if assignment}


def _resident_ci_weights(table: _CIWeightTable) -> _CIWeightTable:
    return _CIWeightTable(
        optimizer_state=_resident_rule(table.optimizer_state),
        compute_weights=_resident_rule(table.operands),
        operands=_resident_rule(table.operands),
        ns_compute=_resident_rule(table.ns_compute),
    )


def _resident_weight(table: _TargetWeightTable) -> _TargetWeightTable:
    return _TargetWeightTable(
        persist=_resident_rule(table.operand), operand=_resident_rule(table.operand)
    )


def _resident_linear(table: _TargetLinearTable) -> _TargetLinearTable:
    return _TargetLinearTable(
        persist=_resident_rule(table.operand),
        operand=_resident_rule(table.operand),
        input=table.input,
        output=table.output,
    )


def _resident_table(base: PlacementTable) -> PlacementTable:
    """A sharded preset's resident twin: masters keep the base's persistence
    (respelled), every working-copy row equals its operand row, activation refs
    unchanged."""
    components = base.components
    target = base.target
    return PlacementTable(
        components=_ComponentsTable(
            optimizer_state=_resident_rule(components.optimizer_state),
            compute_weights=_resident_rule(components.operands),
            faithfulness_weights=_resident_rule(components.faithfulness_weights),
            faithfulness_deltas=_resident_rule(components.faithfulness_deltas),
            operands=_resident_rule(components.operands),
            ns_compute=_resident_rule(components.ns_compute),
        ),
        ci_fn=_CIFnTable(
            attention=_resident_ci_weights(base.ci_fn.attention),
            ffn=_resident_ci_weights(base.ci_fn.ffn),
            input=_resident_ci_weights(base.ci_fn.input),
            output=_resident_ci_weights(base.ci_fn.output),
            vectors=_resident_rule(base.ci_fn.vectors),
            activations=_resident_rule(base.ci_fn.activations),
            moe=None
            if base.ci_fn.moe is None
            else _CIMoETable(
                expert_ffn=_resident_ci_weights(base.ci_fn.moe.expert_ffn),
                expert_head=_resident_ci_weights(base.ci_fn.moe.expert_head),
            ),
        ),
        activations=_ActivationsTable(
            external=_resident_rule(base.activations.external),
            component=_resident_rule(base.activations.component),
        ),
        target=_TargetTable(
            embedding=_resident_weight(target.embedding),
            normalization=_resident_rule(target.normalization),
            position_encoding=_resident_rule(target.position_encoding),
            column=_resident_linear(target.column),
            row=_resident_linear(target.row),
            output=_resident_weight(target.output),
            intermediate=_resident_rule(target.intermediate),
            component=target.component,
        ),
    )


_OWNER_TABLE = PlacementTable(
    components=_OWNER_COMPONENTS,
    ci_fn=_STACK_OWNER_CI,
    activations=_SHARDED_ACTIVATIONS,
    target=_SHARDED_TARGET,
)
_ZERO1_TABLE = PlacementTable(
    components=_ZERO1_COMPONENTS,
    ci_fn=_ZERO1_CI,
    activations=_SHARDED_ACTIVATIONS,
    target=_SHARDED_TARGET,
)

# ── the MoE resident presets ──────────────────────────────────────────────────
# The `*-replicated-resident-moe` twins extend the resident presets with the rows
# expert-blocked component groups need. In BOTH flavors V/U expert blocks CO-LOCATE
# with their frozen experts (`expert: tp` on residents and operands — no rank ever
# touches another shard's expert components), the faithfulness rows ARE the master
# layout (identity transition), and — keyed on `expert` (+`C_block` where used) — the
# preset binds only site sets that actually hold expert-blocked groups; the rule-key
# check refuses dense-only runs, whose presets are the dense twins.
#
# `zero1-replicated-resident-moe`: masters intra-matrix ÷N in the zero1 spirit
# (`C_block: data`), so any stack length stays placeable. The costs of that freedom:
# the faithfulness V·U contraction runs over a data-sharded `C_block` (its delta lands
# through a cross-`data` reduce-scatter at entry), and the layout is muon-incompatible
# (`assert_stacked_muon_component_staging` — NS staging would round-trip the full
# master bytes across `data` per step). One faithfulness-delta row serves both
# factorization kinds, and an expert delta carries `expert` and `d_in` together, so
# `d_in` cannot also ride `tp`: the dense (shared-kind) deltas' tp contraction
# all-reduces at entry instead of scattering — small matrices, off the hot loop.
#
# `owner-replicated-resident-moe`: COMPONENT masters stack-cut in the owner spirit —
# `{stack: data, expert: tp}` rests every V/U block WHOLE on one device on both axes.
# The entry gather becomes the layout-preserving stack-axis all-gather over `data`
# (GLU owner's shape), the faithfulness V·U contraction is fully rank-local (its
# C/C_block contraction is unsharded, so the delta path carries NO cross-`data`
# collective), and stacked-muon NS staging keeps owner's story: identity on the
# cross-node stack axis, only intra-node tp gathers of the co-located block axes.
# Owner cuts the stack ÷data: a stack length that does not tile it is placed by
# padding the persist stack to the next multiple (`GroupCensus.stack_pad` — e.g. the
# whole 40-layer grid pads to 48 at data=16, +16.7% persist bytes/optimizer work,
# zero compute-side cost). The flavor is the COMPONENTS masters' — the CI-fn
# rows stay the zero1 twin's (intra-matrix, no chunk-stack pad ever resolves): the
# ruling that motivates owner masters is the components muon pairing, and owner-cut CI
# masters would pad the chunk stack to ÷data for nothing.
_EXPERT_RESIDENT: Rule = {"C": ("tp",), "expert": ("tp",)}
_MOE_ZERO1_RESIDENT_COMPONENTS = _ComponentsTable(
    optimizer_state={"C": ("tp", "data"), "expert": ("tp",), "C_block": ("data",)},
    compute_weights=_EXPERT_RESIDENT,
    faithfulness_weights={"C": ("tp", "data"), "expert": ("tp",), "C_block": ("data",)},
    faithfulness_deltas={"d_in": ("data",), "expert": ("tp",)},
    operands=_EXPERT_RESIDENT,
    ns_compute={"stack": ("data",)},
)
_MOE_OWNER_RESIDENT_COMPONENTS = _ComponentsTable(
    optimizer_state={"stack": ("data",), "C": ("tp",), "expert": ("tp",)},
    compute_weights=_EXPERT_RESIDENT,
    faithfulness_weights={"stack": ("data",), "C": ("tp",), "expert": ("tp",)},
    faithfulness_deltas={"stack": ("data",), "expert": ("tp",)},
    operands=_EXPERT_RESIDENT,
    ns_compute={"stack": ("data",)},
)


_MOE_RESIDENT_CI = _CIMoETable(
    # The MoE CI fn's expert families mirror the V/U expert blocks: residents/operands
    # rest whole per expert shard, CO-LOCATED with the target's frozen experts and the
    # V/U blocks (`expert: tp` — the CI banks' routed compute and the fused heads ride
    # `ExpertShardedJobs` on the same axis with zero weight movement); masters ÷(tp·data)
    # in the zero1 spirit, so entry is a pure all-gather over `data` with `expert`
    # staying put. NS staging folds layer/expert into the canonical stack ({stack: data},
    # whole matrices per device). BOTH moe flavors carry these same rows: the owner
    # spirit's stack-cut masters would pad the CI chunk stack to tile `data`
    # (n_chunks = 10 pads to 16 at data=8), and the intra-matrix cut costs owner nothing
    # it claims — no faithfulness row exists for CI weights, and the seats keep the CI
    # group on adamw.
    expert_ffn=_CIWeightTable(
        optimizer_state={"expert": ("tp",), "ffn_hidden": ("data",)},
        compute_weights={"expert": ("tp",)},
        operands={"expert": ("tp",)},
        ns_compute={"stack": ("data",)},
    ),
    expert_head=_CIWeightTable(
        optimizer_state={"expert": ("tp",), "C_block": ("data",)},
        compute_weights={"expert": ("tp",)},
        operands={"expert": ("tp",)},
        ns_compute={"stack": ("data",)},
    ),
)


def _moe_resident_table(base: PlacementTable, components: _ComponentsTable) -> PlacementTable:
    """A resident twin re-armed for expert-blocked groups: the flavor's component rows,
    the CI MoE weight families, and the component waist additionally keyed `expert`, so
    the expert-blocked view of a component activation ([*lead, expert, C_block]) derives
    from the same row as its flat [*lead, C] view — C is expert-major, so the two spell
    one layout."""
    resident = _resident_table(base)
    return replace(
        resident,
        components=components,
        ci_fn=replace(resident.ci_fn, moe=_MOE_RESIDENT_CI),
        activations=replace(
            resident.activations,
            component={**resident.activations.component, "expert": ("tp",)},
        ),
    )


# The built-in tables: `zero1` (intra-matrix ÷N over the full data mesh — no row shards
# the stack axis, so every semantic group is placeable pad-free; ~equivalent comms to
# `owner` under elementwise optimizers), `owner` (stack ÷replicate, d ÷fsdp — the
# muon-motivated layout, node-local NS; a stack that doesn't tile ÷replicate
# pads its persist stack to the next multiple — `GroupCensus.stack_pad`),
# `zero1-replicated-resident` (`zero1` masters, resident working copy — classic ZeRO-1;
# every semantic group is placeable, and the faithfulness rows ARE the master layout so
# that transition stays the identity), `owner-replicated-resident` (`owner` stack-cut
# masters × the resident working copy: the faithfulness transition is the identity AND
# nothing is gathered in-loop; the V/U stacks AND the CI-fn chunk stack cut ÷data, each
# padded where it does not tile),
# `zero1-replicated-resident-moe` / `owner-replicated-resident-moe` (the
# resident twins plus expert-blocked component rows — expert axis over tp, co-located
# with the frozen experts; binding only expert-carrying site sets; zero1 masters rest
# `C_block: data` and place any stack length, owner masters `{stack: data, expert: tp}`
# rest whole blocks per device — see the MoE resident presets block), `ddp` (everything
# replicated — single-node / small-model runs). The `*-replicated-resident*` presets
# live on the two-axis `(data, tp)` mesh; their tables are the shared rules under
# `_resident_table`'s respelling.
PRESETS: Mapping[PlacementPresetName, PlacementTable] = {
    "owner": _OWNER_TABLE,
    "zero1": _ZERO1_TABLE,
    "zero1-replicated-resident": _resident_table(_ZERO1_TABLE),
    "owner-replicated-resident": _resident_table(_OWNER_TABLE),
    "zero1-replicated-resident-moe": _moe_resident_table(
        _ZERO1_TABLE, _MOE_ZERO1_RESIDENT_COMPONENTS
    ),
    "owner-replicated-resident-moe": _moe_resident_table(
        _ZERO1_TABLE, _MOE_OWNER_RESIDENT_COMPONENTS
    ),
    "ddp": PlacementTable(
        components=_DDP_COMPONENTS,
        ci_fn=_DDP_CI,
        activations=_SHARDED_ACTIVATIONS,
        target=_REPLICATED_TARGET,
    ),
}
PRESET_NAMES = tuple(PRESETS)
assert set(PRESET_NAMES) == set(get_args(PlacementPresetName)), PRESET_NAMES


def _bind(
    desc: str,
    table: PlacementTable,
    mesh: Mesh | AbstractMesh,
    sites: tuple[SiteSpec, ...],
    sequence_sharding: SequenceSharding,
) -> PlacementRules:
    """The one constructor: bind the table's rules to the mesh, stamp the printed
    labels, fail closed on rule keys no tensor consumes, and refuse any semantic group
    the component rows cannot place."""
    components = table.components
    target = table.target

    # THE mesh-vocabulary check: the table's rows must be spelled in the mesh's axes,
    # and every mesh axis with more than one device along it must be sharded over by
    # some row — an unnamed multi-device axis would replicate the whole run across it.
    # A size-1 axis is exempt: the mesh schema requires all of a shape's axes, so a run
    # without tensor parallelism spells `tp: 1`, and a table need not name a one-device
    # axis to say nothing about it.
    named = table.mesh_axes
    declared = frozenset(mesh.axis_names)
    assert named <= declared, (
        f"{desc} names mesh axes {sorted(named - declared)} the mesh does not declare "
        f"(mesh axes: {list(mesh.axis_names)}); a table binds only a mesh spelled in its "
        f"own axis vocabulary — author `runtime.mesh` with the axes its rows name"
    )
    unsharded = sorted(axis for axis in declared - named if mesh.shape[axis] > 1)
    assert not unsharded, (
        f"{desc} shards nothing over mesh axes {unsharded} "
        f"(sizes {[mesh.shape[axis] for axis in unsharded]}) — every device along them "
        f"would hold a full replica. Name them in some row, or author a mesh without them"
    )

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

    def scanned_row(label: str, rule: Rule, consumes: frozenset[SemanticAxis]) -> PlacedRule:
        # The arrays at these rows are iterated slot by slot by the per-layer / per-chunk
        # scans, so their stack axis rests whole on every device: the CI vector leaves'
        # pad exit (`strip_stack_pad`) is device-local there, and a stack-sharded scan
        # operand has no lowering at all.
        assert "stack" not in rule, (
            f"{desc}: row {label!r} assigns `stack` ({rule['stack']!r}); the arrays at "
            f"this row are scanned slot by slot and rest whole on the stack axis — only "
            f"persist rows (optimizer_state, faithfulness_*) cut the stack"
        )
        return row(label, rule, consumes)

    def ns_row(label: str, rule: Rule) -> PlacedRule:
        assert set(rule) <= {"stack"}, (
            f"{desc}: {label} may assign only `stack` (got {sorted(rule)}) — the batched NS"
            f" needs whole matrices per device (`ns_staging_sharding`), and a"
            f" matrix-axis-carrying waypoint re-triggers the SPMD"
            f" involuntary-full-rematerialization fallback"
        )
        return PlacedRule(mesh=mesh, label=label, rule=rule)

    # The component rows' consumable keys are the union of the RUN's factorizations'
    # leaf axes: a table keyed on `expert`/`C_block` refuses unless some group actually
    # holds expert-blocked leaves, and `C` refuses when no group holds dense ones.
    run_factorizations = {group.factorization for group in vu_groups(sites).values()}
    vu_axes: frozenset[SemanticAxis] = frozenset(
        axis for f in run_factorizations for axes in (f.v_axes, f.u_axes) for axis in axes
    )
    delta_axes: frozenset[SemanticAxis] = frozenset(
        axis for f in run_factorizations for axis in f.delta_axes
    )
    optimizer_state_row = row("components/optimizer_state", components.optimizer_state, vu_axes)
    compute_weights_row = scanned_row(
        "components/compute_weights", components.compute_weights, vu_axes
    )
    faithfulness_weights_row = row(
        "components/faithfulness_weights", components.faithfulness_weights, vu_axes
    )
    faithfulness_deltas_row = row(
        "components/faithfulness_deltas", components.faithfulness_deltas, delta_axes
    )
    group_census = _resolve_group_census(
        optimizer_state_row,
        compute_weights_row,
        faithfulness_weights_row,
        faithfulness_deltas_row,
        sites,
    )
    ns_compute_row = ns_row("components/ns_compute", components.ns_compute)
    external_row = row("activations/external", table.activations.external, _WAIST_AXES)
    intermediate_row = row("target/intermediate", target.intermediate, _WAIST_AXES)
    match sequence_sharding:
        case "replicate":
            masked_external_row = external_row
        case "sequence_parallel":
            assert "position" not in table.activations.external, (
                f"{desc}: sequence_sharding 'sequence_parallel' adds `position -> tp` to "
                f"the external row, which already assigns position "
                f"({table.activations.external['position']!r})"
            )
            masked_external_row = row(
                "activations/masked_external",
                {**table.activations.external, "position": ("tp",)},
                _WAIST_AXES,
            )

    def activation(ref: TargetActivationRef) -> PlacedRule:
        match ref:
            case "external":
                return external_row
            case "intermediate":
                return intermediate_row

    def ci_family(
        label: str, rules: _CIWeightTable, tensor_axes: tuple[Axes, ...]
    ) -> CIWeightPlacement:
        # This family's consumable keys ARE its declared tensor axes (the operand rows
        # apply to the compute residents, whose stack axis the scan has already split).
        stored: frozenset[SemanticAxis] = frozenset(axis for axes in tensor_axes for axis in axes)
        placement = CIWeightPlacement(
            optimizer_state=row(f"ci_fn/{label}.optimizer_state", rules.optimizer_state, stored),
            compute_weights=scanned_row(
                f"ci_fn/{label}.compute_weights", rules.compute_weights, stored
            ),
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
            compute_weights=compute_weights_row,
            faithfulness_weights=faithfulness_weights_row,
            faithfulness_deltas=faithfulness_deltas_row,
            operands=row("components/operands", components.operands, vu_axes.difference({"stack"})),
            ns_compute=ns_compute_row,
            group_census=group_census,
        ),
        ci_fn=CIFnRows(
            attention=ci_family(
                "attention",
                table.ci_fn.attention,
                (
                    ("stack", "q_head", "d_model"),
                    ("stack", "kv_head", "d_model"),
                    ("stack", "d_model", "q_head"),
                ),
            ),
            ffn=ci_family(
                "ffn",
                table.ci_fn.ffn,
                (("stack", "d_model", "ffn_hidden"), ("stack", "ffn_hidden", "d_model")),
            ),
            input=ci_family("input", table.ci_fn.input, (("stack", "input", "d_model"),)),
            output=ci_family("output", table.ci_fn.output, (("stack", "d_model", "C"),)),
            vectors=scanned_row("ci_fn/vectors", table.ci_fn.vectors, _CI_VECTOR_AXES),
            activations=row("ci_fn/activations", table.ci_fn.activations, _CI_ACTIVATION_AXES),
            moe=None
            if table.ci_fn.moe is None
            else CIMoEPlacement(
                expert_ffn=ci_family(
                    "moe.expert_ffn",
                    table.ci_fn.moe.expert_ffn,
                    (
                        ("stack", "expert", "d_model", "ffn_hidden"),
                        ("stack", "expert", "ffn_hidden", "d_model"),
                    ),
                ),
                expert_head=ci_family(
                    "moe.expert_head",
                    table.ci_fn.moe.expert_head,
                    (("stack", "expert", "ffn_hidden", "C_block"),),
                ),
            ),
        ),
        activations=ActivationsPlacement(
            external=external_row,
            masked_external=masked_external_row,
            component=row(
                "activations/component",
                table.activations.component,
                # An expert-carrying run consumes the waist's expert-blocked view
                # ([*lead, expert, C_block] — `expert_component_linear_plan`) at this
                # row too; dense-only runs keep the flat vocabulary and refuse those
                # keys like any other unconsumed axis.
                _COMPONENT_WAIST_AXES
                | (
                    _EXPERT_WAIST_AXES
                    if any(isinstance(f, ExpertBlocked) for f in run_factorizations)
                    else frozenset()
                ),
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


def _assignment(spelling: MeshAxis | list[MeshAxis] | None) -> MeshAssignment:
    """The parse boundary for a rule value: the schema's three authoring spellings fold
    into the one in-code tuple. YAML list ORDER is semantics (PLACEMENT_DESIGN.md
    invariant 5)."""
    match spelling:
        case None:
            return ()
        case str():
            return (spelling,)
        case list():
            return tuple(spelling)


def _rule(config: RuleConfig) -> Rule:
    return {axis: _assignment(spelling) for axis, spelling in config.items()}


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
            moe=None
            if config.ci_fn.moe is None
            else _CIMoETable(
                expert_ffn=_ci_weight_table(config.ci_fn.moe.expert_ffn),
                expert_head=_ci_weight_table(config.ci_fn.moe.expert_head),
            ),
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
    spec: PlacementSpec,
    mesh: Mesh | AbstractMesh,
    sites: tuple[SiteSpec, ...],
    *,
    sequence_sharding: SequenceSharding = "replicate",
) -> PlacementRules:
    """The configured placement spec (`runtime.sharding`) + the run's resolved site set →
    the TOTAL placement policy: a preset name, or an explicit table (already
    parse-validated by `PlacementTableConfig` — closed row vocabulary). Construction is
    the decision point: a stack length that does not tile the stack-sharding component
    rows resolves to a persist-stack pad (`GroupCensus.stack_pad`); any other group the
    rows cannot place refuses, with the remedies spelled out — there is no fallback arm.
    The same construction serves the run's own topology (config build via
    `sharding.hsdp_abstract_mesh`, pre-submission; the composition roots) and a consumer
    re-placing a finished run on its own mesh — NOTE a padded run's persist arrays carry
    their pads, so a re-placing consumer's mesh must resolve the same pad counts.
    `sequence_sharding` is `runtime.sequence_sharding` — the default is the standing
    `replicate` spelling (`masked_external` IS the external row)."""
    match spec:
        case str():
            assert spec in PRESETS, f"unknown placement preset {spec!r} (have {PRESET_NAMES})"
            table, desc = PRESETS[spec], f"sharding preset {spec!r}"
        case PlacementTableConfig():
            table, desc = _table_from_config(spec), "explicit placement table"
    return _bind(desc, table, mesh, sites, sequence_sharding)


# ── component-stack placement (lookup + boundary validation) ─────────────────
# `ComponentStacks` is placement-free; this is where its persistence placement is read
# off the rules.


def _validate_component_stacks(stacks: ComponentStacks, placement: ComponentsPlacement) -> None:
    """BOUNDARY VALIDATION of the build-time group census against the stacks actually
    held — validation of received data at a trust boundary. Disagreement between the two
    worlds (different semantic groups, a different stack length for one, or a pad
    enumeration that does not match the census) is an upstream bug and dies here."""
    lengths = stacks.group_lengths()
    census = placement.group_census
    assert census.keys() == lengths.keys(), (
        f"placement was built for component groups {sorted(census)}; "
        f"these stacks hold {sorted(lengths)}"
    )
    for group, g in lengths.items():
        assert census[group].stack_len == g, (
            f"placement expects a {census[group].stack_len}-stack for "
            f"component group {group!r}; these stacks hold {g}"
        )
        assert census[group].stack_pad == stacks.pad_of(group), (
            f"placement expects a stack pad of {census[group].stack_pad} for component "
            f"group {group!r}; these stacks enumerate {stacks.pad_of(group)}"
        )


def component_stacks_shardings(
    stacks: ComponentStacks[Array], rules: PlacementRules
) -> ComponentStacks[NamedSharding]:
    """The V/U persistence placement, each group at its factorization's leaf axes
    (`owner` is the hybrid HSDP layout).
    Boundary-validated by `_validate_component_stacks`; divisibility was validated at
    rules construction against these same (padded) shapes."""
    _validate_component_stacks(stacks, rules.components)
    row = rules.components.optimizer_state
    placed: dict[str, tuple[NamedSharding, NamedSharding]] = {}
    for group, (Vs, _) in stacks.stacks.items():
        census = rules.components.group_census[group]
        assert Vs.shape[0] == census.padded_stack_len, (
            "stacks disagree with the padded census extent — the audit would lie",
            group,
            Vs.shape,
        )
        placed[group] = (
            row.sharding_for(census.factorization.v_axes),
            row.sharding_for(census.factorization.u_axes),
        )
    return ComponentStacks(
        stacks=placed, site_slots=stacks.site_slots, stack_pads=stacks.stack_pads
    )


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
    eval-shape trees). A padded group's label names its pad count so the audit shows
    the persist arrays' extra slots explicitly."""
    _validate_component_stacks(stacks, rules.components)
    row = rules.components.optimizer_state
    out: dict[str, tuple[PlacedRule, Axes, tuple[int, ...]]] = {}
    for group, (Vs, Us) in stacks.stacks.items():
        census = rules.components.group_census[group]
        suffix = f" (stack pad +{census.stack_pad})" if census.stack_pad else ""
        out[f"V {group}{suffix}"] = (row, census.factorization.v_axes, Vs.shape)
        out[f"U {group}{suffix}"] = (row, census.factorization.u_axes, Us.shape)
    return out
