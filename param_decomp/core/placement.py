"""Declarative placement: semantic axis names + a typed rules table → derived PartitionSpecs.

The ab-initio sharding design (see PLACEMENT_DESIGN.md). Three vocabularies:

- **Semantic axes** — dimension NAMES declared once by the code that owns each tensor
  (`("stack", "d_in", "C")` for a V stack; `("batch", "seq", "d")` for the waist).
- **Mesh axes** — the physical grid the run config declares (`{node: 4, device: 8}`).
- **Rules** — the config-owned mapping `semantic axis -> mesh axes`, one `Rule` per
  placement row. The rows are TYPED FIELDS of `PlacementRules` (`params.persist`,
  `params.zero1`, `params.forward`, `activations`) — never string keys; future rows
  (an `optim/muon.ns` phase) become future fields.

A tensor's PartitionSpec at a row is DERIVED: look up each of its dim names in that
row's rule; unlisted names are replicated. Phase transitions (persist→forward at ENTRY,
persist→optimizer at the update step) are then mechanical reshards between two rows of
the table — `transition_bytes` prices them, `describe` prints the whole policy as one
table (the startup lint AND the documentation).

Construction is per-RUN: `from_config(spec, mesh, sites)` resolves the TOTAL per-shape-
group row assignment (persist vs the opt-in zero1 arm) and asserts the spec's
bidirectional claim, so the decision exists exactly once — at config build — and flows
downward as data (`ParamsPlacement.groups`). Consumers validate what they receive; they
never re-decide (PLACEMENT_DESIGN.md "Decision at build time").

The rule language is deliberately WEAK: exact-name lookup (semantic axis name → mesh
axes), no patterns, no conditionals, no expressions. Weird cases get a literal spec
override on a named row, not a smarter language. Load-bearing surfaces only: rules pin
the persist trees, phase entries, and the activation waist; GSPMD propagates between
pins as today.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from math import prod

import jax.numpy as jnp
from jax.sharding import AbstractMesh, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array

from param_decomp.core.components import ComponentStacks, SiteSpec, VUShape, vu_shape_groups
from param_decomp.core.configs import PlacementTableConfig, RuleConfig

# A semantic dim name -> the mesh axes it shards over (str, tuple of strs, or None).
MeshAssignment = str | tuple[str, ...] | None
Rule = Mapping[str, MeshAssignment]
Axes = tuple[str, ...]


def _mesh_axes_of(assignment: MeshAssignment) -> tuple[str, ...]:
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
        entries = tuple(self.rule.get(name) for name in axes)
        used: list[str] = []
        for e in entries:
            used.extend(_mesh_axes_of(e))
        assert len(used) == len(set(used)), (
            f"{self.label}: tensor axes {axes} derive a spec using a mesh axis twice ({entries})"
        )
        return P(*entries)

    def sharding_for(self, axes: Axes) -> NamedSharding:
        return NamedSharding(self.mesh, self.spec_for(axes))

    def shard_count(self, axis_name: str) -> int:
        """How many ways `axis_name` is split at this row (1 = unsharded)."""
        return prod(self.mesh.shape[a] for a in _mesh_axes_of(self.rule.get(axis_name)))

    def validate_shape(self, axes: Axes, shape: tuple[int, ...]) -> None:
        """Fail-fast divisibility: every dim must tile its assigned mesh-axis product."""
        assert len(axes) == len(shape), (self.label, axes, shape)
        for name, dim in zip(axes, shape, strict=True):
            n = self.shard_count(name)
            assert dim % n == 0, (
                f"{self.label}: semantic axis {name!r} (dim {dim}) does not tile its mesh "
                f"assignment ({self.rule.get(name)!r} = ÷{n})"
            )

    def transition_bytes(
        self, to: "PlacedRule", axes: Axes, shape: tuple[int, ...], dtype: jnp.dtype
    ) -> int:
        """Upper-bound bytes moved PER MATERIALIZATION of the `to` layout (0 when the
        derived specs match — the owner-resident case; else ≤ the full global tensor).
        Deliberately unmodeled: multiplicity (the recon grid materializes the forward
        layout many times per step; remat may replay it), residency-vs-regather, op-count,
        and axis locality — see PLACEMENT_DESIGN.md lessons 3-4. Good enough to rank
        resting layouts; NOT sufficient to drive an automatic search."""
        assert self.mesh == to.mesh, (self.label, to.label)
        if self.spec_for(axes) == to.spec_for(axes):
            return 0
        return prod(shape) * jnp.dtype(dtype).itemsize


@dataclass(frozen=True)
class GroupPlacement:
    """One V/U shape group's placement, decided at rules CONSTRUCTION: the group's stack
    length as resolved from the run's site set (the consumer boundary re-checks its
    arrays against it — `_component_stacks_rows`) and the row it persists on."""

    stack_len: int
    row: PlacedRule


@dataclass(frozen=True)
class ParamsPlacement:
    """The trainable V/U placement rows, plus the per-shape-group row ASSIGNMENT resolved
    at construction from the run's site set. The rules object is TOTAL: consumers look
    rows up (`row_for`) and validate what they received — the persist-vs-zero1 decision
    happens exactly once, in `_assign_groups`, never downstream. `persist` and the opt-in
    `zero1` (intra-matrix ZeRO-1 behind the stack axis; declaring it CLAIMS some group
    needs it — see `from_config`) are the rows the assignment draws from; `forward` is
    declared intent for the staged migration (PLACEMENT_DESIGN.md stage 3 — the
    compute-weight reconstruct is hand-written today)."""

    persist: PlacedRule
    zero1: PlacedRule | None
    forward: PlacedRule
    groups: Mapping[VUShape, GroupPlacement]

    def row_for(self, shape: VUShape) -> PlacedRule:
        assert shape in self.groups, (
            f"shape group {shape} is not in this placement's assignment "
            f"(built for {sorted(self.groups)})"
        )
        return self.groups[shape].row


@dataclass(frozen=True)
class PlacementRules:
    """The resolved placement policy for one run: a mesh plus one `PlacedRule` per row."""

    mesh: Mesh | AbstractMesh
    params: ParamsPlacement
    activations: PlacedRule

    def __post_init__(self) -> None:
        for row, _ in self._rows():
            assert row.mesh is self.mesh, (row.label, "row bound to a different mesh")

    def _rows(self) -> tuple[tuple[PlacedRule, bool], ...]:
        """`(row, enforced)` in table order. `enforced` = some code consumes the row
        today; everything else is declared intent for the staged migration, which
        `describe` marks loudly so the printed policy never overstates enforcement."""
        zero1 = () if self.params.zero1 is None else ((self.params.zero1, True),)
        return (
            (self.params.persist, True),
            *zero1,
            (self.params.forward, False),
            (self.activations, False),
        )

    def describe(
        self,
        tensors: Mapping[str, tuple[PlacedRule, Axes, tuple[int, ...]]] | None = None,
        not_audited: tuple[str, ...] = (),
    ) -> str:
        """The policy as one printable table (startup log + documentation). With `tensors`
        (`{label: (row, axes, shape)}`) it also prints each tensor's derived spec and
        per-device share — the placement audit a human or agent reads before a run.

        Honesty rules: rows no code consumes yet are marked (an audit that overstates
        enforcement is worse than none); any audited tensor larger than
        `REPLICATION_FLAG_ELEMS` that derives to fully replicated is flagged; tensor
        families still placed by legacy mesh-vocabulary `.shardings` are listed as
        NOT AUDITED rather than silently omitted."""
        lines = ["placement rules:"]
        mesh_desc = ", ".join(f"{a}={s}" for a, s in self.mesh.shape.items())
        lines.append(f"  mesh: {mesh_desc}")
        for row, enforced in self._rows():
            body = (
                ", ".join(f"{k}->{v}" for k, v in row.rule.items()) if row.rule else "(replicated)"
            )
            mark = "" if enforced else "   [declared — NOT yet enforced]"
            lines.append(f"  {row.label:<22} {body}{mark}")
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
        if not_audited:
            lines.append(
                "  NOT AUDITED (legacy mesh-vocabulary .shardings): " + ", ".join(not_audited)
            )
        return "\n".join(lines)


# ── presets ──────────────────────────────────────────────────────────────────
# Named rule tables, one per deliberately supported layout. `stack` is the V/U
# shape-group stack axis (components.ComponentStacks); `d` covers both d_in and d_out via the
# per-tensor axes tuples. All presets share the activation waist rule (batch over the
# full data mesh) — that surface is layout-invariant (SPEC §4.1 pins).

# ÷N master rows linearize FSDP-MAJOR (compute axes first, `replicate` last): each fsdp
# group's ÷fsdp compute shard is then the contiguous concat of its own replicate-group's
# ÷N shards, so the persist→forward reconstruct (and its grad reverse) partitions as a
# pure all-gather / reduce-scatter over `replicate`. Replicate-major scatters the ÷N
# shards across the WRONG fsdp groups and GSPMD legalizes both directions as a full
# (replicate, fsdp) grid-transpose collective-permute. Nested-axis order is semantics
# (PLACEMENT_DESIGN.md lesson 4); this constant is where the trainer says so.
_ZERO1_DATA = ("fsdp", "replicate")

# The activation waist stays replicate-major: it matches the live batch pins (token /
# bsc-source device_puts). Batch never reconstructs to an fsdp-only layout, so the order
# carries consistency weight only, no comms cost.
_BATCH = ("replicate", "fsdp")

PRESET_NAMES = ("owner", "owner+zero1", "zero1", "ddp")

# describe() flags any audited tensor this large that ends up fully replicated — the
# design's precondition for the quiet unlisted-axis-replicates default (lesson 2).
REPLICATION_FLAG_ELEMS = 10_000_000


def _assign_groups(
    desc: str, persist: PlacedRule, zero1: PlacedRule | None, sites: tuple[SiteSpec, ...]
) -> dict[VUShape, GroupPlacement]:
    """THE tiles-or-fallback decision (SPEC D4, amended 2026-07-15 / 2026-07-21), made
    once, at construction: a shape group persists on `persist` when its stack length
    tiles the row's stack sharding, else on the declared opt-in `zero1` row — with no row
    to fall to, it fails closed here. Everything below receives the result as data and
    only validates it; there is deliberately no second implementation of this branch."""
    n = persist.shard_count("stack")
    groups: dict[VUShape, GroupPlacement] = {}
    for (d_in, d_out, c), specs in vu_shape_groups(sites).items():
        g = len(specs)
        if g % n == 0:
            row = persist
        else:
            assert zero1 is not None, (
                f"{desc}: shape group (d_in={d_in}, d_out={d_out}, C={c}) stacks {g} "
                f"matrices, which does not tile the params.persist stack sharding (÷{n}), "
                f"and this placement declares no params.zero1 row. Intended (e.g. a "
                f"single-layer decomposition at multi-node dp)? Opt in: `sharding: "
                f"owner+zero1`, or declare `params: {{zero1: ...}}` in an explicit table. "
                f"Not intended? Your site set and mesh disagree — fix one."
            )
            row = zero1
        row.validate_shape(ComponentStacks.V_AXES, (g, d_in, c))
        row.validate_shape(ComponentStacks.U_AXES, (g, c, d_out))
        groups[(d_in, d_out, c)] = GroupPlacement(stack_len=g, row=row)
    return groups


def _build(
    desc: str,
    mesh: Mesh | AbstractMesh,
    sites: tuple[SiteSpec, ...],
    *,
    persist: Rule,
    zero1: Rule | None,
    forward: Rule,
    activations: Rule,
    launch_claims: bool,
) -> PlacementRules:
    """The one constructor: bind rules to the mesh, stamp the printed labels, resolve the
    per-shape-group assignment, and (with `launch_claims`) assert the declared zero1 arm
    is reachable — a spec is a bidirectional CLAIM about the run it launches."""

    def row(label: str, rule: Rule) -> PlacedRule:
        return PlacedRule(mesh=mesh, label=label, rule=rule)

    persist_row = row("params/persist", persist)
    zero1_row = None if zero1 is None else row("params/persist.zero1", zero1)
    groups = _assign_groups(desc, persist_row, zero1_row, sites)
    if launch_claims and zero1_row is not None:
        n = persist_row.shard_count("stack")
        lengths = {shape: gp.stack_len for shape, gp in sorted(groups.items())}
        assert any(gp.row is zero1_row for gp in groups.values()), (
            f"{desc} declares a params.zero1 row — the claim that some shape group's "
            f"stack does NOT tile the params.persist stack sharding — but at this mesh "
            f"every group tiles it (÷{n}; stack lengths {lengths}). A declared-but-"
            f"unreachable arm is a misconfiguration, not a no-op. If the run genuinely "
            f"has no non-tiling groups, declare that: `sharding: owner` (or drop the "
            f"zero1 row). If this was meant to exercise owner+zero1 and dp was lowered "
            f"for a smoke (e.g. `dp: 1`), smoke at a multi-device topology instead — "
            f"an inline single-device smoke cannot exercise the owner+zero1 layout."
        )
    return PlacementRules(
        mesh=mesh,
        params=ParamsPlacement(
            persist=persist_row,
            zero1=zero1_row,
            forward=row("params/forward", forward),
            groups=groups,
        ),
        activations=row("activations", activations),
    )


def _preset(
    name: str, mesh: Mesh | AbstractMesh, sites: tuple[SiteSpec, ...], *, launch_claims: bool
) -> PlacementRules:
    """The built-in tables: `zero1` (intra-matrix ÷N over the full data mesh; ~equivalent
    comms to `owner` under elementwise optimizers),
    `owner` (stack ÷replicate, d ÷fsdp — the muon-motivated D4-amended layout,
    node-local NS; STRICT: a stack that doesn't tile ÷replicate is an error), `owner+zero1`
    (`owner` plus the `params.zero1` opt-in: non-tiling groups take intra-matrix ZeRO-1
    behind the stack axis), `ddp` (everything replicated — single-node / small-model runs)."""
    activations: Rule = {"batch": _BATCH, "C": "tp"}
    owner_persist: Rule = {"stack": "replicate", "d_in": "fsdp", "d_out": "fsdp", "C": "tp"}
    zero1_persist: Rule = {"d_in": _ZERO1_DATA, "d_out": _ZERO1_DATA, "C": "tp"}
    forward: Rule = {"d_in": "fsdp", "d_out": "fsdp", "C": "tp"}
    replicated: Rule = {}
    persist: Rule
    zero1: Rule | None
    match name:
        case "owner":
            persist, zero1 = owner_persist, None
        case "owner+zero1":
            persist, zero1 = owner_persist, zero1_persist
        case "zero1":
            persist, zero1 = zero1_persist, None
        case "ddp":
            persist, zero1, forward = replicated, None, replicated
        case _:
            raise AssertionError(f"unknown placement preset {name!r}")
    return _build(
        f"sharding preset {name!r}",
        mesh,
        sites,
        persist=persist,
        zero1=zero1,
        forward=forward,
        activations=activations,
        launch_claims=launch_claims,
    )


def _rule(config: RuleConfig) -> Rule:
    """YAML lists become ordered tuples — nested-axis ORDER is semantics
    (PLACEMENT_DESIGN.md lesson 4)."""
    return {axis: tuple(v) if isinstance(v, list) else v for axis, v in config.items()}


def _from_spec(
    spec: str | PlacementTableConfig,
    mesh: Mesh | AbstractMesh,
    sites: tuple[SiteSpec, ...],
    *,
    launch_claims: bool,
) -> PlacementRules:
    match spec:
        case str():
            assert spec in PRESET_NAMES, f"unknown placement preset {spec!r} (have {PRESET_NAMES})"
            return _preset(spec, mesh, sites, launch_claims=launch_claims)
        case PlacementTableConfig():
            return _build(
                "explicit placement table",
                mesh,
                sites,
                persist=_rule(spec.params.persist),
                zero1=None if spec.params.zero1 is None else _rule(spec.params.zero1),
                forward=_rule(spec.params.forward),
                activations=_rule(spec.activations),
                launch_claims=launch_claims,
            )


def from_config(
    spec: str | PlacementTableConfig, mesh: Mesh | AbstractMesh, sites: tuple[SiteSpec, ...]
) -> PlacementRules:
    """`RuntimeConfig.sharding` + the run's resolved site set → the TOTAL placement
    policy: a preset name, or an explicit table (already parse-validated by
    `PlacementTableConfig` — closed row vocabulary). Construction is the decision point
    AND the claim check: a table WITHOUT a zero1 row claims every shape group tiles the
    persist stack sharding (a non-tiling group refuses); a table WITH one claims some
    group does not (every-group-tiles refuses). Call this wherever the mesh is the run's
    OWN topology — config build (`sharding.hsdp_abstract_mesh`, pre-sbatch) and the
    composition roots. Consumers re-placing a finished run on a foreign mesh use
    `from_config_for_consumer`."""
    return _from_spec(spec, mesh, sites, launch_claims=True)


def from_config_for_consumer(
    spec: str | PlacementTableConfig, mesh: Mesh | AbstractMesh, sites: tuple[SiteSpec, ...]
) -> PlacementRules:
    """`from_config` minus the zero1-reachability claim — for a CONSUMER re-placing a
    finished run's arrays on its own mesh (`open_jax_run` on one device), where a zero1
    row declared for the run's launch topology is legitimately unreachable (everything
    tiles at replicate=1). The fail-closed direction is unchanged: a non-tiling group
    still requires the declared row."""
    return _from_spec(spec, mesh, sites, launch_claims=False)


# ── component-stack placement (lookup + boundary validation) ─────────────────
# `ComponentStacks` is placement-free; this is where its persistence placement is read
# off the rules. Deliberately LOOKUP-ONLY: the per-group persist-vs-zero1 decision was
# made once, at construction above, and arrives here as `ParamsPlacement.groups`.


def _component_stacks_rows(
    stacks: ComponentStacks, rules: PlacementRules
) -> dict[VUShape, PlacedRule]:
    """BOUNDARY VALIDATION of the build-time row assignment against the stacks actually
    held — validation of received data at a trust boundary, NOT a re-decision; this
    function must never grow a tiles-or-fallback branch. Disagreement between the two
    worlds (different shape groups, or a different stack length for one) is an upstream
    bug and dies here."""
    lengths = stacks.group_lengths()
    groups = rules.params.groups
    assert groups.keys() == lengths.keys(), (
        f"placement assignment was built for shape groups {sorted(groups)}; "
        f"these stacks hold {sorted(lengths)}"
    )
    for shape, g in lengths.items():
        assert groups[shape].stack_len == g, (
            f"placement assignment expects a {groups[shape].stack_len}-stack for "
            f"shape group {shape}; these stacks hold {g}"
        )
    return {shape: rules.params.row_for(shape) for shape in lengths}


def component_stacks_shardings(
    stacks: ComponentStacks[Array], rules: PlacementRules
) -> ComponentStacks[NamedSharding]:
    """The V/U persistence placement, LOOKED UP from the rules' build-time assignment
    (semantic axes `V (stack, d_in, C)` / `U (stack, C, d_out)`; `owner` is the hybrid
    HSDP layout of the 2026-07-15 SPEC D4 amendment). Boundary-validated by
    `_component_stacks_rows`; divisibility was validated at rules construction against
    these same shapes."""
    rows = _component_stacks_rows(stacks, rules)
    lengths = stacks.group_lengths()
    placed: dict[VUShape, tuple[NamedSharding, NamedSharding]] = {}
    for shape, (Vs, _) in stacks.stacks.items():
        assert Vs.shape[0] == lengths[shape], (
            "stacks disagree with the static site_slots index — the audit would lie"
        )
        placed[shape] = (
            rows[shape].sharding_for(ComponentStacks.V_AXES),
            rows[shape].sharding_for(ComponentStacks.U_AXES),
        )
    return ComponentStacks(stacks=placed, site_slots=stacks.site_slots)


def component_stacks_audit(
    stacks: ComponentStacks, rules: PlacementRules
) -> dict[str, tuple[PlacedRule, Axes, tuple[int, ...]]]:
    """`{label: (row, axes, shape)}` for `rules.describe(...)` — the startup audit. Rows
    come from the rules' build-time assignment (boundary-validated); group lengths from
    the static `site_slots` (works on eval-shape trees)."""
    rows = _component_stacks_rows(stacks, rules)
    out: dict[str, tuple[PlacedRule, Axes, tuple[int, ...]]] = {}
    for (d_in, d_out, c), g in stacks.group_lengths().items():
        row = rows[(d_in, d_out, c)]
        group = f"(d_in={d_in}, d_out={d_out}, C={c})"
        out[f"V {group}"] = (row, ComponentStacks.V_AXES, (g, d_in, c))
        out[f"U {group}"] = (row, ComponentStacks.U_AXES, (g, c, d_out))
    return out
