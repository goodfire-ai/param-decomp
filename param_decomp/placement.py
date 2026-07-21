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

from param_decomp.configs import PlacementTableConfig, RuleConfig

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
class ParamsPlacement:
    """The trainable V/U placement rows. `persist` and `zero1` are enforced (consumed by
    `ComponentStacks.shardings`); `forward` is declared intent for the staged migration
    (PLACEMENT_DESIGN.md stage 3 — the compute-weight reconstruct is hand-written today)."""

    persist: PlacedRule
    # The OPT-IN row for shape groups whose stack does not tile the persist stack sharding
    # (intra-matrix ZeRO-1 behind the stack axis). Absence IS strictness: without it a
    # non-tiling group is a loud error (`ComponentStacks._row_for_group`).
    zero1: PlacedRule | None
    forward: PlacedRule


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
# Named rule tables for the layouts this trainer has actually run. `stack` is the V/U
# shape-group stack axis (components.ComponentStacks); `d` covers both d_in and d_out via the
# per-tensor axes tuples. All presets share the activation waist rule (batch over the
# full data mesh) — that surface is layout-invariant (SPEC §4.1 pins).

# ÷N master rows linearize FSDP-MAJOR (compute axes first, `replicate` last): each fsdp
# group's ÷fsdp compute shard is then the contiguous concat of its own replicate-group's
# ÷N shards, so the persist→forward reconstruct (and its grad reverse) partitions as a
# pure all-gather / reduce-scatter over `replicate`. Replicate-major scatters the ÷N
# shards across the WRONG fsdp groups and GSPMD legalizes both directions as a full
# (replicate, fsdp) grid-transpose collective-permute — measured at dp32: 117 permutes,
# ~13 GiB/rank/step cross-node, −14% step time when fixed (PR #927). Nested-axis order
# is semantics (PLACEMENT_DESIGN.md lesson 4); this constant is where the trainer says so.
_ZERO1_DATA = ("fsdp", "replicate")

# The activation waist stays replicate-major: it matches the live batch pins (token /
# bsc-source device_puts). Batch never reconstructs to an fsdp-only layout, so the order
# carries consistency weight only, no comms cost.
_BATCH = ("replicate", "fsdp")

PRESET_NAMES = ("owner", "owner+zero1", "zero1", "ddp")

# describe() flags any audited tensor this large that ends up fully replicated — the
# design's precondition for the quiet unlisted-axis-replicates default (lesson 2).
REPLICATION_FLAG_ELEMS = 10_000_000


def _rules(
    mesh: Mesh | AbstractMesh,
    *,
    persist: Rule,
    zero1: Rule | None,
    forward: Rule,
    activations: Rule,
) -> PlacementRules:
    """The one constructor binding rules to the mesh and stamping the printed labels."""

    def row(label: str, rule: Rule) -> PlacedRule:
        return PlacedRule(mesh=mesh, label=label, rule=rule)

    return PlacementRules(
        mesh=mesh,
        params=ParamsPlacement(
            persist=row("params/persist", persist),
            zero1=None if zero1 is None else row("params/persist.zero1", zero1),
            forward=row("params/forward", forward),
        ),
        activations=row("activations", activations),
    )


def preset(name: str, mesh: Mesh | AbstractMesh) -> PlacementRules:
    """The built-in tables: `zero1` (intra-matrix ÷N over the full data mesh — the proven
    layout, all production mileage to date; ~equivalent comms to `owner` under elementwise
    optimizers), `owner` (stack ÷replicate, d ÷fsdp — the muon-motivated D4-amended layout,
    node-local NS; STRICT: a stack that doesn't tile ÷replicate is an error), `owner+zero1`
    (`owner` plus the `params.zero1` opt-in: non-tiling groups take intra-matrix ZeRO-1
    behind the stack axis), `ddp` (everything replicated — single-node / small-model runs)."""
    activations: Rule = {"batch": _BATCH, "C": "tp"}
    owner_persist: Rule = {"stack": "replicate", "d_in": "fsdp", "d_out": "fsdp", "C": "tp"}
    zero1_persist: Rule = {"d_in": _ZERO1_DATA, "d_out": _ZERO1_DATA, "C": "tp"}
    forward: Rule = {"d_in": "fsdp", "d_out": "fsdp", "C": "tp"}
    match name:
        case "owner":
            return _rules(
                mesh, persist=owner_persist, zero1=None, forward=forward, activations=activations
            )
        case "owner+zero1":
            # groups whose stack length does not tile `replicate` (row subsets): the
            # CONSUMER picks this row instead — conditionals are row choices in code,
            # never expressions in rules.
            return _rules(
                mesh,
                persist=owner_persist,
                zero1=zero1_persist,
                forward=forward,
                activations=activations,
            )
        case "zero1":
            return _rules(
                mesh, persist=zero1_persist, zero1=None, forward=forward, activations=activations
            )
        case "ddp":
            return _rules(mesh, persist={}, zero1=None, forward={}, activations=activations)
        case _:
            raise AssertionError(f"unknown placement preset {name!r}")


def _rule(config: RuleConfig) -> Rule:
    """YAML lists become ordered tuples — nested-axis ORDER is semantics (PR #927)."""
    return {axis: tuple(v) if isinstance(v, list) else v for axis, v in config.items()}


def from_config(spec: str | PlacementTableConfig, mesh: Mesh | AbstractMesh) -> PlacementRules:
    """`RuntimeConfig.sharding` → `PlacementRules`: a preset name, or an explicit table
    (already parse-validated by `PlacementTableConfig` — closed row vocabulary)."""
    match spec:
        case str():
            assert spec in PRESET_NAMES, f"unknown placement preset {spec!r} (have {PRESET_NAMES})"
            return preset(spec, mesh)
        case PlacementTableConfig():
            return _rules(
                mesh,
                persist=_rule(spec.params.persist),
                zero1=None if spec.params.zero1 is None else _rule(spec.params.zero1),
                forward=_rule(spec.params.forward),
                activations=_rule(spec.activations),
            )
