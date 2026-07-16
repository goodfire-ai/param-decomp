"""Declarative placement: semantic axis names + a rules table → derived PartitionSpecs.

The ab-initio sharding design (see PLACEMENT_DESIGN.md). Three vocabularies:

- **Semantic axes** — dimension NAMES declared once by the code that owns each tensor
  (`("stack", "d_in", "C")` for a V stack; `("batch", "seq", "d")` for the waist).
- **Mesh axes** — the physical grid the run config declares (`{node: 4, device: 8}`).
- **Rules** — the config-owned mapping `semantic axis -> mesh axes`, per placement SITE
  (a `role/phase` string like `params/persist`, `params/forward`, `optim/muon.ns`).

A tensor's PartitionSpec at a site is DERIVED: look up each of its dim names in that
site's rule; unlisted names are replicated. Phase transitions (persist→forward at ENTRY,
persist→optimizer at the update step) are then mechanical reshards between two rows of
the table — `transition_bytes` prices them, `describe` prints the whole policy as one
table (the startup lint AND the documentation).

The rule language is deliberately WEAK: exact-name lookup (semantic axis name → mesh
axes), no patterns, no conditionals, no expressions. Weird cases get a literal spec override on a named site, not a smarter
language. Load-bearing surfaces only: rules pin the persist trees, phase entries, and the
activation waist; GSPMD propagates between pins as today.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from math import prod

import jax.numpy as jnp
from jax.sharding import AbstractMesh, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

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
class PlacementRules:
    """The resolved placement policy for one run: a mesh plus one `Rule` per site.

    Sites are `role/phase` strings (`params/persist`, `params/forward`, `optim/muon.ns`,
    `activations`). `spec_for` derives a PartitionSpec from a tensor's semantic axes;
    unknown SITES are a loud error (a site must be declared, even if `{}` = replicated),
    while unlisted AXIS NAMES within a declared site are replicated — the quiet default
    is per-axis, never per-site."""

    mesh: Mesh | AbstractMesh
    sites: Mapping[str, Rule]

    def __post_init__(self) -> None:
        mesh_axes = set(self.mesh.axis_names)
        for site, rule in self.sites.items():
            for axis, assignment in rule.items():
                unknown = set(_mesh_axes_of(assignment)) - mesh_axes
                assert not unknown, (
                    f"placement rule {site!r}: axis {axis!r} maps to unknown mesh "
                    f"axes {sorted(unknown)} (mesh has {sorted(mesh_axes)})"
                )
                per = _mesh_axes_of(assignment)
                assert len(per) == len(set(per)), (
                    f"placement rule {site!r}: axis {axis!r} repeats a mesh axis ({assignment})"
                )
            # NOTE: one mesh axis MAY appear under several semantic names in a rule
            # (`d_in -> fsdp, d_out -> fsdp`) because no single tensor carries both;
            # uniqueness is a per-TENSOR invariant, checked in `spec_for`.

    def rule(self, site: str) -> Rule:
        assert site in self.sites, (
            f"no placement rule for site {site!r}; declared sites: {sorted(self.sites)}"
        )
        return self.sites[site]

    def spec_for(self, site: str, axes: Axes) -> P:
        """The derived PartitionSpec for a tensor with semantic `axes` at `site`."""
        rule = self.rule(site)
        entries = tuple(rule.get(name) for name in axes)
        used: list[str] = []
        for e in entries:
            used.extend(_mesh_axes_of(e))
        assert len(used) == len(set(used)), (
            f"{site}: tensor axes {axes} derive a spec using a mesh axis twice ({entries})"
        )
        return P(*entries)

    def sharding_for(self, site: str, axes: Axes) -> NamedSharding:
        return NamedSharding(self.mesh, self.spec_for(site, axes))

    def shard_count(self, site: str, axis_name: str) -> int:
        """How many ways `axis_name` is split at `site` (1 = unsharded)."""
        return prod(self.mesh.shape[a] for a in _mesh_axes_of(self.rule(site).get(axis_name)))

    def validate_shape(self, site: str, axes: Axes, shape: tuple[int, ...]) -> None:
        """Fail-fast divisibility: every dim must tile its assigned mesh-axis product."""
        assert len(axes) == len(shape), (site, axes, shape)
        for name, dim in zip(axes, shape, strict=True):
            n = self.shard_count(site, name)
            assert dim % n == 0, (
                f"{site}: semantic axis {name!r} (dim {dim}) does not tile its mesh "
                f"assignment ({self.rule(site).get(name)!r} = ÷{n})"
            )

    def transition_bytes(
        self, site_from: str, site_to: str, axes: Axes, shape: tuple[int, ...], dtype: jnp.dtype
    ) -> int:
        """Upper-bound bytes moved PER MATERIALIZATION of the `site_to` layout (0 when the
        derived specs match — the owner-resident case; else ≤ the full global tensor).
        Deliberately unmodeled: multiplicity (the recon grid materializes the forward
        layout many times per step; remat may replay it), residency-vs-regather, op-count,
        and axis locality — see PLACEMENT_DESIGN.md lessons 3-4. Good enough to rank
        resting layouts; NOT sufficient to drive an automatic search."""
        if self.spec_for(site_from, axes) == self.spec_for(site_to, axes):
            return 0
        return prod(shape) * jnp.dtype(dtype).itemsize

    def describe(
        self,
        tensors: Mapping[str, tuple[str, Axes, tuple[int, ...]]] | None = None,
        not_audited: tuple[str, ...] = (),
    ) -> str:
        """The policy as one printable table (startup log + documentation). With `tensors`
        (`{label: (site, axes, shape)}`) it also prints each tensor's derived spec and
        per-device share — the placement audit a human or agent reads before a run.

        Honesty rules: rows for sites no code consumes yet are marked (an audit that
        overstates enforcement is worse than none); any audited tensor larger than
        `REPLICATION_FLAG_ELEMS` that derives to fully replicated is flagged; tensor
        families still placed by legacy mesh-vocabulary `.shardings` are listed as
        NOT AUDITED rather than silently omitted."""
        lines = ["placement rules:"]
        mesh_desc = ", ".join(f"{a}={s}" for a, s in self.mesh.shape.items())
        lines.append(f"  mesh: {mesh_desc}")
        for site in sorted(self.sites):
            rule = self.sites[site]
            body = ", ".join(f"{k}->{v}" for k, v in rule.items()) if rule else "(replicated)"
            mark = "" if site in ENFORCED_SITES else "   [declared — NOT yet enforced]"
            lines.append(f"  {site:<22} {body}{mark}")
        if tensors:
            lines.append("derived placements:")
            for label, (site, axes, shape) in tensors.items():
                self.validate_shape(site, axes, shape)
                spec = self.spec_for(site, axes)
                n_shards = prod(self.shard_count(site, n) for n in axes)
                share = prod(shape) // n_shards
                flag = (
                    "   ⚠ FULLY REPLICATED (large)"
                    if n_shards == 1 and prod(shape) > REPLICATION_FLAG_ELEMS
                    else ""
                )
                lines.append(
                    f"  {label:<36} {site:<22} {str(spec):<40} per-device {share:,} elems{flag}"
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

PRESET_NAMES = ("owner", "zero1", "ddp")

# Sites consumers dereference TODAY. A table (preset or explicit) must declare these;
# `from_config` validates the manifest up front so a custom table fails at parse time,
# not at whatever step first hits the missing site.
REQUIRED_SITES = ("params/persist", "params/persist.subset")

# Sites whose rows are actually CONSUMED by code today. Everything else in a table is
# declared intent for the staged migration (PLACEMENT_DESIGN.md) — `describe` marks such
# rows loudly so the printed policy never overstates what is enforced.
ENFORCED_SITES = frozenset(REQUIRED_SITES)

# describe() flags any audited tensor this large that ends up fully replicated — the
# design's precondition for the quiet unlisted-axis-replicates default (lesson 2).
REPLICATION_FLAG_ELEMS = 10_000_000


def preset(name: str, mesh: Mesh | AbstractMesh) -> PlacementRules:
    """The built-in tables: `owner` (stack ÷replicate, d ÷fsdp — the D4-amended layout),
    `zero1` (intra-matrix ÷N — the retired layout, kept for A/B), `ddp` (everything
    replicated — single-node / small-model runs)."""
    activations: Rule = {"batch": _BATCH, "C": "tp"}
    match name:
        case "owner":
            sites: dict[str, Rule] = {
                "activations": activations,
                "params/persist": {
                    "stack": "replicate",
                    "d_in": "fsdp",
                    "d_out": "fsdp",
                    "C": "tp",
                },
                # groups whose stack length does not tile `replicate` (site subsets): the
                # CONSUMER picks this site instead — conditionals are site choices in code,
                # never expressions in rules.
                "params/persist.subset": {"d_in": _ZERO1_DATA, "d_out": _ZERO1_DATA, "C": "tp"},
                "params/forward": {"d_in": "fsdp", "d_out": "fsdp", "C": "tp"},
            }
        case "zero1":
            intra: Rule = {"d_in": _ZERO1_DATA, "d_out": _ZERO1_DATA, "C": "tp"}
            sites = {
                "activations": activations,
                "params/persist": intra,
                "params/persist.subset": intra,
                "params/forward": {"d_in": "fsdp", "d_out": "fsdp", "C": "tp"},
            }
        case "ddp":
            sites = {
                "activations": activations,
                "params/persist": {},
                "params/persist.subset": {},
                "params/forward": {},
            }
        case _:
            raise AssertionError(f"unknown placement preset {name!r}")
    return PlacementRules(mesh=mesh, sites=sites)


def from_config(
    spec: "str | Mapping[str, Mapping[str, str | list[str] | None]]",
    mesh: Mesh | AbstractMesh,
) -> PlacementRules:
    """`RuntimeConfig.sharding` → `PlacementRules`: a preset name, or an explicit sites
    table (YAML lists become ordered tuples — nested-axis ORDER is semantics, PR #927)."""
    if isinstance(spec, str):
        assert spec in PRESET_NAMES, f"unknown placement preset {spec!r} (have {PRESET_NAMES})"
        return preset(spec, mesh)
    sites = {
        site: {axis: tuple(v) if isinstance(v, list) else v for axis, v in rule.items()}
        for site, rule in spec.items()
    }
    missing = [s for s in REQUIRED_SITES if s not in sites]
    assert not missing, (
        f"placement table missing required sites {missing} — consumers dereference these "
        f"(declare them even if empty; see placement.REQUIRED_SITES)"
    )
    return PlacementRules(mesh=mesh, sites=sites)
