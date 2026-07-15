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

The rule language is deliberately WEAK: name → mesh axes, first-match, no conditionals,
no expressions. Weird cases get a literal spec override on a named site, not a smarter
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
        """Upper-bound bytes moved PER STEP resharding a tensor between two sites' layouts
        (0 when the derived specs match — the owner-resident case). The honest cost model
        for the startup lint: a differing spec moves ≤ the full global tensor."""
        if self.spec_for(site_from, axes) == self.spec_for(site_to, axes):
            return 0
        return prod(shape) * jnp.dtype(dtype).itemsize

    def describe(
        self, tensors: Mapping[str, tuple[str, Axes, tuple[int, ...]]] | None = None
    ) -> str:
        """The policy as one printable table (startup log + documentation). With `tensors`
        (`{label: (site, axes, shape)}`) it also prints each tensor's derived spec and
        per-device share — the placement audit a human or agent reads before a run."""
        lines = ["placement rules:"]
        mesh_desc = ", ".join(f"{a}={s}" for a, s in self.mesh.shape.items())
        lines.append(f"  mesh: {mesh_desc}")
        for site in sorted(self.sites):
            rule = self.sites[site]
            body = ", ".join(f"{k}->{v}" for k, v in rule.items()) if rule else "(replicated)"
            lines.append(f"  {site:<18} {body}")
        if tensors:
            lines.append("derived placements:")
            for label, (site, axes, shape) in tensors.items():
                self.validate_shape(site, axes, shape)
                spec = self.spec_for(site, axes)
                share = prod(shape) // prod(self.shard_count(site, n) for n in axes)
                lines.append(f"  {label:<28} {site:<18} {str(spec):<40} per-device {share:,} elems")
        return "\n".join(lines)


# ── presets ──────────────────────────────────────────────────────────────────
# Named rule tables for the layouts this trainer has actually run. `stack` is the V/U
# shape-group stack axis (components.DecompVU); `d` covers both d_in and d_out via the
# per-tensor axes tuples. All presets share the activation waist rule (batch over the
# full data mesh) — that surface is layout-invariant (SPEC §4.1 pins).

_DATA = ("replicate", "fsdp")


def preset(name: str, mesh: Mesh | AbstractMesh) -> PlacementRules:
    """The built-in tables: `owner` (stack ÷replicate, d ÷fsdp — the D4-amended layout),
    `zero1` (intra-matrix ÷N — the retired layout, kept for A/B), `ddp` (everything
    replicated — single-node / small-model runs)."""
    activations: Rule = {"batch": _DATA, "C": "tp"}
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
                "params/persist.subset": {"d_in": _DATA, "d_out": _DATA, "C": "tp"},
                "params/forward": {"d_in": "fsdp", "d_out": "fsdp", "C": "tp"},
            }
        case "zero1":
            sites = {
                "activations": activations,
                "params/persist": {"d_in": _DATA, "d_out": _DATA, "C": "tp"},
                "params/forward": {"d_in": "fsdp", "d_out": "fsdp", "C": "tp"},
            }
        case "ddp":
            sites = {
                "activations": activations,
                "params/persist": {},
                "params/forward": {},
            }
        case _:
            raise AssertionError(f"unknown placement preset {name!r}")
    return PlacementRules(mesh=mesh, sites=sites)


def vu_axes(ndim_stack: bool = True) -> tuple[Axes, Axes]:
    """Semantic axes of the V/U stacks: `(stack, d_in, C)` and `(stack, C, d_out)`."""
    assert ndim_stack
    return ("stack", "d_in", "C"), ("stack", "C", "d_out")
