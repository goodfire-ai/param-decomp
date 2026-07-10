"""Structured, block-indexed decomposition sites + arch-family resolution.

`SiteTree` is the block-structured form of a decomposition — the layer index is carried as
DATA, never parsed back out of a site name. The tiled LM site specs resolve INTO it, and the
chunkwise CI resolver consumes it directly (a chunk = a slice of consecutive `BlockSites`), so
nothing downstream recovers block structure by regex-ing site-name strings.

`ArchFamily` is a target's matrix grammar as data: the ordered, exhaustive matrix set (canonical
within-block order) + the `(layer, matrix) -> site name` renderer. Each target module builds its
OWN family (`llama8b.FAMILY`, `llama_simple_mlp.FAMILY`), with `matrices` derived from the same
config-side `Literal` vocabulary its c-spec keys are typed by — so a c-spec key outside the
family's vocabulary is unrepresentable, not merely asserted.
"""

from collections.abc import Callable
from dataclasses import dataclass

from param_decomp.components import SiteC, SiteSpec
from param_decomp.configs import (
    AllLayers,
    GluTransformerCSpec,
    LayerList,
    LayerRange,
    LayerSelection,
    SimpleMlpCSpec,
)


@dataclass(frozen=True)
class BlockSites:
    """One transformer block's decomposed matrices, in canonical within-block (family) order.
    `layer_idx` is a field — the whole point is that structure is never thrown into a string."""

    layer_idx: int
    slots: tuple[tuple[str, int], ...]  # (matrix_type, C)


@dataclass(frozen=True)
class SiteTree:
    """A decomposition as blocks, strictly layer-ascending. The flat `SiteC` view is DERIVED
    via the family's name grammar — construction only, no inverse parse."""

    blocks: tuple[BlockSites, ...]

    def site_cs(self, name_of: Callable[[int, str], str]) -> tuple[SiteC, ...]:
        return tuple(
            SiteC(name_of(b.layer_idx, kind), c) for b in self.blocks for kind, c in b.slots
        )


@dataclass(frozen=True)
class ArchFamily:
    """A target's matrix grammar as data. `matrices` is the ordered matrix vocabulary
    (canonical within-block order); `name_of(layer, matrix)` renders a site name and
    `parse(name)` inverts it (asserting on non-site names). The config→sites→chunks path
    only ever renders; `parse` serves the flat-site-name boundary the targets keep
    (`canonical_site_cs` / `site_specs` / `tap_layer`)."""

    key: str
    matrices: tuple[str, ...]
    name_of: Callable[[int, str], str]
    parse: Callable[[str], tuple[int, str]]


def canonical_site_cs(family: ArchFamily, site_cs: tuple[SiteC, ...]) -> tuple[SiteC, ...]:
    """Canonical site order: layer-ascending, family order within a layer. Names must
    parse and be unique."""
    names = [site.name for site in site_cs]
    assert len(set(names)) == len(names), f"duplicate sites in {names}"
    rank = {matrix: i for i, matrix in enumerate(family.matrices)}

    def order_key(site: SiteC) -> tuple[int, int]:
        layer, kind = family.parse(site.name)
        return layer, rank[kind]

    return tuple(sorted(site_cs, key=order_key))


def site_specs(
    family: ArchFamily,
    site_cs: tuple[SiteC, ...],
    dims_of: Callable[[str], tuple[int, int]],
    n_layer: int,
) -> tuple[SiteSpec, ...]:
    """Shape-resolved specs in canonical order (input must already be canonical);
    `dims_of(matrix)` is the target's shape table, closed over its config."""
    assert site_cs == canonical_site_cs(family, site_cs), f"sites not in canonical order: {site_cs}"
    specs = []
    for site in site_cs:
        layer, kind = family.parse(site.name)
        assert 0 <= layer < n_layer, (site.name, n_layer)
        assert site.C >= 1, site
        specs.append(SiteSpec(site.name, *dims_of(kind), site.C))
    return tuple(specs)


def tap_layer(family: ArchFamily, key: str) -> int:
    """Global block index a `read_activations` key reads at: the block a `resid.{L}` tap
    enters, or the block a decomposed site lives in."""
    if key.startswith("resid."):
        return int(key.split(".")[1])
    return family.parse(key)[0]


def _select_layers(sel: LayerSelection, n_layer: int) -> tuple[int, ...]:
    match sel:
        case AllLayers():
            return tuple(range(n_layer))
        case LayerRange(start=start, end=end):
            assert end <= n_layer, f"layer range end {end} exceeds n_layer {n_layer}"
            return tuple(range(start, end))
        case LayerList(indices=indices):
            assert indices[-1] < n_layer, f"layer {indices[-1]} exceeds n_layer {n_layer}"
            return tuple(indices)


def resolve_site_tree(
    sites: GluTransformerCSpec | SimpleMlpCSpec, family: ArchFamily, n_layer: int
) -> SiteTree:
    """Tile the per-matrix-type `cs` across the selected layers into a `SiteTree`. Every block
    shares ONE `slots` tuple (canonical family order, only the requested matrices), so the tree
    is homogeneous by construction — which is exactly what makes the chunkwise CI fn's chunks
    homogeneous. Asserts the spec's declared family matches the target's."""
    assert sites.kind == family.key, f"c-spec family {sites.kind!r} != target family {family.key!r}"
    # cs keys are Literal-typed by the family vocabulary `matrices` derives from, so every
    # key is a family matrix by construction — ordering is the only work left.
    rank = {matrix: i for i, matrix in enumerate(family.matrices)}
    slots = tuple(sorted(sites.cs.items(), key=lambda slot: rank[slot[0]]))
    layers = _select_layers(sites.layers, n_layer)
    return SiteTree(tuple(BlockSites(layer, slots) for layer in layers))
