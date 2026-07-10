"""Structured, block-indexed decomposition sites + arch-family resolution.

`SiteTree` is the block-structured form of a decomposition — the layer index is carried as
DATA, never parsed back out of a site name. The tiled LM site specs resolve INTO it, and the
chunkwise CI resolver consumes it directly (a chunk = a slice of consecutive `BlockSites`), so
nothing downstream recovers block structure by regex-ing site-name strings.

`ArchFamily` is a target's matrix grammar as data: the ordered, exhaustive matrix set (canonical
within-block order) + the `(layer, matrix) -> site name` renderer. The concrete families (GLU
for llama8b, plain-MLP for LlamaSimpleMLP) are built lab-side, binding each target's `site_name`.
"""

from collections.abc import Callable
from dataclasses import dataclass

from param_decomp.components import SiteC
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
    """A target's matrix grammar as data. `matrices` is the ordered matrix vocabulary (canonical
    within-block order); `name_of(layer, matrix)` renders the site name. Names are generated,
    never parsed."""

    key: str
    matrices: tuple[str, ...]
    name_of: Callable[[int, str], str]


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
    cs: dict[str, int] = {k: v for k, v in sites.cs.items()}
    assert set(cs) <= set(family.matrices), (
        f"c-spec matrices {set(cs) - set(family.matrices)} not in the {family.key} vocabulary "
        f"{family.matrices} (GluMatrix/SimpleMlpMatrix and the target KIND_ORDER have drifted)"
    )
    slots = tuple((m, cs[m]) for m in family.matrices if m in cs)
    layers = _select_layers(sites.layers, n_layer)
    return SiteTree(tuple(BlockSites(layer, slots) for layer in layers))
