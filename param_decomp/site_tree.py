"""Structured, block-indexed decomposition sites + arch-family resolution.

`SiteTree` is the block-structured form of a decomposition — the layer index is carried as
DATA, never parsed back out of a site name. The tiled LM site specs resolve INTO it, and the
chunkwise CI resolver consumes it directly (a chunk = a slice of consecutive `BlockSites`), so
nothing downstream recovers block structure by regex-ing site-name strings.

`ArchFamily` is a target's matrix grammar as data: the ordered, exhaustive matrix set (canonical
within-block order) + the `(layer, matrix) -> site name` renderer. Each target module builds its
OWN family (`glu_transformer.FAMILY`, `llama_simple_mlp.FAMILY`), with `matrices` derived from the
same config-side `Literal` vocabulary its c-spec keys are typed by — so a c-spec key outside the
family's vocabulary is unrepresentable, not merely asserted.

The activation-TAP address grammar lives here too (`ResidIn | SiteInput`, `tap_key`/`parse_tap`,
`tap_layer`/`tap_width`): taps name locations in the same target grammar as sites, so one module
owns both vocabularies. The string forms are the wire format (`Chunk.input_taps` carries them as
pytree-static keys) — consumers (the CI-fn chunk resolver lab-side, the targets'
`read_activations`, the hidden-acts eval) must not mint or split key strings themselves.
Deliberately separate, still: `param_decomp_lab/topology/` (the consumer-side canonical
weight/component address space) — the third naming system, out of scope here.
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


_RESID_PREFIX = "resid."


@dataclass(frozen=True)
class ResidIn:
    """The residual stream ENTERING block `layer` (before its attention norm)."""

    layer: int


@dataclass(frozen=True)
class SiteInput:
    """The input activation of decomposed site `name` — the vector its matrix multiplies.

    Width is the site's d_in, NOT the residual width (down/o projections read
    intermediates) — `tap_width` consults the site dims."""

    name: str


TapAddress = ResidIn | SiteInput


def tap_key(tap: TapAddress) -> str:
    """The canonical string form — the only place these strings are minted."""
    match tap:
        case ResidIn(layer):
            return f"{_RESID_PREFIX}{layer}"
        case SiteInput(name):
            return name


def parse_tap(key: str) -> TapAddress:
    """Inverse of `tap_key`. Anything that isn't a `resid.{L}` form is a site name; the
    caller validates site names against its family grammar (they are target-specific)."""
    if key.startswith(_RESID_PREFIX):
        return ResidIn(int(key.removeprefix(_RESID_PREFIX)))
    return SiteInput(key)


def tap_layer(family: ArchFamily, key: str) -> int:
    """Global block index a `read_activations` key reads at: the block a `resid.{L}` tap
    enters, or the block a decomposed site lives in."""
    match parse_tap(key):
        case ResidIn(layer):
            return layer
        case SiteInput(name):
            return family.parse(name)[0]


def tap_width(tap: TapAddress, d_resid: int, d_in_of: Callable[[str], int]) -> int:
    """Concat width contribution of one tap: a residual tap is `d_resid` wide; a site-input
    tap is as wide as the site's d_in (`d_in_of(site_name)` — down/o projections read
    intermediates, not the residual)."""
    match tap:
        case ResidIn():
            return d_resid
        case SiteInput(name):
            return d_in_of(name)


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
