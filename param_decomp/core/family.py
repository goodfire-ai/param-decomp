"""A target's matrix grammar as data: arch families + the family-parameterized site helpers.

A SITE is one decomposed weight matrix of one BLOCK, and its name is always
LAYER-INDEXED — `layers.{i}.self_attn.q_proj` for the GLU family,
`h.{i}.attn.q_proj` for `simple_mlp`. The layer index is the structure, the spelling is
the family's own; `name_of(layer, matrix)` renders it and `parse(name)` inverts it,
asserting on anything malformed (a site name is never string-manipulated elsewhere).

`ArchFamily` is the ordered, exhaustive matrix set (canonical within-block order) plus the
`(layer, matrix) -> site name` renderer. Each target module builds its OWN family
(`glu_transformer.FAMILY`, `llama_simple_mlp.FAMILY`), with `matrices` derived from the
`Literal` matrix vocabulary the target module itself owns — the same vocabulary the
authored c-spec keys (lab-side, `param_decomp/experiments/lm/config.py`) are typed
by, so a c-spec key outside the family's vocabulary is unrepresentable, not merely
asserted. The family is the whole grammar: `canonical_site_cs` derives the canonical
order from it (layer-ascending, then family order) and `site_specs` resolves the shapes.

WHICH layers carry sites is therefore a CONFIG choice (the c-spec's `layers` selection),
never a target property: a target declares its family and serves any layer subset, and
layers without sites run the plain frozen block. A new target gets multi-block
decomposition by declaring an `ArchFamily` — hardcoding a block index is always the
wrong shape.

The activation-TAP grammar does NOT live here: tap keys are opaque strings to everything
generic (`Chunk.input_taps` carries them as pytree-static keys), and their structure is the
transformer families' own vocabulary (`param_decomp/targets/transformer_taps.py`). Deliberately separate,
still: `param_decomp/topology/` (the consumer-side canonical weight/component address
space) — the third naming system, out of scope here.
"""

from collections.abc import Callable
from dataclasses import dataclass

from param_decomp.core.components import SiteC, SiteDims, SiteSpec
from param_decomp.core.nonlinearity import NonlinearityPartition


@dataclass(frozen=True)
class ArchFamily:
    """A target's matrix grammar as data. `matrices` is the ordered matrix vocabulary
    (canonical within-block order); `name_of(layer, matrix)` renders a site name and
    `parse(name)` inverts it (asserting on non-site names). The config→sites→chunks path
    only ever renders; `parse` serves the flat-site-name boundary the targets keep
    (`canonical_site_cs` / `site_specs` / the family tap grammar in
    `param_decomp/targets/transformer_taps.py`)."""

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
    dims_of: Callable[[str], SiteDims],
    nonlinearity_partition_of: Callable[[str], NonlinearityPartition | None],
    n_layer: int,
) -> tuple[SiteSpec, ...]:
    """Shape-resolved specs in canonical order (input must already be canonical);
    `dims_of(matrix)` / `nonlinearity_partition_of(matrix)` are the target's shape and
    nonlinearity-unit tables, closed over its config."""
    assert site_cs == canonical_site_cs(family, site_cs), f"sites not in canonical order: {site_cs}"
    specs = []
    for site in site_cs:
        layer, kind = family.parse(site.name)
        assert 0 <= layer < n_layer, (site.name, n_layer)
        assert site.C >= 1, site
        dims = dims_of(kind)
        specs.append(
            SiteSpec(
                name=site.name,
                d_in=dims.d_in,
                d_out=dims.d_out,
                C=site.C,
                group=kind,
                nonlinearity_partition=nonlinearity_partition_of(kind),
            )
        )
    return tuple(specs)
