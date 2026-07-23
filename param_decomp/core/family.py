"""A target's matrix grammar as data: arch families + the family-parameterized site helpers.

`ArchFamily` is the ordered, exhaustive matrix set (canonical within-block order) + the
`(layer, matrix) -> site name` renderer. Each target module builds its OWN family
(`glu_transformer.FAMILY`, `llama_simple_mlp.FAMILY`), with `matrices` derived from the
`Literal` matrix vocabulary the target module itself owns — the same vocabulary the
authored c-spec keys (lab-side, `param_decomp/experiments/lm/config.py`) are typed
by, so a c-spec key outside the family's vocabulary is unrepresentable, not merely
asserted.

The activation-TAP grammar does NOT live here: tap keys are opaque strings to everything
generic (`Chunk.input_taps` carries them as pytree-static keys), and their structure is the
transformer families' own vocabulary (`param_decomp/targets/transformer_taps.py`). Deliberately separate,
still: `param_decomp/topology/` (the consumer-side canonical weight/component address
space) — the third naming system, out of scope here.
"""

from collections.abc import Callable
from dataclasses import dataclass

from param_decomp.core.components import SiteC, SiteSpec


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
