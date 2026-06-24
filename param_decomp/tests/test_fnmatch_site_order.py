"""fnmatch site-resolution: JAX canonical order vs torch first-match order (E21, S5/S7/S10).

torch (`param_decomp/decomposition_targets.py::resolve_decomposition_targets`, the
ORACLE) resolves a target list by looping patterns in CONFIG order (outer) over
`named_modules()` in DFS pre-order (inner), first-match-wins into an insertion-ordered
dict. The resolved site ORDER is therefore pattern-major, then named_modules-major.

JAX (`canonical_site_cs` / `expand_wildcard_site_cs`) instead CANONICALIZES the resolved
set to layer-ascending, then `KIND_ORDER` within a layer — independent of the pattern
order in the yaml.

Site order is RNG- and concat/split-load-bearing (S10), so the resolved SET must match
torch exactly. The ORDER convention is a separate, still-open decision: this test
asserts SET-equality unconditionally and PINS the
ORDER divergence for the configs where it bites, so a silent convergence/regression is
caught.

The torch oracle here is reimplemented torch-free from the same algorithm + the vendored
module-attribute order (which IS each arch's `named_modules()` DFS order); the test asserts
that vendored attribute order equals each arch's `KIND_ORDER`, so the reimplementation
stays honest if either side moves.
"""

import fnmatch

import pytest

from param_decomp.components import SiteC
from param_decomp.targets import llama8b, llama_simple_mlp


def _named_modules_order(
    layer_prefix: str,
    attn_submodule: str,
    mlp_submodule: str,
    kind_order: tuple[str, ...],
    attn_kinds: tuple[str, ...],
    n_layer: int,
) -> tuple[str, ...]:
    """Decomposable leaf-module names in torch `named_modules()` DFS pre-order.

    Per layer torch visits the attention submodule (its q/k/v/o children in definition
    order) then the mlp submodule (its children) — i.e. `kind_order` within a layer,
    layers ascending. We assert `kind_order` is exactly attn-kinds-then-mlp-kinds so this
    mirror of the module tree can't drift from the arch's real child order."""
    assert kind_order == attn_kinds + tuple(k for k in kind_order if k not in attn_kinds)
    names: list[str] = []
    for layer in range(n_layer):
        for kind in kind_order:
            submodule = attn_submodule if kind in attn_kinds else mlp_submodule
            names.append(f"{layer_prefix}.{layer}.{submodule}.{kind}")
    return tuple(names)


def _torch_resolve(targets: tuple[SiteC, ...], module_names: tuple[str, ...]) -> tuple[SiteC, ...]:
    """Replica of torch `resolve_decomposition_targets`: patterns outer (config order),
    named_modules inner, first-match-wins into an insertion-ordered dict, dedup raises."""
    resolved: dict[str, int] = {}
    for target in targets:
        matched_any = False
        for name in module_names:
            if fnmatch.fnmatch(name, target.name):
                matched_any = True
                assert name not in resolved, f"module {name!r} matches multiple patterns"
                resolved[name] = target.C
        assert matched_any, f"pattern {target.name!r} matched no modules"
    return tuple(SiteC(name, c) for name, c in resolved.items())


def _llama8b_module_names(n_layer: int) -> tuple[str, ...]:
    return _named_modules_order(
        "layers",
        "self_attn",
        "mlp",
        tuple(f"{k}_proj" for k in llama8b.KIND_ORDER),
        tuple(f"{k}_proj" for k in llama8b.ATTN_KINDS),
        n_layer,
    )


def _simple_mlp_module_names(n_layer: int) -> tuple[str, ...]:
    return _named_modules_order(
        "h",
        "attn",
        "mlp",
        llama_simple_mlp.KIND_ORDER,
        llama_simple_mlp.ATTN_KINDS,
        n_layer,
    )


# ----- production yaml: llama8b L18 gate/up/down at one C (b128 / C49k families) -----


def test_llama8b_single_layer_mlp_set_matches_torch():
    """`layers.18.mlp.{gate,up,down}_proj` — JAX canonical vs torch first-match."""
    targets = (
        SiteC("layers.18.mlp.gate_proj", 24576),
        SiteC("layers.18.mlp.up_proj", 24576),
        SiteC("layers.18.mlp.down_proj", 24576),
    )
    jax_sites = llama8b.canonical_site_cs(targets)
    torch_sites = _torch_resolve(targets, _llama8b_module_names(32))

    assert set(jax_sites) == set(torch_sites)
    # Single C-equal family with one pattern per module in computation order: orders agree.
    assert jax_sites == torch_sites


# ----- multi-layer mixed attn+mlp: the order divergence actually bites here -----


def test_simple_mlp_wildcard_mixed_set_matches_torch():
    """Pile config (`pile_llama_simple_mlp_4l_pgd1`): `h.*` wildcards over c_fc, down_proj,
    q/k/v/o at different Cs, in a pattern order that is NOT canonical."""
    n_layer = 4
    wildcard_targets = (
        SiteC("h.*.mlp.c_fc", 3072),
        SiteC("h.*.mlp.down_proj", 3584),
        SiteC("h.*.attn.q_proj", 512),
        SiteC("h.*.attn.k_proj", 512),
        SiteC("h.*.attn.v_proj", 1024),
        SiteC("h.*.attn.o_proj", 1024),
    )
    jax_sites = llama_simple_mlp.expand_wildcard_site_cs(wildcard_targets, n_layer)
    torch_sites = _torch_resolve(wildcard_targets, _simple_mlp_module_names(n_layer))

    assert set(jax_sites) == set(torch_sites)

    # ORDER divergence — pinned, not yet reconciled.
    # JAX: layer-ascending then KIND_ORDER. torch: pattern-major (all c_fc, then all
    # down_proj, ...), then layer-ascending within a pattern.
    assert jax_sites != torch_sites
    assert jax_sites[:2] == (SiteC("h.0.attn.q_proj", 512), SiteC("h.0.attn.k_proj", 512))
    assert torch_sites[:2] == (SiteC("h.0.mlp.c_fc", 3072), SiteC("h.1.mlp.c_fc", 3072))


def test_simple_mlp_canonical_pattern_order_matches_torch():
    """When the yaml lists per-kind wildcards in canonical KIND_ORDER, torch's
    pattern-major order coincides with JAX canonical only up to the layer/kind nesting
    swap — torch is kind-major-across-layers, JAX is layer-major-across-kinds — so they
    still differ for >1 layer. Documents that pattern order alone can't make them agree."""
    n_layer = 2
    targets = tuple(
        SiteC(f"h.*.attn.{kind}", 512)
        if kind in llama_simple_mlp.ATTN_KINDS
        else SiteC(f"h.*.mlp.{kind}", 512)
        for kind in llama_simple_mlp.KIND_ORDER
    )
    jax_sites = llama_simple_mlp.expand_wildcard_site_cs(targets, n_layer)
    torch_sites = _torch_resolve(targets, _simple_mlp_module_names(n_layer))

    assert set(jax_sites) == set(torch_sites)
    assert jax_sites[1] == SiteC("h.0.attn.k_proj", 512)  # layer-major: next kind, same layer
    assert torch_sites[1] == SiteC("h.1.attn.q_proj", 512)  # kind-major: same kind, next layer
    assert jax_sites != torch_sites


def test_torch_resolve_rejects_overlapping_patterns():
    """Oracle fidelity: torch raises when two patterns match the same module."""
    targets = (SiteC("h.*.attn.q_proj", 512), SiteC("h.0.attn.q_proj", 256))
    with pytest.raises(AssertionError, match="multiple patterns"):
        _torch_resolve(targets, _simple_mlp_module_names(2))
