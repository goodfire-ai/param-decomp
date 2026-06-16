"""Lock the export site-order permutation for ARBITRARY multi-matrix, multi-layer
site sets — the case SPEC_AUDIT R-8 / EDGES I7 flags as load-bearing.

JAX lays the CI-fn concat blocks out in computation order (`KIND_ORDER` per layer,
layers ascending); torch concatenates in `sorted()` lexicographic module-path order.
On the production single-MLP-layer family the two coincide; for multi-layer or mixed
attn+mlp configs they DIVERGE (lexicographically `layers.10... < layers.2...`, and
within a layer `mlp.* < self_attn.*` with attention kinds sorting k,o,q,v). These are
JAX-only STRUCTURAL checks on `_concat_permutation` — no torch golden (the torch
generators are deleted; push-1 keeps `param_decomp_jax` torch-free). They prove the
permutation reorders the per-site blocks from JAX layout onto `sorted(site_names)`
and is a true bijection.
"""

import dataclasses

import numpy as np

from jax_single_pool.export import _concat_permutation, torch_site_order
from jax_single_pool.llama8b import (
    ATTN_KINDS,
    MLP_KINDS,
    canonical_site_cs,
    llama_site_specs,
    site_name,
)
from jax_single_pool.lm import SiteC, SiteSpec
from jax_single_pool.tests.test_llama8b import _tiny_cfg

# `_tiny_cfg` is 8 layers; layer 10 (chosen so its string sorts before layer 2) needs more.
_CFG = dataclasses.replace(_tiny_cfg(), n_layer=12)

# Layers chosen so lexicographic string sort REORDERS them (10 < 2 < 5 as strings),
# every layer carries a different attn+mlp mix, and per-site C varies so column-block
# offsets are non-uniform.
_MULTISITE_CS = canonical_site_cs(
    (
        SiteC(site_name(2, "q"), 3),
        SiteC(site_name(2, "k"), 4),
        SiteC(site_name(2, "v"), 5),
        SiteC(site_name(2, "o"), 6),
        SiteC(site_name(2, "gate"), 7),
        SiteC(site_name(2, "up"), 8),
        SiteC(site_name(2, "down"), 9),
        SiteC(site_name(5, "q"), 2),
        SiteC(site_name(5, "o"), 10),
        SiteC(site_name(5, "down"), 11),
        SiteC(site_name(10, "v"), 12),
        SiteC(site_name(10, "gate"), 13),
        SiteC(site_name(10, "up"), 14),
    )
)


def _sites() -> tuple[SiteSpec, ...]:
    return llama_site_specs(_CFG, _MULTISITE_CS)


def _block_slices(order: tuple[str, ...], sizes: dict[str, int]) -> dict[str, slice]:
    out: dict[str, slice] = {}
    offset = 0
    for site in order:
        out[site] = slice(offset, offset + sizes[site])
        offset += sizes[site]
    return out


def test_multisite_orders_actually_diverge():
    """Guard the premise: on this config JAX (canonical) order != torch (sorted) order,
    with the specific layer-string and within-layer reorderings R-8 calls out."""
    jax_order = tuple(s.name for s in _sites())
    sorted_order = torch_site_order(jax_order)
    assert jax_order != sorted_order

    layer2 = tuple(s for s in jax_order if s.startswith("layers.2."))
    assert layer2 == tuple(site_name(2, kind) for kind in (*ATTN_KINDS, *MLP_KINDS)), (
        "JAX within-layer order is KIND_ORDER (q,k,v,o,gate,up,down)"
    )

    # Lexicographic: layers.10.* < layers.2.* < layers.5.*, and within layer.2 the
    # mlp block (down,gate,up) precedes self_attn (k,o,q,v).
    assert sorted_order == (
        site_name(10, "gate"),
        site_name(10, "up"),
        site_name(10, "v"),
        site_name(2, "down"),
        site_name(2, "gate"),
        site_name(2, "up"),
        site_name(2, "k"),
        site_name(2, "o"),
        site_name(2, "q"),
        site_name(2, "v"),
        site_name(5, "down"),
        site_name(5, "o"),
        site_name(5, "q"),
    )


def test_permutation_maps_jax_blocks_onto_sorted_order():
    """For both the row axis (by d_in) and the column axis (by C), the permutation must
    map each site's JAX-layout block to its `sorted()`-layout block, index for index."""
    sites = _sites()
    jax_order = tuple(s.name for s in sites)
    sorted_order = torch_site_order(jax_order)

    for sizes in ({s.name: s.d_in for s in sites}, {s.name: s.C for s in sites}):
        perm = _concat_permutation(jax_order, sizes)
        jax_slices = _block_slices(jax_order, sizes)
        sorted_slices = _block_slices(sorted_order, sizes)
        # `concat_jax[perm] == concat_torch`: the rows perm pulls into a site's
        # torch-order block must be exactly that site's jax-order block.
        for site in jax_order:
            np.testing.assert_array_equal(
                perm[sorted_slices[site]],
                np.arange(jax_slices[site].start, jax_slices[site].stop),
            )


def test_permutation_is_a_bijection():
    """permute then invert is identity — the reorder loses no block and duplicates none
    (proves the row/col perms are true permutations of [0, total))."""
    sites = _sites()
    jax_order = tuple(s.name for s in sites)
    for sizes in ({s.name: s.d_in for s in sites}, {s.name: s.C for s in sites}):
        perm = _concat_permutation(jax_order, sizes)
        total = sum(sizes.values())
        assert sorted(perm.tolist()) == list(range(total))
        inverse = np.argsort(perm)
        np.testing.assert_array_equal(perm[inverse], np.arange(total))


def test_permutation_applied_to_data_reorders_blocks():
    """End-to-end on a synthetic concat: tag each block with its site, permute, and read
    back `sorted()` order — the concrete failure mode if KIND_ORDER and sorted disagree."""
    sites = _sites()
    jax_order = tuple(s.name for s in sites)
    sorted_order = torch_site_order(jax_order)
    sizes = {s.name: s.C for s in sites}

    site_index = {name: i for i, name in enumerate(jax_order)}
    tagged = np.concatenate([np.full(sizes[name], site_index[name]) for name in jax_order])
    permuted = tagged[_concat_permutation(jax_order, sizes)]

    expected = np.concatenate([np.full(sizes[name], site_index[name]) for name in sorted_order])
    np.testing.assert_array_equal(permuted, expected)
