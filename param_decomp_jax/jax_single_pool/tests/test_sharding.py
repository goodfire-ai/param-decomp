"""Sharding tests. Run under simulated multi-device CPU via the env in conftest.

These guard the harness pitfall (NOTES): `shard_batch` must reconstruct the FULL
global array across the mesh, not replicate a per-process slice. The GPU-count
invariance of the whole step is validated end-to-end by
`experiments/distributed_stacked_sites.py` at 1 vs N devices (bit-identical);
that needs distinct process-level device counts so it lives in the runnable
experiment, not here.
"""

import jax
import jax.numpy as jnp

from jax_single_pool.sharding import dp_mesh, shard_batch


def test_shard_batch_preserves_global_data():
    mesh = dp_mesh()
    n = mesh.devices.size
    B = 8 * n
    full = jax.random.normal(jax.random.PRNGKey(0), (3, B, 5))
    sharded = shard_batch(full, mesh, batch_axis=1)
    assert sharded.shape == full.shape
    # the sharded array must equal the original global array (the harness pitfall
    # replicated a single slice instead, which this catches when n > 1).
    assert jnp.allclose(jnp.asarray(sharded), full)


def test_shard_batch_requires_divisible_batch():
    mesh = dp_mesh()
    n = mesh.devices.size
    if n == 1:
        return  # any batch divides 1
    full = jax.random.normal(jax.random.PRNGKey(1), (2, n + 1, 4))
    try:
        shard_batch(full, mesh, batch_axis=1)
    except AssertionError:
        return
    raise AssertionError("expected non-divisible batch to fail")


def test_jitted_sharded_inits_match_eager_values():
    """`init_*_sharded` must be a placement-only change: same values as the eager init
    fns (threefry is partitionable, so generating under jit with `out_shardings` cannot
    perturb the stream — only op fusion can reassociate the scaling, SPEC D4: rel ~1e-7),
    with the expected per-site placements (V shards C on axis 1, U on axis 0) — for a
    heterogeneous-C site set spanning attention and MLP matrices."""
    import pytest
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from jax_single_pool.adversary import init_persistent_sources
    from jax_single_pool.ci_fn import CIArch, init_ci_fn
    from jax_single_pool.llama8b import canonical_site_cs, init_decomp_vu, llama_site_specs
    from jax_single_pool.llama8b_sharding import (
        init_ci_fn_sharded,
        init_decomp_vu_sharded,
        init_sources_sharded,
    )
    from jax_single_pool.lm import SiteC, SiteSpec
    from jax_single_pool.tests.test_llama8b import _tiny_cfg
    from param_decomp_config.losses import BSCScope, SCScope

    mesh = dp_mesh()
    n = mesh.devices.size
    cfg = _tiny_cfg()
    sites = llama_site_specs(
        cfg,
        canonical_site_cs(
            (
                SiteC("layers.2.self_attn.q_proj", 8 * n),
                SiteC("layers.2.self_attn.o_proj", 16 * n),
                SiteC("layers.2.mlp.gate_proj", 8 * n),
                SiteC("layers.3.mlp.down_proj", 16 * n),
            )
        ),
    )

    vu_sharded = init_decomp_vu_sharded(sites, jax.random.PRNGKey(1), mesh)
    vu_eager = init_decomp_vu(sites, jax.random.PRNGKey(1))
    for spec in sites:
        V, U = vu_sharded.site(spec.name)
        assert isinstance(V.sharding, NamedSharding) and isinstance(U.sharding, NamedSharding)
        assert V.sharding.spec == P(None, "dp"), spec.name
        assert U.sharding.spec == P("dp", None), spec.name
    for got, want in zip(jax.tree.leaves(vu_sharded), jax.tree.leaves(vu_eager), strict=True):
        assert got.shape == want.shape and got.dtype == want.dtype
        assert jnp.allclose(jnp.asarray(got), want, rtol=1e-6, atol=0)

    if n > 1:
        indivisible = (SiteSpec("layers.2.mlp.gate_proj", cfg.n_embd, cfg.n_intermediate, n + 1),)
        with pytest.raises(AssertionError, match="not divisible"):
            init_decomp_vu_sharded(indivisible, jax.random.PRNGKey(1), mesh)

    arch = CIArch(d_model=16, n_blocks=1, n_heads=2, mlp_hidden=8 * n)
    ci_sharded = init_ci_fn_sharded(arch, sites, jax.random.PRNGKey(2), mesh)
    ci_eager = init_ci_fn(arch, sites, jax.random.PRNGKey(2))
    for got, want in zip(jax.tree.leaves(ci_sharded), jax.tree.leaves(ci_eager), strict=True):
        assert got.shape == want.shape and got.dtype == want.dtype
        assert jnp.allclose(jnp.asarray(got), want, rtol=1e-6, atol=0)

    site_names = tuple(s.name for s in sites)
    site_Cs = tuple(s.C for s in sites)
    src_sharded = init_sources_sharded(
        site_names, site_Cs, 16, SCScope(), 1, jax.random.PRNGKey(3), mesh
    )
    src_eager = init_persistent_sources(site_names, site_Cs, 16, 1, jax.random.PRNGKey(3))
    for name in site_names:
        src_sharding = src_sharded[name].sharding
        assert isinstance(src_sharding, NamedSharding)
        assert src_sharding.spec == P()
        assert jnp.allclose(jnp.asarray(src_sharded[name]), src_eager[name], rtol=1e-6, atol=0)

    # bsc: one source per batch element, batch-sharded over dp (axis 0), no cross-rank sync.
    bsc_global_batch = 4 * n
    src_bsc = init_sources_sharded(
        site_names, site_Cs, 16, BSCScope(), bsc_global_batch, jax.random.PRNGKey(3), mesh
    )
    for name, C in zip(site_names, site_Cs, strict=True):
        assert src_bsc[name].shape == (bsc_global_batch, 16, C + 1), name
        bsc_sharding = src_bsc[name].sharding
        assert isinstance(bsc_sharding, NamedSharding)
        assert bsc_sharding.spec == P("dp", None, None), name
