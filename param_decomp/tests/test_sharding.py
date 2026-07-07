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
import pytest

from param_decomp.sharding import hsdp_mesh, shard_batch

# Needs >1 jax device; hangs at the default 1 device, so gated behind --runmultidevice.
# Run via `make test-multidevice` (sets XLA_FLAGS for simulated CPU devices). See conftest.
pytestmark = pytest.mark.multidevice


def test_shard_batch_preserves_global_data():
    mesh = hsdp_mesh()
    n = mesh.devices.size
    B = 8 * n
    full = jax.random.normal(jax.random.PRNGKey(0), (3, B, 5))
    sharded = shard_batch(full, mesh, batch_axis=1)
    assert sharded.shape == full.shape
    # the sharded array must equal the original global array (the harness pitfall
    # replicated a single slice instead, which this catches when n > 1).
    assert jnp.allclose(jnp.asarray(sharded), full)


def test_shard_batch_requires_divisible_batch():
    mesh = hsdp_mesh()
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
    """`init_*_placed` (model-owned `.shardings`) must be a placement-only change: same
    values as the host (unsharded) init fns (threefry is partitionable, so generating under
    jit with `out_shardings` cannot perturb the stream — only op fusion can reassociate the
    scaling, SPEC D4: rel ~1e-7), with the expected pure-HSDP placements (V FSDP d_in on
    `fsdp`, U FSDP d_out on `fsdp`, C never sharded) — for a heterogeneous-C site set
    spanning attention and MLP matrices."""
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from param_decomp.adversary import init_persistent_sources
    from param_decomp.ci_fn import (
        Chunk,
        ChunkwiseTransformerCIArch,
        build_ci_fn,
    )
    from param_decomp.components import SiteC, init_decomp_vu
    from param_decomp.configs import BSCScope, SCScope
    from param_decomp.targets.llama8b import canonical_site_cs, llama_site_specs
    from param_decomp.targets.llama8b_sharding import (
        init_ci_fn_placed,
        init_decomp_vu_placed,
        init_sources_sharded,
    )
    from param_decomp.tests.test_llama8b import _tiny_cfg

    # The HSDP mesh `(replicate, fsdp)`: on the n-device CPU sim with n not a multiple of 8,
    # `fsdp` takes the full count and `replicate` is 1. V FSDP-shards d_in on `fsdp`, U FSDP
    # d_out on `fsdp`; C is never sharded. The tiny target's matrix dims (n_embd=32,
    # n_intermediate=64, qkv head dims) all tile the sim device counts (1/2/4).
    n = jax.device_count()
    mesh = hsdp_mesh()
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
    # Placement is MODEL-OWNED and true ÷N ZeRO-1, split across the data + TP axes: V shards
    # d_in over the data axes and C over `tp` (`P(("replicate","fsdp"), "tp")`); U shards C
    # over `tp` and d_out over the data axes (`P("tp", ("replicate","fsdp"))`). C now carries
    # the Megatron-C `tp` factor, so master + Adam still shard ÷(replicate·fsdp·tp) = ÷N. At
    # tp=1 the tp axis is size 1 (C effectively unsharded).
    full = ("replicate", "fsdp")
    vu_placed = init_decomp_vu_placed(sites, jax.random.PRNGKey(1), mesh)
    vu_eager = init_decomp_vu(sites, jax.random.PRNGKey(1))
    for spec in sites:
        V, U = vu_placed.site(spec.name)
        assert isinstance(V.sharding, NamedSharding) and isinstance(U.sharding, NamedSharding)
        assert V.sharding.spec == P(full, "tp"), (spec.name, V.sharding.spec)
        assert U.sharding.spec == P("tp", full), (spec.name, U.sharding.spec)
    # The placed init runs vmap-stacked per shape group (compile-time optimization) and
    # must be BIT-identical to the jitted per-site `init_decomp_vu` it replaced (vmap over
    # the same per-site keys) — the trajectory anchor. The unjitted eager reference differs
    # from EITHER jitted path by ~1 ULP (XLA fusion), so it only gets allclose.
    from functools import partial

    import equinox as eqx

    per_site_shardings = eqx.filter_eval_shape(
        partial(init_decomp_vu, sites), jax.random.PRNGKey(1)
    ).shardings(mesh)
    vu_jitted_per_site = jax.jit(partial(init_decomp_vu, sites), out_shardings=per_site_shardings)(
        jax.random.PRNGKey(1)
    )
    for got, want in zip(
        jax.tree.leaves(vu_placed), jax.tree.leaves(vu_jitted_per_site), strict=True
    ):
        assert got.shape == want.shape and got.dtype == want.dtype
        assert jnp.array_equal(jnp.asarray(got), jnp.asarray(want))
    for got, want in zip(jax.tree.leaves(vu_placed), jax.tree.leaves(vu_eager), strict=True):
        assert jnp.allclose(jnp.asarray(got), want, rtol=1e-6, atol=0)

    # A declared ÷N shard dim that does NOT tile the device count is a loud crash inside
    # `.shardings` (fail-fast), not a silent replicate. (Only observable at n > 1.)
    if n > 1:
        from param_decomp.components import SiteSpec

        # d_in = n+1 (does not tile N=n) -> DecompVU.shardings must crash.
        indivisible = (SiteSpec("layers.2.mlp.gate_proj", n + 1, 8 * n, 16),)
        try:
            init_decomp_vu_placed(indivisible, jax.random.PRNGKey(1), mesh)
        except AssertionError:
            pass
        else:
            raise AssertionError("expected a non-dividing d_in to fail in DecompVU.shardings")

    first_block = min(int(s.name.split(".")[1]) for s in sites)
    arch = ChunkwiseTransformerCIArch(
        chunks=(
            Chunk(input_taps=(f"resid.{first_block}",), output_sites=tuple(s.name for s in sites)),
        ),
        input_dim=cfg.n_embd,
        d_model=16,
        n_blocks=1,
        n_heads=2,
        mlp_hidden=8 * n,
    )
    ci_placed = init_ci_fn_placed(arch, sites, jax.random.PRNGKey(2), mesh)
    ci_eager = build_ci_fn(arch, sites, jax.random.PRNGKey(2))
    for got, want in zip(jax.tree.leaves(ci_placed), jax.tree.leaves(ci_eager), strict=True):
        assert got.shape == want.shape and got.dtype == want.dtype
        assert jnp.allclose(jnp.asarray(got), want, rtol=1e-6, atol=0)

    site_names = tuple(s.name for s in sites)
    site_Cs = tuple(s.C for s in sites)
    src_sharded = init_sources_sharded(
        site_names, site_Cs, 16, SCScope(), 1, jnp.float32, jax.random.PRNGKey(3), mesh
    )
    src_eager = init_persistent_sources(
        site_names, site_Cs, (1, 16), jnp.float32, jax.random.PRNGKey(3)
    )
    # The sharded init runs vmap-stacked per C group (compile-time optimization) and must
    # be BIT-identical to the per-site init (same per-site keys): uniform draws are pure
    # threefry bit-ops, so even eager-vs-jit is exact.
    for name in site_names:
        src_sharding = src_sharded[name].sharding
        assert isinstance(src_sharding, NamedSharding)
        assert src_sharding.spec == P()
        assert jnp.array_equal(jnp.asarray(src_sharded[name]), src_eager[name]), name

    # bsc: one source per batch element, batch-sharded over the full mesh (axis 0), no
    # cross-rank sync.
    bsc_global_batch = 4 * n
    src_bsc = init_sources_sharded(
        site_names,
        site_Cs,
        16,
        BSCScope(),
        bsc_global_batch,
        jnp.float32,
        jax.random.PRNGKey(3),
        mesh,
    )
    for name, C in zip(site_names, site_Cs, strict=True):
        assert src_bsc[name].shape == (bsc_global_batch, 16, C + 1), name
        bsc_sharding = src_bsc[name].sharding
        assert isinstance(bsc_sharding, NamedSharding)
        assert bsc_sharding.spec == P(("replicate", "fsdp"), None, None), name


def test_fresh_pgd_c_bc_sources_are_replica_identical():
    """Fresh-PGD `c`/`bc` sources must be REPLICA-IDENTICAL across every shard (issue
    #660; SPEC S16, D4): the `c` -> `(1,1,C+1)` / `bc` -> `(B,1,C+1)` leaf carries no
    sharded leading axis, so the adversarial source the masks see must hold the same
    values on every device. Replica-identity follows from the init key being replicated
    (the trainer derives it from `fold_in(run_key, step)`, identical on all processes).

    Asserts: (a) the per-shard buffers of the replicated init all equal the eager
    single-device init, and (b) the placement is fully replicated (`P()`), at the
    test's device count (run at 1 AND `--xla_force_host_platform_device_count=4`).
    """
    from functools import partial

    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from param_decomp.adversary import init_fresh_pgd_sources
    from param_decomp.components import SiteSpec

    mesh = hsdp_mesh()
    n = mesh.devices.size
    batch = 4 * n
    seq = 7
    sites = (
        SiteSpec("layers.2.self_attn.q_proj", 16, 16, 8),
        SiteSpec("layers.3.mlp.down_proj", 8, 16, 13),
    )

    for scope in ("c", "bc"):
        key = jax.random.PRNGKey(660)
        eager = init_fresh_pgd_sources(sites, "random", scope, (batch, seq), key)
        init = partial(init_fresh_pgd_sources, sites, "random", scope, (batch, seq))
        repl = NamedSharding(mesh, P())
        sharded = jax.jit(init, out_shardings=repl)(key)
        for site in sites:
            leaf = sharded[site.name]
            assert isinstance(leaf.sharding, NamedSharding)
            assert leaf.sharding.spec == P(), (scope, site.name)
            for shard in leaf.addressable_shards:
                assert jnp.array_equal(jnp.asarray(shard.data), eager[site.name]), (
                    scope,
                    site.name,
                )
