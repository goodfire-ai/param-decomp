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

from param_decomp.sharding import dp_mesh, shard_batch

# Needs >1 jax device; hangs at the default 1 device, so gated behind --runmultidevice.
# Run via `make test-multidevice` (sets XLA_FLAGS for simulated CPU devices). See conftest.
pytestmark = pytest.mark.multidevice


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


def test_mesh_is_3d_hsdp_tp():
    """The mesh is 3-D `(replicate, dp, tp)`: `dp · tp` tiles the node size, `replicate`
    carries the rest. ZeRO-1 ÷N: V/U master + Adam state shard the FSDP dim over BOTH
    data-parallel axes (`DATA_AXES = replicate × dp`) + C on `tp`, so the master is ÷ the
    FULL mesh (carries `replicate`). The batch shards over `DATA_AXES`. Most informative at
    >8 sim devices (replicate>1)."""
    import equinox as eqx
    import optax
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from param_decomp.components import SiteC, init_decomp_vu
    from param_decomp.sharding import DATA_AXES
    from param_decomp.targets.llama8b import canonical_site_cs, llama_site_specs
    from param_decomp.targets.llama8b_sharding import init_decomp_vu_placed
    from param_decomp.tests.test_llama8b import _tiny_cfg

    n = jax.device_count()
    node = min(n, 8)
    tp = 2 if node % 2 == 0 else 1
    mesh = dp_mesh(tp=tp)
    assert mesh.axis_names == ("replicate", "dp", "tp")
    assert mesh.shape["tp"] == tp
    assert mesh.shape["dp"] == node // tp
    assert mesh.shape["replicate"] == n // node
    assert mesh.shape["replicate"] * mesh.shape["dp"] * mesh.shape["tp"] == n

    # batch shards over replicate × dp, NOT tp.
    n_data = mesh.shape["replicate"] * mesh.shape["dp"]
    full = jax.random.normal(jax.random.PRNGKey(0), (1, 8 * n_data, 4))
    sharded = shard_batch(full, mesh, batch_axis=1)
    assert isinstance(sharded.sharding, NamedSharding)
    assert sharded.sharding.spec == P(None, DATA_AXES, None)
    assert jnp.allclose(jnp.asarray(sharded), full)  # global data preserved

    # ZeRO-1 ÷N: V FSDP-shards d_in over DATA_AXES (carries replicate) + C on tp; the Adam
    # mu/nu inherit the master spec (so the optimizer state is ÷ the full mesh).
    cfg = _tiny_cfg()
    c = 8 * mesh.shape["tp"]
    sites = llama_site_specs(cfg, canonical_site_cs((SiteC("layers.0.mlp.gate_proj", c),)))
    vu = init_decomp_vu_placed(sites, jax.random.PRNGKey(1), mesh)
    eager = init_decomp_vu(sites, jax.random.PRNGKey(1))
    for name, (V, U) in vu.vu.items():
        assert isinstance(V.sharding, NamedSharding) and isinstance(U.sharding, NamedSharding)
        assert V.sharding.spec == P(DATA_AXES, "tp"), name  # d_in ÷ replicate·dp, C on tp
        assert U.sharding.spec == P("tp", DATA_AXES), name  # gate_proj (MLP): d_out ÷ DATA_AXES
    for got, want in zip(jax.tree.leaves(vu), jax.tree.leaves(eager), strict=True):
        assert jnp.allclose(jnp.asarray(got), want, rtol=1e-6, atol=0)  # placement-only

    # the V/U Adam moments carry the master sharding => ÷N optimizer state.
    opt = optax.adamw(1e-3, weight_decay=0.0)
    ostate = opt.init(eqx.filter(vu, eqx.is_array))
    moment_specs = {
        str(leaf.sharding.spec)
        for leaf in jax.tree.leaves(ostate)
        if eqx.is_array(leaf) and leaf.ndim == 2  # mu/nu for the 2-D V/U, not scalar counts
    }
    assert moment_specs == {str(P(DATA_AXES, "tp")), str(P("tp", DATA_AXES))}, moment_specs

    # the CI fn (the ~32B transformer) is ALSO ÷N: its weights' FSDP leg carries DATA_AXES,
    # so its Adam mu/nu shard ÷ the full mesh too — the dominant optimizer-state memory.
    from param_decomp.ci_fn import (
        Chunk,
        ChunkwiseTransformerCIArch,
    )
    from param_decomp.targets.llama8b import parse_site_name
    from param_decomp.targets.llama8b_sharding import init_ci_fn_placed

    first_block = min(parse_site_name(s.name)[0] for s in sites)
    arch = ChunkwiseTransformerCIArch(
        chunks=(
            Chunk(input_taps=(f"resid.{first_block}",), output_sites=tuple(s.name for s in sites)),
        ),
        input_dim=cfg.n_embd,
        d_model=8 * mesh.shape["tp"] * mesh.shape["replicate"] * mesh.shape["dp"],
        n_blocks=1,
        n_heads=2,
        mlp_hidden=8 * mesh.shape["tp"],
    )  # d_model tiles tp AND replicate·dp (so it shards both ways)
    ci_fn = init_ci_fn_placed(arch, sites, jax.random.PRNGKey(2), mesh)
    ci_ostate = optax.adamw(1e-3, weight_decay=0.0).init(eqx.filter(ci_fn, eqx.is_array))
    # every CI-fn weight whose FSDP (d_model / total_d_in) leg is sharded must name DATA_AXES
    # in its mu/nu spec (the leading n_chunks axis is unsharded; tp on the Megatron dim).
    ci_specs = {
        str(leaf.sharding.spec)
        for leaf in jax.tree.leaves(ci_ostate)
        if eqx.is_array(leaf) and leaf.ndim == 3  # the [n_chunks, in, out] stacked weights
    }
    assert all("replicate" in s and "dp" in s for s in ci_specs), ci_specs
    # at least one weight is sharded both ways (FSDP ÷N + tp Megatron):
    assert any(("replicate" in s and "tp" in s) for s in ci_specs), ci_specs


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
    """`init_*_placed` (model-owned `.shardings`) must be a placement-only change: same
    values as the host (unsharded) init fns (threefry is partitionable, so generating under
    jit with `out_shardings` cannot perturb the stream — only op fusion can reassociate the
    scaling, SPEC D4: rel ~1e-7), with the expected per-site placements (V shards C on axis
    1, U on axis 0) — for a heterogeneous-C site set spanning attention and MLP matrices."""
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

    # All devices on the `tp` axis (dp=1): V/U C-shard and the CI fn's within-chunk Megatron
    # both live on `tp`, and the single test chunk (n_chunks=1) trivially tiles dp=1.
    # chunk-on-dp tiling (n_chunks % dp) is exercised by the CPU multi-device mesh sim.
    n = jax.device_count()
    mesh = dp_mesh(tp=n)
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
    # Placement is MODEL-OWNED and uniform (ZeRO-1 ÷N): V FSDP-shards d_in over BOTH
    # data-parallel axes `DATA_AXES = replicate × dp` + C on `tp` (`P(DATA_AXES,"tp")`); U
    # shards C on `tp` + d_out FSDP over `DATA_AXES` (`P("tp",DATA_AXES)`), EXCEPT the
    # attention q/k/v sites keep d_out REPLICATED (`P("tp",None)`) — d_out there is the head
    # dim, re-sharded to `tp` at the attention seam. At an axis of size 1 the shard is
    # trivially replicated; the SPEC is unchanged.
    from param_decomp.sharding import DATA_AXES

    vu_placed = init_decomp_vu_placed(sites, jax.random.PRNGKey(1), mesh)
    vu_eager = init_decomp_vu(sites, jax.random.PRNGKey(1))
    qkv = {"layers.2.self_attn.q_proj"}  # the only q/k/v site in this set (d_out replicated)
    for spec in sites:
        V, U = vu_placed.site(spec.name)
        assert isinstance(V.sharding, NamedSharding) and isinstance(U.sharding, NamedSharding)
        assert V.sharding.spec == P(DATA_AXES, "tp"), spec.name
        want_u = P("tp", None) if spec.name in qkv else P("tp", DATA_AXES)
        assert U.sharding.spec == want_u, spec.name
    for got, want in zip(jax.tree.leaves(vu_placed), jax.tree.leaves(vu_eager), strict=True):
        assert got.shape == want.shape and got.dtype == want.dtype
        assert jnp.allclose(jnp.asarray(got), want, rtol=1e-6, atol=0)

    # A declared shard axis that does NOT tile the mesh is a loud crash inside `.shardings`
    # (fail-fast), not a silent replicate. (Only observable at n > 1.)
    if n > 1:
        indivisible = llama_site_specs(cfg, (SiteC("layers.2.mlp.gate_proj", n + 1),))
        try:
            init_decomp_vu_placed(indivisible, jax.random.PRNGKey(1), mesh)
        except AssertionError:
            pass
        else:
            raise AssertionError("expected a non-dividing C to fail in DecompVU.shardings")

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
    for name in site_names:
        src_sharding = src_sharded[name].sharding
        assert isinstance(src_sharding, NamedSharding)
        assert src_sharding.spec == P()
        assert jnp.allclose(jnp.asarray(src_sharded[name]), src_eager[name], rtol=1e-6, atol=0)

    # bsc: one source per batch element, batch-sharded over the data axes (replicate × dp,
    # axis 0), no cross-rank sync.
    from param_decomp.sharding import DATA_AXES

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
        assert bsc_sharding.spec == P(DATA_AXES, None, None), name


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

    mesh = dp_mesh()
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
