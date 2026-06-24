"""Global-chunkwise-hybrid CI fn: shape/partition contract, gradient flow, the
`block_attention` ablation, and the defining behavioral property — block-axis attention
lets one block's CI depend on ANOTHER block's residual input (cross-block flow), which the
ablated (block_attention=False) and chunkwise CI fns cannot do."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from param_decomp.ci_fn import (
    GlobalChunkwiseHybridCIArch,
    GlobalChunkwiseHybridCIFn,
    build_ci_fn,
    init_global_chunkwise_hybrid_ci_fn,
)
from param_decomp.components import SiteSpec
from param_decomp.sharding import dp_mesh
from param_decomp.targets.llama8b_sharding import init_ci_fn_placed

N_EMBD = 8
SITE_LAYOUT = (("attn.q_proj", 3), ("mlp.down_proj", 4))  # c_chunk = 7
PREFIXES = ("h.0", "h.1")
SITES = tuple(
    SiteSpec(f"{prefix}.{suffix}", d_in=N_EMBD, d_out=N_EMBD, C=c)
    for prefix in PREFIXES
    for suffix, c in SITE_LAYOUT
)


def _arch(block_attention: bool) -> GlobalChunkwiseHybridCIArch:
    return GlobalChunkwiseHybridCIArch(
        block_taps=("resid.0", "resid.1"),
        block_site_prefixes=PREFIXES,
        site_layout=SITE_LAYOUT,
        n_embd=N_EMBD,
        n_model_blocks=2,
        d_model=16,
        n_blocks=2,
        n_heads=2,
        mlp_hidden=32,
        block_attention=block_attention,
    )


def _ci_fn(block_attention: bool, seed: int) -> GlobalChunkwiseHybridCIFn:
    return init_global_chunkwise_hybrid_ci_fn(
        _arch(block_attention), SITES, jax.random.PRNGKey(seed)
    )


def _taps(key: jax.Array, b: int = 2, t: int = 5) -> dict[str, jax.Array]:
    k0, k1 = jax.random.split(key)
    return {
        "resid.0": jax.random.normal(k0, (b, t, N_EMBD)),
        "resid.1": jax.random.normal(k1, (b, t, N_EMBD)),
    }


def test_shapes_and_partition():
    ci_fn = build_ci_fn(_arch(block_attention=True), SITES, jax.random.PRNGKey(0))
    assert set(ci_fn.input_names) == {"resid.0", "resid.1"}
    assert set(ci_fn.output_names) == {s.name for s in SITES}
    b, t = 2, 5
    ci = ci_fn(_taps(jax.random.PRNGKey(1), b, t))
    for s in SITES:
        for view in (ci.logits, ci.lower, ci.upper):
            assert view[s.name].shape == (b, t, s.C), (s.name, view[s.name].shape)
        assert jnp.all(ci.lower[s.name] >= 0.0) and jnp.all(ci.lower[s.name] <= 1.0)


def test_gradients_flow():
    ci_fn = _ci_fn(block_attention=True, seed=2)
    taps = _taps(jax.random.PRNGKey(3))

    def loss(fn: GlobalChunkwiseHybridCIFn) -> jax.Array:
        ci = fn(taps)
        return sum((jnp.sum(v**2) for v in ci.logits.values()), start=jnp.zeros(()))

    grads = eqx.filter_grad(loss)(ci_fn)
    tr = grads.transformer
    leaves = [tr.in_proj_w, tr.block_pos_emb, tr.out_w]
    leaves += [b.t_wq for b in tr.blocks]
    leaves += [b.b_wq for b in tr.blocks]
    for g in leaves:
        assert g is not None and float(jnp.sum(jnp.abs(g))) > 0.0


def test_block_attention_ablation_drops_leaves():
    on = _ci_fn(block_attention=True, seed=4)
    off = _ci_fn(block_attention=False, seed=4)
    for block in on.transformer.blocks:
        assert block.b_wq is not None and block.b_wo is not None
    for block in off.transformer.blocks:
        assert block.b_wq is None and block.b_wk is None
        assert block.b_wv is None and block.b_wo is None
    ci = off(_taps(jax.random.PRNGKey(5)))  # the ablated fn still produces the full contract
    assert set(ci.logits) == {s.name for s in SITES}


def _block0_logits(ci_fn: GlobalChunkwiseHybridCIFn, taps: dict[str, jax.Array]) -> jax.Array:
    logits = ci_fn(taps).logits
    return jnp.concatenate([logits[f"h.0.{suffix}"] for suffix, _ in SITE_LAYOUT], axis=-1)


def test_block_attention_enables_cross_block_flow():
    """With block attention ON, perturbing block 1's residual changes block 0's CI; with it
    OFF (shared params but no block-axis mixing), block 0's CI is invariant to block 1's
    input — the property that motivates the hybrid over the chunkwise CI fn."""
    taps = _taps(jax.random.PRNGKey(6))
    perturbed = {**taps, "resid.1": taps["resid.1"] + 1.0}

    on = _ci_fn(block_attention=True, seed=7)
    d_on = float(jnp.max(jnp.abs(_block0_logits(on, taps) - _block0_logits(on, perturbed))))
    assert d_on > 1e-4, d_on

    off = _ci_fn(block_attention=False, seed=7)
    d_off = float(jnp.max(jnp.abs(_block0_logits(off, taps) - _block0_logits(off, perturbed))))
    assert d_off < 1e-6, d_off


# c_chunk / mlp_hidden / d_model must tile the mesh; keep them multiples of 4 so the test
# is meaningful at both 1 device AND `--xla_force_host_platform_device_count=4`.
_SHARD_LAYOUT = (("attn.q_proj", 4), ("mlp.down_proj", 4))  # c_chunk = 8
_SHARD_SITES = tuple(
    SiteSpec(f"{prefix}.{suffix}", d_in=N_EMBD, d_out=N_EMBD, C=c)
    for prefix in PREFIXES
    for suffix, c in _SHARD_LAYOUT
)


def _shard_arch(d_model: int, n_heads: int, mlp_hidden: int) -> GlobalChunkwiseHybridCIArch:
    return GlobalChunkwiseHybridCIArch(
        block_taps=("resid.0", "resid.1"),
        block_site_prefixes=PREFIXES,
        site_layout=_SHARD_LAYOUT,
        n_embd=N_EMBD,
        n_model_blocks=2,
        d_model=d_model,
        n_blocks=2,
        n_heads=n_heads,
        mlp_hidden=mlp_hidden,
        block_attention=True,
    )


def test_shardings_place_like_eager():
    """`init_ci_fn_placed` (jit init with `.shardings(mesh)` out_shardings) must yield the
    SAME values as the single-device eager build, and every leaf a `NamedSharding` — pins the
    Megatron 2-D layout (qkv/out-proj/MLP) + replicated block_pos_emb/biases/inv_freq. Run at
    1 AND 4 simulated devices."""
    mesh = dp_mesh()
    n = mesh.devices.size
    arch = _shard_arch(d_model=16, n_heads=2, mlp_hidden=8 * n)
    placed = init_ci_fn_placed(arch, _SHARD_SITES, jax.random.PRNGKey(2), mesh)
    eager = build_ci_fn(arch, _SHARD_SITES, jax.random.PRNGKey(2))
    for got, want in zip(jax.tree.leaves(placed), jax.tree.leaves(eager), strict=True):
        assert got.shape == want.shape and got.dtype == want.dtype
        assert isinstance(got.sharding, NamedSharding)
        assert jnp.allclose(jnp.asarray(got), want, rtol=1e-6, atol=0)
    # block_pos_emb replicates
    assert isinstance(placed, GlobalChunkwiseHybridCIFn)
    pos_sharding = placed.transformer.block_pos_emb.sharding
    assert isinstance(pos_sharding, NamedSharding)
    assert pos_sharding.spec == P()


def test_shardings_fail_fast_on_indivisible_dim():
    """A shard axis that doesn't tile the mesh is a loud crash in `.shardings`, not a silent
    replicate (only observable at n > 1)."""
    mesh = dp_mesh()
    n = mesh.devices.size
    if n == 1:
        return
    bad = _shard_arch(d_model=16, n_heads=2, mlp_hidden=8 * n + 1)  # mlp_hidden not divisible
    try:
        init_ci_fn_placed(bad, _SHARD_SITES, jax.random.PRNGKey(2), mesh)
    except AssertionError:
        pass
    else:
        raise AssertionError("expected a non-dividing mlp_hidden to fail in .shardings")
