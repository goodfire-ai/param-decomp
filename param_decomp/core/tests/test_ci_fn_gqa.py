"""GQA in the chunkwise CI transformer: grouping semantics, param shapes, MHA identity."""

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from param_decomp.core.ci_fn import (
    CI_FN_RMS_EPS,
    Chunk,
    ChunkwiseTransformerCIArch,
    CIAttention,
    CIBlock,
    GQACIAttention,
    MHACIAttention,
    _weightless_rms_norm,
    build_ci_fn,
    init_chunkwise_transformer_ci_fn,
)
from param_decomp.core.components import SiteSpec
from param_decomp.vendored_jax.llama import apply_rope, repeat_kv, rope_cos_sin


def _arch(attention: CIAttention, sites: tuple[SiteSpec, ...]) -> ChunkwiseTransformerCIArch:
    return ChunkwiseTransformerCIArch(
        chunks=(Chunk(input_taps=("resid.0",), output_sites=tuple(s.name for s in sites)),),
        input_dim=12,
        d_model=16,
        n_blocks=2,
        attention=attention,
        ffn_hidden=32,
        ffn_kind="gelu",
        learned_norm_scale=False,
    )


SITES = (SiteSpec("layers.0.q_proj", 12, 12, 3), SiteSpec("layers.0.mlp", 12, 12, 5))


def _block(n_head: int, n_kv_head: int, key: jax.Array) -> CIBlock:
    """A block with distinct random weights per projection, at the GQA head counts."""
    d, hd, mlp = 16, 16 // n_head, 32
    kq, kk, kv, ko, k1, k2 = jax.random.split(key, 6)
    return CIBlock(
        wq=jax.random.normal(kq, (d, d)) * 0.1,
        wk=jax.random.normal(kk, (n_kv_head * hd, d)) * 0.1,
        wv=jax.random.normal(kv, (n_kv_head * hd, d)) * 0.1,
        wo=jax.random.normal(ko, (d, d)) * 0.1,
        w1=jax.random.normal(k1, (d, mlp)) * 0.1,
        b1=jnp.zeros((mlp,)),
        w2=jax.random.normal(k2, (mlp, d)) * 0.1,
        b2=jnp.zeros((d,)),
        gate=None,
        norm_scales=None,
        attention=(
            MHACIAttention(n_heads=n_head)
            if n_kv_head == n_head
            else GQACIAttention(n_heads=n_head, n_kv_heads=n_kv_head)
        ),
        eps=CI_FN_RMS_EPS,
    )


def _reference_gqa_attn_out(block: CIBlock, x: jax.Array, inv_freq: jax.Array) -> jax.Array:
    """The GQA attention sublayer computed via EXPLICIT repeat_kv + plain MHA math.

    Independent of `jax.nn.dot_product_attention`'s native grouping: this is the semantics
    we intend (query head i reads K/V group `i // (n_head // n_kv_head)`, the `repeat_kv`
    convention the vendored Llama target uses).
    """
    t = x.shape[1]
    h = _weightless_rms_norm(x, block.eps)

    def heads(w: jax.Array, nh: int) -> jax.Array:
        proj = einops.einsum(h, w, "b t i, o i -> b t o")
        return einops.rearrange(proj, "b t (nh hd) -> b nh t hd", nh=nh)

    q = heads(block.wq, block.attention.n_heads)
    k, v = heads(block.wk, block.attention.n_kv_heads), heads(block.wv, block.attention.n_kv_heads)
    cos, sin = rope_cos_sin(inv_freq, t, x.dtype)
    q, k = apply_rope(q, k, cos, sin)
    k = repeat_kv(k, block.attention.n_heads // block.attention.n_kv_heads)
    v = repeat_kv(v, block.attention.n_heads // block.attention.n_kv_heads)
    hd = q.shape[-1]
    logits = einops.einsum(q, k, "b nh tq hd, b nh tk hd -> b nh tq tk") / hd**0.5
    y = einops.einsum(jax.nn.softmax(logits, axis=-1), v, "b nh tq tk, b nh tk hd -> b nh tq hd")
    return einops.einsum(
        einops.rearrange(y, "b nh t hd -> b t (nh hd)"), block.wo, "b t i, o i -> b t o"
    )


def _attn_sublayer_via_block(block: CIBlock, x: jax.Array, inv_freq: jax.Array) -> jax.Array:
    """The production attention sublayer (`jax.nn.dot_product_attention`) in isolation: with
    the MLP weights zeroed the block returns `x + attn(x)`, so subtracting `x` leaves attn."""
    zeroed = eqx.tree_at(
        lambda b: (b.w1, b.w2), block, (jnp.zeros_like(block.w1), jnp.zeros_like(block.w2))
    )
    return zeroed(x, inv_freq) - x


@pytest.mark.parametrize("n_kv_head", [1, 2, 4])
def test_gqa_matches_explicit_repeat_kv_reference(n_kv_head: int):
    """`dot_product_attention`'s native grouping == repeat_kv + explicit MHA math.

    Pins the grouping CONVENTION (query head i ← K/V head i // group_size). A transposed
    convention (i % n_kv_head) would still run and still train — it would just silently pair
    the wrong heads — so this needs a reference, not a shape check.
    """
    n_head = 4
    block = _block(n_head, n_kv_head, jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 6, 16))
    hd = 16 // n_head
    inv_freq = 1.0 / (10000.0 ** (jnp.arange(0, hd, 2, dtype=jnp.float32) / hd))

    want = _reference_gqa_attn_out(block, x, inv_freq)
    got = _attn_sublayer_via_block(block, x, inv_freq)
    assert jnp.allclose(got, want, rtol=1e-5, atol=1e-5), jnp.abs(got - want).max()


def test_gqa_narrows_kv_projections_only():
    arch = _arch(attention=GQACIAttention(n_heads=4, n_kv_heads=1), sites=SITES)
    ci_fn = init_chunkwise_transformer_ci_fn(arch, SITES, jax.random.PRNGKey(0))
    hd = arch.d_model // arch.attention.n_heads
    for b in ci_fn.chunks.blocks:
        assert b.wq.shape[1:] == (arch.d_model, arch.d_model), b.wq.shape
        assert b.wo.shape[1:] == (arch.d_model, arch.d_model), b.wo.shape
        assert b.wk.shape[1:] == (arch.attention.n_kv_heads * hd, arch.d_model), b.wk.shape
        assert b.wv.shape[1:] == (arch.attention.n_kv_heads * hd, arch.d_model), b.wv.shape


def test_mha_arch_is_unchanged_by_the_gqa_seam():
    """`MHACIAttention` draws the K/V projections at full `[d_model, d_model]`, so existing
    runs' params and RNG consumption are untouched by the GQA seam."""
    arch = _arch(attention=MHACIAttention(n_heads=4), sites=SITES)
    ci_fn = init_chunkwise_transformer_ci_fn(arch, SITES, jax.random.PRNGKey(0))
    for b in ci_fn.chunks.blocks:
        assert b.wk.shape[1:] == (arch.d_model, arch.d_model), b.wk.shape
        assert b.wv.shape[1:] == (arch.d_model, arch.d_model), b.wv.shape


def test_gqa_ci_fn_runs_end_to_end():
    arch = _arch(attention=GQACIAttention(n_heads=4, n_kv_heads=2), sites=SITES)
    ci_fn = build_ci_fn(arch, SITES, jax.random.PRNGKey(0))
    taps = {"resid.0": jax.random.normal(jax.random.PRNGKey(1), (2, 6, 12))}
    ci = ci_fn(taps, remat=False)
    for site in SITES:
        for squashed in (ci.logits[site.name], ci.lower[site.name], ci.upper[site.name]):
            assert squashed.shape == (2, 6, site.C), squashed.shape
            assert jnp.isfinite(squashed).all()


# The authored-schema parse tests (`ChunkwiseTransformerCiConfig.attention` arms) live with
# the schema, lab-side: `param_decomp/tests/test_lm_ci_schema.py`.
