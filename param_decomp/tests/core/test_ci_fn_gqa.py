"""GQA in the chunkwise CI transformer: grouping semantics, param shapes, MHA identity."""

from dataclasses import replace

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from param_decomp.core.ci_fn.implementations.chunkwise.arch import (
    Chunk,
    ChunkwiseTransformerCIFnArch,
    UnplacedChunkwiseCIFn,
    init_chunkwise_transformer_backbone,
)
from param_decomp.core.ci_fn.implementations.transformer.layers import (
    CI_FN_RMS_EPS,
    LOCAL_CI_FN_TRANSFORMER_PLACEMENT,
    CIFnAttention,
    CIFnBlock,
    GQACIFnAttention,
    MHACIFnAttention,
    weightless_rms_norm,
)
from param_decomp.core.components import (
    DenseFactorization,
    SiteSpec,
    init_component_stacks,
    require_full_emission,
)
from param_decomp.sequence import SequenceLayout
from param_decomp.target_ports.llama import apply_rope, repeat_kv, rope_cos_sin
from param_decomp.tests.sequence import unsegmented_sequence_layout


def _arch(attention: CIFnAttention, sites: tuple[SiteSpec, ...]) -> ChunkwiseTransformerCIFnArch:
    return ChunkwiseTransformerCIFnArch(
        chunks=(Chunk(input_taps=("resid.0",), output_sites=tuple(s.name for s in sites)),),
        input_dim=12,
        d_model=16,
        n_blocks=2,
        attention=attention,
        ffn_hidden=32,
        ffn_kind="gelu",
        learned_norm_scale=False,
    )


SITES = (
    SiteSpec("layers.0.q_proj", DenseFactorization(d_in=12, d_out=12, C=3), "q_proj"),
    SiteSpec("layers.0.mlp", DenseFactorization(d_in=12, d_out=12, C=5), "mlp"),
)
COMPONENTS = init_component_stacks(SITES, jax.random.PRNGKey(3))


def _block(n_head: int, n_kv_head: int, key: jax.Array) -> CIFnBlock:
    """A block with distinct random weights per projection, at the GQA head counts."""
    d, hd, mlp = 16, 16 // n_head, 32
    kq, kk, kv, ko, k1, k2 = jax.random.split(key, 6)
    return CIFnBlock(
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
            MHACIFnAttention(mask="bidirectional", implementation="xla", n_heads=n_head)
            if n_kv_head == n_head
            else GQACIFnAttention(
                mask="bidirectional", implementation="xla", n_heads=n_head, n_kv_heads=n_kv_head
            )
        ),
        eps=CI_FN_RMS_EPS,
    )


def _reference_gqa_attn_out(block: CIFnBlock, x: jax.Array, inv_freq: jax.Array) -> jax.Array:
    """The GQA attention sublayer computed via EXPLICIT repeat_kv + plain MHA math.

    Independent of `jax.nn.dot_product_attention`'s native grouping: this is the semantics
    we intend (query head i reads K/V group `i // (n_head // n_kv_head)`, the `repeat_kv`
    convention the vendored Llama target uses).
    """
    t = x.shape[1]
    h = weightless_rms_norm(x, block.eps)

    def heads(w: jax.Array, nh: int) -> jax.Array:
        proj = einops.einsum(h, w, "b t i, o i -> b t o")
        return einops.rearrange(proj, "b t (nh hd) -> b nh t hd", nh=nh)

    q = heads(block.wq, block.attention.n_heads)
    k, v = heads(block.wk, block.attention.n_kv_heads), heads(block.wv, block.attention.n_kv_heads)
    cos, sin = rope_cos_sin(inv_freq, jnp.broadcast_to(jnp.arange(t), x.shape[:2]), x.dtype)
    q, k = apply_rope(q, k, cos, sin)
    k = repeat_kv(k, block.attention.n_heads // block.attention.n_kv_heads)
    v = repeat_kv(v, block.attention.n_heads // block.attention.n_kv_heads)
    hd = q.shape[-1]
    logits = einops.einsum(q, k, "b nh tq hd, b nh tk hd -> b nh tq tk") / hd**0.5
    y = einops.einsum(jax.nn.softmax(logits, axis=-1), v, "b nh tq tk, b nh tk hd -> b nh tq hd")
    return einops.einsum(
        einops.rearrange(y, "b nh t hd -> b t (nh hd)"), block.wo, "b t i, o i -> b t o"
    )


def _attn_sublayer_via_block(block: CIFnBlock, x: jax.Array, inv_freq: jax.Array) -> jax.Array:
    """The production attention sublayer (`jax.nn.dot_product_attention`) in isolation: with
    the MLP weights zeroed the block returns `x + attn(x)`, so subtracting `x` leaves attn."""
    zeroed = eqx.tree_at(
        lambda b: (b.w1, b.w2), block, (jnp.zeros_like(block.w1), jnp.zeros_like(block.w2))
    )
    return (
        zeroed(
            x,
            inv_freq,
            placement=LOCAL_CI_FN_TRANSFORMER_PLACEMENT,
            sequence=SequenceLayout(jnp.zeros(x.shape[:2], dtype=jnp.int32)),
        )
        - x
    )


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
    arch = _arch(
        attention=GQACIFnAttention(
            mask="bidirectional", implementation="xla", n_heads=4, n_kv_heads=1
        ),
        sites=SITES,
    )
    backbone = init_chunkwise_transformer_backbone(
        arch, SITES, UnplacedChunkwiseCIFn(), jax.random.PRNGKey(0)
    )
    hd = arch.d_model // arch.attention.n_heads
    for b in backbone.chunks.blocks:
        assert b.wq.shape[1:] == (arch.d_model, arch.d_model), b.wq.shape
        assert b.wo.shape[1:] == (arch.d_model, arch.d_model), b.wo.shape
        assert b.wk.shape[1:] == (arch.attention.n_kv_heads * hd, arch.d_model), b.wk.shape
        assert b.wv.shape[1:] == (arch.attention.n_kv_heads * hd, arch.d_model), b.wv.shape


def test_mha_arch_is_unchanged_by_the_gqa_seam():
    """`MHACIFnAttention` draws the K/V projections at full `[d_model, d_model]`, so existing
    runs' params and RNG consumption are untouched by the GQA seam."""
    arch = _arch(
        attention=MHACIFnAttention(mask="bidirectional", implementation="xla", n_heads=4),
        sites=SITES,
    )
    backbone = init_chunkwise_transformer_backbone(
        arch, SITES, UnplacedChunkwiseCIFn(), jax.random.PRNGKey(0)
    )
    for b in backbone.chunks.blocks:
        assert b.wk.shape[1:] == (arch.d_model, arch.d_model), b.wk.shape
        assert b.wv.shape[1:] == (arch.d_model, arch.d_model), b.wv.shape


def test_gqa_ci_fn_runs_end_to_end():
    arch = _arch(
        attention=GQACIFnAttention(
            mask="bidirectional", implementation="xla", n_heads=4, n_kv_heads=2
        ),
        sites=SITES,
    )
    ci_fn = arch.initialize(SITES, None, jax.random.PRNGKey(0))
    taps = {"resid.0": jax.random.normal(jax.random.PRNGKey(1), (2, 6, 12))}
    ci = ci_fn.prepare()(
        taps, None, COMPONENTS, sequence=unsegmented_sequence_layout(taps), remat=False
    )
    for site in SITES:
        for value in (ci.preactivations[site.name], ci.lower[site.name], ci.upper[site.name]):
            squashed = require_full_emission(value)
            assert squashed.shape == (2, 6, site.C), squashed.shape
            assert jnp.isfinite(squashed).all()


# The authored-schema parse tests (`ChunkwiseTransformerCIFnConfig.attention` arms) live with
# the schema, lab-side: `param_decomp/tests/experiments/lm/test_lm_ci_schema.py`.


@pytest.mark.skipif(jax.default_backend() != "cpu", reason="exercises unavailable cuDNN on CPU")
@pytest.mark.parametrize(
    "attention",
    [
        MHACIFnAttention(mask="bidirectional", n_heads=4, implementation="xla"),
        GQACIFnAttention(mask="bidirectional", n_heads=4, n_kv_heads=2, implementation="xla"),
    ],
)
def test_ci_forward_honors_the_requested_attention_backend(attention: CIFnAttention):
    taps = {"resid.0": jax.random.normal(jax.random.PRNGKey(1), (2, 64, 12))}

    def evaluate(requested: CIFnAttention):
        fn = _arch(attention=requested, sites=SITES).initialize(SITES, None, jax.random.PRNGKey(0))
        return fn.prepare()(taps, None, COMPONENTS, sequence=None, remat=False)

    ci = evaluate(attention)
    for site in SITES:
        values = require_full_emission(ci.lower[site.name])
        assert values.shape == (2, 64, site.C)
        assert jnp.isfinite(values).all()

    with pytest.raises(AssertionError, match="cuDNN flash attention requires a GPU"):
        evaluate(replace(attention, implementation="flash"))
