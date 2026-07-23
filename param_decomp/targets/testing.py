"""Tiny random targets — the standard CPU-test fixtures.

One tiny random target per LM family (the llama8b-flavored GLU transformer and the
`LlamaSimpleMLP`), plus a one-chunk chunkwise CI fn over each. Engine tests use these as
the concrete target behind the `DecomposedModel` protocol; the per-target suites
(`tests/`) use them as the system under test. Toy dims throughout — no real weights, no
GPU.
"""

import jax
import jax.numpy as jnp

from param_decomp.core.ci_fn import (
    Chunk,
    ChunkwiseTransformerCIArch,
    ChunkwiseTransformerCIFn,
    MHACIAttention,
    build_ci_fn,
)
from param_decomp.core.components import SiteC, SiteSpec
from param_decomp.core.model import DecomposedModel
from param_decomp.targets import llama_simple_mlp
from param_decomp.targets.glu_transformer import (
    FrozenAttn,
    GLUDecomposedModel,
    GLULayer,
    build_decomposed_lm,
    parse_site_name,
)
from param_decomp.targets.llama_simple_mlp import (
    LlamaSimpleMLPConfig,
    SimpleMLPDecomposedModel,
    SimpleMLPLayer,
    build_decomposed_simple_mlp,
)
from param_decomp.vendored_jax.llama import LlamaConfig, llama3_inv_freq


def _tiny_chunkwise_ci_fn(
    model: DecomposedModel, key: jax.Array, first_block: int, input_dim: int, n_blocks: int
) -> ChunkwiseTransformerCIFn:
    """One chunk reading the residual entering the first decomposed block, emitting CI
    for every site. `input_dim` is the target residual width (`n_embd`)."""
    arch = ChunkwiseTransformerCIArch(
        chunks=(Chunk(input_taps=(f"resid.{first_block}",), output_sites=model.site_names),),
        input_dim=input_dim,
        d_model=16,
        n_blocks=n_blocks,
        attention=MHACIAttention(n_heads=2),
        ffn_hidden=32,
        ffn_kind="gelu",
        learned_norm_scale=False,
    )
    ci_fn = build_ci_fn(arch, model.sites, key)
    assert isinstance(ci_fn, ChunkwiseTransformerCIFn)
    return ci_fn


def tiny_glu_cfg() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=64,
        n_layer=8,
        n_head=4,
        n_kv_head=2,
        n_embd=32,
        n_intermediate=64,
        rope_theta=500000.0,
        rms_norm_eps=1e-5,
        max_position_embeddings=512,
        rope_factor=8.0,
        rope_low_freq_factor=1.0,
        rope_high_freq_factor=4.0,
        rope_original_max_position_embeddings=128,
    )


def tiny_glu_decomposed_lm(
    cfg: LlamaConfig, sites: tuple[SiteSpec, ...], key: jax.Array
) -> GLUDecomposedModel:
    """A tiny random `GLUDecomposedModel` (random embedding + full frozen layer stack
    plus the decomposition `sites`) — the CPU-test analog of `load_decomposed_lm_from_hf`."""
    ks = iter(jax.random.split(key, 1024))
    d, di = cfg.n_embd, cfg.n_intermediate
    qd, kvd = cfg.n_head * cfg.head_dim, cfg.n_kv_head * cfg.head_dim

    def n(shape: tuple[int, ...], s: float | None = None) -> jax.Array:
        return jax.random.normal(next(ks), shape) * (s or d**-0.5)

    def fattn():
        return FrozenAttn(
            n((qd, d)), n((kvd, d)), n((kvd, d)), n((d, qd)),
            cfg.n_head, cfg.n_kv_head, cfg.head_dim, cfg.n_rep,
        )  # fmt: skip

    def layer():
        return GLULayer(jnp.ones((d,)), jnp.ones((d,)), fattn(), n((di, d)), n((di, d)), n((d, di)))

    return build_decomposed_lm(
        embed=n((cfg.vocab_size, d), 0.02),
        layers=[layer() for _ in range(cfg.n_layer)],
        norm=jnp.ones((d,)),
        lm_head=n((cfg.vocab_size, d), 0.02),
        inv_freq=llama3_inv_freq(cfg),
        cfg=cfg,
        sites=sites,
    )


def tiny_glu_chunkwise_ci_fn(
    model: DecomposedModel, key: jax.Array, n_blocks: int
) -> ChunkwiseTransformerCIFn:
    first_block = min(parse_site_name(n)[0] for n in model.site_names)
    return _tiny_chunkwise_ci_fn(model, key, first_block, tiny_glu_cfg().n_embd, n_blocks)


def tiny_simple_mlp_cfg() -> LlamaSimpleMLPConfig:
    return LlamaSimpleMLPConfig(
        vocab_size=64,
        n_layer=6,
        n_head=4,
        n_kv_head=2,
        n_embd=32,
        n_intermediate=64,
        rotary_base=10000.0,
        rms_norm_eps=1e-6,
        n_ctx=64,
    )


def _tiny_simple_mlp_layers(
    cfg: LlamaSimpleMLPConfig, n: int, key: jax.Array
) -> list[SimpleMLPLayer]:
    ks = iter(jax.random.split(key, 1024))
    d, di = cfg.n_embd, cfg.n_intermediate
    qd, kvd = cfg.n_head * cfg.head_dim, cfg.n_kv_head * cfg.head_dim

    def rand(shape: tuple[int, ...]) -> jax.Array:
        return jax.random.normal(next(ks), shape) * d**-0.5

    return [
        SimpleMLPLayer(
            ln1=jnp.ones((d,)),
            ln2=jnp.ones((d,)),
            attn=FrozenAttn(
                rand((qd, d)),
                rand((kvd, d)),
                rand((kvd, d)),
                rand((d, qd)),
                cfg.n_head,
                cfg.n_kv_head,
                cfg.head_dim,
                cfg.n_rep,
            ),  # fmt: skip
            Wfc=rand((di, d)),
            Wdown=rand((d, di)),
        )
        for _ in range(n)
    ]


def tiny_simple_mlp_decomposed_model(
    cfg: LlamaSimpleMLPConfig, sites: tuple[SiteSpec, ...], key: jax.Array
) -> SimpleMLPDecomposedModel:
    """A tiny random `SimpleMLPDecomposedModel` carrying a random embedding + full frozen
    layer stack plus the decomposition `sites`."""
    layers_key, embed_key = jax.random.split(key)
    layers = _tiny_simple_mlp_layers(cfg, cfg.n_layer, layers_key)
    embed = jax.random.normal(embed_key, (cfg.vocab_size, cfg.n_embd)) * 0.02
    return build_decomposed_simple_mlp(
        embed=embed, layers=layers, norm=jnp.ones((cfg.n_embd,)), lm_head=embed,
        cfg=cfg, sites=sites,
    )  # fmt: skip


SIMPLE_MLP_MIXED_SITE_CS = (
    SiteC("h.2.attn.q_proj", 8),
    SiteC("h.2.attn.v_proj", 12),
    SiteC("h.2.mlp.c_fc", 8),
    SiteC("h.3.mlp.down_proj", 16),
)
"""Attention + MLP sites across two layers with heterogeneous per-site C."""


def tiny_simple_mlp_chunkwise_ci_fn(
    model: DecomposedModel, key: jax.Array
) -> ChunkwiseTransformerCIFn:
    first_block = min(llama_simple_mlp.parse_site_name(n)[0] for n in model.site_names)
    return _tiny_chunkwise_ci_fn(model, key, first_block, tiny_simple_mlp_cfg().n_embd, 2)
