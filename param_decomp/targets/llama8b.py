"""The Llama-3.1-8B family target: its arch config (the vendored `LlamaConfig`, llama3
rope scaling) and its HF loader over the shared GLU-transformer machinery
(`targets/glu_transformer.py` — plain `FrozenAttn`, no pre-RoPE extras)."""

import jax.numpy as jnp

from param_decomp.components import SiteSpec
from param_decomp.targets.glu_transformer import (
    FrozenAttn,
    GLUDecomposedModel,
    HFWeights,
    load_decomposed_glu_from_hf,
)
from vendored_jax.llama import LlamaConfig, llama3_inv_freq


def llama31_8b_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=128256,
        n_layer=32,
        n_head=32,
        n_kv_head=8,
        n_embd=4096,
        n_intermediate=14336,
        rope_theta=500000.0,
        rms_norm_eps=1e-5,
        max_position_embeddings=131072,
        rope_factor=8.0,
        rope_low_freq_factor=1.0,
        rope_high_freq_factor=4.0,
        rope_original_max_position_embeddings=8192,
    )


def _load_attn(w: HFWeights, i: int, cfg: LlamaConfig) -> FrozenAttn:
    pre = "model.layers"
    return FrozenAttn(
        wq=w.get(f"{pre}.{i}.self_attn.q_proj.weight"),
        wk=w.get(f"{pre}.{i}.self_attn.k_proj.weight"),
        wv=w.get(f"{pre}.{i}.self_attn.v_proj.weight"),
        wo=w.get(f"{pre}.{i}.self_attn.o_proj.weight"),
        n_head=cfg.n_head,
        n_kv_head=cfg.n_kv_head,
        head_dim=cfg.head_dim,
        n_rep=cfg.n_rep,
    )


def load_decomposed_llama_from_hf(
    model_name: str,
    cfg: LlamaConfig,
    sites: tuple[SiteSpec, ...],
    scan_unroll: int = 1,
    gather_fp8: bool = False,
) -> GLUDecomposedModel:
    """The Llama family HF load: plain attention + llama3-rescaled RoPE frequencies."""
    return load_decomposed_glu_from_hf(
        model_name,
        cfg,
        sites,
        load_attn=lambda w, i: _load_attn(w, i, cfg),
        weights_dtype=jnp.bfloat16,  # the family is bf16-only (TargetConfig.supported_weights_dtypes)
        inv_freq=llama3_inv_freq(cfg),
        scan_unroll=scan_unroll,
        gather_fp8=gather_fp8,
    )
