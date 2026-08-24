"""Direct JAX-vs-HuggingFace parity for the Qwen3 target.

The tiny test rebuilds `gen_hf_fixtures.py`'s seeded random `Qwen3ForCausalLM` as a
`GLUDecomposedModel` (Qwen3 family) from the golden's own state dict, fp32, and matches
HF's residual boundary after every layer plus its logits — the exact-architecture check
(rectangular query projection, QK-norm, GQA, plain RoPE, and tied embeddings). The slow
test loads the REAL `Qwen/Qwen3-8B-Base` snapshot through the
production loader (`load_decomposed_lm_from_hf`, bf16) and matches HF's bf16
final-position logits in distribution (KL + argmax) — the weight-loading check.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from param_decomp.targets.glu_transformer import (
    GatedMLP,
    GLUConfig,
    GLUDecomposedModel,
    GLULayer,
    TiedHead,
    build_decomposed_lm,
    default_inv_freq,
    hf_snapshot_dir,
)
from param_decomp.targets.qwen3 import (
    Qwen3FrozenAttn,
    load_decomposed_qwen3_from_hf,
    qwen3_8b_base_config,
)
from param_decomp.targets.testing import capture_clean, run_clean
from param_decomp.targets.transformer_taps import resid_tap_key

HERE = Path(__file__).resolve().parent


def _tiny_cfg_from_golden(config_json: str) -> GLUConfig:
    hf = json.loads(config_json)
    assert hf["head_dim"] * hf["num_attention_heads"] > hf["hidden_size"], hf
    assert hf["tie_word_embeddings"] and not hf["attention_bias"], hf
    return GLUConfig(
        vocab_size=hf["vocab_size"],
        n_layer=hf["num_hidden_layers"],
        n_head=hf["num_attention_heads"],
        n_kv_head=hf["num_key_value_heads"],
        n_embd=hf["hidden_size"],
        n_intermediate=hf["intermediate_size"],
        head_dim=hf["head_dim"],
        rope_theta=hf["rope_theta"],
        rms_norm_eps=hf["rms_norm_eps"],
        max_position_embeddings=hf["max_position_embeddings"],
        tie_word_embeddings=hf["tie_word_embeddings"],
    )


def _build_from_hf_state(cfg: GLUConfig, sd: dict[str, np.ndarray]) -> GLUDecomposedModel:
    """The Qwen3 family loader mirrored over an in-memory fp32 HF state dict (the
    production loader reads sharded safetensors and casts bf16; here we stay fp32 so the
    comparison isolates the MATH from rounding)."""
    a = lambda key: jnp.asarray(sd[key], jnp.float32)  # noqa: E731
    pre = "model.layers"
    layers = [
        GLULayer(
            ln1=a(f"{pre}.{i}.input_layernorm.weight"),
            ln2=a(f"{pre}.{i}.post_attention_layernorm.weight"),
            attn=Qwen3FrozenAttn(
                wq=a(f"{pre}.{i}.self_attn.q_proj.weight"),
                wk=a(f"{pre}.{i}.self_attn.k_proj.weight"),
                wv=a(f"{pre}.{i}.self_attn.v_proj.weight"),
                wo=a(f"{pre}.{i}.self_attn.o_proj.weight"),
                n_head=cfg.n_head,
                n_kv_head=cfg.n_kv_head,
                head_dim=cfg.head_dim,
                n_rep=cfg.n_rep,
                implementation="auto",
                q_norm=a(f"{pre}.{i}.self_attn.q_norm.weight"),
                k_norm=a(f"{pre}.{i}.self_attn.k_norm.weight"),
                eps=cfg.rms_norm_eps,
            ),
            mlp=GatedMLP(
                Wg=a(f"{pre}.{i}.mlp.gate_proj.weight"),
                Wu=a(f"{pre}.{i}.mlp.up_proj.weight"),
                Wd=a(f"{pre}.{i}.mlp.down_proj.weight"),
            ),
        )
        for i in range(cfg.n_layer)
    ]
    return build_decomposed_lm(
        embed=a("model.embed_tokens.weight"),
        layers=layers,
        norm=a("model.norm.weight"),
        lm_head=TiedHead() if cfg.tie_word_embeddings else a("lm_head.weight"),
        inv_freq=default_inv_freq(cfg.head_dim, cfg.rope_theta),
        cfg=cfg,
        sites=(),
    )


def test_tiny_random_qwen3_matches_hf():
    f = np.load(HERE / "qwen3_tiny_hf_fixtures.npz")
    cfg = _tiny_cfg_from_golden(str(f["config_json"]))
    sd = {k.removeprefix("sd::"): f[k] for k in f.files if k.startswith("sd::")}
    model = _build_from_hf_state(cfg, sd)
    tokens = jnp.asarray(f["tokens"])
    residual_keys = tuple(resid_tap_key(i) for i in range(cfg.n_layer + 1))
    residuals = capture_clean(model, tokens, residual_keys)
    for i, key in enumerate(residual_keys):
        np.testing.assert_allclose(residuals[key], f[f"resid::{i}"], rtol=2e-4, atol=1e-5)
    np.testing.assert_allclose(run_clean(model, tokens), f["logits"], rtol=2e-4, atol=1e-5)


@pytest.mark.slow
def test_real_qwen3_8b_matches_hf():
    """The production HF loader against real-weights HF logits: bf16 both sides, so the
    comparison is distributional — small KL at the final position and argmax agreement —
    not elementwise."""
    try:
        hf_snapshot_dir("Qwen/Qwen3-8B-Base")
    except (AssertionError, FileNotFoundError, KeyError):
        pytest.skip("no local Qwen/Qwen3-8B-Base snapshot")
    f = np.load(HERE / "qwen3_8b_real_logits.npz")
    model = load_decomposed_qwen3_from_hf(
        "Qwen/Qwen3-8B-Base", qwen3_8b_base_config(), (), jnp.bfloat16
    )
    logits = np.asarray(run_clean(model, jnp.asarray(f["tokens"]))[:, -1, :].astype(jnp.float32))
    ref = f["final_logits"]
    # tie-aware argmax: the golden is bf16, so distinct plausible tokens often carry the
    # IDENTICAL quantized top logit (observed: a 3-way 32.25 tie) — require JAX's argmax
    # to sit at the golden's top logit level, not to match one arbitrary tie-winner.
    jax_pick_ref_logit = np.take_along_axis(ref, logits.argmax(-1, keepdims=True), -1)[:, 0]
    assert (jax_pick_ref_logit >= ref.max(-1) - 0.25).all(), (
        logits.argmax(-1),
        ref.argmax(-1),
        jax_pick_ref_logit,
        ref.max(-1),
    )
    ref_logp = np.asarray(jax.nn.log_softmax(jnp.asarray(ref), axis=-1))
    jax_logp = np.asarray(jax.nn.log_softmax(jnp.asarray(logits), axis=-1))
    kl = (np.exp(ref_logp) * (ref_logp - jax_logp)).sum(-1)
    assert (kl < 5e-3).all(), kl
