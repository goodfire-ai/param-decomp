"""Direct JAX-vs-HuggingFace parity for the qwen36_moe target.

The tiny test rebuilds `gen_hf_fixtures.py`'s seeded random `Qwen3_5MoeForCausalLM` as a
`Qwen36MoeDecomposedModel` from the golden's own state dict, fp32, and matches HF's
residual boundary after every layer plus its logits — the exact-architecture check
(gated DeltaNet, gated attention + partial RoPE, zero-centered norms, the MoE router,
fused experts, shared expert, untied head). The slow test loads the REAL
`Qwen/Qwen3.6-35B-A3B` snapshot through the production loader (bf16) and matches HF's
bf16 final-position logits in distribution (KL + argmax) — the weight-loading check.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from param_decomp.lm.batch import LMBatch
from param_decomp.targets.lm_output import (
    MaterializedOutputEdge,
    OutputEdge,
    StreamedLinearOutput,
    StreamedOutputEdge,
)
from param_decomp.targets.qwen36_moe import (
    Qwen36MoeConfig,
    build_qwen36_moe_from_weights,
    load_decomposed_qwen36_moe_from_hf,
    qwen36_35b_a3b_config,
)
from param_decomp.targets.testing import capture_clean, materialized_logits, run_clean
from param_decomp.targets.transformer import hf_snapshot_dir
from param_decomp.targets.transformer_taps import resid_tap_key

HERE = Path(__file__).resolve().parent


def _tiny_cfg_from_golden(config_json: str) -> Qwen36MoeConfig:
    hf = json.loads(config_json)
    interval = len(hf["layer_types"]) // hf["layer_types"].count("full_attention")
    assert hf["layer_types"] == (["linear_attention"] * (interval - 1) + ["full_attention"]) * (
        hf["num_hidden_layers"] // interval
    ), hf["layer_types"]
    assert not hf["tie_word_embeddings"] and not hf["attention_bias"], hf
    rope = hf["rope_parameters"]
    assert rope["rope_type"] == "default", rope
    return Qwen36MoeConfig(
        vocab_size=hf["vocab_size"],
        n_layer=hf["num_hidden_layers"],
        full_attention_interval=interval,
        n_embd=hf["hidden_size"],
        n_head=hf["num_attention_heads"],
        n_kv_head=hf["num_key_value_heads"],
        head_dim=hf["head_dim"],
        partial_rotary_factor=rope["partial_rotary_factor"],
        rope_theta=rope["rope_theta"],
        linear_num_key_heads=hf["linear_num_key_heads"],
        linear_key_head_dim=hf["linear_key_head_dim"],
        linear_num_value_heads=hf["linear_num_value_heads"],
        linear_value_head_dim=hf["linear_value_head_dim"],
        linear_conv_kernel_dim=hf["linear_conv_kernel_dim"],
        n_experts=hf["num_experts"],
        n_experts_per_token=hf["num_experts_per_tok"],
        moe_intermediate=hf["moe_intermediate_size"],
        shared_expert_intermediate=hf["shared_expert_intermediate_size"],
        rms_norm_eps=hf["rms_norm_eps"],
        max_position_embeddings=hf["max_position_embeddings"],
    )


@pytest.mark.parametrize("output_edge", [MaterializedOutputEdge(), StreamedOutputEdge(4)])
def test_tiny_random_qwen36_moe_matches_hf(output_edge: OutputEdge):
    f = np.load(HERE / "qwen36_moe_tiny_hf_fixtures.npz")
    cfg = _tiny_cfg_from_golden(str(f["config_json"]))
    sd = {k.removeprefix("sd::"): f[k] for k in f.files if k.startswith("sd::")}

    # The production loader mirrored over the in-memory fp32 state dict: the
    # `Qwen3_5MoeForCausalLM` fixture nests the decoder under `model` (the real
    # conditional-generation checkpoint uses `model.language_model`).
    def get(key: str) -> jax.Array:
        return jnp.asarray(sd[key], jnp.float32)

    model = build_qwen36_moe_from_weights(
        cfg,
        (),
        get,
        decoder_prefix="model",
        implementation="xla",
        expert_implementation="tokamax_split_vjp",
        output_edge=output_edge,
    )
    assert jax.tree.leaves(model)
    assert all(
        isinstance(weight, jax.Array)
        and all(device.platform == "cpu" for device in weight.devices())
        for weight in jax.tree.leaves(model)
    )
    tokens = jnp.asarray(f["tokens"])
    residual_keys = tuple(resid_tap_key(i) for i in range(cfg.n_layer + 1))
    residuals = capture_clean(model, LMBatch(tokens), residual_keys)
    for i, key in enumerate(residual_keys):
        np.testing.assert_allclose(residuals[key], f[f"resid::{i}"], rtol=2e-4, atol=1e-5)
    assert model.output_edge == output_edge
    output = run_clean(model, LMBatch(tokens))
    match output:
        case jax.Array():
            logits = output
        case StreamedLinearOutput(activations=activations, head=head):
            logits = activations @ head.T
    np.testing.assert_allclose(logits, f["logits"], rtol=2e-4, atol=1e-5)


@pytest.mark.slow
def test_real_qwen36_35b_matches_hf():
    """The production HF loader against real-weights HF logits: bf16 both sides, so the
    comparison is distributional — small KL at the final position and a tie-aware argmax
    check — not elementwise. Runs only when the 35B snapshot is available locally."""
    try:
        hf_snapshot_dir("Qwen/Qwen3.6-35B-A3B")
    except (AssertionError, FileNotFoundError, KeyError):
        pytest.skip("no local Qwen/Qwen3.6-35B-A3B snapshot")
    f = np.load(HERE / "qwen36_35b_real_logits.npz")
    model = load_decomposed_qwen36_moe_from_hf(
        "Qwen/Qwen3.6-35B-A3B",
        qwen36_35b_a3b_config(),
        (),
        jnp.bfloat16,
        "xla",
        "tokamax_split_vjp",
        MaterializedOutputEdge(),
    )
    # fp32 accumulations must not silently drop to TF32 on GPU: the drift grows with
    # depth and at 40 layers exceeds the KL threshold, faking a broken port.
    with jax.default_matmul_precision("highest"):
        logits = np.asarray(
            materialized_logits(run_clean(model, LMBatch(jnp.asarray(f["tokens"]))))[
                :, -1, :
            ].astype(jnp.float32)
        )
    ref = f["final_logits"]
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
