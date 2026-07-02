"""CPU tests for the CI-fn training-health metrics (`ci_health.py`).

Pins: the instrumented telemetry forward returns the SAME logits as `__call__` (it is
the same math, just with stats taps); the power-iteration spectral norm matches a full
SVD on tiny weights; and the activation/logit/component health scalars are finite and
in their documented ranges.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np

from param_decomp.ci_fn import (
    Chunk,
    ChunkwiseTransformerCIArch,
    ChunkwiseTransformerCIFn,
    build_ci_fn,
)
from param_decomp.ci_health import (
    ci_fn_weight_health,
    make_ci_activation_health_step,
)
from param_decomp.lm import DecomposedModel
from param_decomp.targets.llama8b import llama_site_specs, mlp_family_site_cs
from param_decomp.tests.test_llama8b import _tiny_cfg, _tiny_decomposed_lm
from param_decomp.train import COMPUTE_DT, cast_floating


def _tiny_setup() -> tuple[DecomposedModel, ChunkwiseTransformerCIFn, jax.Array, int]:
    cfg = _tiny_cfg()
    sites = llama_site_specs(cfg, mlp_family_site_cs(4, 5, 8))
    lm = _tiny_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
    first_block = min(int(name.split(".")[1]) for name in lm.site_names)
    arch = ChunkwiseTransformerCIArch(
        chunks=(Chunk(input_taps=(f"resid.{first_block}",), output_sites=lm.site_names),),
        input_dim=cfg.n_embd,
        d_model=16,
        n_blocks=2,
        n_heads=2,
        mlp_hidden=32,
    )
    ci_fn = build_ci_fn(arch, lm.sites, jax.random.PRNGKey(2))
    assert isinstance(ci_fn, ChunkwiseTransformerCIFn)
    tokens = jax.random.randint(jax.random.PRNGKey(4), (3, 16), 0, cfg.vocab_size)
    return lm, ci_fn, tokens, cfg.vocab_size


def test_telemetry_logits_match_call():
    lm, ci_fn, tokens, _ = _tiny_setup()
    ci_fn_bf16 = cast_floating(ci_fn, COMPUTE_DT)
    taps = {
        k: x.astype(COMPUTE_DT) for k, x in lm.read_activations(tokens, ci_fn.input_names).items()
    }
    reference = ci_fn_bf16(taps, remat=False).logits
    telemetry_logits, chunk_stats = ci_fn_bf16.telemetry(taps)
    assert set(telemetry_logits) == set(reference)
    for site in reference:
        np.testing.assert_array_equal(
            np.asarray(telemetry_logits[site], np.float32),
            np.asarray(reference[site], np.float32),
        )
    n_chunks = len(ci_fn.chunk_meta)
    for key, per_chunk in chunk_stats.items():
        assert per_chunk.shape == (n_chunks,), (key, per_chunk.shape)
        assert bool(jnp.isfinite(per_chunk).all()), key
    assert "in_proj_rms" in chunk_stats
    for block_idx in range(2):
        for stat in (
            "q_rms", "k_rms", "attn_logit_max", "attn_logit_rms", "attn_entropy",
            "mlp_hidden_rms", "resid_rms",
        ):  # fmt: skip
            assert f"block{block_idx}/{stat}" in chunk_stats


def test_weight_health_sigma_matches_svd():
    _, ci_fn, _, _ = _tiny_setup()
    health = ci_fn_weight_health(ci_fn)
    for key, value in health.items():
        assert bool(jnp.isfinite(value)), key
    block0 = ci_fn.chunks.blocks[0]
    for matrix_name in ("wq", "wk", "wv", "wo", "w1", "w2"):
        w = np.asarray(getattr(block0, matrix_name), np.float32)
        true_sigma = max(np.linalg.svd(w[c], compute_uv=False)[0] for c in range(w.shape[0]))
        # Power iteration converges slowly on a fresh init's near-degenerate spectrum
        # (always an under-estimate); ~1% is fine for a trend scalar.
        np.testing.assert_allclose(
            float(health[f"ci_health/weights/sigma_max/block0/{matrix_name}"]),
            true_sigma,
            rtol=1e-2,
        )
        stable_rank = float(health[f"ci_health/weights/stable_rank/block0/{matrix_name}"])
        assert 1.0 - 1e-3 <= stable_rank <= min(w.shape[1], w.shape[2]) + 1e-3
        assert float(health[f"ci_health/weights/frob/block0/{matrix_name}"]) > 0.0
    for family in ("in_proj", "out_heads"):
        assert f"ci_health/weights/frob/{family}" in health


def test_activation_health_step_ranges():
    lm, ci_fn, tokens, _ = _tiny_setup()
    step = make_ci_activation_health_step(mesh=None)
    health = step(lm, ci_fn, tokens)
    for key, value in health.items():
        assert value.shape == (), key
        assert bool(jnp.isfinite(value)), key
    seq_len = tokens.shape[1]
    for block_idx in range(2):
        entropy = float(health[f"ci_health/act/block{block_idx}/attn_entropy"])
        assert 0.0 <= entropy <= math.log(seq_len) + 1e-4
        assert float(health[f"ci_health/act/block{block_idx}/q_rms"]) > 0.0
    for frac_key in (
        "ci_health/logits/frac_above_1",
        "ci_health/logits/frac_below_0",
        "ci_health/components/dead_frac_mean",
        "ci_health/components/dead_frac_max",
        "ci_health/components/participation_frac_mean",
        "ci_health/components/participation_frac_min",
    ):
        assert 0.0 <= float(health[frac_key]) <= 1.0, frac_key
    assert float(health["ci_health/logits/std"]) >= 0.0
