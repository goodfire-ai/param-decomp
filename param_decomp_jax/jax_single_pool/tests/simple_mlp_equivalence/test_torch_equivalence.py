"""Torch-reference equivalence for the LlamaSimpleMLP target.

Fixtures are FROZEN committed goldens (`*_fixture.npz`); the torch generator that drew
them (`gen_torch_fixtures.py`) is deleted — `param_decomp_jax` imports no torch. To
regen, check out the `torch-oracle` git tag and run that revision's
`gen_torch_fixtures.py` in the torch venv. Both sides fp32; this is the test that
catches RoPE / GELU-flavor / GQA / norm-eps mismatches:

  * tiny (hermetic): the JAX target is rebuilt from the fixture's state dict; the
    fixture's GQA repeat=2 exercises the kv-head repeat path.
  * real: the actual t-9d2b8f02 weights via the converted safetensors cache (skipped
    where the cluster cache is absent).
"""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array

from jax_single_pool.llama_simple_mlp import (
    clean_suffix_logits,
    config_from_model_config_dict,
    load_model_config,
    load_target_from_pretrain_cache,
    prefix_from_weights,
    prefix_residual,
    target_from_weights,
)

FIXTURE_DIR = Path(__file__).parent
REAL_CACHE_DIR = Path("/mnt/data/artifacts/mechanisms/param-decomp/pretrain_cache/spd-t-9d2b8f02")


def _max_abs_diff(a: Array, b: np.ndarray) -> float:
    return float(jnp.max(jnp.abs(a - jnp.asarray(b))))


def test_tiny_random_model_matches_torch_logits():
    fixture = np.load(FIXTURE_DIR / "tiny_fixture.npz")
    cfg = config_from_model_config_dict(json.loads(str(fixture["config_json"])))
    assert cfg.n_rep == 2, "tiny fixture must exercise the GQA kv-head repeat"

    def get(key: str) -> Array:
        return jnp.asarray(fixture[f"weights.{key}"], dtype=jnp.float32)

    target = target_from_weights(get, cfg, first_layer=0)
    prefix = prefix_from_weights(get, cfg, first_layer=0)
    idx = jnp.asarray(fixture["idx"])

    logits = clean_suffix_logits(target, prefix_residual(prefix, idx))
    assert logits.shape == fixture["logits"].shape
    assert _max_abs_diff(logits, fixture["logits"]) < 1e-5


@pytest.mark.skipif(not REAL_CACHE_DIR.exists(), reason="t-9d2b8f02 pretrain cache not mounted")
def test_real_t9d2b8f02_weights_match_torch_logits():
    fixture = np.load(FIXTURE_DIR / "real_t-9d2b8f02_fixture.npz")
    cfg = load_model_config(REAL_CACHE_DIR)
    target = load_target_from_pretrain_cache(REAL_CACHE_DIR, cfg, 0, jnp.float32)

    embed = target.lm_head  # tied: wte.weight serves embedding and head
    resid = embed[jnp.asarray(fixture["idx"])]
    logits = clean_suffix_logits(target, resid)

    assert logits.shape == fixture["logits"].shape
    # fp32 end to end; |logits| ~ 15, observed max abs diff ~1e-4 (matmul reassociation)
    assert _max_abs_diff(logits, fixture["logits"]) < 2e-3
