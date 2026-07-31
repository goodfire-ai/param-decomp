"""`HFWeights` over a synthetic sharded snapshot — no network, no cached HF checkpoint.

Real HF checkpoints (SmolLM2, Llama-3.1-8B, Qwen3-8B) are bf16 on disk, so the read path
must handle bf16 without relying on numpy resolving the "bfloat16" dtype name (stock
numpy only can when ml_dtypes registration has run as an import side effect).
"""

import json
from pathlib import Path

import jax.numpy as jnp
from safetensors.flax import save_file

from param_decomp.targets.glu_transformer import HFWeights

_EMBED_KEY = "model.embed_tokens.weight"
_NORM_KEY = "model.norm.weight"


def _write_snapshot(snapshot: Path) -> tuple[jnp.ndarray, jnp.ndarray]:
    """A minimal two-shard snapshot: a bf16 tensor and an fp32 tensor, plus the
    `model.safetensors.index.json` weight_map `HFWeights.__init__` reads."""
    embed = (jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 7).astype(jnp.bfloat16)
    norm = jnp.linspace(0.5, 1.5, 4, dtype=jnp.float32)
    shard_of = {
        _EMBED_KEY: "model-00001-of-00002.safetensors",
        _NORM_KEY: "model-00002-of-00002.safetensors",
    }
    save_file({_EMBED_KEY: embed}, snapshot / shard_of[_EMBED_KEY])
    save_file({_NORM_KEY: norm}, snapshot / shard_of[_NORM_KEY])
    (snapshot / "model.safetensors.index.json").write_text(json.dumps({"weight_map": shard_of}))
    return embed, norm


def test_hf_weights_reads_bf16_shards(tmp_path: Path):
    embed, norm = _write_snapshot(tmp_path)

    w = HFWeights(tmp_path, jnp.bfloat16)
    got_embed = w.get(_EMBED_KEY)
    assert got_embed.dtype == jnp.bfloat16
    assert jnp.array_equal(got_embed, embed)
    got_norm = w.get(_NORM_KEY)
    assert got_norm.dtype == jnp.bfloat16
    assert jnp.array_equal(got_norm, norm.astype(jnp.bfloat16))


def test_hf_weights_casts_bf16_up_to_fp32(tmp_path: Path):
    embed, norm = _write_snapshot(tmp_path)

    w = HFWeights(tmp_path, jnp.float32)
    got_embed = w.get(_EMBED_KEY)
    assert got_embed.dtype == jnp.float32
    assert jnp.array_equal(got_embed, embed.astype(jnp.float32))
    assert jnp.array_equal(w.get(_NORM_KEY), norm)


def test_hf_weights_stages_reads_on_host_cpu(tmp_path: Path):
    """Loaded leaves must be host-resident: `place_via_shardings` serves device shards
    from the loaded copy, and a multi-GB checkpoint must not pass through a single
    accelerator first."""
    _write_snapshot(tmp_path)
    got = HFWeights(tmp_path, jnp.bfloat16).get(_EMBED_KEY)
    assert all(d.platform == "cpu" for d in got.devices())
