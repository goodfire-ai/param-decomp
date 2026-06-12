"""Generate the torch-reference fixtures for the LlamaSimpleMLP equivalence tests.

Runs in the TORCH venv (imports the read-only torch reference model):

    /mnt/home/oli/param-decomp/.venv/bin/python \
        jax_single_pool/tests/simple_mlp_equivalence/gen_torch_fixtures.py

Writes two npz fixtures next to this script:

  * `tiny_fixture.npz` — a seeded random tiny config (GQA repeat=2, exercising the
    kv-head repeat the real model doesn't): config json + full state dict + token ids
    + fp32 logits. Hermetic — the JAX test rebuilds the model from these weights.
  * `real_t-9d2b8f02_fixture.npz` — the real pile checkpoint on a short input: token
    ids + fp32 logits only (the JAX side loads the cached safetensors).

Regenerate only if the torch reference model changes.
"""

import json
from pathlib import Path

import numpy as np
import torch
import yaml

from param_decomp_lab.experiments.lm.pretrain.models.llama_simple_mlp import (
    LlamaSimpleMLP,
    LlamaSimpleMLPConfig,
)

FIXTURE_DIR = Path(__file__).parent
REAL_CACHE_DIR = Path("/mnt/data/artifacts/mechanisms/param-decomp/pretrain_cache/spd-t-9d2b8f02")

TINY_CONFIG = dict(
    model_type="LlamaSimpleMLP",
    block_size=64,
    vocab_size=64,
    n_layer=3,
    n_head=4,
    n_embd=32,
    n_intermediate=64,
    mlp_bias=False,
    attn_bias=False,
    rotary_adjacent_pairs=False,
    rotary_dim=8,
    rotary_base=10000,
    n_ctx=64,
    n_key_value_heads=2,
    use_grouped_query_attention=True,
    flash_attention=False,
    rms_norm_eps=1e-6,
)


def gen_tiny() -> None:
    model = LlamaSimpleMLP(LlamaSimpleMLPConfig(**TINY_CONFIG))
    model.eval()
    generator = torch.Generator().manual_seed(7)
    idx = torch.randint(0, TINY_CONFIG["vocab_size"], (2, 16), generator=generator)
    with torch.no_grad():
        logits, _ = model(idx)
    assert logits is not None
    arrays = {f"weights.{k}": v.numpy() for k, v in model.state_dict().items()}
    np.savez(
        FIXTURE_DIR / "tiny_fixture.npz",
        config_json=np.array(json.dumps(TINY_CONFIG)),
        idx=idx.numpy(),
        logits=logits.float().numpy(),
        **arrays,
    )
    print(f"tiny: logits {tuple(logits.shape)}, |logits|max {logits.abs().max():.3f}")


def gen_real() -> None:
    model_config = yaml.safe_load((REAL_CACHE_DIR / "model_config.yaml").read_text())
    model = LlamaSimpleMLP(LlamaSimpleMLPConfig(**model_config))
    state_dict = torch.load(
        REAL_CACHE_DIR / "model_step_99999.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    generator = torch.Generator().manual_seed(11)
    idx = torch.randint(0, model_config["vocab_size"], (1, 16), generator=generator)
    with torch.no_grad():
        logits, _ = model(idx)
    assert logits is not None
    np.savez(
        FIXTURE_DIR / "real_t-9d2b8f02_fixture.npz",
        idx=idx.numpy(),
        logits=logits.float().numpy(),
        checkpoint=np.array("model_step_99999.pt"),
    )
    print(f"real: logits {tuple(logits.shape)}, |logits|max {logits.abs().max():.3f}")


if __name__ == "__main__":
    gen_tiny()
    gen_real()
