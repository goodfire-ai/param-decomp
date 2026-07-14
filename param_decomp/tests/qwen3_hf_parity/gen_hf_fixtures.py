"""Generate the Qwen3 HF-parity goldens — a TORCH-ENV script (the repo venv is
torch-free; run it in a throwaway venv, like the `tests/equivalence` torch generators):

    uv venv /tmp/qwen3-golden --python 3.12
    VIRTUAL_ENV=/tmp/qwen3-golden uv pip install torch --index-url https://download.pytorch.org/whl/cpu
    VIRTUAL_ENV=/tmp/qwen3-golden uv pip install transformers numpy
    /tmp/qwen3-golden/bin/python param_decomp/tests/qwen3_hf_parity/gen_hf_fixtures.py         # tiny
    /tmp/qwen3-golden/bin/python param_decomp/tests/qwen3_hf_parity/gen_hf_fixtures.py --real  # 8B

Tiny golden (`qwen3_tiny_hf_fixtures.npz`): a seeded random `Qwen3ForCausalLM` at a
4-layer toy config, fp32, eager attention — the exact-architecture check (QK-norm, GQA,
RoPE base) `test_qwen3_hf_parity.py` compares against at fp32 tolerance. The full state
dict rides in the npz under `sd::`-prefixed keys.

Real golden (`qwen3_8b_real_logits.npz`): `Qwen/Qwen3-8B-Base` bf16 from the HF cache,
a few fixed prompts, final-position fp32 logits — the weight-loading end-to-end check
(slow test). Regenerate only if the MATH changes; record the transformers version bump
in the commit message.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

HERE = Path(__file__).resolve().parent

TINY_CONFIG = dict(
    vocab_size=64,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    rope_theta=1000000.0,
    rms_norm_eps=1e-6,
    max_position_embeddings=512,
    tie_word_embeddings=False,
    attention_bias=False,
    use_cache=False,
)

REAL_MODEL = "Qwen/Qwen3-8B-Base"
REAL_PROMPTS = (
    "The capital of France is Paris, and the capital of Germany is",
    "In 1859, Charles Darwin published On the Origin of Species, which",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "The mitochondria is the powerhouse of the cell, and the nucleus stores the",
)
REAL_SEQ_LEN = 12


def gen_tiny() -> None:
    torch.manual_seed(0)
    cfg = Qwen3Config(**TINY_CONFIG)
    model = Qwen3ForCausalLM._from_config(cfg, attn_implementation="eager").eval()
    tokens = torch.randint(0, cfg.vocab_size, (2, 16), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        logits = model(tokens).logits
    arrays = {f"sd::{k}": v.numpy() for k, v in model.state_dict().items()}
    import transformers

    np.savez_compressed(
        HERE / "qwen3_tiny_hf_fixtures.npz",
        **arrays,
        tokens=tokens.numpy(),
        logits=logits.numpy(),
        config_json=np.array(json.dumps(TINY_CONFIG)),
        transformers_version=np.array(transformers.__version__),
    )
    print(f"tiny golden: {len(arrays)} tensors, logits {tuple(logits.shape)}")


def gen_real() -> None:
    tokenizer = AutoTokenizer.from_pretrained(REAL_MODEL)
    ids = []
    for prompt in REAL_PROMPTS:
        prompt_ids = tokenizer(prompt).input_ids
        assert len(prompt_ids) >= REAL_SEQ_LEN, (prompt, len(prompt_ids))
        ids.append(prompt_ids[:REAL_SEQ_LEN])
    tokens = torch.tensor(ids)
    model = Qwen3ForCausalLM.from_pretrained(REAL_MODEL, dtype=torch.bfloat16).eval()
    with torch.no_grad():
        final_logits = model(tokens).logits[:, -1, :].float()
    import transformers

    np.savez_compressed(
        HERE / "qwen3_8b_real_logits.npz",
        tokens=tokens.numpy(),
        final_logits=final_logits.numpy(),
        transformers_version=np.array(transformers.__version__),
    )
    print(f"real golden: tokens {tuple(tokens.shape)}, final logits {tuple(final_logits.shape)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="generate the Qwen3-8B-Base golden")
    if parser.parse_args().real:
        gen_real()
    else:
        gen_tiny()
