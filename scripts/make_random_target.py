"""Create a randomly-initialized LlamaSimpleMLP checkpoint at a chosen scale.

For Phase 5 of the scaling investigation: we need a 1B-parameter target to stress-
test the per-rank memory math from the report's §3 table, but training one is
unnecessary for memory measurement. This builds a random-init model and writes
the directory layout PretrainRunInfo expects so it can be loaded as a target.

Output layout (so PretrainRunInfo.from_path(<output>/checkpoints/model.pt) works):
    <output>/
      checkpoints/model.pt
      final_config.yaml          (synthetic — only the bits the loader reads)
      model_config.yaml          (LlamaSimpleMLPConfig as a dict)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from param_decomp.pretrain.models.llama_simple_mlp import (
    LlamaSimpleMLP,
    LlamaSimpleMLPConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n_layer", type=int, default=18)
    parser.add_argument("--n_embd", type=int, default=2048)
    parser.add_argument("--n_head", type=int, default=16)
    parser.add_argument("--n_intermediate", type=int, default=8192)
    parser.add_argument("--vocab_size", type=int, default=50277)
    parser.add_argument("--n_ctx", type=int, default=512)
    parser.add_argument("--tokenizer_name", type=str, default="EleutherAI/gpt-neox-20b")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "checkpoints").mkdir(exist_ok=True)

    config = LlamaSimpleMLPConfig(
        model_type="LlamaSimpleMLP",
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_intermediate=args.n_intermediate,
        vocab_size=args.vocab_size,
        n_ctx=args.n_ctx,
        block_size=args.n_ctx,
        n_key_value_heads=args.n_head,  # no GQA — keeps math simple
        rotary_dim=args.n_embd // args.n_head,
        use_grouped_query_attention=True,
    )
    print("Building model...")
    model = LlamaSimpleMLP(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Built {n_params / 1e9:.3f} B params (target = {args.n_embd}d × {args.n_layer}L)")

    ckpt_path = args.output / "checkpoints" / "model_random.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Wrote checkpoint to {ckpt_path}")

    # PretrainRunInfo.from_path expects model_config.yaml and final_config.yaml in the
    # parent.parent of the checkpoint. We synthesize the smallest config that the
    # PretrainRunInfo loader will accept (it only reads the dicts; we stash a tokenizer
    # entry in final_config so _extract_hf_tokenizer_path returns something reasonable).
    model_config_dict = config.model_dump()
    (args.output / "model_config.yaml").write_text(yaml.dump(model_config_dict))

    final_config_dict = {
        "tokenizer_name": args.tokenizer_name,
        "model_type": "LlamaSimpleMLP",
        "model_config": model_config_dict,
    }
    (args.output / "final_config.yaml").write_text(yaml.dump(final_config_dict))

    print(f"Done. Use as: pretrained_model_name={ckpt_path}")
    summary = {
        "n_params": n_params,
        "n_params_billions": n_params / 1e9,
        "config": model_config_dict,
        "checkpoint_path": str(ckpt_path),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
