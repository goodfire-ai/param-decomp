"""One-off pretrain-checkpoint conversion: `model_step_<N>.pt` -> safetensors.

Runs in the TORCH venv (the JAX venv has no torch); the JAX trainer's
`llama_simple_mlp` loader reads only the safetensors:

    /mnt/home/oli/param-decomp/.venv/bin/python \
        jax_single_pool/tools/convert_llama_simple_mlp_checkpoint.py <pretrain_cache_dir>

The tied `lm_head.weight` is dropped (safetensors refuses shared storage); the JAX
loader reads `wte.weight` for both the embedding and the head.
"""

import argparse
import re
from pathlib import Path

import torch
from safetensors.torch import save_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_dir", type=Path, help="pretrain_cache/<project>-<run_id> dir")
    args = parser.parse_args()

    step_pattern = re.compile(r"^model_step_(\d+)\.pt$")
    checkpoints = sorted(
        (int(m.group(1)), p)
        for p in args.cache_dir.glob("model_step_*.pt")
        if (m := step_pattern.match(p.name)) is not None
    )
    assert checkpoints, f"no model_step_*.pt under {args.cache_dir}"
    _, checkpoint_path = checkpoints[-1]

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert torch.equal(state_dict["lm_head.weight"], state_dict["wte.weight"]), "head not tied"
    del state_dict["lm_head.weight"]

    out_path = checkpoint_path.with_suffix(".safetensors")
    save_file({k: v.contiguous() for k, v in state_dict.items()}, str(out_path))
    print(f"wrote {out_path} ({len(state_dict)} tensors)")


if __name__ == "__main__":
    main()
