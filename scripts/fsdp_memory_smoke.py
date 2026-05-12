"""Standalone FSDP-fit smoke test.

The full training loop has many `model(...)` calls that bypass the FSDP wrapper
(losses, metrics, PPGD). Plumbing `wrapped_model` through all of those is real
work; this script bypasses that question and answers the more important one:
*does FSDP-wrapping the new fused-decomposition-sites ComponentModel actually
fit a target that ZeRO-1+ckpt OOMed on?*

What it does:
  - Loads a target from `--target_checkpoint`
  - Builds a ComponentModel at Jose's CI-fn shape
  - Wraps with `fsdp_wrap`
  - Runs a single forward + backward through the wrapped model with a tiny batch
  - Reports `max_memory_allocated()` per rank

Use with the existing 4B / 2B / 1B random target checkpoints from Phase 5.
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist

from param_decomp.configs import (
    AttnConfig,
    GlobalCiConfig,
    GlobalSharedTransformerCiConfig,
    ModulePatternInfoConfig,
)
from param_decomp.models.batch_and_loss_fns import make_run_batch
from param_decomp.models.component_model import ComponentModel
from param_decomp.models.mask_info import make_mask_infos
from param_decomp.pretrain.models.llama_simple_mlp import LlamaSimpleMLP
from param_decomp.pretrain.run_info import PretrainRunInfo
from param_decomp.utils.fsdp import fsdp_wrap
from param_decomp.utils.module_utils import expand_module_patterns

JOSE_MODULE_INFO = [
    ModulePatternInfoConfig(module_pattern="h.*.mlp.c_fc", C=3072),
    ModulePatternInfoConfig(module_pattern="h.*.mlp.down_proj", C=3584),
    ModulePatternInfoConfig(module_pattern="h.*.attn.q_proj", C=512),
    ModulePatternInfoConfig(module_pattern="h.*.attn.k_proj", C=512),
    ModulePatternInfoConfig(module_pattern="h.*.attn.v_proj", C=1024),
    ModulePatternInfoConfig(module_pattern="h.*.attn.o_proj", C=1024),
]


def build_ci_config() -> GlobalCiConfig:
    """Jose-shape CI fn (d_model=2048, 8 blocks, 8192 mlp)."""
    return GlobalCiConfig(
        mode="global",
        fn_type="global_shared_transformer",
        simple_transformer_ci_cfg=GlobalSharedTransformerCiConfig(
            d_model=2048,
            n_blocks=8,
            mlp_hidden_dim=[8192],
            attn_config=AttnConfig(n_heads=16, max_len=512, rope_base=10000.0),
            gradient_checkpointing=True,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_checkpoint", type=str, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=512)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl")

    def rprint(*a: object) -> None:
        if rank == 0:
            print(*a, flush=True)

    rprint(f"world_size={world_size} batch={args.batch} seq={args.seq}")
    rprint(f"target_checkpoint={args.target_checkpoint}")

    # 1. Build target on the right device, frozen.
    rinfo = PretrainRunInfo.from_path(args.target_checkpoint)
    target = LlamaSimpleMLP.from_run_info(rinfo).to(device).eval()
    target.requires_grad_(False)

    n_target = sum(p.numel() for p in target.parameters())
    rprint(f"target params: {n_target / 1e9:.3f} B ({n_target * 4 / 1e9:.2f} GB fp32)")

    # 2. Build ComponentModel with fused sites.
    module_path_info = expand_module_patterns(target, JOSE_MODULE_INFO)
    cm = ComponentModel(
        target_model=target,
        run_batch=make_run_batch(output_extract=0),
        module_path_info=module_path_info,
        ci_config=build_ci_config(),
        sigmoid_type="leaky_hard",
    ).to(device)

    n_trainable = sum(p.numel() for p in cm.parameters() if p.requires_grad)
    rprint(f"trainable: {n_trainable / 1e9:.3f} B ({n_trainable * 4 / 1e9:.2f} GB fp32)")

    # 3. FSDP-wrap.
    rprint("FSDP-wrapping...")
    wrapped = fsdp_wrap(cm, device_id=local_rank, autocast_bf16=True)
    torch.cuda.synchronize()
    rprint(f"post-wrap memory: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")

    # 4. Forward + backward through the wrapped model.
    optimizer = torch.optim.AdamW([p for p in wrapped.parameters() if p.requires_grad], lr=1e-5)

    rprint("\nrun #1: target-only forward (no mask_infos, no backward — target is frozen)")
    torch.cuda.reset_peak_memory_stats(device)
    idx = torch.randint(0, 50000, (args.batch, args.seq), device=device)
    with torch.no_grad():
        wrapped(idx)
    torch.cuda.synchronize()
    rprint(f"  peak: {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB")

    # 5. Decomposed forward — push every site through its decomposed path.
    rprint("\nrun #2: decomposed forward (mask=1, no delta)")
    torch.cuda.reset_peak_memory_stats(device)
    # Need component_mask shapes per site. Use ones, which mirror what an "all alive" run looks like.
    masks: dict[str, torch.Tensor] = {}
    for site_name, site in cm.components.items():
        masks[site_name] = torch.ones(args.batch, args.seq, site.C, device=device)
    mask_infos = make_mask_infos(masks)
    out = wrapped(idx, mask_infos=mask_infos)
    loss = out.float().pow(2).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    rprint(f"  peak: {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
