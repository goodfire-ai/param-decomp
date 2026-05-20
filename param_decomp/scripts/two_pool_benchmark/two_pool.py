"""2-pool benchmark — thin wrapper over `param_decomp.two_pool.optimize_two_pool`.

Demonstrates how the public 2-pool entry point composes with the rest of the
param_decomp infrastructure (ComponentModel, PersistentPGDState, ReconstructionLoss,
configs). The benchmark is intentionally minimal: build a tiny target, configure the
2-pool layout, and let `optimize_two_pool` run the training loop while we record
per-step wall-clock around the dispatcher.

Run on 8 GPUs single-node:
    .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=8 \\
        -m param_decomp.scripts.two_pool_benchmark.two_pool
"""

# pyright: reportArgumentType=false, reportOperatorIssue=false, reportIndexIssue=false

import os
import time

import torch
import torch.distributed as dist
from torch import Tensor

from param_decomp.configs import (
    AttnConfig,
    GlobalCiConfig,
    GlobalSharedTransformerCiConfig,
    PerBatchPerPositionScope,
    PersistentPGDReconLossConfig,
    ScheduleConfig,
    SignPGDConfig,
)
from param_decomp.models.batch_and_loss_fns import recon_loss_kl, run_batch_passthrough
from param_decomp.scripts.two_pool_benchmark._tiny_model import TinyTransformer, sites_for_block
from param_decomp.two_pool import BlockGroup, TwoPoolConfig, optimize_two_pool

VOCAB = 8192
D_MODEL = 768
N_HEADS = 12
D_MLP = 3072
N_TRANSFORMER_BLOCKS = 6
BATCH = 8
SEQ_LEN = 64
C = 32
# Per-rank CI fn config. Each pool-A rank gets a `GlobalSharedTransformerCiFn`
# instantiated over its owned sites (singleton in maxA, 3-site group here in
# wider). The fn is shared across whatever sites the rank owns — kept modest
# so it isn't the bottleneck. See `param_decomp/models/components.py`.
CI_D_MODEL = 128
CI_N_BLOCKS = 2
CI_N_HEADS = 4

# Topology: 3 block groups × 2 ranks (in-block DDP-2) + 2 pool B ranks (DP-2) = 8 GPUs.
BLOCK_GROUP_RANKS: tuple[tuple[int, ...], ...] = ((0, 1), (2, 3), (4, 5))
POOL_B_RANKS: tuple[int, ...] = (6, 7)
BLOCKS_PER_GROUP = N_TRANSFORMER_BLOCKS // len(BLOCK_GROUP_RANKS)

WARMUP_STEPS = 2
PROFILE_STEPS = 4


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 8
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    torch.manual_seed(0)
    target = TinyTransformer(VOCAB, D_MODEL, N_TRANSFORMER_BLOCKS, N_HEADS, D_MLP).to(device)
    target.requires_grad_(False)

    block_groups = tuple(
        BlockGroup(
            ranks=ranks,
            owned_sites=tuple(
                s
                for tb in range(g * BLOCKS_PER_GROUP, (g + 1) * BLOCKS_PER_GROUP)
                for s in sites_for_block(tb)
            ),
        )
        for g, ranks in enumerate(BLOCK_GROUP_RANKS)
    )
    all_sites = [s for bg in block_groups for s in bg.owned_sites]
    c_per_site = {s: C for s in all_sites}

    ppgd_cfg = PersistentPGDReconLossConfig(
        coeff=1.0,
        scope=PerBatchPerPositionScope(),
        optimizer=SignPGDConfig(lr_schedule=ScheduleConfig(start_val=0.01)),
        n_warmup_steps=2,
        n_samples=1,
        use_sigmoid_parameterization=False,
    )

    pool_config = TwoPoolConfig(
        block_groups=block_groups,
        pool_b_ranks=POOL_B_RANKS,
        batch_global=BATCH,
        c_per_site=c_per_site,
        ci_config=GlobalCiConfig(
            fn_type="global_shared_transformer",
            simple_transformer_ci_cfg=GlobalSharedTransformerCiConfig(
                d_model=CI_D_MODEL,
                n_blocks=CI_N_BLOCKS,
                attn_config=AttnConfig(n_heads=CI_N_HEADS),
            ),
        ),
        sigmoid_type="leaky_hard",
        run_batch=run_batch_passthrough,
        reconstruction_loss=recon_loss_kl,
        ppgd_cfg=ppgd_cfg,
        bf16_autocast=True,
    )

    if rank == 0:
        print("[two_pool] 2-POOL via optimize_two_pool (param_decomp.two_pool.run)", flush=True)
        print(
            f"[two_pool] batch={BATCH} (A_local={BATCH // 2} B_local={BATCH // 2}) "
            f"seq={SEQ_LEN} d={D_MODEL} d_mlp={D_MLP} n_blocks={N_TRANSFORMER_BLOCKS} "
            f"ci_d_model={CI_D_MODEL} ci_n_blocks={CI_N_BLOCKS} ci_n_heads={CI_N_HEADS}",
            flush=True,
        )

    data_rng = torch.Generator(device=device).manual_seed(0)

    def batch_iter(step: int) -> Tensor:
        data_rng.manual_seed(step * 7919 + 17)
        return torch.randint(0, VOCAB, (BATCH, SEQ_LEN), device=device, generator=data_rng)

    step_times: list[float] = []

    def on_step(step: int, metrics: dict[str, float]) -> None:
        torch.cuda.synchronize()
        now = time.perf_counter()
        step_times.append(now)
        if rank in (0, POOL_B_RANKS[0]):
            mem = torch.cuda.memory_allocated(device) / 1e9
            peak = torch.cuda.max_memory_allocated(device) / 1e9
            metrics_str = " ".join(f"{k}={v:.4g}" for k, v in metrics.items())
            print(
                f"[two_pool rank{rank}] step={step} mem={mem:.2f}GB peak={peak:.2f}GB {metrics_str}",
                flush=True,
            )

    # Warm up cuda + record t=0 right before the loop starts.
    torch.cuda.synchronize()
    step_times.append(time.perf_counter())

    optimize_two_pool(
        target_model=target,
        pool_config=pool_config,
        device=device,
        n_steps=WARMUP_STEPS + PROFILE_STEPS,
        batch_iter=batch_iter,
        on_step=on_step,
    )

    # Compute per-step wall-clock skipping warmup.
    intervals = [step_times[i + 1] - step_times[i] for i in range(len(step_times) - 1)]
    profile = intervals[WARMUP_STEPS:]
    if profile and rank == 0:
        avg_ms = 1000 * sum(profile) / len(profile)
        print(
            f"\n[two_pool rank0] STEP_TOTAL avg={avg_ms:.2f}ms  min={1000 * min(profile):.2f}ms  "
            f"max={1000 * max(profile):.2f}ms  (n={len(profile)})",
            flush=True,
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
