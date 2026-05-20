"""Wider 2-pool: 14 pool-A ranks × 1 site-group each + 2 pool-B ranks DP-2.

Same model as `two_pool.py` but with much more pool-A parallelism:

  - 14 block groups × 1 rank each (no in-block DDP, so the in-block all-reduce is
    a no-op single-rank group).
  - Each pool-A rank owns 3 sites (42 sites / 14 groups). Layerwise loop is 3
    serial forwards per rank instead of 14.
  - Pool A rank's batch is the full batch (batch_local_a = batch_global since
    n_per_block_group = 1).
  - Pool B DP-2 unchanged.

Total: 14 + 2 = 16 GPUs across 2 nodes.

Goal: validate that pool A was the bottleneck in `two_pool.py` (6A + 2B with
14 sites/rank). If wall-clock drops here, the answer is yes and "pool A's job
is to not be the bottleneck" is now achieved.

Run:
    sbatch param_decomp/scripts/two_pool_benchmark/two_pool_wider.sbatch
"""

# pyright: reportArgumentType=false, reportOperatorIssue=false, reportIndexIssue=false

import os
import time
from collections import defaultdict

import torch
import torch.distributed as dist
from torch import Tensor

from param_decomp.configs import (
    LayerwiseCiConfig,
    PerBatchPerPositionScope,
    PersistentPGDReconLossConfig,
    ScheduleConfig,
    SignPGDConfig,
)
from param_decomp.models.batch_and_loss_fns import recon_loss_kl, run_batch_passthrough
from param_decomp.scripts.two_pool_benchmark._tiny_model import TinyTransformer, sites_for_block
from param_decomp.two_pool import TwoPoolConfig, optimize_two_pool

# Same model as two_pool.py — only the topology and batch slicing change.
VOCAB = 8192
D_MODEL = 768
N_HEADS = 12
D_MLP = 3072
N_TRANSFORMER_BLOCKS = 6
BATCH = 8
SEQ_LEN = 64
C = 32
CI_HIDDEN = 1024

# Topology: 14 single-rank block groups + 2 pool B ranks. Total 16 GPUs.
N_BLOCK_GROUPS = 14
N_PER_BLOCK_GROUP = 1
N_POOL_B = 2
SITES_PER_GROUP = (N_TRANSFORMER_BLOCKS * 7) // N_BLOCK_GROUPS  # 42 / 14 = 3
assert N_TRANSFORMER_BLOCKS * 7 == N_BLOCK_GROUPS * SITES_PER_GROUP, (
    f"sites must divide evenly across groups: {N_TRANSFORMER_BLOCKS * 7} != "
    f"{N_BLOCK_GROUPS * SITES_PER_GROUP}"
)

# Spread the 14 pool-A ranks across 2 nodes (alternating) so cross-pool comm sees
# both intra- and inter-node sends — same kind of layout we used in the nano
# multi-node profiles.
#
# SLURM places ranks 0-7 on node 0, ranks 8-15 on node 1.
# Pool A:  {0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14}   (7 on node 0, 7 on node 1) — wait, 14 ranks
# Actually we have ranks 0-15. Reserve 2 ranks for pool B; the rest for pool A.
# Put pool B as one-per-node so pool_b_allreduce is also cross-node.
POOL_B_RANKS = (6, 14)
POOL_A_RANKS = tuple(r for r in range(16) if r not in POOL_B_RANKS)
assert len(POOL_A_RANKS) == N_BLOCK_GROUPS, (
    f"need {N_BLOCK_GROUPS} pool A ranks, got {len(POOL_A_RANKS)}"
)
BLOCK_GROUPS: tuple[tuple[int, ...], ...] = tuple((r,) for r in POOL_A_RANKS)

WARMUP_STEPS = 2
PROFILE_STEPS = 4


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 16
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    torch.manual_seed(0)
    target = TinyTransformer(VOCAB, D_MODEL, N_TRANSFORMER_BLOCKS, N_HEADS, D_MLP).to(device)
    target.requires_grad_(False)

    all_sites_list = [s for b in range(N_TRANSFORMER_BLOCKS) for s in sites_for_block(b)]
    # Contiguous chunks of SITES_PER_GROUP sites per block group, in canonical order.
    # A group may span a transformer-block boundary, which is fine — block_owned_sites
    # is just a logical grouping for ownership, independent of the target's structure.
    block_owned_sites: tuple[tuple[str, ...], ...] = tuple(
        tuple(all_sites_list[i * SITES_PER_GROUP : (i + 1) * SITES_PER_GROUP])
        for i in range(N_BLOCK_GROUPS)
    )
    c_per_site = {s: C for s in all_sites_list}

    ppgd_cfg = PersistentPGDReconLossConfig(
        coeff=1.0,
        scope=PerBatchPerPositionScope(),
        optimizer=SignPGDConfig(lr_schedule=ScheduleConfig(start_val=0.01)),
        n_warmup_steps=2,
        n_samples=1,
        use_sigmoid_parameterization=False,
    )

    pool_config = TwoPoolConfig(
        block_groups=BLOCK_GROUPS,
        block_owned_sites=block_owned_sites,
        pool_b_ranks=POOL_B_RANKS,
        batch_global=BATCH,
        c_per_site=c_per_site,
        ci_config=LayerwiseCiConfig(fn_type="vector_mlp", hidden_dims=[CI_HIDDEN]),
        sigmoid_type="leaky_hard",
        run_batch=run_batch_passthrough,
        reconstruction_loss=recon_loss_kl,
        ppgd_cfg=ppgd_cfg,
    )

    if rank == 0:
        print(
            f"[wider] 2-POOL WIDER  ({N_BLOCK_GROUPS}A + {N_POOL_B}B = {world_size} GPUs, "
            f"{SITES_PER_GROUP} sites / pool-A rank)",
            flush=True,
        )
        print(
            f"[wider] batch={BATCH} (A_local={BATCH // N_PER_BLOCK_GROUP} "
            f"B_local={BATCH // N_POOL_B}) seq={SEQ_LEN} d={D_MODEL} d_mlp={D_MLP} "
            f"n_blocks={N_TRANSFORMER_BLOCKS} ci_hidden={CI_HIDDEN}",
            flush=True,
        )

    data_rng = torch.Generator(device=device).manual_seed(0)

    def batch_iter(step: int) -> Tensor:
        data_rng.manual_seed(step * 7919 + 17)
        return torch.randint(0, VOCAB, (BATCH, SEQ_LEN), device=device, generator=data_rng)

    step_times: list[float] = []

    def on_step(step: int, metrics: dict[str, float]) -> None:
        torch.cuda.synchronize()
        step_times.append(time.perf_counter())
        if rank in (0, POOL_B_RANKS[0]):
            mem = torch.cuda.memory_allocated(device) / 1e9
            peak = torch.cuda.max_memory_allocated(device) / 1e9
            metrics_str = " ".join(f"{k}={v:.4g}" for k, v in metrics.items())
            print(
                f"[wider rank{rank}] step={step} mem={mem:.2f}GB peak={peak:.2f}GB {metrics_str}",
                flush=True,
            )

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

    intervals = [step_times[i + 1] - step_times[i] for i in range(len(step_times) - 1)]
    profile = intervals[WARMUP_STEPS:]
    if profile and rank == 0:
        avg_ms = 1000 * sum(profile) / len(profile)
        print(
            f"\n[wider rank0] STEP_TOTAL avg={avg_ms:.2f}ms  "
            f"min={1000*min(profile):.2f}ms  max={1000*max(profile):.2f}ms  (n={len(profile)})",
            flush=True,
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
