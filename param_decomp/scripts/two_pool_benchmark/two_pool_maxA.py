"""Max-pool-A 2-pool: 42 pool-A ranks × 1 site each + 6 pool-B DP-6 = 48 GPUs.

Same model as the other two_pool_benchmark variants. Pushes pool A parallelism
to the maximum the model admits — one decomposable site per rank, so the
layerwise loop on each rank is exactly 1 forward+backward.

Block groups distributed across 6 nodes so cross-pool sends, pool_b_allreduce,
and the (now no-op) in-block all-reduce all see realistic multi-node topology.

Pool B is DP-6 (one rank per node, all cross-IB). batch_global=12 gives each
pool-B rank a slice of 2; each pool-A rank sees the full batch=12.

Run:
    sbatch param_decomp/scripts/two_pool_benchmark/two_pool_maxA.sbatch
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

# Same model as two_pool.py / two_pool_wider.py.
VOCAB = 8192
D_MODEL = 768
N_HEADS = 12
D_MLP = 3072
N_TRANSFORMER_BLOCKS = 6   # → 42 sites total
BATCH = 12                 # divisible by both N_PER_BLOCK_GROUP (=1) and N_POOL_B (=6)
SEQ_LEN = 64
C = 32
CI_HIDDEN = 1024

# Topology: 48 GPUs across 6 nodes.
#   Per node: 7 pool-A ranks + 1 pool-B rank (the last rank on each node)
#   Pool A: 42 ranks total = 42 block groups × 1 rank, 1 site/group.
#   Pool B: 6 ranks (one per node, all cross-IB for pool_b_allreduce).
N_NODES = 6
GPUS_PER_NODE = 8
WORLD_SIZE = N_NODES * GPUS_PER_NODE  # 48
N_BLOCK_GROUPS = 42
N_PER_BLOCK_GROUP = 1
N_POOL_B = 6

# Reserve rank 7, 15, 23, 31, 39, 47 (last on each node) for pool B.
POOL_B_RANKS: tuple[int, ...] = tuple(node * GPUS_PER_NODE + (GPUS_PER_NODE - 1) for node in range(N_NODES))
POOL_A_RANKS: tuple[int, ...] = tuple(r for r in range(WORLD_SIZE) if r not in POOL_B_RANKS)
assert len(POOL_A_RANKS) == N_BLOCK_GROUPS

# Block groups: one rank per group.
BLOCK_GROUPS: tuple[tuple[int, ...], ...] = tuple((r,) for r in POOL_A_RANKS)

WARMUP_STEPS = 2
PROFILE_STEPS = 4


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == WORLD_SIZE
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    torch.manual_seed(0)
    target = TinyTransformer(VOCAB, D_MODEL, N_TRANSFORMER_BLOCKS, N_HEADS, D_MLP).to(device)
    target.requires_grad_(False)

    all_sites_list = [s for b in range(N_TRANSFORMER_BLOCKS) for s in sites_for_block(b)]
    # 1 site per block group, in canonical order.
    assert len(all_sites_list) == N_BLOCK_GROUPS
    block_owned_sites: tuple[tuple[str, ...], ...] = tuple((s,) for s in all_sites_list)
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
            f"[maxA] 2-POOL MAX-A  ({N_BLOCK_GROUPS}A + {N_POOL_B}B = {world_size} GPUs across "
            f"{N_NODES} nodes; 1 site per pool-A rank)",
            flush=True,
        )
        print(
            f"[maxA] batch={BATCH} (A_local={BATCH // N_PER_BLOCK_GROUP} "
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
                f"[maxA rank{rank}] step={step} mem={mem:.2f}GB peak={peak:.2f}GB {metrics_str}",
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
        per_sample = avg_ms / BATCH
        print(
            f"\n[maxA rank0] STEP_TOTAL avg={avg_ms:.2f}ms  "
            f"min={1000*min(profile):.2f}ms  max={1000*max(profile):.2f}ms  (n={len(profile)})",
            flush=True,
        )
        print(
            f"[maxA rank0] per-sample throughput: {1000/per_sample:.1f} samples/sec/world",
            flush=True,
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
