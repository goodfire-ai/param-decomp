"""2-pool launcher using the LM driver for model + dataloader materialization.

  - Model / tokenizer / data loader / module_info come from a regular RunConfig
    YAML (`--run_config`), loaded via the LM driver — same machinery as a
    standard `pd-run` invocation.
  - Topology comes from a separate `TwoPoolConfig` YAML (`--topology`).
  - ``run_two_pool`` glues them together and runs ``optimize_two_pool``.

Drop in any pair: `qwen3_0p6b.yaml + qwen3_0p6b_two_pool_topology.yaml`,
`llama_simple_mlp_12L.yaml + llama_simple_mlp_12L_two_pool_topology.yaml`, etc.
"""

# pyright: reportArgumentType=false

import argparse
import os
from pathlib import Path

import torch.distributed as dist
import yaml

from param_decomp.run import RunConfig
from param_decomp.run_pd import materialize_run
from param_decomp.two_pool import TwoPoolConfig
from param_decomp.two_pool.driver_entry import run_two_pool
from param_decomp.two_pool.run import PhaseProfiler
from param_decomp.utils.distributed_utils import init_distributed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_config", type=str, required=True)
    parser.add_argument("--topology", type=str, required=True)
    parser.add_argument("--wandb_project", type=str, default=None)
    args = parser.parse_args()

    dist_state = init_distributed()
    assert dist_state is not None, "lm_2pool_launcher requires a distributed launch"
    rank = dist_state.rank
    world_size = dist_state.world_size
    device = f"cuda:{dist_state.local_rank}"

    run_cfg = RunConfig.from_file(args.run_config)
    with open(args.topology) as f:
        two_pool_cfg = TwoPoolConfig.model_validate(yaml.safe_load(f))

    if rank == 0:
        n_a = len([r for bg in two_pool_cfg.block_groups for r in bg.ranks])
        n_b = len(two_pool_cfg.pool_b_ranks)
        print(
            f"[2pool-launcher] {n_a}A + {n_b}B = {world_size} GPUs  "
            f"({args.run_config} + {args.topology})",
            flush=True,
        )
        print(f"[2pool-launcher] driver={run_cfg.driver_path}", flush=True)
        print(
            f"[2pool-launcher] batch_size={run_cfg.pd.batch_size} steps={run_cfg.pd.steps}",
            flush=True,
        )

    # No dist_state passed: 2-pool wants the FULL global batch on every rank.
    # Pool A ranks all compute the same target+CI forward against the full
    # batch; pool B ranks slice locally via my_batch_slice_b. Passing dist_state
    # would shard batch_size across world_size (and seed shuffling per-rank),
    # which is wrong here.
    target, train_loader, _eval_loader = materialize_run(run_cfg, device=device)

    profile_mode = os.environ.get("PROFILE_MODE", "off")
    assert profile_mode in ("on", "off"), f"PROFILE_MODE={profile_mode}"
    profile_enabled = profile_mode == "on"
    # Profile a small subset of ranks (rank 0 and the first pool-B rank) — full
    # 64-rank traces are huge and Perfetto chokes on them.
    profiled_ranks = {0, two_pool_cfg.pool_b_ranks[0]}
    profiler = (
        PhaseProfiler(
            enabled=profile_enabled and rank in profiled_ranks,
            out_dir=Path(os.environ.get("PROFILE_OUT_DIR", "/tmp/two_pool_profile")),
            rank=rank,
            pool="a" if rank not in two_pool_cfg.pool_b_ranks else "b",
        )
        if profile_enabled
        else None
    )
    if rank == 0 and profile_enabled:
        print(f"[2pool-launcher] PROFILE_MODE=on (ranks {sorted(profiled_ranks)})", flush=True)

    run_two_pool(
        run_cfg=run_cfg,
        two_pool_cfg=two_pool_cfg,
        target=target,
        train_loader=train_loader,
        device=device,
        wandb_project=args.wandb_project,
        profiler=profiler,
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
