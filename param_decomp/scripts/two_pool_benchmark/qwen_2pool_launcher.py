"""2-pool launcher using the LM driver for model + dataloader materialization.

Replaces the bespoke ``qwen_2pool.py`` benchmark for the real-experiment path:
  - Model / tokenizer / pile loader / module_info come from a regular RunConfig
    YAML (`qwen3_0p6b.yaml`), loaded via the LM driver — same machinery as a
    standard `pd-run` invocation.
  - Topology + 2-pool loss coefficients + PPGD config come from a separate
    `TwoPoolConfig` YAML (`qwen3_0p6b_two_pool_topology.yaml`).
  - ``run_two_pool`` glues them together and runs ``optimize_two_pool``.

Run with the same sbatch as the bespoke benchmark — just point torchrun at
this module instead.
"""

# pyright: reportArgumentType=false

import os
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

from param_decomp.run import RunConfig
from param_decomp.run_pd import materialize_run
from param_decomp.two_pool import TwoPoolConfig
from param_decomp.two_pool.driver_entry import run_two_pool
from param_decomp.utils.distributed_utils import DistributedState

REPO = Path(__file__).resolve().parents[3]
RUN_CONFIG = REPO / "param_decomp/experiments/lm/qwen3_0p6b.yaml"
TWO_POOL_CONFIG = REPO / "param_decomp/experiments/lm/qwen3_0p6b_two_pool_topology.yaml"


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"

    run_cfg = RunConfig.from_file(RUN_CONFIG)
    with open(TWO_POOL_CONFIG) as f:
        two_pool_cfg = TwoPoolConfig.model_validate(yaml.safe_load(f))

    if rank == 0:
        print(
            f"[qwen-launcher] {len([r for bg in two_pool_cfg.block_groups for r in bg.ranks])}A "
            f"+ {len(two_pool_cfg.pool_b_ranks)}B = {world_size} GPUs",
            flush=True,
        )
        print(f"[qwen-launcher] driver={run_cfg.driver_path}", flush=True)
        print(
            f"[qwen-launcher] batch_global={two_pool_cfg.batch_global} steps={run_cfg.pd.steps}",
            flush=True,
        )

    dist_state = DistributedState(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        backend="nccl",
    )
    target, train_loader, _eval_loader = materialize_run(
        run_cfg, device=device, dist_state=dist_state
    )

    run_two_pool(
        run_cfg=run_cfg,
        two_pool_cfg=two_pool_cfg,
        target=target,
        train_loader=train_loader,
        device=device,
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
