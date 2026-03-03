"""Harvest worker: collects component statistics on a single GPU.

Usage:
    python -m spd.harvest.scripts.run_worker --config_json '{"n_batches": 100}'
    python -m spd.harvest.scripts.run_worker --config_json '...' --rank 0 --world_size 4 --subrun_id h-20260211_120000
"""

from datetime import datetime
from typing import Any

import fire
import torch

from spd.adapters import adapter_from_config
from spd.harvest.config import HarvestConfig
from spd.harvest.harvest import harvest
from spd.harvest.harvest_fn import make_harvest_fn
from spd.harvest.schemas import get_harvest_subrun_dir
from spd.log import logger
from spd.utils.distributed_utils import get_device


def main(
    config_json: dict[str, Any],
    rank: int | None = None,
    world_size: int | None = None,
    subrun_id: str | None = None,
) -> None:
    assert isinstance(config_json, dict), f"Expected dict from fire, got {type(config_json)}"
    assert (rank is not None) == (world_size is not None)

    if subrun_id is None:
        subrun_id = "h-" + datetime.now().strftime("%Y%m%d_%H%M%S")
    device = torch.device(get_device())

    config = HarvestConfig.model_validate(config_json)

    adapter = adapter_from_config(config.method_config)

    output_dir = get_harvest_subrun_dir(adapter.decomposition_id, subrun_id)

    if rank is not None:
        logger.info(f"Distributed harvest: rank {rank}/{world_size}, subrun {subrun_id}")
    else:
        logger.info(f"Single-GPU harvest: subrun {subrun_id}")

    harvest(
        layers=adapter.layer_activation_sizes,
        vocab_size=adapter.vocab_size,
        dataloader=adapter.dataloader(config.batch_size),
        harvest_fn=make_harvest_fn(device, config.method_config, adapter),
        config=config,
        output_dir=output_dir,
        rank_world_size=(rank, world_size) if rank is not None and world_size is not None else None,
        device=device,
    )


def get_command(config: HarvestConfig, rank: int, world_size: int, subrun_id: str) -> str:
    config_json = config.model_dump_json(exclude_none=True)
    cmd = (
        f"python -m spd.harvest.scripts.run_worker "
        f"--config_json '{config_json}' "
        f"--rank {rank} "
        f"--world_size {world_size} "
        f"--subrun_id {subrun_id}"
    )
    return cmd


if __name__ == "__main__":
    fire.Fire(main)
