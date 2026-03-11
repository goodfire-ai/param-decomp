"""Worker script for dataset attribution computation.

Called by SLURM jobs submitted via spd-attributions.

Usage:
    python -m spd.dataset_attributions.scripts.run_worker <wandb_path> \
        --config_json '{"n_batches": 500}' \
        --rank 0 --world_size 4 --subrun_id da-xxx
"""

from typing import Any

from spd.dataset_attributions.config import DatasetAttributionConfig
from spd.dataset_attributions.harvest import harvest_attributions
from spd.dataset_attributions.repo import get_attributions_subrun_dir
from spd.utils.wandb_utils import parse_wandb_run_path


def main(
    wandb_path: str,
    config_json: dict[str, Any],
    harvest_subrun_id: str,
    rank: int,
    world_size: int,
    subrun_id: str,
) -> None:
    # Fire parses JSON strings into dicts automatically
    assert isinstance(config_json, dict), f"Expected dict from Fire, got {type(config_json)}"
    _, _, run_id = parse_wandb_run_path(wandb_path)

    config = DatasetAttributionConfig.model_validate(config_json)
    assert config.spd_run_wandb_path == wandb_path
    output_dir = get_attributions_subrun_dir(run_id, subrun_id)

    harvest_attributions(
        config=config,
        output_dir=output_dir,
        harvest_subrun_id=harvest_subrun_id,
        rank=rank,
        world_size=world_size,
    )


def get_command(
    wandb_path: str,
    config_json: str,
    harvest_subrun_id: str,
    rank: int,
    world_size: int,
    subrun_id: str,
) -> str:
    return (
        f"python -m spd.dataset_attributions.scripts.run_worker "
        f'"{wandb_path}" '
        f"--config_json '{config_json}' "
        f"--harvest_subrun_id {harvest_subrun_id} "
        f"--rank {rank} "
        f"--world_size {world_size} "
        f"--subrun_id {subrun_id}"
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
