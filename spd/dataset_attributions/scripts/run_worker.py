"""Worker script for dataset attribution computation.

Called by SLURM jobs submitted via spd-attributions, or run directly for non-SLURM environments.

Usage:
    # Single GPU
    python -m spd.dataset_attributions.scripts.run_worker <path>

    # Single GPU with config
    python -m spd.dataset_attributions.scripts.run_worker <path> --config_json '{"n_batches": 500}'

    # Multi-GPU (run in parallel)
    python -m spd.dataset_attributions.scripts.run_worker <path> --rank 0 --world_size 4 --subrun_id da-xxx
"""

from datetime import datetime
from typing import Any

from spd.dataset_attributions.config import DatasetAttributionConfig
from spd.dataset_attributions.harvest import harvest_attributions
from spd.dataset_attributions.repo import get_attributions_subrun_dir
from spd.utils.wandb_utils import parse_wandb_run_path


def main(
    wandb_path: str,
    config_json: dict[str, Any],
    rank: int,
    world_size: int,
    subrun_id: str | None = None,
    harvest_subrun_id: str | None = None,
) -> None:
    _, _, run_id = parse_wandb_run_path(wandb_path)

    if subrun_id is None:
        subrun_id = "da-" + datetime.now().strftime("%Y%m%d_%H%M%S")

    base = config_json or {}
    base.setdefault("spd_run_wandb_path", wandb_path)
    if harvest_subrun_id is not None:
        base.setdefault("harvest_subrun_id", harvest_subrun_id)
    config = DatasetAttributionConfig.model_validate(base)
    output_dir = get_attributions_subrun_dir(run_id, subrun_id)

    harvest_attributions(
        config=config,
        output_dir=output_dir,
        rank=rank,
        world_size=world_size,
    )


def get_command(
    wandb_path: str,
    config_json: str,
    rank: int,
    world_size: int,
    subrun_id: str,
    harvest_subrun_id: str | None = None,
) -> str:
    cmd = (
        f"python -m spd.dataset_attributions.scripts.run_worker "
        f'"{wandb_path}" '
        f"--config_json '{config_json}' "
        f"--rank {rank} "
        f"--world_size {world_size} "
        f"--subrun_id {subrun_id}"
    )
    if harvest_subrun_id is not None:
        cmd += f" --harvest_subrun_id {harvest_subrun_id}"
    return cmd


if __name__ == "__main__":
    import fire

    fire.Fire(main)
