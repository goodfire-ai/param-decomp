"""Merge script for dataset attribution rank files.

Combines per-rank attribution files into a single merged result.

Usage:
    python -m spd.dataset_attributions.scripts.run_merge --wandb_path <path> --subrun_id da-xxx
"""

from spd.dataset_attributions.harvest import merge_attributions
from spd.dataset_attributions.repo import get_attributions_subrun_dir
from spd.log import logger
from spd.utils.wandb_utils import parse_wandb_run_path


def main(
    *,
    wandb_path: str,
    subrun_id: str,
) -> None:
    _, _, run_id = parse_wandb_run_path(wandb_path)
    output_dir = get_attributions_subrun_dir(run_id, subrun_id)
    logger.info(f"Merging attribution results for {wandb_path} (subrun {subrun_id})")
    merge_attributions(output_dir)


def get_command(wandb_path: str, subrun_id: str) -> str:
    return (
        f"python -m spd.dataset_attributions.scripts.run_merge "
        f'--wandb_path "{wandb_path}" '
        f"--subrun_id {subrun_id}"
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
