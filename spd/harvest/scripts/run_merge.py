"""Harvest merge: combines worker states into final harvest results.

Usage:
    python -m spd.harvest.scripts.run_merge --subrun_id h-20260211_120000 --config_json '...'
"""

from typing import Any

import fire

from spd.harvest.config import HarvestConfig
from spd.harvest.harvest import merge_harvest
from spd.harvest.schemas import get_harvest_subrun_dir
from spd.log import logger


def main(subrun_id: str, config_json: dict[str, Any]) -> None:
    assert isinstance(config_json, dict), f"Expected dict from fire, got {type(config_json)}"
    config = HarvestConfig.model_validate(config_json)
    output_dir = get_harvest_subrun_dir(config.method_config.id, subrun_id)
    logger.info(f"Merging harvest results for (subrun {subrun_id})")
    merge_harvest(output_dir, config)


def get_command(subrun_id: str, config: HarvestConfig) -> str:
    config_json = config.model_dump_json(exclude_none=True)
    cmd = (
        f"python -m spd.harvest.scripts.run_merge "
        f"--subrun_id {subrun_id} "
        f"--config_json '{config_json}'"
    )
    return cmd


if __name__ == "__main__":
    fire.Fire(main)
