"""CLI for autointerp pipeline.

Usage:
    python -m spd.autointerp.scripts.run_interpret <wandb_path> --config_json '...'
    spd-autointerp <wandb_path>  # SLURM submission
"""

import os
from datetime import datetime
from typing import Any

from dotenv import load_dotenv

from spd.adapters import adapter_from_id
from spd.autointerp.config import AutointerpConfig
from spd.autointerp.interpret import run_interpret
from spd.autointerp.schemas import get_autointerp_dir, get_autointerp_subrun_dir
from spd.harvest.repo import HarvestRepo
from spd.log import logger


def main(
    decomposition_id: str,
    config_json: dict[str, Any],
    harvest_subrun_id: str,
    autointerp_subrun_id: str | None = None,
) -> None:
    assert isinstance(config_json, dict), f"Expected dict from fire, got {type(config_json)}"
    interp_config = AutointerpConfig.model_validate(config_json)

    load_dotenv()
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    assert openrouter_api_key, "OPENROUTER_API_KEY not set"

    harvest = HarvestRepo(decomposition_id, subrun_id=harvest_subrun_id, readonly=False)

    if autointerp_subrun_id is not None:
        subrun_dir = get_autointerp_dir(decomposition_id) / autointerp_subrun_id
        assert subrun_dir.exists(), f"Subrun dir not found: {subrun_dir}"
        logger.info(f"Resuming existing subrun: {autointerp_subrun_id}")
    else:
        autointerp_subrun_id = "a-" + datetime.now().strftime("%Y%m%d_%H%M%S")
        subrun_dir = get_autointerp_subrun_dir(decomposition_id, autointerp_subrun_id)
        subrun_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    interp_config.to_file(subrun_dir / "config.yaml")

    db_path = subrun_dir / "interp.db"

    logger.info(f"Autointerp run: {subrun_dir}")

    adapter = adapter_from_id(decomposition_id)

    run_interpret(
        openrouter_api_key=openrouter_api_key,
        model=interp_config.model,
        reasoning_effort=interp_config.reasoning_effort,
        limit=interp_config.limit,
        cost_limit_usd=interp_config.cost_limit_usd,
        max_requests_per_minute=interp_config.max_requests_per_minute,
        model_metadata=adapter.model_metadata,
        template_strategy=interp_config.template_strategy,
        harvest=harvest,
        db_path=db_path,
        tokenizer_name=adapter.tokenizer_name,
        max_concurrent=interp_config.max_concurrent,
    )


def get_command(
    decomposition_id: str,
    config: AutointerpConfig,
    harvest_subrun_id: str,
    autointerp_subrun_id: str | None = None,
) -> str:
    config_json = config.model_dump_json(exclude_none=True)
    cmd = (
        "python -m spd.autointerp.scripts.run_interpret "
        f"--decomposition_id {decomposition_id} "
        f"--config_json '{config_json}' "
        f"--harvest_subrun_id {harvest_subrun_id} "
    )
    if autointerp_subrun_id is not None:
        cmd += f"--autointerp_subrun_id {autointerp_subrun_id} "
    return cmd


if __name__ == "__main__":
    import fire

    fire.Fire(main)
