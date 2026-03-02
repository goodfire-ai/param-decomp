import asyncio
import os
from typing import Any

from dotenv import load_dotenv

from spd.adapters import adapter_from_id
from spd.harvest.config import IntruderEvalConfig
from spd.harvest.db import HarvestDB
from spd.harvest.intruder import run_intruder_scoring
from spd.harvest.repo import HarvestRepo
from spd.log import logger


def main(
    decomposition_id: str,
    config_json: dict[str, Any],
    harvest_subrun_id: str,
) -> None:
    assert isinstance(config_json, dict), f"Expected dict from fire, got {type(config_json)}"
    load_dotenv()
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    assert openrouter_api_key, "OPENROUTER_API_KEY not set"

    eval_config = IntruderEvalConfig.model_validate(config_json)

    tokenizer_name = adapter_from_id(decomposition_id).tokenizer_name

    harvest = HarvestRepo(decomposition_id, subrun_id=harvest_subrun_id, readonly=True)
    score_db = HarvestDB(harvest._dir / "harvest.db")

    logger.info("Loading components from harvest DB...")
    components = harvest.get_all_components()
    logger.info(f"Loaded {len(components)} components")

    asyncio.run(
        run_intruder_scoring(
            components=components,
            model=eval_config.model,
            openrouter_api_key=openrouter_api_key,
            tokenizer_name=tokenizer_name,
            score_db=score_db,
            eval_config=eval_config,
            limit=eval_config.limit,
            cost_limit_usd=eval_config.cost_limit_usd,
        )
    )
    score_db.close()


def get_command(decomposition_id: str, config: IntruderEvalConfig, harvest_subrun_id: str) -> str:
    config_json = config.model_dump_json(exclude_none=True)
    return (
        f"python -m spd.harvest.scripts.run_intruder {decomposition_id} "
        f"--config_json '{config_json}' "
        f"--harvest_subrun_id {harvest_subrun_id}"
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
