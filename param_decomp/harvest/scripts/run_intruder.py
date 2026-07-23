import asyncio
from typing import Any

from dotenv import load_dotenv

from param_decomp.adapters import adapter_from_id
from param_decomp.harvest.config import IntruderEvalConfig
from param_decomp.harvest.db import HarvestDB
from param_decomp.harvest.intruder import run_intruder_scoring
from param_decomp.harvest.repo import HarvestRepo


def main(
    decomposition_id: str,
    config_json: dict[str, Any],
    harvest_subrun_id: str,
) -> None:
    assert isinstance(config_json, dict), f"Expected dict from fire, got {type(config_json)}"
    load_dotenv()

    eval_config = IntruderEvalConfig.model_validate(config_json)

    from param_decomp.autointerp.providers import create_provider

    provider = create_provider(eval_config.llm)

    tokenizer_name = adapter_from_id(decomposition_id).tokenizer_name

    harvest = HarvestRepo(decomposition_id, subrun_id=harvest_subrun_id, readonly=True)
    db = HarvestDB(harvest._dir / "harvest.db")

    asyncio.run(
        run_intruder_scoring(
            db=db,
            provider=provider,
            tokenizer_name=tokenizer_name,
            eval_config=eval_config,
            limit=eval_config.limit,
            cost_limit_usd=eval_config.cost_limit_usd,
        )
    )
    db.close()


def get_command(decomposition_id: str, config: IntruderEvalConfig, harvest_subrun_id: str) -> str:
    config_json = config.model_dump_json(exclude_none=True)
    return (
        f"python -m param_decomp.harvest.scripts.run_intruder {decomposition_id} "
        f"--config_json '{config_json}' "
        f"--harvest_subrun_id {harvest_subrun_id}"
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
