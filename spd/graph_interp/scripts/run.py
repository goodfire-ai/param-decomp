"""CLI entry point for graph interpretation.

Called by SLURM or directly:
    python -m spd.graph_interp.scripts.run <decomposition_id> --config_json '{...}'
"""

import os
from datetime import datetime
from typing import Any

from dotenv import load_dotenv

from spd.adapters import adapter_from_id
from spd.adapters.spd import SPDAdapter
from spd.dataset_attributions.repo import AttributionRepo
from spd.graph_interp.config import GraphInterpConfig
from spd.graph_interp.interpret import run_graph_interp
from spd.graph_interp.schemas import get_graph_interp_subrun_dir
from spd.harvest.repo import HarvestRepo
from spd.log import logger


def main(
    decomposition_id: str,
    config_json: dict[str, Any],
    harvest_subrun_id: str | None = None,
    subrun_id: str | None = None,
) -> None:
    assert isinstance(config_json, dict), f"Expected dict from fire, got {type(config_json)}"
    config = GraphInterpConfig.model_validate(config_json)

    load_dotenv()
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    assert openrouter_api_key, "OPENROUTER_API_KEY not set"

    if subrun_id is None:
        subrun_id = "ti-" + datetime.now().strftime("%Y%m%d_%H%M%S")
    subrun_dir = get_graph_interp_subrun_dir(decomposition_id, subrun_id)
    subrun_dir.mkdir(parents=True, exist_ok=True)
    config.to_file(subrun_dir / "config.yaml")
    db_path = subrun_dir / "interp.db"
    logger.info(f"Graph interp run: {subrun_dir}")

    logger.info("Loading adapter and model metadata...")
    adapter = adapter_from_id(decomposition_id)
    assert isinstance(adapter, SPDAdapter)
    logger.info("Loading harvest data...")
    if harvest_subrun_id is not None:
        harvest = HarvestRepo(decomposition_id, subrun_id=harvest_subrun_id, readonly=True)
    else:
        harvest = HarvestRepo.open_most_recent(decomposition_id, readonly=True)
        assert harvest is not None, f"No harvest data for {decomposition_id}"

    logger.info("Loading dataset attributions...")
    attributions = AttributionRepo.open(decomposition_id)
    assert attributions is not None, f"Dataset attributions required for {decomposition_id}"
    attribution_storage = attributions.get_attributions()
    logger.info(
        f"  {attribution_storage.n_components} components, {attribution_storage.n_tokens_processed:,} tokens"
    )

    logger.info("Loading component correlations...")
    correlations = harvest.get_correlations()
    assert correlations is not None, f"Component correlations required for {decomposition_id}"

    logger.info("Loading token stats...")
    token_stats = harvest.get_token_stats()
    assert token_stats is not None, f"Token stats required for {decomposition_id}"

    logger.info("Data loading complete")

    run_graph_interp(
        openrouter_api_key=openrouter_api_key,
        config=config,
        harvest=harvest,
        attribution_storage=attribution_storage,
        correlation_storage=correlations,
        token_stats=token_stats,
        model_metadata=adapter.model_metadata,
        db_path=db_path,
        tokenizer_name=adapter.tokenizer_name,
    )


def get_command(
    decomposition_id: str,
    config: GraphInterpConfig,
    harvest_subrun_id: str | None = None,
) -> str:
    config_json = config.model_dump_json(exclude_none=True)
    cmd = (
        "python -m spd.graph_interp.scripts.run "
        f"--decomposition_id {decomposition_id} "
        f"--config_json '{config_json}' "
    )
    if harvest_subrun_id is not None:
        cmd += f"--harvest_subrun_id {harvest_subrun_id} "
    return cmd


if __name__ == "__main__":
    import fire

    fire.Fire(main)
