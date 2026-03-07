"""Data sources provenance endpoint.

Shows where harvest/autointerp/attribution data came from: subrun IDs, configs, counts.
"""

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from spd.app.backend.dependencies import DepLoadedRun
from spd.app.backend.utils import log_errors


class HarvestInfo(BaseModel):
    subrun_id: str
    config: dict[str, Any]
    n_components: int
    has_intruder_scores: bool


class AutointerpInfo(BaseModel):
    subrun_id: str
    config: dict[str, Any]
    n_interpretations: int
    eval_scores: list[str]


class AttributionsInfo(BaseModel):
    subrun_id: str
    n_tokens_processed: int
    ci_threshold: float


class GraphInterpInfo(BaseModel):
    subrun_id: str
    config: dict[str, Any] | None
    label_counts: dict[str, int]


class DataSourcesResponse(BaseModel):
    harvest: HarvestInfo | None
    autointerp: AutointerpInfo | None
    attributions: AttributionsInfo | None
    graph_interp: GraphInterpInfo | None


router = APIRouter(prefix="/api/data_sources", tags=["data_sources"])


@router.get("")
@log_errors
def get_data_sources(loaded: DepLoadedRun) -> DataSourcesResponse:
    harvest_info: HarvestInfo | None = None
    if loaded.harvest is not None:
        harvest_info = HarvestInfo(
            subrun_id=loaded.harvest.subrun_id,
            config=loaded.harvest.get_config(),
            n_components=loaded.harvest.get_component_count(),
            has_intruder_scores=bool(loaded.harvest.get_scores("intruder")),
        )

    autointerp_info: AutointerpInfo | None = None
    if loaded.interp is not None:
        config = loaded.interp.get_config()
        if config is not None:
            autointerp_info = AutointerpInfo(
                subrun_id=loaded.interp.subrun_id,
                config=config,
                n_interpretations=loaded.interp.get_interpretation_count(),
                eval_scores=loaded.interp.get_available_score_types(),
            )

    attributions_info: AttributionsInfo | None = None
    if loaded.attributions is not None:
        storage = loaded.attributions.get_attributions()
        attributions_info = AttributionsInfo(
            subrun_id=loaded.attributions.subrun_id,
            n_tokens_processed=storage.n_tokens_processed,
            ci_threshold=storage.ci_threshold,
        )

    graph_interp_info: GraphInterpInfo | None = None
    if loaded.graph_interp is not None:
        graph_interp_info = GraphInterpInfo(
            subrun_id=loaded.graph_interp.subrun_id,
            config=loaded.graph_interp.get_config(),
            label_counts=loaded.graph_interp.get_label_counts(),
        )

    return DataSourcesResponse(
        harvest=harvest_info,
        autointerp=autointerp_info,
        attributions=attributions_info,
        graph_interp=graph_interp_info,
    )
