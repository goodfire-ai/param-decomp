"""Data sources provenance endpoint.

Shows where harvest/autointerp data came from: subrun IDs, configs, counts.
"""

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from param_decomp_lab.app.backend.dependencies import DepLoadedRun
from param_decomp_lab.app.backend.utils import log_errors


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


class DataSourcesResponse(BaseModel):
    harvest: HarvestInfo | None
    autointerp: AutointerpInfo | None


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

    return DataSourcesResponse(
        harvest=harvest_info,
        autointerp=autointerp_info,
    )
