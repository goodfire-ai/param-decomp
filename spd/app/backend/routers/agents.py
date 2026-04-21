"""Agent-friendly API endpoints for direct graph access."""

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from spd.app.backend.dependencies import DepLoadedRun, DepStateManager
from spd.app.backend.routers.graphs import (
    GraphData,
    GraphDataWithOptimization,
    NormalizeType,
    stored_graph_to_response,
)
from spd.app.backend.utils import log_errors

router = APIRouter(prefix="/api/agents", tags=["agents"])


@router.get("/graphs/{graph_id}")
@log_errors
def get_graph_by_id(
    graph_id: int,
    normalize: Annotated[NormalizeType, Query()],
    ci_threshold: Annotated[float, Query(ge=0)],
    loaded: DepLoadedRun,
    manager: DepStateManager,
) -> GraphData | GraphDataWithOptimization:
    """Get a stored graph by its ID."""
    db = manager.db

    result = db.get_graph(graph_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Graph {graph_id} not found")

    graph, prompt_id = result
    prompt = db.get_prompt(prompt_id)
    assert prompt is not None, f"Prompt {prompt_id} not found for graph {graph_id}"

    return stored_graph_to_response(
        graph=graph,
        token_ids=prompt.token_ids,
        tokenizer=loaded.tokenizer,
        normalize=normalize,
        ci_threshold=ci_threshold,
    )
