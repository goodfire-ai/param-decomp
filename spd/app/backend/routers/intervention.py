"""Intervention forward pass endpoint."""

import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from spd.app.backend.compute import (
    InterventionResult,
    compute_intervention,
)
from spd.app.backend.dependencies import DepDB, DepLoadedRun, DepStateManager
from spd.app.backend.optim_cis import AdvPGDConfig, LossConfig, MeanKLLossConfig
from spd.app.backend.utils import log_errors
from spd.topology import TransformerTopology
from spd.utils.distributed_utils import get_device

# =============================================================================
# Schemas
# =============================================================================


class AdvPgdParams(BaseModel):
    n_steps: int
    step_size: float


class RunInterventionRequest(BaseModel):
    """Request to run and save an intervention."""

    graph_id: int
    selected_nodes: list[str]  # node keys (layer:seq:cIdx)
    nodes_to_ablate: list[str] | None = None  # node keys to ablate in ablated (omit to skip)
    top_k: int
    adv_pgd: AdvPgdParams


class InterventionRunSummary(BaseModel):
    """Summary of a saved intervention run."""

    id: int
    selected_nodes: list[str]
    result: InterventionResult
    created_at: str


router = APIRouter(prefix="/api/intervention", tags=["intervention"])

DEVICE = get_device()


def _parse_node_key(key: str, topology: TransformerTopology) -> tuple[str, int, int]:
    """Parse canonical node key into (concrete_path, seq_pos, component_idx).

    Translates canonical layer address back to concrete module path for ComponentModel.
    """
    parts = key.split(":")
    assert len(parts) == 3, f"Invalid node key format: {key!r} (expected 'layer:seq:cIdx')"
    canonical_layer, seq_str, cidx_str = parts
    assert canonical_layer not in ("embed", "output"), (
        f"Cannot intervene on {canonical_layer!r} nodes - only internal layers are interventable"
    )
    concrete_path = topology.canon_to_target(canonical_layer)
    return concrete_path, int(seq_str), int(cidx_str)


def _parse_and_validate_active_nodes(
    selected_nodes: list[str], topology: TransformerTopology, seq_len: int
) -> list[tuple[str, int, int]]:
    """Parse node keys and validate sequence bounds for the current prompt."""
    active_nodes = [_parse_node_key(key, topology) for key in selected_nodes]
    for _, seq_pos, _ in active_nodes:
        if seq_pos >= seq_len:
            raise ValueError(f"seq_pos {seq_pos} out of bounds for text with {seq_len} tokens")
    return active_nodes


@router.post("/run")
@log_errors
def run_and_save_intervention(
    request: RunInterventionRequest,
    loaded: DepLoadedRun,
    db: DepDB,
    manager: DepStateManager,
) -> InterventionRunSummary:
    """Run an intervention and save the result."""
    with manager.gpu_lock():
        graph_record = db.get_graph(request.graph_id)
        if graph_record is None:
            raise HTTPException(status_code=404, detail="Graph not found")
        graph, prompt_id = graph_record

        prompt = db.get_prompt(prompt_id)
        if prompt is None:
            raise HTTPException(status_code=404, detail="Prompt not found")

        token_ids = prompt.token_ids
        active_nodes = _parse_and_validate_active_nodes(
            request.selected_nodes, loaded.topology, len(token_ids)
        )
        nodes_to_ablate = (
            _parse_and_validate_active_nodes(
                request.nodes_to_ablate, loaded.topology, len(token_ids)
            )
            if request.nodes_to_ablate is not None
            else None
        )
        tokens = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)

        # Use graph's loss config if optimized, else mean KL
        loss_config: LossConfig = (
            graph.optimization_params.loss
            if graph.optimization_params is not None
            else MeanKLLossConfig()
        )

        result = compute_intervention(
            model=loaded.model,
            tokens=tokens,
            active_nodes=active_nodes,
            nodes_to_ablate=nodes_to_ablate,
            tokenizer=loaded.tokenizer,
            adv_pgd_config=AdvPGDConfig(
                n_steps=request.adv_pgd.n_steps,
                step_size=request.adv_pgd.step_size,
                init="random",
            ),
            loss_config=loss_config,
            sampling=loaded.config.sampling,
            top_k=request.top_k,
        )

    run_id = db.save_intervention_run(
        graph_id=request.graph_id,
        selected_nodes=request.selected_nodes,
        result_json=result.model_dump_json(),
    )

    record = db.get_intervention_runs(request.graph_id)
    saved_run = next((r for r in record if r.id == run_id), None)
    assert saved_run is not None

    return InterventionRunSummary(
        id=run_id,
        selected_nodes=request.selected_nodes,
        result=result,
        created_at=saved_run.created_at,
    )


@router.get("/runs/{graph_id}")
@log_errors
def get_intervention_runs(graph_id: int, db: DepDB) -> list[InterventionRunSummary]:
    """Get all intervention runs for a graph."""
    records = db.get_intervention_runs(graph_id)
    return [
        InterventionRunSummary(
            id=r.id,
            selected_nodes=r.selected_nodes,
            result=InterventionResult.model_validate_json(r.result_json),
            created_at=r.created_at,
        )
        for r in records
    ]


@router.delete("/runs/{run_id}")
@log_errors
def delete_intervention_run(run_id: int, db: DepDB) -> dict[str, bool]:
    """Delete an intervention run."""
    db.delete_intervention_run(run_id)
    return {"success": True}
