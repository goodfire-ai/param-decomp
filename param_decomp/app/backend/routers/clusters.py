"""Cluster mapping endpoints."""

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError, model_validator

from param_decomp.app.backend.state import StateManager
from param_decomp.app.backend.utils import log_errors
from param_decomp.base_config import BaseConfig
from param_decomp.settings import PARAM_DECOMP_OUT_DIR
from param_decomp.topology import TransformerTopology

router = APIRouter(prefix="/api/clusters", tags=["clusters"])


class ClusterMapping(BaseConfig):
    """Cluster mapping from component keys (layer:component_idx) to cluster IDs.

    Singleton clusters (components not grouped with others) have null values.
    """

    mapping: dict[str, int | None]


class ClusterMappingFile(BaseConfig):
    """Schema for the on-disk cluster mapping JSON file."""

    clustering_run_id: str
    notes: str
    pd_run: str
    iteration: int
    clusters: dict[str, int | None]

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_keys(cls, data: dict[str, Any]) -> dict[str, Any]:
        if "spd_run" in data:
            data["pd_run"] = data.pop("spd_run")
        return data


@router.post("/load")
@log_errors
def load_cluster_mapping(file_path: str) -> ClusterMapping:
    """Load a cluster mapping JSON file from the given path.

    Paths are resolved relative to PARAM_DECOMP_OUT_DIR unless they are absolute.

    The file should contain a JSON object with:
    - clustering_run_id: string
    - notes: string
    - pd_run: wandb path (must match currently loaded run)
    - clusters: dict mapping component keys to cluster IDs
    """
    state = StateManager.get()
    run_state = state.run_state
    if run_state is None:
        raise HTTPException(status_code=400, detail="No run loaded. Load a run first.")

    path = Path(file_path)
    if not path.is_absolute():
        path = PARAM_DECOMP_OUT_DIR / path
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    if not path.is_file():
        raise HTTPException(status_code=400, detail=f"Not a file: {file_path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid JSON in cluster mapping file: {file_path} ({exc})",
        ) from exc

    try:
        parsed = ClusterMappingFile.model_validate(data)
    except ValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Invalid cluster mapping file schema",
                "errors": exc.errors(),
            },
        ) from exc

    if parsed.pd_run != run_state.run.wandb_path:
        raise HTTPException(
            status_code=409,
            detail=f"Run ID mismatch: cluster file is for '{parsed.pd_run}', "
            f"but loaded run is '{run_state.run.wandb_path}'",
        )

    canonical_clusters = _to_canonical_keys(parsed.clusters, run_state.topology)
    return ClusterMapping(mapping=canonical_clusters)


def _to_canonical_keys(
    clusters: dict[str, int | None], topology: TransformerTopology
) -> dict[str, int | None]:
    """Convert concrete component keys (e.g. 'h.3.mlp.down_proj:5') to canonical (e.g. '3.mlp.down:5')."""
    result: dict[str, int | None] = {}
    for key, cluster_id in clusters.items():
        layer, idx = key.rsplit(":", 1)
        canonical_layer = topology.target_to_canon(layer)
        result[f"{canonical_layer}:{idx}"] = cluster_id
    return result
