"""Autointerp comparison endpoints.

Lists all completed autointerp subruns for a decomposition and serves
interpretations from each, enabling side-by-side comparison of different
strategies/models.
"""

from datetime import datetime
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from spd.app.backend.dependencies import DepLoadedRun
from spd.app.backend.utils import log_errors
from spd.autointerp.db import InterpDB
from spd.autointerp.schemas import get_autointerp_dir
from spd.topology import TransformerTopology

router = APIRouter(prefix="/api/autointerp_compare", tags=["autointerp_compare"])


class SubrunSummary(BaseModel):
    subrun_id: str
    strategy: str
    llm_model: str
    timestamp: str
    n_completed: int
    mean_detection_score: float | None
    mean_fuzzing_score: float | None
    note: str | None
    harvest_subrun_id: str | None
    harvest_mismatch: bool


class InterpretationHeadline(BaseModel):
    label: str
    detection_score: float | None = None
    fuzzing_score: float | None = None


class InterpretationDetail(BaseModel):
    reasoning: str
    prompt: str


def _concrete_to_canonical_key(concrete_key: str, topology: TransformerTopology) -> str:
    layer, idx = concrete_key.rsplit(":", 1)
    canonical = topology.target_to_canon(layer)
    return f"{canonical}:{idx}"


def _canonical_to_concrete_key(
    canonical_layer: str, component_idx: int, topology: TransformerTopology
) -> str:
    concrete = topology.canon_to_target(canonical_layer)
    return f"{concrete}:{component_idx}"


def _parse_subrun_timestamp(subrun_id: str) -> str:
    """Parse 'a-YYYYMMDD_HHMMSS' into a readable timestamp."""
    raw = subrun_id.removeprefix("a-")
    try:
        dt = datetime.strptime(raw, "%Y%m%d_%H%M%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return raw


def _parse_config(config_path: Path) -> tuple[str, str]:
    """Extract strategy type and LLM model from a subrun's config.yaml."""
    if not config_path.exists():
        return "unknown", "unknown"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    strategy = cfg.get("template_strategy", {}).get("type", "unknown")
    llm = cfg.get("llm", {})
    llm_model = llm.get("model", "unknown")
    return strategy, llm_model


def _mean_score(db: InterpDB, score_type: str) -> float | None:
    scores = db.get_scores(score_type)
    if not scores:
        return None
    return sum(scores.values()) / len(scores)


def _get_run_id(loaded: DepLoadedRun) -> str:
    return loaded.run.wandb_path.split("/")[-1]


@router.get("/subruns")
@log_errors
def list_subruns(loaded: DepLoadedRun) -> list[SubrunSummary]:
    """List all completed autointerp subruns with metadata."""
    run_id = _get_run_id(loaded)
    autointerp_dir = get_autointerp_dir(run_id)
    if not autointerp_dir.exists():
        return []

    active_harvest_id = loaded.harvest.subrun_id if loaded.harvest else None

    results: list[SubrunSummary] = []
    for d in sorted(autointerp_dir.iterdir(), key=lambda d: d.name):
        if not d.is_dir() or not d.name.startswith("a-"):
            continue
        db_path = d / "interp.db"
        if not db_path.exists():
            continue

        strategy, llm_model = _parse_config(d / "config.yaml")
        db = InterpDB(db_path, readonly=True)
        try:
            if not db.has_interpretations_table():
                continue
            n_completed = db.get_interpretation_count()
            if n_completed == 0:
                continue
            mean_detection = _mean_score(db, "detection")
            mean_fuzzing = _mean_score(db, "fuzzing")
            harvest_subrun_id_raw = db.get_config_value("harvest_subrun_id")
        finally:
            db.close()

        notes_path = d / "notes.txt"
        note = notes_path.read_text().strip() if notes_path.exists() else None

        harvest_subrun_id = str(harvest_subrun_id_raw) if harvest_subrun_id_raw else None
        harvest_mismatch = (
            harvest_subrun_id is not None
            and active_harvest_id is not None
            and harvest_subrun_id != active_harvest_id
        )

        results.append(
            SubrunSummary(
                subrun_id=d.name,
                strategy=strategy,
                llm_model=llm_model,
                timestamp=_parse_subrun_timestamp(d.name),
                n_completed=n_completed,
                mean_detection_score=mean_detection,
                mean_fuzzing_score=mean_fuzzing,
                note=note,
                harvest_subrun_id=harvest_subrun_id,
                harvest_mismatch=harvest_mismatch,
            )
        )

    return results


@router.get("/subruns/{subrun_id}/interpretations")
@log_errors
def get_subrun_interpretations(
    subrun_id: str,
    loaded: DepLoadedRun,
) -> dict[str, InterpretationHeadline]:
    """Get all interpretation headlines for a subrun (canonical keys)."""
    run_id = _get_run_id(loaded)
    subrun_dir = get_autointerp_dir(run_id) / subrun_id
    db_path = subrun_dir / "interp.db"
    assert db_path.exists(), f"No interp.db at {subrun_dir}"

    db = InterpDB(db_path, readonly=True)
    try:
        interpretations = db.get_all_interpretations()
        detection_scores = db.get_scores("detection")
        fuzzing_scores = db.get_scores("fuzzing")
    finally:
        db.close()

    return {
        _concrete_to_canonical_key(key, loaded.topology): InterpretationHeadline(
            label=result.label,
            detection_score=detection_scores.get(key),
            fuzzing_score=fuzzing_scores.get(key),
        )
        for key, result in interpretations.items()
    }


@router.get("/subruns/{subrun_id}/interpretations/{layer}/{c_idx}")
@log_errors
def get_subrun_interpretation_detail(
    subrun_id: str,
    layer: str,
    c_idx: int,
    loaded: DepLoadedRun,
) -> InterpretationDetail:
    """Get full interpretation detail (reasoning + prompt) for one component in a subrun."""
    run_id = _get_run_id(loaded)
    subrun_dir = get_autointerp_dir(run_id) / subrun_id
    db_path = subrun_dir / "interp.db"
    assert db_path.exists(), f"No interp.db at {subrun_dir}"

    concrete_key = _canonical_to_concrete_key(layer, c_idx, loaded.topology)

    db = InterpDB(db_path, readonly=True)
    try:
        result = db.get_interpretation(concrete_key)
    finally:
        db.close()

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"No interpretation for {layer}:{c_idx} in subrun {subrun_id}",
        )

    return InterpretationDetail(reasoning=result.reasoning, prompt=result.prompt)
