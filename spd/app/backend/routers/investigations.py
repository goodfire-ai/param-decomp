"""Investigations endpoint for viewing agent investigation results.

Lists and serves investigation data from SPD_OUT_DIR/investigations/.
Each investigation directory contains findings from a single agent run.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from spd.app.backend.dependencies import DepLoadedRun
from spd.settings import SPD_OUT_DIR
from spd.utils.wandb_utils import parse_wandb_run_path

router = APIRouter(prefix="/api/investigations", tags=["investigations"])

INVESTIGATIONS_DIR = SPD_OUT_DIR / "investigations"


class InvestigationSummary(BaseModel):
    """Summary of a single investigation."""

    id: str
    wandb_path: str | None
    prompt: str | None
    created_at: str
    has_research_log: bool
    has_explanations: bool
    event_count: int
    last_event_time: str | None
    last_event_message: str | None
    title: str | None
    summary: str | None
    status: str | None


class EventEntry(BaseModel):
    """A single event from events.jsonl."""

    event_type: str
    timestamp: str
    message: str
    details: dict[str, Any] | None = None


class InvestigationDetail(BaseModel):
    """Full detail of an investigation including logs."""

    id: str
    wandb_path: str | None
    prompt: str | None
    created_at: str
    research_log: str | None
    events: list[EventEntry]
    explanations: list[dict[str, Any]]
    artifact_ids: list[str]
    title: str | None
    summary: str | None
    status: str | None


def _parse_metadata(inv_path: Path) -> dict[str, Any] | None:
    """Parse metadata.json from an investigation directory."""
    metadata_path = inv_path / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        data: dict[str, Any] = json.loads(metadata_path.read_text())
        return data
    except Exception:
        return None


def _get_last_event(events_path: Path) -> tuple[str | None, str | None, int]:
    """Get the last event timestamp, message, and total count from events.jsonl."""
    if not events_path.exists():
        return None, None, 0

    last_time = None
    last_msg = None
    count = 0

    try:
        with open(events_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                count += 1
                try:
                    event = json.loads(line)
                    last_time = event.get("timestamp")
                    last_msg = event.get("message")
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass

    return last_time, last_msg, count


def _parse_task_summary(inv_path: Path) -> tuple[str | None, str | None, str | None]:
    """Parse summary.json from an investigation directory. Returns (title, summary, status)."""
    summary_path = inv_path / "summary.json"
    if not summary_path.exists():
        return None, None, None
    try:
        data: dict[str, Any] = json.loads(summary_path.read_text())
        return data.get("title"), data.get("summary"), data.get("status")
    except Exception:
        return None, None, None


def _list_artifact_ids(inv_path: Path) -> list[str]:
    """List all artifact IDs for an investigation."""
    artifacts_dir = inv_path / "artifacts"
    if not artifacts_dir.exists():
        return []
    return [f.stem for f in sorted(artifacts_dir.glob("graph_*.json"))]


def _get_created_at(inv_path: Path, metadata: dict[str, Any] | None) -> str:
    """Get creation time for an investigation."""
    events_path = inv_path / "events.jsonl"
    if events_path.exists():
        try:
            with open(events_path) as f:
                first_line = f.readline().strip()
                if first_line:
                    event = json.loads(first_line)
                    if "timestamp" in event:
                        return event["timestamp"]
        except Exception:
            pass

    if metadata and "created_at" in metadata:
        return metadata["created_at"]

    return datetime.fromtimestamp(inv_path.stat().st_mtime).isoformat()


@router.get("")
def list_investigations(loaded: DepLoadedRun) -> list[InvestigationSummary]:
    """List investigations for the currently loaded run."""
    if not INVESTIGATIONS_DIR.exists():
        return []

    wandb_path = loaded.run.wandb_path
    results = []

    for inv_path in INVESTIGATIONS_DIR.iterdir():
        if not inv_path.is_dir() or not inv_path.name.startswith("inv-"):
            continue

        inv_id = inv_path.name
        metadata = _parse_metadata(inv_path)

        meta_wandb_path = metadata.get("wandb_path") if metadata else None
        if meta_wandb_path is None:
            continue
        # Normalize to canonical form for comparison (strips "runs/", "wandb:" prefix, etc.)
        try:
            e, p, r = parse_wandb_run_path(meta_wandb_path)
            canonical_meta_path = f"{e}/{p}/{r}"
        except ValueError:
            continue
        if canonical_meta_path != wandb_path:
            continue

        events_path = inv_path / "events.jsonl"
        last_time, last_msg, event_count = _get_last_event(events_path)
        title, summary, status = _parse_task_summary(inv_path)

        explanations_path = inv_path / "explanations.jsonl"

        results.append(
            InvestigationSummary(
                id=inv_id,
                wandb_path=meta_wandb_path,
                prompt=metadata.get("prompt") if metadata else None,
                created_at=_get_created_at(inv_path, metadata),
                has_research_log=(inv_path / "research_log.md").exists(),
                has_explanations=explanations_path.exists()
                and explanations_path.stat().st_size > 0,
                event_count=event_count,
                last_event_time=last_time,
                last_event_message=last_msg,
                title=title,
                summary=summary,
                status=status,
            )
        )

    results.sort(key=lambda x: x.created_at, reverse=True)
    return results


@router.get("/{inv_id}")
def get_investigation(inv_id: str) -> InvestigationDetail:
    """Get full details of an investigation."""
    inv_path = INVESTIGATIONS_DIR / inv_id

    if not inv_path.exists() or not inv_path.is_dir():
        raise HTTPException(status_code=404, detail=f"Investigation {inv_id} not found")

    metadata = _parse_metadata(inv_path)

    research_log = None
    research_log_path = inv_path / "research_log.md"
    if research_log_path.exists():
        research_log = research_log_path.read_text()

    events = []
    events_path = inv_path / "events.jsonl"
    if events_path.exists():
        with open(events_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    events.append(
                        EventEntry(
                            event_type=event.get("event_type", "unknown"),
                            timestamp=event.get("timestamp", ""),
                            message=event.get("message", ""),
                            details=event.get("details"),
                        )
                    )
                except json.JSONDecodeError:
                    continue

    explanations: list[dict[str, Any]] = []
    explanations_path = inv_path / "explanations.jsonl"
    if explanations_path.exists():
        with open(explanations_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    explanations.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    title, summary, status = _parse_task_summary(inv_path)
    artifact_ids = _list_artifact_ids(inv_path)

    return InvestigationDetail(
        id=inv_id,
        wandb_path=metadata.get("wandb_path") if metadata else None,
        prompt=metadata.get("prompt") if metadata else None,
        created_at=_get_created_at(inv_path, metadata),
        research_log=research_log,
        events=events,
        explanations=explanations,
        artifact_ids=artifact_ids,
        title=title,
        summary=summary,
        status=status,
    )


class LaunchRequest(BaseModel):
    prompt: str


class LaunchResponse(BaseModel):
    inv_id: str
    job_id: str


@router.post("/launch")
def launch_investigation_endpoint(request: LaunchRequest, loaded: DepLoadedRun) -> LaunchResponse:
    """Launch a new investigation for the currently loaded run."""
    from spd.investigate.scripts.run_slurm import launch_investigation

    result = launch_investigation(
        wandb_path=loaded.run.wandb_path,
        prompt=request.prompt,
        context_length=loaded.context_length,
        max_turns=50,
        partition="h200-reserved",
        time="8:00:00",
        job_suffix=None,
    )
    return LaunchResponse(inv_id=result.inv_id, job_id=result.job_id)


@router.get("/{inv_id}/artifacts")
def list_artifacts(inv_id: str) -> list[str]:
    """List all artifact IDs for an investigation."""
    inv_path = INVESTIGATIONS_DIR / inv_id
    if not inv_path.exists():
        raise HTTPException(status_code=404, detail=f"Investigation {inv_id} not found")
    return _list_artifact_ids(inv_path)


@router.get("/{inv_id}/artifacts/{artifact_id}")
def get_artifact(inv_id: str, artifact_id: str) -> dict[str, Any]:
    """Get a specific artifact by ID."""
    inv_path = INVESTIGATIONS_DIR / inv_id
    artifact_path = inv_path / "artifacts" / f"{artifact_id}.json"

    if not artifact_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Artifact {artifact_id} not found in {inv_id}",
        )

    data: dict[str, Any] = json.loads(artifact_path.read_text())
    return data
