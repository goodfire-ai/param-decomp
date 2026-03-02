"""Run registry endpoint.

Returns canonical SPD runs with lightweight data availability checks
so the run picker can show what post-processing data exists at a glance.
"""

from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from spd.app.backend.routers.pretrain_info import _get_pretrain_info, _load_spd_config_lightweight
from spd.app.backend.utils import log_errors
from spd.log import logger
from spd.settings import SPD_OUT_DIR
from spd.utils.wandb_utils import parse_wandb_run_path

router = APIRouter(prefix="/api/run_registry", tags=["run_registry"])


class RegistryEntry(BaseModel):
    wandb_run_id: str
    name: str | None = None
    notes: str | None = None


CANONICAL_RUNS: list[RegistryEntry] = [
    RegistryEntry(
        name="Thomas",
        wandb_run_id="goodfire/spd/runs/s-82ffb969",
        notes="pile_llama_simple_mlp-4L",
    ),
    RegistryEntry(
        name="Jose",
        wandb_run_id="goodfire/spd/s-55ea3f9b",
        notes="pile_llama_simple_mlp-4L",
    ),
    RegistryEntry(
        wandb_run_id="goodfire/spd/s-275c8f21",
        notes="Lucius' pile run Feb 11",
    ),
    RegistryEntry(
        wandb_run_id="goodfire/spd/s-eab2ace8",
        notes="Oli's PPGD run, great metrics",
    ),
    RegistryEntry(
        wandb_run_id="goodfire/spd/s-892f140b",
        notes="Lucius run, Jan 22",
    ),
    RegistryEntry(
        wandb_run_id="goodfire/spd/s-7884efcc",
        notes="Lucius' new run, Jan 8",
    ),
]


class DataAvailability(BaseModel):
    harvest: bool
    autointerp: bool
    attributions: bool
    graph_interp: bool


class RegistryRunInfo(BaseModel):
    wandb_run_id: str
    name: str | None
    notes: str | None
    architecture: str | None
    availability: DataAvailability


def _has_glob_match(pattern_dir: Path, glob_pattern: str) -> bool:
    """Check if any file matches a glob pattern under a directory."""
    if not pattern_dir.exists():
        return False
    return next(pattern_dir.glob(glob_pattern), None) is not None


def _check_availability(run_id: str) -> DataAvailability:
    """Lightweight filesystem checks for post-processing data availability."""
    harvest_dir = SPD_OUT_DIR / "harvest" / run_id
    autointerp_dir = SPD_OUT_DIR / "autointerp" / run_id
    attributions_dir = SPD_OUT_DIR / "dataset_attributions" / run_id
    graph_interp_dir = SPD_OUT_DIR / "graph_interp" / run_id

    return DataAvailability(
        harvest=_has_glob_match(harvest_dir, "h-*/harvest.db"),
        autointerp=_has_glob_match(autointerp_dir, "a-*/.done"),
        attributions=_has_glob_match(attributions_dir, "da-*/dataset_attributions.pt"),
        graph_interp=_has_glob_match(graph_interp_dir, "*/interp.db"),
    )


def _get_architecture_summary(wandb_path: str) -> str | None:
    """Get a short architecture label for a run. Returns None on failure."""
    try:
        spd_config = _load_spd_config_lightweight(wandb_path)
        info = _get_pretrain_info(spd_config)
        parts: list[str] = []
        if info.dataset_short:
            parts.append(info.dataset_short)
        parts.append(info.model_type)
        cfg = info.target_model_config
        if cfg:
            n_layer = cfg.get("n_layer")
            n_embd = cfg.get("n_embd")
            if n_layer is not None:
                parts.append(f"{n_layer}L")
            if n_embd is not None:
                parts.append(f"d{n_embd}")
        return " ".join(parts)
    except Exception:
        logger.exception(f"[run_registry] Failed to get architecture for {wandb_path}")
        return None


@router.get("")
@log_errors
def get_run_registry() -> list[RegistryRunInfo]:
    """Return all canonical runs with data availability."""
    results: list[RegistryRunInfo] = []
    for entry in CANONICAL_RUNS:
        _, _, run_id = parse_wandb_run_path(entry.wandb_run_id)
        availability = _check_availability(run_id)
        architecture = _get_architecture_summary(entry.wandb_run_id)

        results.append(
            RegistryRunInfo(
                wandb_run_id=entry.wandb_run_id,
                name=entry.name,
                notes=entry.notes,
                architecture=architecture,
                availability=availability,
            )
        )
    return results
