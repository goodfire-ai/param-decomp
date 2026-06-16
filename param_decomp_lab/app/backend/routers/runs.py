"""Run management endpoints."""

import getpass
from pathlib import Path
from urllib.parse import unquote

import yaml
from fastapi import APIRouter, HTTPException
from jax_single_pool.load_run import open_jax_run
from pydantic import BaseModel, ValidationError

from param_decomp.log import logger
from param_decomp_config.lm import LMExperimentConfig
from param_decomp_lab.app.backend.app_tokenizer import AppTokenizer
from param_decomp_lab.app.backend.dependencies import DepStateManager
from param_decomp_lab.app.backend.state import RunState
from param_decomp_lab.app.backend.topology import AppTopology
from param_decomp_lab.app.backend.utils import log_errors
from param_decomp_lab.autointerp.repo import InterpRepo
from param_decomp_lab.dataset_attributions.repo import AttributionRepo
from param_decomp_lab.experiments.utils import EXPERIMENT_CONFIG_FILENAME
from param_decomp_lab.graph_interp.repo import GraphInterpRepo
from param_decomp_lab.harvest.repo import HarvestRepo
from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR
from param_decomp_lab.infra.wandb import parse_wandb_run_path

# Datasets small enough to load into memory for search
_SEARCHABLE_DATASETS = {"SimpleStories/SimpleStories"}

# =============================================================================
# Schemas
# =============================================================================


class LoadedRun(BaseModel):
    """Info about the currently loaded run."""

    id: int
    wandb_path: str
    config_yaml: str
    has_prompts: bool
    prompt_count: int
    context_length: int
    backend_user: str
    dataset_attributions_available: bool
    dataset_search_enabled: bool
    graph_interp_available: bool
    autointerp_available: bool


router = APIRouter(prefix="/api", tags=["runs"])


def _model_type(cfg: LMExperimentConfig) -> str:
    """Target-model class name (last segment of the dotted `model_class` path)."""
    return cfg.target.spec.model_class.rsplit(".", 1)[-1]


@router.post("/runs/load")
@log_errors
def load_run(wandb_path: str, context_length: int, manager: DepStateManager):
    """Load a JAX run by its wandb path / run id. Creates the run in DB if not found.

    Accepts various W&B run reference formats:
    - "entity/project/runId" (compact form)
    - "entity/project/runs/runId" (with /runs/)
    - "https://wandb.ai/entity/project/runs/runId..." (URL)

    Opens the run's orbax checkpoint via `open_jax_run` (the model forward) and reads its
    pinned `LMExperimentConfig` for target/data/algorithm metadata. Read-only: no GPU
    training, no torch.
    """
    db = manager.db

    entity, project, run_id = parse_wandb_run_path(unquote(wandb_path))
    clean_wandb_path = f"{entity}/{project}/{run_id}"

    logger.info(f"[API] Loading {clean_wandb_path}")
    run_dir = PARAM_DECOMP_OUT_DIR / "runs" / run_id
    assert run_dir.is_dir(), f"run dir not found: {run_dir}"

    config_path = run_dir / EXPERIMENT_CONFIG_FILENAME
    try:
        cfg = LMExperimentConfig.from_file(config_path)
    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=(
                f"This run is not a valid LM run and is not compatible with the "
                f"token-based app. Use an LM run.\n\n{e}"
            ),
        ) from e

    run = db.get_run_by_wandb_path(clean_wandb_path)
    if run is None:
        new_run_id = db.create_run(clean_wandb_path)
        run = db.get_run(new_run_id)
        assert run is not None
        logger.info(f"[API] Created new run in DB: {run.id}")
    else:
        logger.info(f"[API] Found existing run in DB: {run.id}")

    # If already loaded with same context_length, skip model load
    if (
        manager.run_state is not None
        and manager.run_state.run.id == run.id
        and manager.run_state.context_length == context_length
    ):
        logger.info(
            f"[API] Run {run.id} already loaded with context_length={context_length}, skipping"
        )
        return {"status": "already_loaded", "run_id": run.id, "wandb_path": run.wandb_path}

    manager.run_state = None

    logger.info(f"[API] Opening JAX run {run.id}: {run_dir}")
    jax_run = open_jax_run(Path(run_dir))

    logger.info(f"[API] Loading tokenizer for run {run.id}: {cfg.data.tokenizer_name}")
    app_tokenizer = AppTokenizer.from_pretrained(cfg.data.tokenizer_name)

    topology = AppTopology.from_model_type(_model_type(cfg), jax_run.site_names)

    manager.run_state = RunState(
        run=run,
        jax_run=jax_run,
        topology=topology,
        tokenizer=app_tokenizer,
        config=cfg.pd,
        lm_target=cfg.target,
        lm_data=cfg.data,
        context_length=context_length,
        harvest=HarvestRepo.open_most_recent(run_id),
        interp=InterpRepo.open(run_id),
        attributions=AttributionRepo.open(run_id),
        graph_interp=GraphInterpRepo.open(run_id),
    )

    logger.info(f"[API] Run {run.id} loaded (step {jax_run.step})")
    return {"status": "loaded", "run_id": run.id, "wandb_path": run.wandb_path}


@router.get("/status")
@log_errors
def get_status(manager: DepStateManager) -> LoadedRun | None:
    """Get current server status."""
    if manager.run_state is None:
        return None

    run = manager.run_state.run
    config_yaml = yaml.dump(
        manager.run_state.config.model_dump(), default_flow_style=False, sort_keys=False
    )

    context_length = manager.run_state.context_length

    prompt_count = manager.db.get_prompt_count(run.id, context_length)

    dataset_search_enabled = manager.run_state.lm_data.dataset_name in _SEARCHABLE_DATASETS

    return LoadedRun(
        id=run.id,
        wandb_path=run.wandb_path,
        config_yaml=config_yaml,
        has_prompts=prompt_count > 0,
        prompt_count=prompt_count,
        context_length=context_length,
        backend_user=getpass.getuser(),
        dataset_attributions_available=manager.run_state.attributions is not None,
        dataset_search_enabled=dataset_search_enabled,
        graph_interp_available=manager.run_state.graph_interp is not None,
        autointerp_available=manager.run_state.interp is not None,
    )


@router.get("/health")
@log_errors
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@router.get("/whoami")
@log_errors
def whoami() -> dict[str, str]:
    """Return the current backend user."""
    return {"user": getpass.getuser()}
