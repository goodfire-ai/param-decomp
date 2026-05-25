"""Run management endpoints."""

import getpass
from urllib.parse import unquote

import torch
import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ValidationError

from param_decomp.log import logger
from param_decomp_lab.app.backend.app_tokenizer import AppTokenizer
from param_decomp_lab.app.backend.dependencies import DepStateManager
from param_decomp_lab.app.backend.state import RunState
from param_decomp_lab.app.backend.utils import log_errors
from param_decomp_lab.autointerp.repo import InterpRepo
from param_decomp_lab.dataset_attributions.repo import AttributionRepo
from param_decomp_lab.distributed import get_device
from param_decomp_lab.experiments.lm.run import SavedLMRun
from param_decomp_lab.graph_interp.repo import GraphInterpRepo
from param_decomp_lab.harvest.repo import HarvestRepo
from param_decomp_lab.infra.wandb import parse_wandb_run_path
from param_decomp_lab.topology import TransformerTopology, get_sources_by_target

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

DEVICE = get_device()


@router.post("/runs/load")
@log_errors
def load_run(wandb_path: str, context_length: int, manager: DepStateManager):
    """Load a run by its wandb path. Creates the run in DB if not found.

    Accepts various W&B run reference formats:
    - "entity/project/runId" (compact form)
    - "entity/project/runs/runId" (with /runs/)
    - "https://wandb.ai/entity/project/runs/runId..." (URL)

    This loads the model onto GPU and makes it available for attribution computation.
    """
    db = manager.db

    entity, project, run_id = parse_wandb_run_path(unquote(wandb_path))
    clean_wandb_path = f"{entity}/{project}/{run_id}"

    logger.info(f"[API] Loading {clean_wandb_path}")
    try:
        pd_run = SavedLMRun.from_path(clean_wandb_path)
    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=(
                f"This run is not a valid LM run and is not compatible with the "
                f"token-based app. Use an LM run.\n\n{e}"
            ),
        ) from e
    lm_target = pd_run.cfg.target
    lm_data = pd_run.cfg.data

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

    # Unload previous run if any
    if manager.run_state is not None:
        logger.info(f"[API] Unloading previous run {manager.run_state.run.id}")
        del manager.run_state.model
        torch.cuda.empty_cache()
        manager.run_state = None

    # Load the target + ComponentModel
    logger.info(f"[API] Loading model for run {run.id}: {run.wandb_path}")
    model = pd_run.load_model().to(DEVICE)
    model.eval()

    pd_config = pd_run.cfg.pd
    logger.info(f"[API] Loading tokenizer for run {run.id}: {lm_data.tokenizer_name}")
    app_tokenizer = AppTokenizer.from_pretrained(lm_data.tokenizer_name)

    # Build topology and sources_by_target mapping
    logger.info(f"[API] Building topology for run {run.id}")
    topology = TransformerTopology(model.target_model)

    logger.info(f"[API] Building sources_by_target mapping for run {run.id}")
    sources_by_target = get_sources_by_target(model, topology, DEVICE, pd_config.sampling)

    manager.run_state = RunState(
        run=run,
        model=model,
        topology=topology,
        tokenizer=app_tokenizer,
        sources_by_target=sources_by_target,
        config=pd_config,
        lm_target=lm_target,
        lm_data=lm_data,
        context_length=context_length,
        harvest=HarvestRepo.open_most_recent(run_id),
        interp=InterpRepo.open(run_id),
        attributions=AttributionRepo.open(run_id),
        graph_interp=GraphInterpRepo.open(run_id),
    )

    logger.info(f"[API] Run {run.id} loaded on {DEVICE}")
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
