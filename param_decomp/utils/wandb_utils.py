import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import wandb
import wandb.errors
from dotenv import load_dotenv
from wandb.apis.public import File, Run

from param_decomp.base_config import BaseConfig
from param_decomp.log import logger
from param_decomp.settings import DEFAULT_PROJECT_NAME, REPO_ROOT
from param_decomp.utils.general_utils import fetch_latest_checkpoint_name

# Regex patterns for parsing W&B run references. PD run IDs are formatted as
# `p-<8 hex chars>` (see `RUN_TYPE_ABBREVIATIONS`). Legacy `s-…` IDs predate the
# Run refactor; they still resolve when given as a full `entity/project/runs/id` path.
DEFAULT_WANDB_ENTITY = "goodfire"
DEFAULT_WANDB_PROJECT = DEFAULT_PROJECT_NAME

_RUN_ID_PATTERN = r"(?:[a-z0-9]-)?[a-z0-9]{8}"
_BARE_RUN_ID_RE = re.compile(r"^(p-[a-z0-9]{8})$")
_WANDB_PATH_RE = re.compile(rf"^([^/\s]+)/([^/\s]+)/({_RUN_ID_PATTERN})$")
_WANDB_PATH_WITH_RUNS_RE = re.compile(rf"^([^/\s]+)/([^/\s]+)/runs/({_RUN_ID_PATTERN})$")
_WANDB_URL_RE = re.compile(
    rf"^https://wandb\.ai/([^/]+)/([^/]+)/runs/({_RUN_ID_PATTERN})(?:/[^?]*)?(?:\?.*)?$"
)

# Short names for metric fields (keyed by LossMetricsConfig / EvalMetricsConfig field names),
# used for W&B run names and view names.
METRIC_CONFIG_SHORT_NAMES: dict[str, str] = {
    "faithfulness": "Faith",
    "importance_minimality": "ImpMin",
    "stochastic_recon": "StochRecon",
    "stochastic_recon_subset": "StochReconSub",
    "stochastic_recon_layerwise": "StochReconLayer",
    "ci_masked_recon": "CIMaskRecon",
    "ci_masked_recon_subset": "CIMaskReconSub",
    "ci_masked_recon_layerwise": "CIMaskReconLayer",
    "pgd_recon": "PGDRecon",
    "pgd_recon_subset": "PGDReconSub",
    "pgd_recon_layerwise": "PGDReconLayer",
    "persistent_pgd_recon": "PersistPGDRecon",
    "persistent_pgd_recon_subset": "PersistPGDReconSub",
    "stochastic_hidden_acts_recon": "StochHiddenActRecon",
    "ci_hidden_acts_recon": "CIHiddenActRecon",
    "stochastic_attn_patterns_recon": "StochAttnRecon",
    "ci_masked_attn_patterns_recon": "CIAttnRecon",
    "unmasked_recon": "UnmaskedRecon",
    "ce_and_kl": "CEandKL",
    "ci_histograms": "CIHist",
    "ci_l0": "CI_L0",
    "ci_mean_per_component": "CIMeanPerComp",
    "component_activation_density": "CompActDens",
    "identity_ci_error": "IdCIErr",
    "permuted_ci_plots": "PermCIPlots",
    "uv_plots": "UVPlots",
    "stochastic_recon_subset_ce_and_kl": "StochReconSubCEKL",
    "pgd_multibatch_recon": "PGDMultiBatchRecon",
    "pgd_multibatch_recon_subset": "PGDMultiBatchReconSub",
    "persistent_pgd_recon_eval": "PersistPGDReconEval",
    "persistent_pgd_recon_subset_eval": "PersistPGDReconSubEval",
}


def get_wandb_entity() -> str:
    """Get the WandB entity from env var or the authenticated user's default entity."""
    load_dotenv(override=True)
    entity = os.getenv("WANDB_ENTITY")
    if entity is None:
        entity = wandb.Api().default_entity
    assert entity is not None, (
        "Could not determine WandB entity. Set WANDB_ENTITY in .env or log in with `wandb login`."
    )
    return entity


def get_wandb_run_url(project: str, run_id: str) -> str:
    """Get the direct WandB URL for a run."""
    return f"https://wandb.ai/{get_wandb_entity()}/{project}/runs/{run_id}"


def wandb_path_to_url(wandb_path: str) -> str:
    """Convert a WandB run path to a URL."""
    entity, project, run_id = parse_wandb_run_path(wandb_path)
    return f"https://wandb.ai/{entity}/{project}/runs/{run_id}"


def parse_wandb_run_path(input_path: str) -> tuple[str, str, str]:
    """Parse various W&B run reference formats into (entity, project, run_id).

    Accepts:
    - "p-xxxxxxxx" (bare PD run ID, defaults to goodfire/param-decomp)
    - "entity/project/runId" (compact form)
    - "entity/project/runs/runId" (with /runs/)
    - "wandb:entity/project/runId" (with wandb: prefix)
    - "wandb:entity/project/runs/runId" (full wandb: form)
    - "https://wandb.ai/entity/project/runs/runId..." (URL)

    The bare-ID shortcut only accepts the current `p-…` prefix; legacy `s-…` IDs
    (pre-refactor) still resolve via the full `entity/project/runs/id` form.

    Returns:
        Tuple of (entity, project, run_id)

    Raises:
        ValueError: If the input doesn't match any expected format.
    """
    s = input_path.strip()

    # Strip wandb: prefix if present
    if s.startswith("wandb:"):
        s = s[6:]

    # Bare run ID (e.g. "p-17805b61") → default entity/project
    if m := _BARE_RUN_ID_RE.match(s):
        return DEFAULT_WANDB_ENTITY, DEFAULT_WANDB_PROJECT, m.group(1)

    # Try compact form: entity/project/runid
    if m := _WANDB_PATH_RE.match(s):
        return m.group(1), m.group(2), m.group(3)

    # Try form with /runs/: entity/project/runs/runid
    if m := _WANDB_PATH_WITH_RUNS_RE.match(s):
        return m.group(1), m.group(2), m.group(3)

    # Try full URL
    if m := _WANDB_URL_RE.match(s):
        return m.group(1), m.group(2), m.group(3)

    raise ValueError(
        f"Invalid W&B run reference. Expected one of:\n"
        f' - "p-xxxxxxxx" (bare PD run ID)\n'
        f' - "entity/project/xxxxxxxx"\n'
        f' - "entity/project/runs/xxxxxxxx"\n'
        f' - "wandb:entity/project/runs/xxxxxxxx"\n'
        f' - "https://wandb.ai/entity/project/runs/xxxxxxxx"\n'
        f'Got: "{input_path}"'
    )


def flatten_metric_configs(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Flatten `loss_metrics` and `eval_metrics` into dot-notation for wandb searchability.

    Converts:
        loss_metrics: {"importance_minimality": {"coeff": 0.1, "pnorm": 1.0}}
    To:
        loss.ImpMin.coeff: 0.1
        loss.ImpMin.pnorm: 1.0
    """
    flattened: dict[str, Any] = {}

    for container_name in ("loss_metrics", "eval_metrics"):
        if container_name not in config_dict:
            continue
        container = config_dict[container_name]
        assert isinstance(container, dict), f"{container_name} should be a dict"

        prefix = container_name.split("_")[0]  # "loss" or "eval"
        for metric_field, cfg in container.items():
            if cfg is None:
                continue
            assert isinstance(cfg, dict), f"{container_name}.{metric_field} should be a dict"
            short_name = METRIC_CONFIG_SHORT_NAMES[metric_field]
            for key, value in cfg.items():
                if key == "classname":
                    continue
                flattened[f"{prefix}.{short_name}.{key}"] = value

    return flattened


def fetch_latest_wandb_checkpoint(run: Run, prefix: str | None = None) -> File:
    """Fetch the latest checkpoint from a wandb run."""
    filenames = [file.name for file in run.files() if file.name.endswith((".pth", ".pt"))]
    latest_checkpoint_name = fetch_latest_checkpoint_name(filenames, prefix)
    latest_checkpoint_remote = run.file(latest_checkpoint_name)
    return latest_checkpoint_remote


def fetch_wandb_run_dir(run_id: str) -> Path:
    """Find or create a directory in the W&B cache for a given run.

    We first check if we already have a directory with the suffix "run_id" (if we created the run
    ourselves, a directory of the name "run-<timestamp>-<run_id>" should exist). If not, we create a
    new wandb_run_dir.
    """
    # Default to REPO_ROOT/wandb
    base_cache_dir = REPO_ROOT / "wandb"
    base_cache_dir.mkdir(parents=True, exist_ok=True)

    # Set default wandb_run_dir
    wandb_run_dir = base_cache_dir / run_id / "files"

    # Check if we already have a directory with the suffix "run_id"
    presaved_run_dirs = [
        d for d in base_cache_dir.iterdir() if d.is_dir() and d.name.endswith(run_id)
    ]
    # If there is more than one dir, just ignore the presaved dirs and use the new wandb_run_dir
    if presaved_run_dirs and len(presaved_run_dirs) == 1:
        presaved_file_path = presaved_run_dirs[0] / "files"
        if presaved_file_path.exists():
            # Found a cached run directory, use it
            wandb_run_dir = presaved_file_path

    wandb_run_dir.mkdir(parents=True, exist_ok=True)
    return wandb_run_dir


def download_wandb_file(run: Run, wandb_run_dir: Path, file_name: str) -> Path:
    """Download a file from W&B. Don't overwrite the file if it already exists.

    Args:
        run: The W&B run to download from
        file_name: Name of the file to download
        wandb_run_dir: The directory to download the file to
    Returns:
        Path to the downloaded file
    """
    file_on_wandb = run.file(file_name)
    assert isinstance(file_on_wandb, File)
    file_on_wandb.download(exist_ok=True, replace=False, root=str(wandb_run_dir))
    return wandb_run_dir / file_name


def init_wandb(
    project: str,
    run_id: str,
    configs: dict[str, BaseConfig],
    *,
    name: str | None = None,
    tags: list[str] | None = None,
    view_meta: dict[str, Any] | None = None,
) -> None:
    """Initialize Weights & Biases and log the configs.

    Args:
        project: The wandb project name.
        run_id: The unique run ID (from ExecutionStamp).
        configs: Mapping of prefix to config. Each config is dumped under its prefix
            in ``wandb.config`` (``{prefix}/{field}``). Pass ``""`` as a prefix for a
            flat dump (no key prefix), useful when there's a single config. The PD
            path passes ``{"pd": ..., "logging": ..., "runtime": ...}`` — three
            siblings, none privileged.
        name: The name of the wandb run.
        tags: Optional list of tags to add to the run.
        view_meta: Free-form labels (typically populated by a sweep generator)
            merged into ``wandb.config`` under a ``view_meta/`` prefix so the W&B
            UI can group/color runs by researcher-facing axes.
    """
    wandb.init(
        id=run_id,
        project=project,
        entity=get_wandb_entity(),
        name=name,
        tags=tags,
    )
    assert wandb.run is not None
    wandb.run.log_code(
        root=str(REPO_ROOT / "param_decomp"), exclude_fn=lambda path: "out" in Path(path).parts
    )

    for prefix, cfg in configs.items():
        cfg_dict = cfg.model_dump(mode="json")
        flattened = flatten_metric_configs(cfg_dict)
        cfg_dict.pop("loss_metrics", None)
        cfg_dict.pop("eval_metrics", None)
        key_prefix = f"{prefix}/" if prefix else ""
        wandb.config.update({f"{key_prefix}{k}": v for k, v in cfg_dict.items()})
        wandb.config.update({f"{key_prefix}{k}": v for k, v in flattened.items()})

    if view_meta:
        wandb.config.update({f"view_meta/{k}": v for k, v in view_meta.items()})


_n_try_wandb_comm_errors = 0


# this exists to stop infra issues from crashing training runs
def try_wandb[**P, T](wandb_fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T | None:
    """Attempts to call `wandb_fn` and if it fails with a wandb CommError, logs a warning and returns
    None. The choice of wandb CommError is to catch issues communicating with the wandb server but
    not legitimate logging errors, for example not passing a dict to wandb.log, or the wrong
    arguments to wandb.save."""
    global _n_try_wandb_comm_errors
    try:
        return wandb_fn(*args, **kwargs)
    except wandb.errors.CommError as e:
        _n_try_wandb_comm_errors += 1
        logger.error(
            f"wandb communication error, skipping log (total comm errors: {_n_try_wandb_comm_errors}): {e}"
        )
