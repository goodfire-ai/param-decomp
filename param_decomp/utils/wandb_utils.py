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

# Regex patterns for parsing W&B run references
# Run IDs can be 8 chars (e.g., "d2ec3bfe") or prefixed with char-dash (e.g., "s-d2ec3bfe")
DEFAULT_WANDB_ENTITY = "goodfire"
DEFAULT_WANDB_PROJECT = DEFAULT_PROJECT_NAME

_RUN_ID_PATTERN = r"(?:[a-z0-9]-)?[a-z0-9]{8}"
_BARE_RUN_ID_RE = re.compile(r"^([sp]-[a-z0-9]{8})$")
_WANDB_PATH_RE = re.compile(rf"^([^/\s]+)/([^/\s]+)/({_RUN_ID_PATTERN})$")
_WANDB_PATH_WITH_RUNS_RE = re.compile(rf"^([^/\s]+)/([^/\s]+)/runs/({_RUN_ID_PATTERN})$")
_WANDB_URL_RE = re.compile(
    rf"^https://wandb\.ai/([^/]+)/([^/]+)/runs/({_RUN_ID_PATTERN})(?:/[^?]*)?(?:\?.*)?$"
)


def _build_short_names() -> dict[str, str]:
    """Derive the metric class-name to short-name map from the registry."""
    from param_decomp.metrics import METRIC_REGISTRY, discover_metrics

    discover_metrics()
    return {cls.__name__: cls.short_name for cls in METRIC_REGISTRY.values() if cls.short_name}


METRIC_CONFIG_SHORT_NAMES: dict[str, str] = _build_short_names()


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


def _parse_metric_config_key(key: str) -> tuple[str, str, str] | None:
    """Parse a metric config key into (container, metric_field, param).

    Args:
        key: Flattened key like "loss_metrics.ImportanceMinimalityLoss.pnorm"

    Returns:
        Tuple of (container, metric_field, param) if it's a metric config key, None otherwise
    """
    parts = key.split(".")
    if len(parts) >= 3 and parts[0] in ("loss_metrics", "eval_metrics"):
        container = parts[0]
        metric_field = parts[1]
        param = ".".join(parts[2:])
        return (container, metric_field, param)
    return None


def generate_wandb_run_name(params: dict[str, Any]) -> str:
    """Generate a W&B run name based on sweep parameters.

    Groups parameters under `loss_metrics.<MetricClass>.<param>` or
    `eval_metrics.<MetricClass>.<param>`, abbreviating via METRIC_CONFIG_SHORT_NAMES.

    Args:
        params: Dictionary of flattened sweep parameters

    Returns:
        Formatted run name string

    Example:
        >>> params = {
        ...     "seed": 42,
        ...     "loss_metrics.ImportanceMinimalityLoss.pnorm": 0.9,
        ...     "loss_metrics.ImportanceMinimalityLoss.coeff": 0.001,
        ... }
        >>> generate_wandb_run_name(params)
        "seed-42-ImpMin-coeff-0.001-pnorm-0.9"
    """
    regular_params: list[tuple[str, Any]] = []
    metric_params: dict[str, list[tuple[str, Any]]] = {}

    for key, value in params.items():
        parsed = _parse_metric_config_key(key)
        if parsed:
            _, metric_field, param = parsed
            short_name = METRIC_CONFIG_SHORT_NAMES.get(metric_field, metric_field)
            if short_name not in metric_params:
                metric_params[short_name] = []
            metric_params[short_name].append((param, value))
        else:
            regular_params.append((key, value))

    parts: list[str] = []
    for key, value in sorted(regular_params):
        parts.append(f"{key}-{value}")
    for short_name in sorted(metric_params.keys()):
        parts.append(short_name)
        for param, value in sorted(metric_params[short_name]):
            parts.append(f"{param}-{value}")

    return "-".join(parts)


def parse_wandb_run_path(input_path: str) -> tuple[str, str, str]:
    """Parse various W&B run reference formats into (entity, project, run_id).

    Accepts:
    - "s-xxxxxxxx" (bare PD run ID, defaults to goodfire/param-decomp)
    - "entity/project/runId" (compact form)
    - "entity/project/runs/runId" (with /runs/)
    - "wandb:entity/project/runId" (with wandb: prefix)
    - "wandb:entity/project/runs/runId" (full wandb: form)
    - "https://wandb.ai/entity/project/runs/runId..." (URL)

    Returns:
        Tuple of (entity, project, run_id)

    Raises:
        ValueError: If the input doesn't match any expected format.
    """
    s = input_path.strip()

    # Strip wandb: prefix if present
    if s.startswith("wandb:"):
        s = s[6:]

    # Bare run ID (e.g. "s-17805b61") → default entity/project
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
        f' - "s-xxxxxxxx" (bare run ID)\n'
        f' - "entity/project/xxxxxxxx"\n'
        f' - "entity/project/runs/xxxxxxxx"\n'
        f' - "wandb:entity/project/runs/xxxxxxxx"\n'
        f' - "https://wandb.ai/entity/project/runs/xxxxxxxx"\n'
        f'Got: "{input_path}"'
    )


def flatten_metric_configs(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Flatten `loss_metrics` and `eval_metrics` into dot-notation for wandb searchability.

    Converts:
        loss_metrics: {"ImportanceMinimalityLoss": {"coeff": 0.1, "pnorm": 1.0}}
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
            short_name = METRIC_CONFIG_SHORT_NAMES.get(metric_field, metric_field)
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
    config: BaseConfig,
    project: str,
    run_id: str,
    name: str | None = None,
    tags: list[str] | None = None,
) -> None:
    """Initialize Weights & Biases and log the config.

    Args:
        config: The config to log.
        project: The name of the wandb project.
        run_id: The unique run ID (from ExecutionStamp).
        name: The name of the wandb run.
        tags: Optional list of tags to add to the run.
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

    config_dict = config.model_dump(mode="json")
    # We also want flattened names for easier wandb searchability
    flattened_config_dict = flatten_metric_configs(config_dict)
    # Remove the nested metric configs to avoid duplication (if they exist)
    config_dict.pop("loss_metrics", None)
    config_dict.pop("eval_metrics", None)
    wandb.config.update({**config_dict, **flattened_config_dict})


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
