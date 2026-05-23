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
from param_decomp_lab.infra.settings import DEFAULT_PROJECT_NAME, REPO_ROOT

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


def _build_short_names() -> dict[str, str]:
    """Build the metric class-name to short-name map. Lazy to avoid circular imports."""
    from param_decomp.metrics.dispatch import LOSS_METRIC_CLASSES
    from param_decomp_lab.eval_metrics import EVAL_METRIC_CLASSES

    return {
        cls.__name__: cls.short_name
        for cls in (*LOSS_METRIC_CLASSES.values(), *EVAL_METRIC_CLASSES.values())
        if cls.short_name
    }


_metric_short_names_cache: dict[str, str] | None = None


def _metric_short_names() -> dict[str, str]:
    global _metric_short_names_cache
    if _metric_short_names_cache is None:
        _metric_short_names_cache = _build_short_names()
    return _metric_short_names_cache


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

    Both containers are lists of dicts, each carrying a `type` discriminator alongside the
    metric's config fields. Converts:
        loss_metrics: [{"type": "ImportanceMinimalityLoss", "coeff": 0.1, "pnorm": 1.0}]
    To:
        loss.ImpMin.coeff: 0.1
        loss.ImpMin.pnorm: 1.0
    """
    flattened: dict[str, Any] = {}

    for container_name in ("loss_metrics", "eval_metrics"):
        if container_name not in config_dict:
            continue
        container = config_dict[container_name]
        assert isinstance(container, list), f"{container_name} should be a list"

        prefix = container_name.split("_")[0]  # "loss" or "eval"
        for cfg in container:
            assert isinstance(cfg, dict), f"{container_name} entries should be dicts"
            metric_type = cfg["type"]
            short_name = _metric_short_names().get(metric_type, metric_type)
            for key, value in cfg.items():
                if key == "type":
                    continue
                flattened[f"{prefix}.{short_name}.{key}"] = value

    return flattened


def fetch_latest_checkpoint_name(filenames: list[str], prefix: str | None = None) -> str:
    """Fetch the latest checkpoint name from a list of .pth files.

    Assumes format is <name>_<step>.pth or <name>.pth.
    """
    if prefix:
        filenames = [filename for filename in filenames if filename.startswith(prefix)]
    if not filenames:
        raise ValueError(f"No files found with prefix {prefix}")
    if len(filenames) == 1:
        return filenames[0]
    return sorted(filenames, key=lambda x: int(x.split(".pth")[0].split("_")[-1]))[-1]


def fetch_latest_wandb_checkpoint(run: Run, prefix: str | None = None) -> File:
    """Fetch the latest checkpoint from a wandb run."""
    filenames = [file.name for file in run.files() if file.name.endswith((".pth", ".pt"))]
    latest_checkpoint_name = fetch_latest_checkpoint_name(filenames, prefix)
    latest_checkpoint_remote = run.file(latest_checkpoint_name)
    return latest_checkpoint_remote


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
