import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import wandb
import wandb.errors
from dotenv import load_dotenv
from wandb.apis.public import File, Run

from param_decomp.log import logger
from param_decomp_lab.infra.settings import REPO_ROOT

# Regex patterns for parsing W&B run references. PD run IDs are formatted as
# `p-<8 hex chars>` (see `RUN_TYPE_ABBREVIATIONS`). Legacy `s-…` IDs predate the
# Run refactor; they still resolve when given as a full `entity/project/runs/id` path.
DEFAULT_WANDB_ENTITY = "goodfire"
DEFAULT_WANDB_PROJECT = "param-decomp"

_RUN_ID_PATTERN = r"(?:[a-z0-9]-)?[a-z0-9]{8}"
_BARE_RUN_ID_RE = re.compile(r"^(p-[a-z0-9]{8})$")
_WANDB_PATH_RE = re.compile(rf"^([^/\s]+)/([^/\s]+)/({_RUN_ID_PATTERN})$")
_WANDB_PATH_WITH_RUNS_RE = re.compile(rf"^([^/\s]+)/([^/\s]+)/runs/({_RUN_ID_PATTERN})$")
_WANDB_URL_RE = re.compile(
    rf"^https://wandb\.ai/([^/]+)/([^/]+)/runs/({_RUN_ID_PATTERN})(?:/[^?]*)?(?:\?.*)?$"
)


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
    """Parse various W&B run reference formats into `(entity, project, run_id)`.

    Accepts:
    - `"p-xxxxxxxx"` (bare PD run ID, defaults to `goodfire/param-decomp`)
    - `"entity/project/runId"` (compact form)
    - `"entity/project/runs/runId"` (with `/runs/`)
    - `"https://wandb.ai/entity/project/runs/runId..."` (URL)

    The bare-ID shortcut only accepts the current `p-…` prefix; legacy `s-…` IDs
    (pre-refactor) still resolve via the full `entity/project/runs/id` form.
    """
    s = input_path.strip()

    # The legacy "wandb:" prefix is no longer accepted. Reject explicitly so old YAMLs
    # surface a clear error instead of silently parsing with `wandb:foo` as the entity.
    if s.startswith("wandb:"):
        raise ValueError(
            f'Invalid W&B run reference: the "wandb:" prefix is no longer supported. '
            f'Drop it from "{input_path}".'
        )

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
        f' - "https://wandb.ai/entity/project/runs/xxxxxxxx"\n'
        f'Got: "{input_path}"'
    )


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
    """Download `file_name` from `run` to `wandb_run_dir`. No-op when the file already exists."""
    file_on_wandb = run.file(file_name)
    assert isinstance(file_on_wandb, File)
    file_on_wandb.download(exist_ok=True, replace=False, root=str(wandb_run_dir))
    return wandb_run_dir / file_name


def init_wandb(
    project: str,
    run_id: str,
    config_dict: dict[str, Any],
    *,
    entity: str | None = None,
    name: str | None = None,
    tags: list[str] | None = None,
    group: str | None = None,
    view_meta: dict[str, Any] | None = None,
) -> None:
    """Initialise W&B and log `config_dict` (already rendered to a flat-ish dict by the
    caller — see `eval_metrics.wandb_config_dict`).

    `entity` falls back to `get_wandb_entity()`; `view_meta` is merged under a
    `view_meta/` prefix so the UI can group runs by researcher-facing axes.
    """
    wandb.init(
        id=run_id,
        project=project,
        entity=entity or get_wandb_entity(),
        name=name,
        tags=tags,
        group=group,
    )
    assert wandb.run is not None
    wandb.run.log_code(root=str(REPO_ROOT / "param_decomp"))

    # Slow eval keys ride a dedicated `slow_eval/step` axis. The single-pool path
    # logs them in-train (monotonic); the 3-pool path logs them retroactively from
    # the async job (non-monotonic on the default axis). Defining the axis here lets
    # both share the same panels. The async job redefines it too — idempotent.
    wandb.define_metric("slow_eval/step")
    wandb.define_metric("slow_eval/*", step_metric="slow_eval/step")

    wandb.config.update(config_dict)

    if view_meta:
        wandb.config.update({f"view_meta/{k}": v for k, v in view_meta.items()})


_n_try_wandb_comm_errors = 0


# this exists to stop infra issues from crashing training runs
def try_wandb[**P, T](wandb_fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T | None:
    """Call `wandb_fn`, warning and returning `None` on a wandb `CommError`.

    `CommError` is chosen to catch issues communicating with the wandb server but not
    legitimate logging errors (e.g. not passing a dict to `wandb.log`, or the wrong
    arguments to `wandb.save`).
    """
    global _n_try_wandb_comm_errors
    try:
        return wandb_fn(*args, **kwargs)
    except wandb.errors.CommError as e:
        _n_try_wandb_comm_errors += 1
        logger.error(
            f"wandb communication error, skipping log (total comm errors: {_n_try_wandb_comm_errors}): {e}"
        )
