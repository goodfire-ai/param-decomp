import os
import re
from pathlib import Path

import wandb
from dotenv import load_dotenv
from wandb.apis.public import File, Run

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
