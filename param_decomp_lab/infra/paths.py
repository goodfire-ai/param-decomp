"""Repo-relative path types for lab YAMLs.

Lab YAMLs store model checkpoint and config paths as either local paths (resolved
relative to the repo root for portability across machines) or `wandb:`-prefixed
references. `ModelPath` and `RootPath` are pydantic-validated annotations that
normalize these forms; core doesn't need them.
"""

from pathlib import Path
from typing import Annotated

from pydantic import BeforeValidator, PlainSerializer

from param_decomp_lab.infra.settings import REPO_ROOT

WANDB_PATH_PREFIX = "wandb:"


def to_root_path(path: str | Path) -> Path:
    """Converts relative paths to absolute ones, assuming they are relative to the repo root."""
    return Path(path) if Path(path).is_absolute() else Path(REPO_ROOT / path)


def from_root_path(path: str | Path) -> Path:
    """Converts absolute paths to relative ones, relative to the repo root."""
    path = Path(path)
    try:
        return path.relative_to(REPO_ROOT)
    except ValueError:
        return path


def validate_path(v: str | Path) -> str | Path:
    """Check if wandb path. If not, convert to relative to repo root."""
    if isinstance(v, str) and v.startswith(WANDB_PATH_PREFIX):
        return v
    return to_root_path(v)


ModelPath = Annotated[
    str | Path,
    BeforeValidator(validate_path),
    PlainSerializer(lambda x: str(from_root_path(x)) if isinstance(x, Path) else x),
]

RootPath = Annotated[
    Path, BeforeValidator(to_root_path), PlainSerializer(lambda x: str(from_root_path(x)))
]
