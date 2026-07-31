"""Repo-relative path types for lab YAMLs.

Lab YAMLs store model checkpoint and config paths as either local paths (resolved
relative to the repo root for portability across machines) or W&B run references
(`entity/project/runId`, `entity/project/runs/runId`, bare `p-xxxxxxxx`, URL).
`ModelPath` recognizes wandb references natively via `parse_wandb_run_path`; anything
that fails to parse as a wandb reference is treated as a repo-relative local path.
"""

from pathlib import Path
from typing import Annotated

from pydantic import BeforeValidator, PlainSerializer

REPO_ROOT = Path(__file__).resolve().parents[2]
"""The checkout this package runs from — a property of the install, not the environment.

`data_root` — the one root of the library's local world — is deliberately NOT declared
here: the library reads no ambient environment for paths and ships no default root, so
every entry edge requires its deployment to pass the real root explicitly."""


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
    """Recognize wandb run references natively; otherwise treat as repo-relative local path."""
    if isinstance(v, str):
        from param_decomp.infra.wandb import parse_wandb_run_path

        try:
            parse_wandb_run_path(v)
        except ValueError:
            pass
        else:
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
