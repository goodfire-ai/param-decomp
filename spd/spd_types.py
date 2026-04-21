from pathlib import Path
from typing import Annotated, Literal

from annotated_types import Ge, Le
from pydantic import BeforeValidator, PlainSerializer

from spd.settings import REPO_ROOT

WANDB_PATH_PREFIX = "wandb:"


def to_root_path(path: str | Path) -> Path:
    """Converts relative paths to absolute ones, assuming they are relative to the rib root."""
    return Path(path) if Path(path).is_absolute() else Path(REPO_ROOT / path)


def from_root_path(path: str | Path) -> Path:
    """Converts absolute paths to relative ones, relative to the repo root."""
    path = Path(path)
    try:
        return path.relative_to(REPO_ROOT)
    except ValueError:
        # If the path is not relative to REPO_ROOT, return the original path
        return path


def validate_path(v: str | Path) -> str | Path:
    """Check if wandb path. If not, convert to relative to repo root."""
    if isinstance(v, str) and v.startswith(WANDB_PATH_PREFIX):
        return v
    return to_root_path(v)


# Type for paths that can either be wandb paths (starting with "wandb:")
# or regular paths (converted to be relative to repo root)
ModelPath = Annotated[
    str | Path,
    BeforeValidator(validate_path),
    PlainSerializer(lambda x: str(from_root_path(x)) if isinstance(x, Path) else x),
]

# This is a type for pydantic configs that will convert all relative paths
# to be relative to the root of this repository
RootPath = Annotated[
    Path, BeforeValidator(to_root_path), PlainSerializer(lambda x: str(from_root_path(x)))
]

Probability = Annotated[float, Ge(0), Le(1)]
TaskName = Literal["tms", "resid_mlp", "lm", "ih"]
LayerwiseCiFnType = Literal["mlp", "vector_mlp", "shared_mlp"]
GlobalCiFnType = Literal[
    "global_shared_mlp", "global_reverse_residual", "global_shared_transformer"
]
