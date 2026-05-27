"""YAML schema for resuming a prior PD run.

A resume YAML is distinct from the run YAML: it doesn't define a run, it points
at one. The schema is deliberately small — `from_run` + `step`. The standard
resume case is "continue with the original config"; mid-trajectory edits to the
saved config (e.g. extending `steps`) are out-of-band — mutate the snapshot's
`pd_config` dict in Python and pass it to `Trainer.from_snapshot` directly.
"""

from pathlib import Path
from typing import Literal

import torch

from param_decomp.base_config import BaseConfig
from param_decomp.training_state import TrainingState


class ResumeConfig(BaseConfig):
    """A resumption YAML: which run to resume, which checkpoint."""

    from_run: Path
    """Path to the parent run directory (the one with `experiment_config.yaml` and
    `training_<step>.pth` files)."""

    step: int | Literal["latest"] = "latest"
    """Which checkpoint to load. `"latest"` picks the highest-numbered
    `training_<step>.pth` under `from_run`."""


def resolve_step(run_dir: Path, step: int | Literal["latest"]) -> int:
    """Resolve `"latest"` to the highest-numbered `training_<step>.pth` under `run_dir`.

    Errors loudly if no training checkpoints exist, or if a specific step was
    requested that isn't on disk.
    """
    candidates: list[int] = []
    for path in run_dir.glob("training_*.pth"):
        try:
            candidates.append(int(path.stem.removeprefix("training_")))
        except ValueError:
            continue
    candidates.sort()
    assert candidates, f"no training_*.pth checkpoints under {run_dir}"
    if step == "latest":
        return candidates[-1]
    assert step in candidates, f"step {step} not on disk under {run_dir}; available: {candidates}"
    return step


def read_training_snapshot(run_dir: Path, step: int) -> TrainingState:
    """Read `<run_dir>/training_<step>.pth` into a `TrainingState` dataclass.

    `weights_only=False` because the payload contains arbitrary cfg dicts
    (model_dump output) alongside tensors.
    """
    path = run_dir / f"training_{step}.pth"
    assert path.is_file(), f"training checkpoint not found: {path}"
    snapshot = torch.load(path, map_location="cpu", weights_only=False)
    assert isinstance(snapshot, TrainingState), (
        f"expected TrainingState in {path}, got {type(snapshot).__name__}"
    )
    return snapshot
