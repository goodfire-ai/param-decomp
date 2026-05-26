"""YAML schema for resuming a prior PD run.

A resume YAML is distinct from the run YAML: it doesn't define a run, it points
at one. The schema is deliberately small — `from_run` + `step` + narrow
`overrides`.
"""

from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import PositiveInt

from param_decomp.base_config import BaseConfig
from param_decomp.training_state import TrainingState


class ResumeOverrides(BaseConfig):
    """Narrow set of fields that may be overridden on resume.

    Resumption is continuous with the parent's step axis, so most config fields
    are inherited verbatim. The fields here are explicitly the ones we know
    are safe to change mid-trajectory.
    """

    extend_to_step: PositiveInt | None = None
    """Extend `pd_config.steps` so the resumed run trains further than the parent's
    original target. Must be > `parent.pd_config.steps`; otherwise use the no-override
    resume to finish the original."""

    def to_pd_config_patch(self) -> dict[str, Any]:
        """Convert to a flat dict patch applied to the saved `pd_config` dict before
        pydantic re-validates it inside `Trainer.from_snapshot`."""
        patch: dict[str, Any] = {}
        if self.extend_to_step is not None:
            patch["steps"] = self.extend_to_step
        return patch


class ResumeConfig(BaseConfig):
    """A resumption YAML: which run to resume, which checkpoint, what to override."""

    from_run: Path
    """Path to the parent run directory (the one with `run_meta.yaml` and
    `training_<step>.pth` files)."""

    step: int | Literal["latest"] = "latest"
    """Which checkpoint to load. `"latest"` picks the highest-numbered
    `training_<step>.pth` under `from_run`."""

    overrides: ResumeOverrides | None = None
    """Optional narrow overrides applied to the saved `pd_config` before constructing
    the resumed trainer."""


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
