"""Typed envelope for the per-run spec carried through the PD pipeline."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

RUN_METADATA_FILENAME = "run_metadata.yaml"


@dataclass
class RunSpec:
    """The per-run spec: built by the launcher (or a sweep generator), passed
    to the worker, written to ``run_metadata.yaml`` beside the checkpoint, and
    re-read on reload. One type, one shape, everywhere.

    Holds everything *about* a run that is not a hyperparameter of the algorithm:

    - ``driver``: import path so the run can be reloaded without external context.
      ``None`` for notebook callers of ``run_pd`` who build their own ``PDTarget``.
    - ``config``: full experiment config dump (driver-validated when ``driver`` is set).
    - ``wandb_project`` / ``wandb_run_name``: where this run is logged. Not part
      of ``PDConfig`` because they affect observation, not training.
      ``wandb_project`` is typically stamped by the launcher from ``--project``;
      ``wandb_run_name=None`` lets W&B auto-name.
    - ``view_meta``: free-form labels for downstream grouping / coloring / reports.
      Populated by sweep generators (e.g. ``{"lr_ratio": 0.1, "size": "medium"}``);
      empty for one-off / notebook runs. Surfaced to W&B under a ``view_meta/``
      prefix for use as grouping/coloring axes in the UI.
    """

    driver: str | None
    config: dict[str, Any]
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    view_meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunSpec":
        return cls(
            driver=data.get("driver"),
            config=data["config"],
            wandb_project=data.get("wandb_project"),
            wandb_run_name=data.get("wandb_run_name"),
            view_meta=data.get("view_meta") or {},
        )

    @classmethod
    def from_file(cls, path: Path) -> "RunSpec":
        assert path.exists(), f"{RUN_METADATA_FILENAME} not found at {path}"
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
