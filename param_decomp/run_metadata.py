"""Typed envelope for the metadata file saved beside every PD run."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

RUN_METADATA_FILENAME = "run_metadata.yaml"


@dataclass
class RunMetadata:
    """Persisted metadata that makes a PD run self-describing and reloadable.

    Written to ``run_metadata.yaml`` beside the checkpoint. Contains everything
    *about* the run that is not a hyperparameter of the algorithm:

    - ``driver``: import path so the run can be reloaded without external context.
    - ``config``: full experiment config dump (driver-validated).
    - ``wandb_project`` / ``wandb_run_name``: where this run is logged. Not part
      of ``PDConfig`` because they affect observation, not training.
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
    def from_dict(cls, data: dict[str, Any]) -> "RunMetadata":
        return cls(
            driver=data.get("driver"),
            config=data["config"],
            wandb_project=data.get("wandb_project"),
            wandb_run_name=data.get("wandb_run_name"),
            view_meta=data.get("view_meta") or {},
        )

    @classmethod
    def from_file(cls, path: Path) -> "RunMetadata":
        assert path.exists(), f"{RUN_METADATA_FILENAME} not found at {path}"
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
