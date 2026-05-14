"""Typed envelope for the metadata file saved beside every PD run."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

RUN_METADATA_FILENAME = "run_metadata.yaml"


@dataclass
class RunMetadata:
    """Persisted metadata that makes a PD run self-describing and reloadable.

    Written to ``run_metadata.yaml`` beside the checkpoint. Contains the driver
    import path (so the run can be reloaded without external context), the full
    experiment config dump, and the list of extra artifact files bundled with
    the run.
    """

    driver: str | None
    config: dict[str, Any]
    artifact_filenames: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunMetadata":
        return cls(
            driver=data.get("driver"),
            config=data["config"],
            artifact_filenames=list(data.get("artifact_filenames", [])),
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
