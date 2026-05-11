"""Open-world experiment driver interface.

The core PD optimizer only needs a target model bundle plus train/eval dataloaders.
Experiment drivers are the boundary layer that turns a serializable experiment config
into those runtime objects.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from param_decomp.models.batch_and_loss_fns import PDTarget
    from param_decomp.utils.distributed_utils import DistributedState

from param_decomp.base_config import BaseConfig
from param_decomp.configs import PDConfig


class ExperimentConfig(BaseConfig):
    """Pure-data config shared by all experiment configs."""

    kind: str
    pd: PDConfig
    metadata: dict[str, Any] | None = None


class ExperimentManifest(BaseConfig):
    """Serializable metadata persisted with a PD run.

    `driver` is optional so direct/custom users can still save a run with an explicit target and
    reload via `load_pd(path, target=...)`. Registered runs set `driver`, enabling tooling to
    reconstruct the target and dataloaders from the saved experiment config.
    """

    kind: str
    experiment_config: dict[str, Any]
    driver: str | None = None
    artifact_filenames: list[str] = []

    @classmethod
    def from_pd_config(
        cls,
        pd_config: PDConfig,
        *,
        kind: str = "manual",
        driver: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExperimentManifest:
        """Build a manifest for direct `run_pd` callers without a full experiment config."""
        experiment_config: dict[str, Any] = {
            "kind": kind,
            "pd": pd_config.model_dump(mode="json"),
        }
        if metadata is not None:
            experiment_config["metadata"] = metadata
        return cls(
            kind=kind,
            driver=driver,
            experiment_config=experiment_config,
        )

    def with_artifacts(self, artifact_filenames: Sequence[str]) -> ExperimentManifest:
        return self.model_copy(update={"artifact_filenames": list(artifact_filenames)})


@dataclass(frozen=True)
class RunArtifact:
    """A file to persist beside the PD checkpoint."""

    filename: str
    data: Any


@dataclass(frozen=True)
class PreparedExperiment:
    """Runtime objects ready for `run_pd`."""

    pd: PDConfig
    target: PDTarget
    train_loader: DataLoader[Any]
    eval_loader: DataLoader[Any]
    artifacts: Sequence[RunArtifact] = ()
    tags: Sequence[str] = ()


class ExperimentDriver[ConfigT: ExperimentConfig](Protocol):
    """Converts a serializable experiment config into runtime PD objects."""

    kind: str
    config_model: type[ConfigT]

    def prepare(
        self,
        experiment_config: ConfigT,
        *,
        device: str,
        dist_state: DistributedState | None = None,
    ) -> PreparedExperiment: ...

    def load_target(
        self, experiment_config: ConfigT, *, run_dir: Path | None = None
    ) -> PDTarget: ...

    def build_dataloaders(
        self,
        experiment_config: ConfigT,
        *,
        seed: int | None = None,
        train_batch_size: int,
        eval_batch_size: int,
        dist_state: DistributedState | None = None,
        device: str = "cpu",
        run_dir: Path | None = None,
    ) -> tuple[DataLoader[Any], DataLoader[Any]]: ...

    def display_name(self, experiment_config: ConfigT) -> str: ...


def load_driver(driver_path: str) -> ExperimentDriver[Any]:
    """Load a driver object or no-arg driver class from `module:attr`."""
    module_path, sep, attr = driver_path.partition(":")
    if sep == "":
        raise ValueError(f"Driver path must be of the form 'module:attr', got {driver_path!r}")
    module = import_module(module_path)
    driver = getattr(module, attr)
    if isinstance(driver, type):
        driver = driver()
    return driver


def parse_manifest_experiment_config(manifest: ExperimentManifest) -> ExperimentConfig:
    """Parse a manifest's raw experiment config with its registered driver."""
    if manifest.driver is None:
        return ExperimentConfig.model_validate(manifest.experiment_config)
    driver = load_driver(manifest.driver)
    return driver.config_model.model_validate(manifest.experiment_config)
