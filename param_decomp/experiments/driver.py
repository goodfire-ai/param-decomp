"""Open-world experiment driver interface.

The core PD optimizer only needs a target model bundle plus train/eval dataloaders.
Experiment drivers are the boundary layer that turns a serializable experiment spec
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

EXPERIMENT_CONFIG_FILENAME = "experiment_config.yaml"


class ExperimentSpec(BaseConfig):
    """Pure-data config shared by all experiment specs."""

    kind: str
    pd: PDConfig
    metadata: dict[str, Any] | None = None


class ExperimentManifest(BaseConfig):
    """Serializable metadata persisted with a PD run.

    `driver` is optional so direct/custom users can still save a run with an explicit target and
    reload via `load_pd(path, target=...)`. Registered runs set `driver`, enabling tooling to
    reconstruct the target and dataloaders from the saved spec.
    """

    kind: str
    spec: dict[str, Any]
    driver: str | None = None
    artifact_filenames: list[str] = []

    @classmethod
    def from_spec(
        cls,
        spec: ExperimentSpec,
        *,
        driver: str | None,
    ) -> ExperimentManifest:
        return cls(
            kind=spec.kind,
            driver=driver,
            spec=spec.model_dump(mode="json"),
        )

    @classmethod
    def from_pd_config(
        cls,
        pd_config: PDConfig,
        *,
        kind: str = "manual",
        driver: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExperimentManifest:
        spec: dict[str, Any] = {
            "kind": kind,
            "pd": pd_config.model_dump(mode="json"),
        }
        if metadata is not None:
            spec["metadata"] = metadata
        return cls(
            kind=kind,
            driver=driver,
            spec=spec,
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
    manifest: ExperimentManifest
    artifacts: Sequence[RunArtifact] = ()
    tags: Sequence[str] = ()


class ExperimentDriver[SpecT: ExperimentSpec](Protocol):
    """Converts a serializable experiment spec into runtime PD objects."""

    kind: str
    spec_model: type[SpecT]
    driver_path: str

    def prepare(
        self,
        spec: SpecT,
        *,
        device: str,
        dist_state: DistributedState | None = None,
    ) -> PreparedExperiment: ...

    def load_target(self, spec: SpecT, *, run_dir: Path | None = None) -> PDTarget: ...

    def build_dataloaders(
        self,
        spec: SpecT,
        *,
        seed: int | None = None,
        train_batch_size: int,
        eval_batch_size: int,
        dist_state: DistributedState | None = None,
        device: str = "cpu",
        run_dir: Path | None = None,
    ) -> tuple[DataLoader[Any], DataLoader[Any]]: ...

    def display_name(self, spec: SpecT) -> str: ...


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


def parse_driver_spec(manifest: ExperimentManifest) -> ExperimentSpec:
    """Parse a manifest's raw spec with its registered driver."""
    if manifest.driver is None:
        return ExperimentSpec.model_validate(manifest.spec)
    driver = load_driver(manifest.driver)
    return driver.spec_model.model_validate(manifest.spec)
