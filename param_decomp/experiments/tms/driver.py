"""TMS experiment driver."""

from pathlib import Path
from typing import Any, ClassVar

import torch
from torch.utils.data import DataLoader

from param_decomp.experiments.driver import (
    ExperimentManifest,
    PreparedExperiment,
    RunArtifact,
)
from param_decomp.experiments.tms.configs import TMSTargetConfig, TMSTrainConfig
from param_decomp.experiments.tms.data import build_tms_dataloaders
from param_decomp.experiments.tms.experiment import TMSExperimentConfig
from param_decomp.experiments.tms.models import TMSModel, TMSTargetRunInfo
from param_decomp.experiments.tms.target import load_tms_target
from param_decomp.models.batch_and_loss_fns import PDTarget
from param_decomp.utils.distributed_utils import DistributedState

TARGET_MODEL_FILENAME = "target_model.pth"
TARGET_TRAIN_CONFIG_FILENAME = "target_train_config.yaml"


def _bundled_train_config(run_dir: Path | None) -> TMSTrainConfig | None:
    if run_dir is None:
        return None
    path = run_dir / TARGET_TRAIN_CONFIG_FILENAME
    if not path.exists():
        return None
    return TMSTrainConfig.from_file(path)


def _load_train_config(target_cfg: TMSTargetConfig, run_dir: Path | None = None) -> TMSTrainConfig:
    bundled = _bundled_train_config(run_dir)
    if bundled is not None:
        return bundled
    return TMSTargetRunInfo.from_path(target_cfg.run_path).config


class TMSDriver:
    kind: ClassVar[str] = "tms"
    spec_model: ClassVar[type[TMSExperimentConfig]] = TMSExperimentConfig
    driver_path: ClassVar[str] = "param_decomp.experiments.tms.driver:DRIVER"

    def prepare(
        self,
        spec: TMSExperimentConfig,
        *,
        device: str,
        dist_state: DistributedState | None = None,
    ) -> PreparedExperiment:
        _ = dist_state
        target, run_info = load_tms_target(spec.target)
        target.model.to(device)
        train_loader, eval_loader = build_tms_dataloaders(
            spec.data,
            run_info.config,
            train_batch_size=spec.pd.batch_size,
            eval_batch_size=spec.pd.eval_batch_size,
            device=device,
        )
        artifacts = (
            RunArtifact(TARGET_MODEL_FILENAME, target.model.state_dict()),
            RunArtifact(TARGET_TRAIN_CONFIG_FILENAME, run_info.config.model_dump(mode="json")),
        )
        manifest = ExperimentManifest.from_spec(
            spec,
            driver=self.driver_path,
            artifact_filenames=[artifact.filename for artifact in artifacts],
        )
        return PreparedExperiment(
            pd=spec.pd,
            target=target,
            train_loader=train_loader,
            eval_loader=eval_loader,
            manifest=manifest,
            artifacts=artifacts,
            tags=(self.kind,),
        )

    def load_target(self, spec: TMSExperimentConfig, *, run_dir: Path | None = None) -> PDTarget:
        if run_dir is None or not (run_dir / TARGET_MODEL_FILENAME).exists():
            return load_tms_target(spec.target)[0]

        train_config = _load_train_config(spec.target, run_dir)
        target_model = TMSModel(train_config.tms_model_config)
        target_model.load_state_dict(
            torch.load(run_dir / TARGET_MODEL_FILENAME, weights_only=True, map_location="cpu")
        )
        target_model.eval()
        if target_model.config.tied_weights:
            target_model.tie_weights_()

        tied_weights: list[tuple[str, str]] | None = None
        if target_model.config.tied_weights:
            tied_weights = [("linear1", "linear2")]

        from param_decomp.models.batch_and_loss_fns import (
            recon_loss_mse,
            run_batch_first_element,
        )

        return PDTarget(
            model=target_model,
            run_batch=run_batch_first_element,
            reconstruction_loss=recon_loss_mse,
            tied_weights=tied_weights,
            name=self.kind,
        )

    def build_dataloaders(
        self,
        spec: TMSExperimentConfig,
        *,
        seed: int,
        train_batch_size: int,
        eval_batch_size: int,
        dist_state: DistributedState | None = None,
        device: str = "cpu",
        run_dir: Path | None = None,
    ) -> tuple[DataLoader[Any], DataLoader[Any]]:
        _ = seed, dist_state
        train_config = _load_train_config(spec.target, run_dir)
        return build_tms_dataloaders(
            spec.data,
            train_config,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            device=device,
        )

    def display_name(self, spec: TMSExperimentConfig) -> str:
        return f"TMS: {spec.target.run_path}"


DRIVER = TMSDriver()
