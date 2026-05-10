"""Induction-head experiment driver."""

from pathlib import Path
from typing import Any, ClassVar

import torch
from torch.utils.data import DataLoader

from param_decomp.experiments.driver import (
    ExperimentManifest,
    PreparedExperiment,
    RunArtifact,
)
from param_decomp.experiments.ih.configs import (
    IHTargetConfig,
    InductionHeadsTrainConfig,
)
from param_decomp.experiments.ih.data import build_ih_dataloaders
from param_decomp.experiments.ih.experiment import IHExperimentConfig
from param_decomp.experiments.ih.model import InductionModelTargetRunInfo, InductionTransformer
from param_decomp.experiments.ih.target import load_ih_target
from param_decomp.models.batch_and_loss_fns import (
    PDTarget,
    recon_loss_kl,
    run_batch_first_element,
)
from param_decomp.utils.distributed_utils import DistributedState

TARGET_MODEL_FILENAME = "target_model.pth"
TARGET_TRAIN_CONFIG_FILENAME = "target_train_config.yaml"


def _bundled_train_config(run_dir: Path | None) -> InductionHeadsTrainConfig | None:
    if run_dir is None:
        return None
    path = run_dir / TARGET_TRAIN_CONFIG_FILENAME
    if not path.exists():
        return None
    return InductionHeadsTrainConfig.from_file(path)


def _load_train_config(
    target_cfg: IHTargetConfig, run_dir: Path | None = None
) -> InductionHeadsTrainConfig:
    bundled = _bundled_train_config(run_dir)
    if bundled is not None:
        return bundled
    return InductionModelTargetRunInfo.from_path(target_cfg.run_path).config


class IHDriver:
    kind: ClassVar[str] = "ih"
    spec_model: ClassVar[type[IHExperimentConfig]] = IHExperimentConfig
    driver_path: ClassVar[str] = "param_decomp.experiments.ih.driver:DRIVER"

    def prepare(
        self,
        spec: IHExperimentConfig,
        *,
        device: str,
        dist_state: DistributedState | None = None,
    ) -> PreparedExperiment:
        _ = dist_state
        target, run_info = load_ih_target(spec.target)
        target.model.to(device)
        train_loader, eval_loader = build_ih_dataloaders(
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

    def load_target(self, spec: IHExperimentConfig, *, run_dir: Path | None = None) -> PDTarget:
        if run_dir is None or not (run_dir / TARGET_MODEL_FILENAME).exists():
            return load_ih_target(spec.target)[0]

        train_config = _load_train_config(spec.target, run_dir)
        target_model = InductionTransformer(train_config.ih_model_config)
        target_model.load_state_dict(
            torch.load(run_dir / TARGET_MODEL_FILENAME, weights_only=True, map_location="cpu")
        )
        target_model.eval()
        return PDTarget(
            model=target_model,
            run_batch=run_batch_first_element,
            reconstruction_loss=recon_loss_kl,
            name=self.kind,
        )

    def build_dataloaders(
        self,
        spec: IHExperimentConfig,
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
        return build_ih_dataloaders(
            spec.data,
            train_config,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            device=device,
        )

    def display_name(self, spec: IHExperimentConfig) -> str:
        return f"IH: {spec.target.run_path}"


DRIVER = IHDriver()
