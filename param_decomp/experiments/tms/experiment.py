"""TMS PD experiment.

This file is the full runtime definition for the TMS experiment: serializable config,
target loading, dataloaders, driver registration, and the CLI entrypoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal

import torch
from pydantic import Field
from torch import Tensor
from torch.utils.data import DataLoader

from param_decomp.base_config import BaseConfig
from param_decomp.experiments.driver import (
    ExperimentManifest,
    ExperimentSpec,
    PreparedExperiment,
    RunArtifact,
)
from param_decomp.experiments.runner import main as run_with_driver
from param_decomp.experiments.tms.models import (
    TMSModel,
    TMSTargetRunInfo,
    TMSTrainConfig,
)
from param_decomp.models.batch_and_loss_fns import (
    PDTarget,
    recon_loss_mse,
    run_batch_first_element,
)
from param_decomp.param_decomp_types import Probability
from param_decomp.utils.data_utils import DatasetGeneratedDataLoader, SparseFeatureDataset
from param_decomp.utils.distributed_utils import DistributedState

TARGET_MODEL_FILENAME = "target_model.pth"
TARGET_TRAIN_CONFIG_FILENAME = "target_train_config.yaml"


class TMSTargetConfig(BaseConfig):
    """Path to the trained TMS target run."""

    run_path: str = Field(
        ...,
        description="Local or wandb path to a TMS pretrain run.",
    )


class TMSDataConfig(BaseConfig):
    """Synthetic-feature dataset settings for TMS PD."""

    feature_probability: Probability
    data_generation_type: Literal["exactly_one_active", "at_least_zero_active"] = (
        "at_least_zero_active"
    )


class TMSExperimentConfig(ExperimentSpec):
    kind: str = "tms"
    target: TMSTargetConfig
    data: TMSDataConfig


# Target and data builders


def _tied_weight_edges(target_model: TMSModel) -> list[tuple[str, str]] | None:
    if target_model.config.tied_weights:
        return [("linear1", "linear2")]
    return None


def load_tms_target(target_cfg: TMSTargetConfig) -> tuple[PDTarget, TMSTargetRunInfo]:
    """Load TMS target weights and wrap them in an MSE reconstruction target."""
    run_info = TMSTargetRunInfo.from_path(target_cfg.run_path)
    target_model = TMSModel.from_run_info(run_info)
    target_model.eval()

    target = PDTarget(
        model=target_model,
        run_batch=run_batch_first_element,
        reconstruction_loss=recon_loss_mse,
        tied_weights=_tied_weight_edges(target_model),
        name="tms",
    )
    return target, run_info


def build_tms_dataloaders(
    data_cfg: TMSDataConfig,
    target_train_config: TMSTrainConfig,
    *,
    train_batch_size: int,
    eval_batch_size: int,
    device: str,
) -> tuple[
    DatasetGeneratedDataLoader[tuple[Tensor, Tensor]],
    DatasetGeneratedDataLoader[tuple[Tensor, Tensor]],
]:
    dataset = SparseFeatureDataset(
        n_features=target_train_config.tms_model_config.n_features,
        feature_probability=data_cfg.feature_probability,
        device=device,
        data_generation_type=data_cfg.data_generation_type,
        value_range=(0.0, 1.0),
        synced_inputs=target_train_config.synced_inputs,
    )
    train_loader = DatasetGeneratedDataLoader(dataset, batch_size=train_batch_size, shuffle=False)
    eval_loader = DatasetGeneratedDataLoader(dataset, batch_size=eval_batch_size, shuffle=False)
    return train_loader, eval_loader


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


# Driver and CLI


class TMSDriver:
    kind: ClassVar[str] = "tms"
    spec_model: ClassVar[type[TMSExperimentConfig]] = TMSExperimentConfig
    driver_path: ClassVar[str] = "param_decomp.experiments.tms.experiment:DRIVER"

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

        return PDTarget(
            model=target_model,
            run_batch=run_batch_first_element,
            reconstruction_loss=recon_loss_mse,
            tied_weights=_tied_weight_edges(target_model),
            name=self.kind,
        )

    def build_dataloaders(
        self,
        spec: TMSExperimentConfig,
        *,
        seed: int | None = None,
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


def main(
    config_path: Path | str | None = None,
    config_json: str | None = None,
    evals_id: str | None = None,
    launch_id: str | None = None,
    sweep_params_json: str | None = None,
    run_id: str | None = None,
) -> None:
    run_with_driver(
        config_path=config_path,
        config_json=config_json,
        driver=DRIVER.driver_path,
        evals_id=evals_id,
        launch_id=launch_id,
        sweep_params_json=sweep_params_json,
        run_id=run_id,
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
