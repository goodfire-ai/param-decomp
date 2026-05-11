"""Residual MLP PD experiment.

This file is the full runtime definition for the ResidMLP experiment: serializable config,
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
from param_decomp.experiments.resid_mlp.models import (
    ResidMLP,
    ResidMLPTargetRunInfo,
    ResidMLPTrainConfig,
)
from param_decomp.experiments.resid_mlp.resid_mlp_dataset import ResidMLPDataset
from param_decomp.experiments.runner import main as run_with_driver
from param_decomp.models.batch_and_loss_fns import (
    PDTarget,
    recon_loss_mse,
    run_batch_first_element,
)
from param_decomp.param_decomp_types import Probability
from param_decomp.utils.data_utils import DatasetGeneratedDataLoader
from param_decomp.utils.distributed_utils import DistributedState

TARGET_MODEL_FILENAME = "target_model.pth"
TARGET_TRAIN_CONFIG_FILENAME = "target_train_config.yaml"
LABEL_COEFFS_FILENAME = "label_coeffs.json"


class ResidMLPTargetConfig(BaseConfig):
    """Path to the trained ResidMLP target run."""

    run_path: str = Field(
        ...,
        description="Local or wandb path to a ResidMLP pretrain run.",
    )


class ResidMLPDataConfig(BaseConfig):
    """Synthetic-feature dataset settings for ResidMLP PD."""

    feature_probability: Probability
    data_generation_type: Literal[
        "exactly_one_active", "exactly_two_active", "at_least_zero_active"
    ] = "at_least_zero_active"


class ResidMLPExperimentConfig(ExperimentSpec):
    kind: str = "resid_mlp"
    target: ResidMLPTargetConfig
    data: ResidMLPDataConfig


# Target and data builders


def load_resid_mlp_target(
    target_cfg: ResidMLPTargetConfig,
) -> tuple[PDTarget, ResidMLPTargetRunInfo]:
    """Load ResidMLP target weights and wrap them in an MSE reconstruction target."""
    run_info = ResidMLPTargetRunInfo.from_path(target_cfg.run_path)
    target_model = ResidMLP.from_run_info(run_info)
    target_model.eval()

    target = PDTarget(
        model=target_model,
        run_batch=run_batch_first_element,
        reconstruction_loss=recon_loss_mse,
        name="resid_mlp",
    )
    return target, run_info


def build_resid_mlp_dataloaders(
    data_cfg: ResidMLPDataConfig,
    target_train_config: ResidMLPTrainConfig,
    *,
    train_batch_size: int,
    eval_batch_size: int,
    device: str,
) -> tuple[
    DatasetGeneratedDataLoader[tuple[Tensor, Tensor]],
    DatasetGeneratedDataLoader[tuple[Tensor, Tensor]],
]:
    dataset = ResidMLPDataset(
        n_features=target_train_config.resid_mlp_model_config.n_features,
        feature_probability=data_cfg.feature_probability,
        device=device,
        calc_labels=False,
        label_type=None,
        act_fn_name=None,
        label_fn_seed=None,
        label_coeffs=None,
        data_generation_type=data_cfg.data_generation_type,
        synced_inputs=target_train_config.synced_inputs,
    )
    train_loader = DatasetGeneratedDataLoader(dataset, batch_size=train_batch_size, shuffle=False)
    eval_loader = DatasetGeneratedDataLoader(dataset, batch_size=eval_batch_size, shuffle=False)
    return train_loader, eval_loader


def _bundled_train_config(run_dir: Path | None) -> ResidMLPTrainConfig | None:
    if run_dir is None:
        return None
    path = run_dir / TARGET_TRAIN_CONFIG_FILENAME
    if not path.exists():
        return None
    return ResidMLPTrainConfig.from_file(path)


def _load_train_config(
    target_cfg: ResidMLPTargetConfig, run_dir: Path | None = None
) -> ResidMLPTrainConfig:
    bundled = _bundled_train_config(run_dir)
    if bundled is not None:
        return bundled

    return ResidMLPTargetRunInfo.from_path(target_cfg.run_path).config


# Driver and CLI


class ResidMLPDriver:
    kind: ClassVar[str] = "resid_mlp"
    spec_model: ClassVar[type[ResidMLPExperimentConfig]] = ResidMLPExperimentConfig
    driver_path: ClassVar[str] = "param_decomp.experiments.resid_mlp.experiment:DRIVER"

    def prepare(
        self,
        spec: ResidMLPExperimentConfig,
        *,
        device: str,
        dist_state: DistributedState | None = None,
    ) -> PreparedExperiment:
        _ = dist_state
        target, run_info = load_resid_mlp_target(spec.target)
        target.model.to(device)
        train_loader, eval_loader = build_resid_mlp_dataloaders(
            spec.data,
            run_info.config,
            train_batch_size=spec.pd.batch_size,
            eval_batch_size=spec.pd.eval_batch_size,
            device=device,
        )
        artifacts = (
            RunArtifact(TARGET_MODEL_FILENAME, target.model.state_dict()),
            RunArtifact(TARGET_TRAIN_CONFIG_FILENAME, run_info.config.model_dump(mode="json")),
            RunArtifact(LABEL_COEFFS_FILENAME, run_info.label_coeffs.detach().cpu().tolist()),
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

    def load_target(
        self, spec: ResidMLPExperimentConfig, *, run_dir: Path | None = None
    ) -> PDTarget:
        if run_dir is None or not (run_dir / TARGET_MODEL_FILENAME).exists():
            return load_resid_mlp_target(spec.target)[0]

        train_config = _load_train_config(spec.target, run_dir)
        target_model = ResidMLP(train_config.resid_mlp_model_config)
        target_model.load_state_dict(
            torch.load(run_dir / TARGET_MODEL_FILENAME, weights_only=True, map_location="cpu")
        )
        target_model.eval()
        return PDTarget(
            model=target_model,
            run_batch=run_batch_first_element,
            reconstruction_loss=recon_loss_mse,
            name=self.kind,
        )

    def build_dataloaders(
        self,
        spec: ResidMLPExperimentConfig,
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
        return build_resid_mlp_dataloaders(
            spec.data,
            train_config,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            device=device,
        )

    def display_name(self, spec: ResidMLPExperimentConfig) -> str:
        return f"ResidMLP: {spec.target.run_path}"


DRIVER = ResidMLPDriver()


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
