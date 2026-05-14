"""Residual MLP PD experiment: serializable config, target loading, dataloaders, and driver."""

from pathlib import Path
from typing import Any, ClassVar, Literal

import torch
from pydantic import Field
from torch import Tensor

from param_decomp.base_config import BaseConfig
from param_decomp.experiments.driver import ExperimentConfig
from param_decomp.experiments.resid_mlp.models import (
    ResidMLP,
    ResidMLPTargetRunInfo,
    ResidMLPTrainConfig,
)
from param_decomp.experiments.resid_mlp.resid_mlp_dataset import ResidMLPDataset
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

    run_path: str = Field(..., description="Local or wandb path to a ResidMLP pretrain run.")


class ResidMLPDataConfig(BaseConfig):
    """Synthetic-feature dataset settings for ResidMLP PD."""

    feature_probability: Probability
    data_generation_type: Literal[
        "exactly_one_active", "exactly_two_active", "at_least_zero_active"
    ] = "at_least_zero_active"


class ResidMLPExperimentConfig(ExperimentConfig):
    target: ResidMLPTargetConfig
    data: ResidMLPDataConfig


def _load_train_config(
    config: ResidMLPExperimentConfig, run_dir: Path | None
) -> ResidMLPTrainConfig:
    if run_dir is None:
        return ResidMLPTargetRunInfo.from_path(config.target.run_path).config

    train_config_path = run_dir / TARGET_TRAIN_CONFIG_FILENAME
    if not train_config_path.exists():
        raise FileNotFoundError(
            f"Saved ResidMLP PD run is missing bundled target train config: {train_config_path}"
        )
    return ResidMLPTrainConfig.from_file(train_config_path)


class Driver:
    name: ClassVar[str] = "resid_mlp"
    config_type: ClassVar[type[ResidMLPExperimentConfig]] = ResidMLPExperimentConfig

    def build_target(
        self, config: ResidMLPExperimentConfig, *, run_dir: Path | None = None
    ) -> PDTarget:
        if run_dir is not None:
            bundled_weights = run_dir / TARGET_MODEL_FILENAME
            if not bundled_weights.exists():
                raise FileNotFoundError(
                    f"Saved ResidMLP PD run is missing bundled target weights: {bundled_weights}"
                )
            train_config = _load_train_config(config, run_dir)
            target_model = ResidMLP(train_config.resid_mlp_model_config)
            target_model.load_state_dict(
                torch.load(bundled_weights, weights_only=True, map_location="cpu")
            )
        else:
            run_info = ResidMLPTargetRunInfo.from_path(config.target.run_path)
            target_model = ResidMLP.from_run_info(run_info)

        target_model.eval()
        return PDTarget(
            model=target_model,
            run_batch=run_batch_first_element,
            reconstruction_loss=recon_loss_mse,
        )

    def build_dataloaders(
        self,
        config: ResidMLPExperimentConfig,
        *,
        train_batch_size: int,
        eval_batch_size: int,
        dist_state: DistributedState | None = None,
        device: str = "cpu",
        run_dir: Path | None = None,
    ) -> tuple[
        DatasetGeneratedDataLoader[tuple[Tensor, Tensor]],
        DatasetGeneratedDataLoader[tuple[Tensor, Tensor]],
    ]:
        _ = dist_state
        train_config = _load_train_config(config, run_dir)
        dataset = ResidMLPDataset(
            n_features=train_config.resid_mlp_model_config.n_features,
            feature_probability=config.data.feature_probability,
            device=device,
            calc_labels=False,
            label_type=None,
            act_fn_name=None,
            label_fn_seed=None,
            label_coeffs=None,
            data_generation_type=config.data.data_generation_type,
            synced_inputs=train_config.synced_inputs,
        )
        train_loader = DatasetGeneratedDataLoader(
            dataset, batch_size=train_batch_size, shuffle=False
        )
        eval_loader = DatasetGeneratedDataLoader(dataset, batch_size=eval_batch_size, shuffle=False)
        return train_loader, eval_loader

    def artifacts(self, config: ResidMLPExperimentConfig, target: PDTarget) -> dict[str, Any]:
        run_info = ResidMLPTargetRunInfo.from_path(config.target.run_path)
        return {
            TARGET_MODEL_FILENAME: target.model.state_dict(),
            TARGET_TRAIN_CONFIG_FILENAME: run_info.config.model_dump(mode="json"),
            LABEL_COEFFS_FILENAME: run_info.label_coeffs.detach().cpu().tolist(),
        }
