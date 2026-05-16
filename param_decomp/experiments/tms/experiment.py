"""TMS PD experiment: serializable config, target loading, dataloaders, and driver."""

from pathlib import Path
from typing import ClassVar, Literal

import torch
from pydantic import Field
from torch import Tensor

from param_decomp.base_config import BaseConfig
from param_decomp.experiments.driver import BuiltTarget, ExperimentConfig
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
from param_decomp.types import Probability
from param_decomp.utils.data_utils import DatasetGeneratedDataLoader, SparseFeatureDataset
from param_decomp.utils.distributed_utils import DistributedState

TARGET_MODEL_FILENAME = "target_model.pth"
TARGET_TRAIN_CONFIG_FILENAME = "target_train_config.yaml"


class TMSTargetConfig(BaseConfig):
    """Path to the trained TMS target run."""

    run_path: str = Field(..., description="Local or wandb path to a TMS pretrain run.")


class TMSDataConfig(BaseConfig):
    """Synthetic-feature dataset settings for TMS PD."""

    feature_probability: Probability
    data_generation_type: Literal["exactly_one_active", "at_least_zero_active"] = (
        "at_least_zero_active"
    )


class TMSExperimentConfig(ExperimentConfig):
    target: TMSTargetConfig
    data: TMSDataConfig


def _load_train_config(config: TMSExperimentConfig, run_dir: Path | None) -> TMSTrainConfig:
    if run_dir is None:
        return TMSTargetRunInfo.from_path(config.target.run_path).config

    train_config_path = run_dir / TARGET_TRAIN_CONFIG_FILENAME
    if not train_config_path.exists():
        raise FileNotFoundError(
            f"Saved TMS PD run is missing bundled target train config: {train_config_path}"
        )
    return TMSTrainConfig.from_file(train_config_path)


def _tied_weights(target_model: TMSModel) -> list[tuple[str, str]] | None:
    return [("linear1", "linear2")] if target_model.config.tied_weights else None


def _make_pd_target(target_model: TMSModel) -> PDTarget:
    target_model.eval()
    return PDTarget(
        model=target_model,
        run_batch=run_batch_first_element,
        reconstruction_loss=recon_loss_mse,
        tied_weights=_tied_weights(target_model),
    )


class Driver:
    name: ClassVar[str] = "tms"
    config_type: ClassVar[type[TMSExperimentConfig]] = TMSExperimentConfig

    def build_target(self, config: TMSExperimentConfig) -> BuiltTarget:
        run_info = TMSTargetRunInfo.from_path(config.target.run_path)
        target_model = TMSModel.from_run_info(run_info)
        return BuiltTarget(
            target=_make_pd_target(target_model),
            artifacts={
                TARGET_MODEL_FILENAME: target_model.state_dict(),
                TARGET_TRAIN_CONFIG_FILENAME: run_info.config.model_dump(mode="json"),
            },
        )

    def load_target(self, config: TMSExperimentConfig, run_dir: Path) -> PDTarget:
        bundled_weights = run_dir / TARGET_MODEL_FILENAME
        if not bundled_weights.exists():
            raise FileNotFoundError(
                f"Saved TMS PD run is missing bundled target weights: {bundled_weights}"
            )
        train_config = _load_train_config(config, run_dir)
        target_model = TMSModel(train_config.tms_model_config)
        target_model.load_state_dict(
            torch.load(bundled_weights, weights_only=True, map_location="cpu")
        )
        if target_model.config.tied_weights:
            target_model.tie_weights_()
        return _make_pd_target(target_model)

    def build_dataloaders(
        self,
        config: TMSExperimentConfig,
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
        dataset = SparseFeatureDataset(
            n_features=train_config.tms_model_config.n_features,
            feature_probability=config.data.feature_probability,
            device=device,
            data_generation_type=config.data.data_generation_type,
            value_range=(0.0, 1.0),
            synced_inputs=train_config.synced_inputs,
        )
        train_loader = DatasetGeneratedDataLoader(
            dataset, batch_size=train_batch_size, shuffle=False
        )
        eval_loader = DatasetGeneratedDataLoader(dataset, batch_size=eval_batch_size, shuffle=False)
        return train_loader, eval_loader
