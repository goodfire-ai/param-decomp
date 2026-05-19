"""Residual MLP PD experiment: serializable config, target loading, dataloaders, and driver."""

from typing import ClassVar, Literal

from pydantic import Field
from torch import Tensor

from param_decomp.base_config import BaseConfig
from param_decomp.experiments.resid_mlp.models import ResidMLP, ResidMLPTargetRunInfo
from param_decomp.experiments.resid_mlp.resid_mlp_dataset import ResidMLPDataset
from param_decomp.models.batch_and_loss_fns import (
    PDTarget,
    recon_loss_mse,
    run_batch_first_element,
)
from param_decomp.run import Run
from param_decomp.types import Probability
from param_decomp.utils.data_utils import DatasetGeneratedDataLoader
from param_decomp.utils.distributed_utils import DistributedState


class ResidMLPTargetConfig(BaseConfig):
    """Path to the trained ResidMLP target run."""

    run_path: str = Field(..., description="Local or wandb path to a ResidMLP pretrain run.")


class ResidMLPDataConfig(BaseConfig):
    """Synthetic-feature dataset settings for ResidMLP PD."""

    feature_probability: Probability
    data_generation_type: Literal[
        "exactly_one_active", "exactly_two_active", "at_least_zero_active"
    ] = "at_least_zero_active"


class ResidMLPRun(Run):
    target: ResidMLPTargetConfig
    data: ResidMLPDataConfig


class Driver:
    name: ClassVar[str] = "resid_mlp"
    config_type: ClassVar[type[ResidMLPRun]] = ResidMLPRun

    def build_target(self, run: ResidMLPRun) -> PDTarget:
        run_info = ResidMLPTargetRunInfo.from_path(run.target.run_path)
        target_model = ResidMLP.from_run_info(run_info)
        target_model.eval()
        return PDTarget(
            model=target_model,
            run_batch=run_batch_first_element,
            reconstruction_loss=recon_loss_mse,
        )

    def build_dataloaders(
        self,
        run: ResidMLPRun,
        *,
        train_batch_size: int,
        eval_batch_size: int,
        dist_state: DistributedState | None = None,
        device: str = "cpu",
    ) -> tuple[
        DatasetGeneratedDataLoader[tuple[Tensor, Tensor]],
        DatasetGeneratedDataLoader[tuple[Tensor, Tensor]],
    ]:
        _ = dist_state
        train_config = ResidMLPTargetRunInfo.from_path(run.target.run_path).config
        dataset = ResidMLPDataset(
            n_features=train_config.resid_mlp_model_config.n_features,
            feature_probability=run.data.feature_probability,
            device=device,
            calc_labels=False,
            label_type=None,
            act_fn_name=None,
            label_fn_seed=None,
            label_coeffs=None,
            data_generation_type=run.data.data_generation_type,
            synced_inputs=train_config.synced_inputs,
        )
        train_loader = DatasetGeneratedDataLoader(
            dataset, batch_size=train_batch_size, shuffle=False
        )
        eval_loader = DatasetGeneratedDataLoader(dataset, batch_size=eval_batch_size, shuffle=False)
        return train_loader, eval_loader
