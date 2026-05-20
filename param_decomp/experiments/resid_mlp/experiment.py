"""Residual MLP PD experiment: serializable config, target loading, dataloaders, and driver."""

from typing import ClassVar, Literal, override

from pydantic import Field
from torch import Tensor

from param_decomp.base_config import BaseConfig
from param_decomp.experiments.driver import ExperimentDriver
from param_decomp.experiments.resid_mlp.models import ResidMLP, ResidMLPTargetRunInfo
from param_decomp.experiments.resid_mlp.resid_mlp_dataset import ResidMLPDataset
from param_decomp.models.batch_and_loss_fns import PDTarget, recon_loss_mse, run_batch_first_element
from param_decomp.run import RunConfig
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


class ResidMLPRunConfig(RunConfig):
    target: ResidMLPTargetConfig
    data: ResidMLPDataConfig


class Driver(ExperimentDriver[ResidMLPRunConfig]):
    name: ClassVar[str] = "resid_mlp"

    @property
    @override
    def config_type(self) -> type[ResidMLPRunConfig]:
        return ResidMLPRunConfig

    @override
    def build_target(self, run_cfg: ResidMLPRunConfig) -> PDTarget:
        run_info = ResidMLPTargetRunInfo.from_path(run_cfg.target.run_path)
        target_model = ResidMLP.from_run_info(run_info)
        target_model.eval()
        return PDTarget(
            model=target_model,
            run_batch=run_batch_first_element,
            reconstruction_loss=recon_loss_mse,
        )

    def _build_dataset(self, run_cfg: ResidMLPRunConfig, device: str) -> ResidMLPDataset:
        train_config = ResidMLPTargetRunInfo.from_path(run_cfg.target.run_path).config
        return ResidMLPDataset(
            n_features=train_config.resid_mlp_model_config.n_features,
            feature_probability=run_cfg.data.feature_probability,
            device=device,
            calc_labels=False,
            label_type=None,
            act_fn_name=None,
            label_fn_seed=None,
            label_coeffs=None,
            data_generation_type=run_cfg.data.data_generation_type,
            synced_inputs=train_config.synced_inputs,
        )

    @override
    def build_train_loader(
        self,
        run_cfg: ResidMLPRunConfig,
        *,
        batch_size_override: int | None = None,
        dist_state: DistributedState | None = None,
        device: str = "cpu",
    ) -> DatasetGeneratedDataLoader[tuple[Tensor, Tensor]]:
        _ = dist_state
        return DatasetGeneratedDataLoader(
            self._build_dataset(run_cfg, device),
            batch_size=batch_size_override or run_cfg.pd.batch_size,
            shuffle=False,
        )

    @override
    def build_eval_loader(
        self,
        run_cfg: ResidMLPRunConfig,
        *,
        batch_size_override: int | None = None,
        dist_state: DistributedState | None = None,
        device: str = "cpu",
    ) -> DatasetGeneratedDataLoader[tuple[Tensor, Tensor]]:
        _ = dist_state
        return DatasetGeneratedDataLoader(
            self._build_dataset(run_cfg, device),
            batch_size=batch_size_override or run_cfg.logging.eval_batch_size,
            shuffle=False,
        )
