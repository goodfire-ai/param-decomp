"""TMS PD experiment: serializable config, target loading, dataloaders, and driver."""

from typing import ClassVar, Literal, override

from pydantic import Field
from torch import Tensor

from param_decomp.base_config import BaseConfig
from param_decomp.experiments.driver import ExperimentDriver
from param_decomp.experiments.tms.models import TMSModel, TMSTargetRunInfo
from param_decomp.models.batch_and_loss_fns import (
    PDTarget,
    recon_loss_mse,
    run_batch_first_element,
)
from param_decomp.run import RunConfig
from param_decomp.types import Probability
from param_decomp.utils.data_utils import DatasetGeneratedDataLoader, SparseFeatureDataset
from param_decomp.utils.distributed_utils import DistributedState


class TMSTargetConfig(BaseConfig):
    """Path to the trained TMS target run."""

    run_path: str = Field(..., description="Local or wandb path to a TMS pretrain run.")


class TMSDataConfig(BaseConfig):
    """Synthetic-feature dataset settings for TMS PD."""

    feature_probability: Probability
    data_generation_type: Literal["exactly_one_active", "at_least_zero_active"] = (
        "at_least_zero_active"
    )


class TMSRunConfig(RunConfig):
    target: TMSTargetConfig
    data: TMSDataConfig


class Driver(ExperimentDriver[TMSRunConfig]):
    name: ClassVar[str] = "tms"

    @property
    @override
    def config_type(self) -> type[TMSRunConfig]:
        return TMSRunConfig

    @override
    def build_target(self, run_cfg: TMSRunConfig) -> PDTarget:
        run_info = TMSTargetRunInfo.from_path(run_cfg.target.run_path)
        target_model = TMSModel.from_run_info(run_info)
        target_model.eval()
        tied_weights = [("linear1", "linear2")] if target_model.config.tied_weights else None
        return PDTarget(
            model=target_model,
            run_batch=run_batch_first_element,
            reconstruction_loss=recon_loss_mse,
            tied_weights=tied_weights,
        )

    def _build_dataset(self, run_cfg: TMSRunConfig, device: str) -> SparseFeatureDataset:
        train_config = TMSTargetRunInfo.from_path(run_cfg.target.run_path).config
        return SparseFeatureDataset(
            n_features=train_config.tms_model_config.n_features,
            feature_probability=run_cfg.data.feature_probability,
            device=device,
            data_generation_type=run_cfg.data.data_generation_type,
            value_range=(0.0, 1.0),
            synced_inputs=train_config.synced_inputs,
        )

    @override
    def build_train_loader(
        self,
        run_cfg: TMSRunConfig,
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
        run_cfg: TMSRunConfig,
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
