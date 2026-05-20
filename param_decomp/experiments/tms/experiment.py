"""TMS PD experiment: serializable config, target loading, dataloaders, and driver."""

from pathlib import Path
from typing import Any, ClassVar, Literal, override

from pydantic import Field
from torch import Tensor
from torch.utils.data import DataLoader

from param_decomp.base_config import BaseConfig
from param_decomp.experiments.driver import ExperimentDriver
from param_decomp.experiments.tms.models import TMSModel, TMSTargetRunInfo
from param_decomp.experiments.tms.optimize import tie_tms_component_weights_, tms_optimize
from param_decomp.models.batch_and_loss_fns import (
    PDTarget,
    recon_loss_mse,
    run_batch_first_element,
)
from param_decomp.models.component_model import ComponentModel
from param_decomp.run import RunConfig
from param_decomp.run_sink import RunSink
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
        return PDTarget(
            model=target_model,
            run_batch=run_batch_first_element,
            reconstruction_loss=recon_loss_mse,
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
        device: str,
        batch_size_override: int | None = None,
        dist_state: DistributedState | None = None,
    ) -> DatasetGeneratedDataLoader[tuple[Tensor, Tensor]]:
        del dist_state
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
        device: str,
        batch_size_override: int | None = None,
        dist_state: DistributedState | None = None,
    ) -> DatasetGeneratedDataLoader[tuple[Tensor, Tensor]]:
        del dist_state
        return DatasetGeneratedDataLoader(
            self._build_dataset(run_cfg, device),
            batch_size=batch_size_override or run_cfg.logging.eval_batch_size,
            shuffle=False,
        )

    @override
    def optimize(
        self,
        run_cfg: TMSRunConfig,
        target: PDTarget,
        train_loader: DataLoader[Any],
        eval_loader: DataLoader[Any],
        *,
        device: str,
        dist_state: DistributedState | None,
        sink: RunSink,
    ) -> None:
        assert dist_state is None, "TMS PD does not support distributed training"
        assert isinstance(target.model, TMSModel)
        tms_optimize(
            target=target,
            train_loader=train_loader,
            eval_loader=eval_loader,
            pd_config=run_cfg.pd,
            logging_config=run_cfg.logging,
            runtime_config=run_cfg.runtime,
            device=device,
            sink=sink,
            tied_weights=target.model.config.tied_weights,
        )

    @override
    def load_model(self, run_cfg: TMSRunConfig, checkpoint_path: Path) -> ComponentModel:
        target = self.build_target(run_cfg)
        assert isinstance(target.model, TMSModel)
        comp_model = ComponentModel.from_checkpoint(
            config=run_cfg.pd,
            checkpoint_path=checkpoint_path,
            target_model=target.model,
            run_batch=target.run_batch,
        )
        if target.model.config.tied_weights:
            tie_tms_component_weights_(comp_model)
        return comp_model
