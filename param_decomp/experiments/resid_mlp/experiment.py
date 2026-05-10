"""Residual MLP experiment config — top of the per-package DAG."""

from typing import Any, Literal, override

from torch.utils.data import DataLoader

from param_decomp.experiments._base import BaseExperimentConfig, LoadedTarget
from param_decomp.experiments.resid_mlp.configs import ResidMLPDataConfig, ResidMLPTargetConfig
from param_decomp.experiments.resid_mlp.data import build_resid_mlp_dataloaders
from param_decomp.experiments.resid_mlp.models import ResidMLPTargetRunInfo
from param_decomp.experiments.resid_mlp.target import load_resid_mlp_target
from param_decomp.utils.distributed_utils import DistributedState


class ResidMLPExperimentConfig(BaseExperimentConfig):
    kind: Literal["resid_mlp"] = "resid_mlp"
    target: ResidMLPTargetConfig
    data: ResidMLPDataConfig

    @override
    def load_target(self) -> LoadedTarget:
        target, run_info = load_resid_mlp_target(self.target)
        return LoadedTarget(target=target, target_train_config=run_info.config)

    @override
    def build_dataloaders(
        self,
        *,
        seed: int,
        train_batch_size: int,
        eval_batch_size: int,
        dist_state: DistributedState | None = None,
        device: str = "cpu",
    ) -> tuple[DataLoader[Any], DataLoader[Any]]:
        run_info = ResidMLPTargetRunInfo.from_path(self.target.run_path)
        return build_resid_mlp_dataloaders(
            self.data,
            run_info,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            device=device,
        )

    @override
    def display_name(self) -> str:
        return f"ResidMLP: {self.target.run_path}"
