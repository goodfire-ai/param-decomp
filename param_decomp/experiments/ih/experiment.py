"""Induction Head experiment config — top of the per-package DAG."""

from typing import Any, Literal, override

from torch.utils.data import DataLoader

from param_decomp.experiments._base import BaseExperimentConfig, LoadedTarget
from param_decomp.experiments.ih.configs import IHDataConfig, IHTargetConfig
from param_decomp.experiments.ih.data import build_ih_dataloaders
from param_decomp.experiments.ih.model import InductionModelTargetRunInfo
from param_decomp.experiments.ih.target import load_ih_target
from param_decomp.utils.distributed_utils import DistributedState


class IHExperimentConfig(BaseExperimentConfig):
    kind: Literal["ih"] = "ih"
    target: IHTargetConfig
    data: IHDataConfig

    @override
    def load_target(self) -> LoadedTarget:
        target, run_info = load_ih_target(self.target)
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
        run_info = InductionModelTargetRunInfo.from_path(self.target.run_path)
        return build_ih_dataloaders(
            self.data,
            run_info,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            device=device,
        )

    @override
    def display_name(self) -> str:
        return f"IH: {self.target.run_path}"
