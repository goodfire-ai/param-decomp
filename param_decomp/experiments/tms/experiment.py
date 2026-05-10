"""TMS experiment config — top of the per-package DAG."""

from typing import Any, Literal, override

from torch.utils.data import DataLoader

from param_decomp.experiments._base import BaseExperimentConfig, LoadedTarget
from param_decomp.experiments.tms.configs import TMSDataConfig, TMSTargetConfig
from param_decomp.experiments.tms.data import build_tms_dataloaders
from param_decomp.experiments.tms.models import TMSTargetRunInfo
from param_decomp.experiments.tms.target import load_tms_target
from param_decomp.utils.distributed_utils import DistributedState


class TMSExperimentConfig(BaseExperimentConfig):
    kind: Literal["tms"] = "tms"
    target: TMSTargetConfig
    data: TMSDataConfig

    @override
    def load_target(self) -> LoadedTarget:
        target, run_info = load_tms_target(self.target)
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
        run_info = TMSTargetRunInfo.from_path(self.target.run_path)
        return build_tms_dataloaders(
            self.data,
            run_info,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            device=device,
        )

    @override
    def display_name(self) -> str:
        return f"TMS: {self.target.run_path}"
