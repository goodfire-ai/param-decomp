"""LM experiment config — top of the per-package DAG.

Imports the data classes (`configs.py`) plus the loader (`target.py`) and the
dataloader builder (`data.py`) at module level. Nothing else in `experiments/lm/`
depends on this file, so there is no cycle.
"""

from typing import Any, Literal, override

from torch.utils.data import DataLoader

from param_decomp.experiments._base import BaseExperimentConfig, LoadedTarget
from param_decomp.experiments.lm.configs import LMDataConfig, LMTargetConfig
from param_decomp.experiments.lm.data import build_lm_dataloaders
from param_decomp.experiments.lm.target import load_lm_target
from param_decomp.utils.distributed_utils import DistributedState


class LMExperimentConfig(BaseExperimentConfig):
    kind: Literal["lm"] = "lm"
    target: LMTargetConfig
    data: LMDataConfig

    @override
    def load_target(self) -> LoadedTarget:
        # LM never bundles target weights into the PD run dir (Llama checkpoints are
        # large and already addressable by HF id / pretrain wandb path), so
        # `target_train_config` is always None — `run_pd` skips the bundling path.
        return LoadedTarget(target=load_lm_target(self.target))

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
        return build_lm_dataloaders(
            self.data,
            seed=seed,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            dist_state=dist_state,
        )

    @override
    def display_name(self) -> str:
        return f"LM: {self.target.model_class.rsplit('.', 1)[-1]} on {self.data.dataset_name}"
