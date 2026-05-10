"""LM experiment driver."""

from pathlib import Path
from typing import Any, ClassVar

from torch.utils.data import DataLoader

from param_decomp.experiments.driver import (
    ExperimentManifest,
    PreparedExperiment,
)
from param_decomp.experiments.lm.data import build_lm_dataloaders
from param_decomp.experiments.lm.experiment import LMExperimentConfig
from param_decomp.experiments.lm.target import load_lm_target
from param_decomp.models.batch_and_loss_fns import PDTarget
from param_decomp.utils.distributed_utils import DistributedState


class LMDriver:
    kind: ClassVar[str] = "lm"
    spec_model: ClassVar[type[LMExperimentConfig]] = LMExperimentConfig
    driver_path: ClassVar[str] = "param_decomp.experiments.lm.driver:DRIVER"

    def prepare(
        self,
        spec: LMExperimentConfig,
        *,
        device: str,
        dist_state: DistributedState | None = None,
    ) -> PreparedExperiment:
        target = self.load_target(spec)
        train_loader, eval_loader = self.build_dataloaders(
            spec,
            seed=spec.pd.seed,
            train_batch_size=spec.pd.batch_size,
            eval_batch_size=spec.pd.eval_batch_size,
            dist_state=dist_state,
            device=device,
        )
        manifest = ExperimentManifest.from_spec(spec, driver=self.driver_path)
        return PreparedExperiment(
            pd=spec.pd,
            target=target,
            train_loader=train_loader,
            eval_loader=eval_loader,
            manifest=manifest,
            tags=(self.kind,),
        )

    def load_target(self, spec: LMExperimentConfig, *, run_dir: Path | None = None) -> PDTarget:
        _ = run_dir
        return load_lm_target(spec.target)

    def build_dataloaders(
        self,
        spec: LMExperimentConfig,
        *,
        seed: int,
        train_batch_size: int,
        eval_batch_size: int,
        dist_state: DistributedState | None = None,
        device: str = "cpu",
        run_dir: Path | None = None,
    ) -> tuple[DataLoader[Any], DataLoader[Any]]:
        _ = device, run_dir
        return build_lm_dataloaders(
            spec.data,
            seed=seed,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            dist_state=dist_state,
        )

    def display_name(self, spec: LMExperimentConfig) -> str:
        return f"LM: {spec.target.model_class.rsplit('.', 1)[-1]} on {spec.data.dataset_name}"


DRIVER = LMDriver()
