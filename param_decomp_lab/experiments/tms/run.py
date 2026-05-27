"""TMS PD experiment: YAML -> `Trainer` glue, plus the `SavedTMSRun` reload class.

Run via `pd-tms path/to/config.yaml`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import fire
from pydantic import Field
from torch.utils.data import DataLoader

from param_decomp.base_config import BaseConfig, Probability
from param_decomp.batch_and_loss_fns import RunBatch
from param_decomp.component_model import ComponentModel
from param_decomp.distributed import DistributedState
from param_decomp.log import logger
from param_decomp.optimize import EvalLoop, Trainer
from param_decomp_lab.batch_and_loss_fns import recon_loss_mse, run_batch_first_element
from param_decomp_lab.component_model_io import load_component_model
from param_decomp_lab.distributed import get_device
from param_decomp_lab.eval_metrics import EVAL_METRIC_CLASSES
from param_decomp_lab.experiments.tms.data import SparseFeatureDataset
from param_decomp_lab.experiments.tms.models import TMSModel, TMSTargetRunInfo
from param_decomp_lab.experiments.utils import (
    EXPERIMENT_CONFIG_FILENAME,
    ExperimentConfig,
    init_pd_run,
)
from param_decomp_lab.infra.paths import ModelPath
from param_decomp_lab.infra.run_files import resolve_run_files
from param_decomp_lab.seed import set_seed


class TMSTargetConfig(BaseConfig):
    run_path: str = Field(..., description="Local or wandb path to a TMS pretrain run.")


class TMSDataConfig(BaseConfig):
    """Synthetic-feature dataset settings for TMS PD."""

    feature_probability: Probability
    data_generation_type: Literal["exactly_one_active", "at_least_zero_active"] = (
        "at_least_zero_active"
    )


class TMSExperimentConfig(ExperimentConfig[TMSTargetConfig, TMSDataConfig]):
    pass


def build_target(target_cfg: TMSTargetConfig) -> TMSModel:
    """Load the pretrained TMS target model in eval mode."""
    run_info = TMSTargetRunInfo.from_path(target_cfg.run_path)
    target_model = TMSModel.from_run_info(run_info)
    target_model.eval()
    return target_model


def build_tms_loader(
    target_cfg: TMSTargetConfig,
    data_cfg: TMSDataConfig,
    *,
    split: Literal["train", "eval"],
    device: str,
    batch_size: int,
    dist_state: DistributedState | None = None,
    seed: int | None = None,
) -> DataLoader[Any]:
    """Synthetic `SparseFeatureDataset` loader for TMS.

    The dataset is infinite, so `split` / `dist_state` / `seed` are ignored — train and
    eval loaders are identical.
    """
    del split, dist_state, seed
    train_config = TMSTargetRunInfo.from_path(target_cfg.run_path).config
    dataset = SparseFeatureDataset(
        n_features=train_config.tms_model_config.n_features,
        feature_probability=data_cfg.feature_probability,
        device=device,
        batch_size=batch_size,
        data_generation_type=data_cfg.data_generation_type,
        value_range=(0.0, 1.0),
        synced_inputs=train_config.synced_inputs,
    )
    return DataLoader(dataset, batch_size=None)


def make_run_batch(target_cfg: TMSTargetConfig) -> RunBatch:
    """`RunBatch` for TMS: unwraps the `(inputs, labels)` tuple."""
    del target_cfg
    return run_batch_first_element


def _tied_weights_for(target_model: TMSModel) -> list[tuple[str, str]] | None:
    return [("linear1", "linear2")] if target_model.config.tied_weights else None


@dataclass(frozen=True)
class SavedTMSRun:
    """Handle to a completed TMS PD run on disk or in W&B."""

    cfg: TMSExperimentConfig
    checkpoint_path: Path

    @classmethod
    def from_path(cls, path: ModelPath) -> "SavedTMSRun":
        """Resolve a run directory or W&B path into a fully-validated `SavedTMSRun`."""
        files = resolve_run_files(
            path, config_filename=EXPERIMENT_CONFIG_FILENAME, checkpoint_prefix="model"
        )
        return cls(
            cfg=TMSExperimentConfig.from_file(files.config_path),
            checkpoint_path=files.checkpoint_path,
        )

    def load_model(self) -> ComponentModel:
        return load_component_model(
            pd_config=self.cfg.pd,
            checkpoint_path=self.checkpoint_path,
            target_model=build_target(self.cfg.target),
            run_batch=make_run_batch(self.cfg.target),
        )


def main(
    config_path: str | Path,
    *,
    group: str | None = None,
    tags: str | None = None,
) -> None:
    """Run a TMS PD experiment end-to-end from a YAML config. `group` / `tags` are wandb-only."""
    cfg = TMSExperimentConfig.from_file(config_path)

    set_seed(cfg.pd.seed)
    device = get_device()
    logger.info(f"Using device: {device}")

    target_model = build_target(cfg.target).to(device)
    cfg = cfg.model_copy(
        update={
            "pd": cfg.pd.model_copy(update={"tied_weights": _tied_weights_for(target_model)}),
            "runtime": cfg.runtime.model_copy(update={"device": device}),
        }
    )

    train_loader = build_tms_loader(
        cfg.target, cfg.data, split="train", device=device, batch_size=cfg.pd.batch_size
    )
    eval_loop = _build_eval_loop(cfg, device)

    sink = init_pd_run(cfg, group=group, tags=tags)

    try:
        trainer = Trainer(
            target_model=target_model,
            run_batch=make_run_batch(cfg.target),
            reconstruction_loss=recon_loss_mse,
            pd_config=cfg.pd,
            runtime_config=cfg.runtime,
        )
        trainer.run(train_loader, sink, cfg.cadence, eval_loop)
    finally:
        sink.finish()


def _build_eval_loop(cfg: TMSExperimentConfig, device: str) -> EvalLoop | None:
    """Build the `EvalLoop` from `cfg.eval`, or `None` when eval is disabled."""
    if cfg.eval is None:
        return None
    return EvalLoop(
        loader=build_tms_loader(
            cfg.target, cfg.data, split="eval", device=device, batch_size=cfg.eval.batch_size
        ),
        metrics=[EVAL_METRIC_CLASSES[m.type](m) for m in cfg.eval.metrics],
        n_steps=cfg.eval.n_steps,
        every=cfg.eval.every,
        slow_every=cfg.eval.slow_every,
        slow_on_first_step=cfg.eval.slow_on_first_step,
    )


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
