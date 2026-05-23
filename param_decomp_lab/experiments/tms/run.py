"""TMS PD experiment: YAML -> `optimize()` glue.

`SavedRun` rebuilds saved TMS runs by accessing this module's dispatch interface
(`TargetConfig`, `DataConfig`, `build_target`, `build_train_loader`, `build_eval_loader`,
`make_run_batch`). Run via ``pd-tms path/to/config.yaml``.
"""

from pathlib import Path
from typing import Literal

import fire
from pydantic import Field
from torch import Tensor

from param_decomp.base_config import BaseConfig, Probability
from param_decomp.batch_and_loss_fns import RunBatch
from param_decomp.distributed import DistributedState
from param_decomp.log import logger
from param_decomp.optimize import optimize
from param_decomp_lab.batch_and_loss_fns import recon_loss_mse, run_batch_first_element
from param_decomp_lab.distributed import get_device
from param_decomp_lab.eval_metrics import EVAL_METRIC_CLASSES
from param_decomp_lab.experiments.synthetic_data import (
    DatasetGeneratedDataLoader,
    SparseFeatureDataset,
)
from param_decomp_lab.experiments.tms.models import TMSModel, TMSTargetRunInfo
from param_decomp_lab.experiments.utils import ExperimentConfig, save_run_meta
from param_decomp_lab.infra.run_files import generate_run_id
from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR
from param_decomp_lab.run_sink import RunSink
from param_decomp_lab.seed import set_seed


class TargetConfig(BaseConfig):
    """Path to the trained TMS target run."""

    run_path: str = Field(..., description="Local or wandb path to a TMS pretrain run.")


class DataConfig(BaseConfig):
    """Synthetic-feature dataset settings for TMS PD."""

    feature_probability: Probability
    data_generation_type: Literal["exactly_one_active", "at_least_zero_active"] = (
        "at_least_zero_active"
    )


class TMSExperimentConfig(ExperimentConfig[TargetConfig, DataConfig]):
    pass


def build_target(target_cfg: TargetConfig) -> TMSModel:
    """Load the TMS target model from disk/wandb. Caller computes tied_weights from the
    returned model's config."""
    run_info = TMSTargetRunInfo.from_path(target_cfg.run_path)
    target_model = TMSModel.from_run_info(run_info)
    target_model.eval()
    return target_model


def _build_loader(
    target_cfg: TargetConfig, data_cfg: DataConfig, *, batch_size: int, device: str
) -> DatasetGeneratedDataLoader[tuple[Tensor, Tensor]]:
    train_config = TMSTargetRunInfo.from_path(target_cfg.run_path).config
    dataset = SparseFeatureDataset(
        n_features=train_config.tms_model_config.n_features,
        feature_probability=data_cfg.feature_probability,
        device=device,
        data_generation_type=data_cfg.data_generation_type,
        value_range=(0.0, 1.0),
        synced_inputs=train_config.synced_inputs,
    )
    return DatasetGeneratedDataLoader(dataset, batch_size=batch_size, shuffle=False)


def build_train_loader(
    target_cfg: TargetConfig,
    data_cfg: DataConfig,
    *,
    batch_size: int,
    device: str,
    dist_state: DistributedState | None = None,
    seed: int = 0,
) -> DatasetGeneratedDataLoader[tuple[Tensor, Tensor]]:
    del dist_state, seed  # synthetic dataset; no per-rank state needed
    return _build_loader(target_cfg, data_cfg, batch_size=batch_size, device=device)


def build_eval_loader(
    target_cfg: TargetConfig,
    data_cfg: DataConfig,
    *,
    batch_size: int,
    device: str,
    dist_state: DistributedState | None = None,
    seed: int = 0,
) -> DatasetGeneratedDataLoader[tuple[Tensor, Tensor]]:
    del dist_state, seed
    return _build_loader(target_cfg, data_cfg, batch_size=batch_size, device=device)


def make_run_batch(target_cfg: TargetConfig) -> RunBatch:
    del target_cfg
    return run_batch_first_element


def _tied_weights_for(target_model: TMSModel) -> list[tuple[str, str]] | None:
    return [("linear1", "linear2")] if target_model.config.tied_weights else None


def main(config_path: str | Path) -> None:
    cfg = TMSExperimentConfig.from_file(config_path)

    set_seed(cfg.pd.seed)
    device = get_device()
    logger.info(f"Using device: {device}")

    target_model = build_target(cfg.target)
    cfg = cfg.model_copy(
        update={
            "pd": cfg.pd.model_copy(update={"tied_weights": _tied_weights_for(target_model)}),
            "runtime": cfg.runtime.model_copy(update={"device": device}),
        }
    )

    train_loader = build_train_loader(
        cfg.target, cfg.data, batch_size=cfg.pd.batch_size, device=device
    )
    eval_loader = build_eval_loader(
        cfg.target, cfg.data, batch_size=cfg.eval.batch_size, device=device
    )
    eval_metrics = [EVAL_METRIC_CLASSES[m.type](m) for m in cfg.eval.metrics]

    run_id = generate_run_id("param_decomp")
    out_dir = PARAM_DECOMP_OUT_DIR / "decompositions" / run_id
    sink = RunSink.local(out_dir)
    save_run_meta(out_dir, experiment_name="tms", cfg=cfg)

    try:
        optimize(
            target_model=target_model,
            train_loader=train_loader,
            eval_loader=eval_loader,
            run_batch=make_run_batch(cfg.target),
            reconstruction_loss=recon_loss_mse,
            pd_config=cfg.pd,
            runtime_config=cfg.runtime,
            cadence=cfg.cadence,
            sink=sink,
            eval_metrics=eval_metrics,
        )
    finally:
        sink.finish()


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
