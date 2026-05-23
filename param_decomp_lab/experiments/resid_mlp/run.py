"""Residual MLP PD experiment: YAML -> `optimize()` glue.

`SavedRun` rebuilds saved ResidMLP runs by accessing this module's dispatch interface
(`TargetConfig`, `DataConfig`, `build_target`, `build_train_loader`, `build_eval_loader`,
`make_run_batch`). Run via ``pd-resid-mlp path/to/config.yaml``.
"""

from pathlib import Path
from typing import Literal

import fire
from pydantic import Field
from torch import Tensor

from param_decomp.base_config import BaseConfig, Probability
from param_decomp.batch_and_loss_fns import RunBatch
from param_decomp.configs import PDConfig, RuntimeConfig
from param_decomp.distributed import DistributedState
from param_decomp.log import logger
from param_decomp.optimize import optimize
from param_decomp_lab.batch_and_loss_fns import recon_loss_mse, run_batch_first_element
from param_decomp_lab.distributed import get_device
from param_decomp_lab.experiments.resid_mlp.models import ResidMLP, ResidMLPTargetRunInfo
from param_decomp_lab.experiments.resid_mlp.resid_mlp_dataset import ResidMLPDataset
from param_decomp_lab.experiments.synthetic_data import DatasetGeneratedDataLoader
from param_decomp_lab.experiments.utils import (
    build_eval_metrics,
    cadence_from_logging_block,
    load_yaml,
    run_sink_from_logging_block,
    save_run_meta,
)
from param_decomp_lab.infra.run_files import generate_run_id
from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR
from param_decomp_lab.seed import set_seed


class TargetConfig(BaseConfig):
    """Path to the trained ResidMLP target run."""

    run_path: str = Field(..., description="Local or wandb path to a ResidMLP pretrain run.")


class DataConfig(BaseConfig):
    """Synthetic-feature dataset settings for ResidMLP PD."""

    feature_probability: Probability
    data_generation_type: Literal[
        "exactly_one_active", "exactly_two_active", "at_least_zero_active"
    ] = "at_least_zero_active"


def build_target(target_cfg: TargetConfig) -> ResidMLP:
    run_info = ResidMLPTargetRunInfo.from_path(target_cfg.run_path)
    target_model = ResidMLP.from_run_info(run_info)
    target_model.eval()
    return target_model


def _build_loader(
    target_cfg: TargetConfig, data_cfg: DataConfig, *, batch_size: int, device: str
) -> DatasetGeneratedDataLoader[tuple[Tensor, Tensor]]:
    train_config = ResidMLPTargetRunInfo.from_path(target_cfg.run_path).config
    dataset = ResidMLPDataset(
        n_features=train_config.resid_mlp_model_config.n_features,
        feature_probability=data_cfg.feature_probability,
        device=device,
        calc_labels=False,
        label_type=None,
        act_fn_name=None,
        label_fn_seed=None,
        label_coeffs=None,
        data_generation_type=data_cfg.data_generation_type,
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
    del dist_state, seed
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


def main(config_path: str | Path) -> None:
    raw = load_yaml(config_path)
    pd_config = PDConfig.model_validate(raw["pd"])
    runtime_config = RuntimeConfig.model_validate(raw["runtime"])
    target_cfg = TargetConfig.model_validate(raw["target"])
    data_cfg = DataConfig.model_validate(raw["data"])
    logging_block = raw["logging"]

    set_seed(pd_config.seed)
    device = get_device()
    runtime_config = RuntimeConfig.model_validate({**runtime_config.model_dump(), "device": device})
    logger.info(f"Using device: {device}")

    target_model = build_target(target_cfg)

    train_loader = build_train_loader(
        target_cfg, data_cfg, batch_size=pd_config.batch_size, device=device
    )
    eval_loader = build_eval_loader(
        target_cfg, data_cfg, batch_size=logging_block["eval_batch_size"], device=device
    )

    eval_metrics = build_eval_metrics(logging_block.get("eval_metrics"))

    run_id = generate_run_id("param_decomp")
    out_dir = PARAM_DECOMP_OUT_DIR / "decompositions" / run_id
    cadence = cadence_from_logging_block(logging_block)
    sink = run_sink_from_logging_block(out_dir)
    save_run_meta(
        out_dir,
        experiment_name="resid_mlp",
        pd_config=pd_config,
        runtime_config=runtime_config,
        target_dict=target_cfg.model_dump(mode="json"),
        data_dict=data_cfg.model_dump(mode="json"),
    )

    try:
        optimize(
            target_model=target_model,
            train_loader=train_loader,
            eval_loader=eval_loader,
            run_batch=make_run_batch(target_cfg),
            reconstruction_loss=recon_loss_mse,
            pd_config=pd_config,
            runtime_config=runtime_config,
            cadence=cadence,
            sink=sink,
            eval_metrics=eval_metrics,
        )
    finally:
        sink.finish()


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
