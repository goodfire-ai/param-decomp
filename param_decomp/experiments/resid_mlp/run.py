"""Residual MLP PD experiment: YAML -> `optimize()` glue.

Composes the target model, dataloaders, eval metrics, configs, and sink, then calls
`optimize()`. Run via `python -m param_decomp.experiments.resid_mlp.run path/to/config.yaml`.
"""

from pathlib import Path
from typing import Literal

import fire
from pydantic import Field
from torch import Tensor

from param_decomp import PDConfig, RuntimeConfig, optimize
from param_decomp.base_config import BaseConfig
from param_decomp.experiments.resid_mlp.models import ResidMLP, ResidMLPTargetRunInfo
from param_decomp.experiments.resid_mlp.resid_mlp_dataset import ResidMLPDataset
from param_decomp.experiments.utils import (
    build_eval_metrics,
    load_yaml,
    run_sink_from_logging_block,
)
from param_decomp.log import logger
from param_decomp.models.batch_and_loss_fns import recon_loss_mse, run_batch_first_element
from param_decomp.settings import PARAM_DECOMP_OUT_DIR
from param_decomp.types import Probability
from param_decomp.utils.data_utils import DatasetGeneratedDataLoader
from param_decomp.utils.distributed_utils import get_device
from param_decomp.utils.general_utils import set_seed
from param_decomp.utils.run_utils import generate_run_id


class ResidMLPTargetConfig(BaseConfig):
    """Path to the trained ResidMLP target run."""

    run_path: str = Field(..., description="Local or wandb path to a ResidMLP pretrain run.")


class ResidMLPDataConfig(BaseConfig):
    """Synthetic-feature dataset settings for ResidMLP PD."""

    feature_probability: Probability
    data_generation_type: Literal[
        "exactly_one_active", "exactly_two_active", "at_least_zero_active"
    ] = "at_least_zero_active"


def build_target(target_cfg: ResidMLPTargetConfig) -> ResidMLP:
    run_info = ResidMLPTargetRunInfo.from_path(target_cfg.run_path)
    target_model = ResidMLP.from_run_info(run_info)
    target_model.eval()
    return target_model


def build_dataset(
    target_cfg: ResidMLPTargetConfig, data_cfg: ResidMLPDataConfig, device: str
) -> ResidMLPDataset:
    train_config = ResidMLPTargetRunInfo.from_path(target_cfg.run_path).config
    return ResidMLPDataset(
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


def build_loader(
    dataset: ResidMLPDataset, batch_size: int
) -> DatasetGeneratedDataLoader[tuple[Tensor, Tensor]]:
    return DatasetGeneratedDataLoader(dataset, batch_size=batch_size, shuffle=False)


def main(config_path: str | Path) -> None:
    raw = load_yaml(config_path)
    pd_config = PDConfig.model_validate(raw["pd"])
    runtime_config = RuntimeConfig.model_validate(raw["runtime"])
    target_cfg = ResidMLPTargetConfig.model_validate(raw["target"])
    data_cfg = ResidMLPDataConfig.model_validate(raw["data"])
    logging_block = raw["logging"]

    set_seed(pd_config.seed)
    device = get_device()
    logger.info(f"Using device: {device}")

    target_model = build_target(target_cfg)

    dataset = build_dataset(target_cfg, data_cfg, device)
    train_loader = build_loader(dataset, pd_config.batch_size)
    eval_loader = build_loader(dataset, logging_block["eval_batch_size"])

    eval_metrics = build_eval_metrics(logging_block.get("eval_metrics"))

    run_id = generate_run_id("param_decomp")
    out_dir = PARAM_DECOMP_OUT_DIR / "decompositions" / run_id
    sink = run_sink_from_logging_block(out_dir, logging_block)

    try:
        optimize(
            target_model=target_model,
            train_loader=train_loader,
            eval_loader=eval_loader,
            run_batch=run_batch_first_element,
            reconstruction_loss=recon_loss_mse,
            pd_config=pd_config,
            runtime_config=runtime_config,
            sink=sink,
            eval_metrics=eval_metrics,
            device=device,
        )
    finally:
        sink.finish()


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
