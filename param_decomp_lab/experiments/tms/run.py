"""TMS PD experiment: YAML -> `optimize()` glue.

Exposes ``EXPERIMENT_SPEC`` for lab-side tools (SavedRun, harvest, etc.) to rebuild the
target model and dataloaders from a saved run.

Run via ``python -m param_decomp_lab.experiments.tms.run path/to/config.yaml``.
"""

from pathlib import Path
from typing import Literal

import fire
from pydantic import Field
from torch import Tensor

from param_decomp import PDConfig, RuntimeConfig, optimize
from param_decomp.base_config import BaseConfig
from param_decomp.log import logger
from param_decomp.models.batch_and_loss_fns import RunBatch, recon_loss_mse, run_batch_first_element
from param_decomp.settings import PARAM_DECOMP_OUT_DIR
from param_decomp.types import Probability
from param_decomp.utils.data_utils import DatasetGeneratedDataLoader, SparseFeatureDataset
from param_decomp.utils.distributed_utils import DistributedState, get_device
from param_decomp.utils.general_utils import set_seed
from param_decomp.utils.run_utils import generate_run_id
from param_decomp_lab.experiments.spec import ExperimentSpec
from param_decomp_lab.experiments.tms.models import TMSModel, TMSTargetRunInfo
from param_decomp_lab.experiments.utils import (
    build_eval_metrics,
    load_yaml,
    run_sink_from_logging_block,
    save_run_meta,
)


class TMSTargetConfig(BaseConfig):
    """Path to the trained TMS target run."""

    run_path: str = Field(..., description="Local or wandb path to a TMS pretrain run.")


class TMSDataConfig(BaseConfig):
    """Synthetic-feature dataset settings for TMS PD."""

    feature_probability: Probability
    data_generation_type: Literal["exactly_one_active", "at_least_zero_active"] = (
        "at_least_zero_active"
    )


def build_target(target_cfg: TMSTargetConfig) -> TMSModel:
    """Load the TMS target model from disk/wandb. Caller computes tied_weights from the
    returned model's config."""
    run_info = TMSTargetRunInfo.from_path(target_cfg.run_path)
    target_model = TMSModel.from_run_info(run_info)
    target_model.eval()
    return target_model


def tied_weights_for(target_model: TMSModel) -> list[tuple[str, str]] | None:
    return [("linear1", "linear2")] if target_model.config.tied_weights else None


def _build_dataset(
    target_cfg: TMSTargetConfig, data_cfg: TMSDataConfig, device: str
) -> SparseFeatureDataset:
    train_config = TMSTargetRunInfo.from_path(target_cfg.run_path).config
    return SparseFeatureDataset(
        n_features=train_config.tms_model_config.n_features,
        feature_probability=data_cfg.feature_probability,
        device=device,
        data_generation_type=data_cfg.data_generation_type,
        value_range=(0.0, 1.0),
        synced_inputs=train_config.synced_inputs,
    )


def build_train_loader(
    target_cfg: TMSTargetConfig,
    data_cfg: TMSDataConfig,
    *,
    batch_size: int,
    device: str,
    dist_state: DistributedState | None = None,
    seed: int = 0,
) -> DatasetGeneratedDataLoader[tuple[Tensor, Tensor]]:
    del dist_state, seed  # synthetic dataset; no per-rank state needed
    dataset = _build_dataset(target_cfg, data_cfg, device)
    return DatasetGeneratedDataLoader(dataset, batch_size=batch_size, shuffle=False)


def build_eval_loader(
    target_cfg: TMSTargetConfig,
    data_cfg: TMSDataConfig,
    *,
    batch_size: int,
    device: str,
    dist_state: DistributedState | None = None,
    seed: int = 0,
) -> DatasetGeneratedDataLoader[tuple[Tensor, Tensor]]:
    del dist_state, seed
    dataset = _build_dataset(target_cfg, data_cfg, device)
    return DatasetGeneratedDataLoader(dataset, batch_size=batch_size, shuffle=False)


def make_run_batch(target_cfg: TMSTargetConfig) -> RunBatch:
    del target_cfg
    return run_batch_first_element


EXPERIMENT_SPEC = ExperimentSpec(
    name="tms",
    target_config_cls=TMSTargetConfig,
    data_config_cls=TMSDataConfig,
    build_target=build_target,
    build_train_loader=build_train_loader,
    build_eval_loader=build_eval_loader,
    make_run_batch=make_run_batch,
    reconstruction_loss=recon_loss_mse,
)


def main(config_path: str | Path) -> None:
    raw = load_yaml(config_path)
    pd_config = PDConfig.model_validate(raw["pd"])
    runtime_config = RuntimeConfig.model_validate(raw["runtime"])
    target_cfg = TMSTargetConfig.model_validate(raw["target"])
    data_cfg = TMSDataConfig.model_validate(raw["data"])
    logging_block = raw["logging"]

    set_seed(pd_config.seed)
    device = get_device()
    logger.info(f"Using device: {device}")

    target_model = build_target(target_cfg)
    pd_config = pd_config.model_copy(update={"tied_weights": tied_weights_for(target_model)})

    train_loader = build_train_loader(
        target_cfg, data_cfg, batch_size=pd_config.batch_size, device=device
    )
    eval_loader = build_eval_loader(
        target_cfg, data_cfg, batch_size=logging_block["eval_batch_size"], device=device
    )

    eval_metrics = build_eval_metrics(logging_block.get("eval_metrics"))

    run_id = generate_run_id("param_decomp")
    out_dir = PARAM_DECOMP_OUT_DIR / "decompositions" / run_id
    sink = run_sink_from_logging_block(out_dir, logging_block)
    save_run_meta(
        out_dir,
        experiment_name=EXPERIMENT_SPEC.name,
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
            reconstruction_loss=EXPERIMENT_SPEC.reconstruction_loss,
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
