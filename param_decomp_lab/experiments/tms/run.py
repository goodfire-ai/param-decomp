"""TMS PD experiment: YAML -> `optimize()` glue.

Exposes module-level `TARGET_CONFIG_TYPE`, `DATA_CONFIG_TYPE`, `build_target`,
`build_loader`, and `make_run_batch` so `SavedRun` can rebuild a run by dispatching
on `run_meta.yaml::experiment_kind`. Run via ``pd-tms path/to/config.yaml``.
"""

from pathlib import Path
from typing import Any, Literal

import fire
from pydantic import Field
from torch.utils.data import DataLoader

from param_decomp.base_config import BaseConfig, Probability
from param_decomp.batch_and_loss_fns import RunBatch
from param_decomp.distributed import DistributedState
from param_decomp.log import logger
from param_decomp.optimize import EvalLoop, optimize
from param_decomp_lab.batch_and_loss_fns import recon_loss_mse, run_batch_first_element
from param_decomp_lab.distributed import get_device
from param_decomp_lab.eval_metrics import EVAL_METRIC_CLASSES
from param_decomp_lab.experiments.tms.data import SparseFeatureDataset
from param_decomp_lab.experiments.tms.models import TMSModel, TMSTargetRunInfo
from param_decomp_lab.experiments.utils import ExperimentConfig, save_run_meta
from param_decomp_lab.infra.run_files import generate_run_id
from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR
from param_decomp_lab.run_sink import RunSink
from param_decomp_lab.seed import set_seed


class TMSTargetConfig(BaseConfig):
    """Path to the trained TMS target run.

    Attributes:
        run_path: Local or wandb path to a TMS pretrain run.
    """

    run_path: str = Field(..., description="Local or wandb path to a TMS pretrain run.")


class TMSDataConfig(BaseConfig):
    """Synthetic-feature dataset settings for TMS PD.

    Attributes:
        feature_probability: Probability that any individual feature is active in a sample.
        data_generation_type: Whether each sample activates exactly one feature or any
            subset (including the empty set).
    """

    feature_probability: Probability
    data_generation_type: Literal["exactly_one_active", "at_least_zero_active"] = (
        "at_least_zero_active"
    )


class TMSExperimentConfig(ExperimentConfig[TMSTargetConfig, TMSDataConfig]):
    """Full YAML schema for a TMS PD run."""

    pass


TARGET_CONFIG_TYPE = TMSTargetConfig
DATA_CONFIG_TYPE = TMSDataConfig


def build_target(target_cfg: TMSTargetConfig) -> TMSModel:
    """Load the pretrained TMS target model from `target_cfg.run_path` in eval mode."""
    run_info = TMSTargetRunInfo.from_path(target_cfg.run_path)
    target_model = TMSModel.from_run_info(run_info)
    target_model.eval()
    return target_model


def build_loader(
    target_cfg: TMSTargetConfig,
    data_cfg: TMSDataConfig,
    *,
    split: Literal["train", "eval"],
    device: str,
    batch_size: int,
    dist_state: DistributedState | None = None,
    seed: int | None = None,
) -> DataLoader[Any]:
    """Build a synthetic `SparseFeatureDataset` loader for TMS.

    The dataset is synthetic and infinite, so `split`, `dist_state`, and `seed` are
    ignored — train and eval loaders are constructed identically.
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
    """Return the `RunBatch` callable for TMS — unwraps the (inputs, labels) tuple."""
    del target_cfg
    return run_batch_first_element


def _tied_weights_for(target_model: TMSModel) -> list[tuple[str, str]] | None:
    return [("linear1", "linear2")] if target_model.config.tied_weights else None


def main(config_path: str | Path) -> None:
    """Run a TMS PD experiment end-to-end from a YAML config.

    Parses the YAML into `TMSExperimentConfig`, builds the target / loaders / eval loop,
    writes `run_meta.yaml`, and calls `optimize(...)`.

    Args:
        config_path: Path to the experiment YAML config.
    """
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

    train_loader = build_loader(
        cfg.target, cfg.data, split="train", device=device, batch_size=cfg.pd.batch_size
    )
    eval_loop = _build_eval_loop(cfg, device)

    run_id = generate_run_id("param_decomp")
    out_dir = PARAM_DECOMP_OUT_DIR / "decompositions" / run_id
    sink = RunSink.local(out_dir)
    save_run_meta(out_dir, kind="tms", cfg=cfg)

    try:
        optimize(
            target_model=target_model,
            train_loader=train_loader,
            run_batch=make_run_batch(cfg.target),
            reconstruction_loss=recon_loss_mse,
            pd_config=cfg.pd,
            runtime_config=cfg.runtime,
            sink=sink,
            cadence=cfg.cadence,
            eval_loop=eval_loop,
        )
    finally:
        sink.finish()


def _build_eval_loop(cfg: TMSExperimentConfig, device: str) -> EvalLoop | None:
    """Build the optional `EvalLoop` from `cfg.eval`, returning None when eval is disabled."""
    if cfg.eval is None:
        return None
    return EvalLoop(
        loader=build_loader(
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
