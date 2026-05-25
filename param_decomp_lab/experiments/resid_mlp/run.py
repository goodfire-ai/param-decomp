"""Residual MLP PD experiment: YAML -> `optimize()` glue, plus the saved-run reload class.

The fresh-run path (`main`) and the reload path (`SavedResidMLPRun`) both consume the
module-level `build_target` / `build_resid_mlp_loader` / `make_run_batch` functions so there's
no duplication between them. Run via ``pd-resid-mlp path/to/config.yaml``.
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
from param_decomp.optimize import EvalLoop, optimize
from param_decomp_lab.batch_and_loss_fns import recon_loss_mse, run_batch_first_element
from param_decomp_lab.component_model_io import load_component_model
from param_decomp_lab.distributed import get_device
from param_decomp_lab.eval_metrics import EVAL_METRIC_CLASSES
from param_decomp_lab.experiments.resid_mlp.data import ResidMLPDataset
from param_decomp_lab.experiments.resid_mlp.models import ResidMLP, ResidMLPTargetRunInfo
from param_decomp_lab.experiments.utils import RUN_META_FILENAME, ExperimentConfig
from param_decomp_lab.infra.paths import ModelPath
from param_decomp_lab.infra.run_files import generate_run_id, resolve_run_files
from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR
from param_decomp_lab.run_sink import RunSink
from param_decomp_lab.seed import set_seed


class ResidMLPTargetConfig(BaseConfig):
    """Path to the trained ResidMLP target run.

    Attributes:
        run_path: Local or wandb path to a ResidMLP pretrain run.
    """

    run_path: str = Field(..., description="Local or wandb path to a ResidMLP pretrain run.")


class ResidMLPDataConfig(BaseConfig):
    """Synthetic-feature dataset settings for ResidMLP PD.

    Attributes:
        feature_probability: Probability that any individual feature is active in a sample.
        data_generation_type: Whether each sample activates exactly one feature, exactly
            two features, or any subset (including the empty set).
    """

    feature_probability: Probability
    data_generation_type: Literal[
        "exactly_one_active", "exactly_two_active", "at_least_zero_active"
    ] = "at_least_zero_active"


class ResidMLPExperimentConfig(ExperimentConfig[ResidMLPTargetConfig, ResidMLPDataConfig]):
    """Full YAML schema for a ResidMLP PD run."""

    pass


def build_target(target_cfg: ResidMLPTargetConfig) -> ResidMLP:
    """Load the pretrained ResidMLP target model from `target_cfg.run_path` in eval mode."""
    run_info = ResidMLPTargetRunInfo.from_path(target_cfg.run_path)
    target_model = ResidMLP.from_run_info(run_info)
    target_model.eval()
    return target_model


def build_resid_mlp_loader(
    target_cfg: ResidMLPTargetConfig,
    data_cfg: ResidMLPDataConfig,
    *,
    split: Literal["train", "eval"],
    device: str,
    batch_size: int,
    dist_state: DistributedState | None = None,
    seed: int | None = None,
) -> DataLoader[Any]:
    """Build a synthetic `ResidMLPDataset` loader.

    The dataset is synthetic and infinite, so `split`, `dist_state`, and `seed` are
    ignored — train and eval loaders are constructed identically.
    """
    del split, dist_state, seed
    train_config = ResidMLPTargetRunInfo.from_path(target_cfg.run_path).config
    dataset = ResidMLPDataset(
        n_features=train_config.resid_mlp_model_config.n_features,
        feature_probability=data_cfg.feature_probability,
        device=device,
        batch_size=batch_size,
        calc_labels=False,
        label_type=None,
        act_fn_name=None,
        label_fn_seed=None,
        label_coeffs=None,
        data_generation_type=data_cfg.data_generation_type,
        synced_inputs=train_config.synced_inputs,
    )
    return DataLoader(dataset, batch_size=None)


def make_run_batch(target_cfg: ResidMLPTargetConfig) -> RunBatch:
    """Return the `RunBatch` callable for ResidMLP — unwraps the (inputs, labels) tuple."""
    del target_cfg
    return run_batch_first_element


@dataclass(frozen=True)
class SavedResidMLPRun:
    """Handle to a completed ResidMLP PD run on disk or in W&B.

    Attributes:
        cfg: The resolved `ResidMLPExperimentConfig` from ``run_meta.yaml``.
        checkpoint_path: Resolved local path to the chosen ``model_<step>.pth`` file.
    """

    cfg: ResidMLPExperimentConfig
    checkpoint_path: Path

    @classmethod
    def from_path(cls, path: ModelPath) -> "SavedResidMLPRun":
        """Resolve a run directory or W&B path into a fully-validated `SavedResidMLPRun`."""
        files = resolve_run_files(
            path, config_filename=RUN_META_FILENAME, checkpoint_prefix="model"
        )
        return cls(
            cfg=ResidMLPExperimentConfig.from_file(files.config_path),
            checkpoint_path=files.checkpoint_path,
        )

    def load_model(self) -> ComponentModel:
        """Materialize the `ComponentModel` from the saved checkpoint."""
        return load_component_model(
            pd_config=self.cfg.pd,
            checkpoint_path=self.checkpoint_path,
            target_model=build_target(self.cfg.target),
            run_batch=make_run_batch(self.cfg.target),
        )


def main(config_path: str | Path) -> None:
    """Run a ResidMLP PD experiment end-to-end from a YAML config.

    Parses the YAML into `ResidMLPExperimentConfig`, builds the target / loaders /
    eval loop, writes ``run_meta.yaml``, and calls `optimize(...)`.

    Args:
        config_path: Path to the experiment YAML config.
    """
    cfg = ResidMLPExperimentConfig.from_file(config_path)

    set_seed(cfg.pd.seed)
    device = get_device()
    logger.info(f"Using device: {device}")
    cfg = cfg.model_copy(update={"runtime": cfg.runtime.model_copy(update={"device": device})})

    target_model = build_target(cfg.target).to(device)

    train_loader = build_resid_mlp_loader(
        cfg.target, cfg.data, split="train", device=device, batch_size=cfg.pd.batch_size
    )
    eval_loop = _build_eval_loop(cfg, device)

    run_id = generate_run_id("param_decomp")
    out_dir = PARAM_DECOMP_OUT_DIR / "decompositions" / run_id
    sink = RunSink.local(out_dir)
    cfg.to_file(out_dir / RUN_META_FILENAME)

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


def _build_eval_loop(cfg: ResidMLPExperimentConfig, device: str) -> EvalLoop | None:
    """Build the optional `EvalLoop` from `cfg.eval`, returning None when eval is disabled."""
    if cfg.eval is None:
        return None
    return EvalLoop(
        loader=build_resid_mlp_loader(
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
