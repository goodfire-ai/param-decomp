"""MNIST MLP PD experiment: YAML -> `Trainer` glue, plus the `SavedMnistRun` reload class.

Run via `pd-mnist path/to/config.yaml`. Uses the categorical KL reconstruction objective
(`recon_loss_kl`) since the target output is a 10-class logit distribution, and unwraps the
`(image, label)` batch tuple via `run_batch_first_element`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import fire
from torch.utils.data import DataLoader

from param_decomp.base_config import BaseConfig
from param_decomp.batch_and_loss_fns import RunBatch
from param_decomp.component_model import ComponentModel
from param_decomp.distributed import DistributedState
from param_decomp.log import logger
from param_decomp.optimize import EvalLoop, Trainer
from param_decomp_lab.batch_and_loss_fns import recon_loss_kl, run_batch_first_element
from param_decomp_lab.component_model_io import load_component_model
from param_decomp_lab.distributed import get_device
from param_decomp_lab.eval_metrics import EVAL_METRIC_CLASSES
from param_decomp_lab.experiments.mnist.data import (
    MnistMemorizedDataset,
    load_raw_mnist,
)
from param_decomp_lab.experiments.mnist.models import MnistMLP, MnistTargetRunInfo
from param_decomp_lab.experiments.utils import (
    EXPERIMENT_CONFIG_FILENAME,
    ExperimentConfig,
    init_pd_run,
)
from param_decomp_lab.infra.paths import ModelPath
from param_decomp_lab.infra.run_files import resolve_run_files
from param_decomp_lab.seed import set_seed

try:
    from silico.slurm_telemetry import register_wandb_url
except Exception:  # pragma: no cover - silico not installed

    def register_wandb_url() -> None:  # type: ignore[misc]
        return None


class MnistTargetConfig(BaseConfig):
    run_path: ModelPath


class MnistDataConfig(BaseConfig):
    """Decomposition reads the target's exact memorized set; the only knob is whether the
    train loader reshuffles each epoch."""

    shuffle_train: bool = True


class MnistExperimentConfig(ExperimentConfig[MnistTargetConfig, MnistDataConfig]):
    pass


def build_target(target_cfg: MnistTargetConfig) -> MnistMLP:
    run_info = MnistTargetRunInfo.from_path(target_cfg.run_path)
    target_model = MnistMLP.from_run_info(run_info)
    target_model.eval()
    return target_model


def build_mnist_loader(
    target_cfg: MnistTargetConfig,
    data_cfg: MnistDataConfig,
    *,
    split: Literal["train", "eval"],
    device: str,
    batch_size: int,
    dist_state: DistributedState | None = None,
    seed: int | None = None,
) -> DataLoader[Any]:
    """Loader over the target's exact memorized set (same images + labels it was trained on).

    Train reshuffles each epoch; eval iterates the whole set in fixed order so per-component
    density / L0 are measured deterministically over every memorized input.
    """
    del dist_state
    run_info = MnistTargetRunInfo.from_path(target_cfg.run_path)
    train_x_full, _, _, _ = load_raw_mnist(
        run_info.config.data_dir, normalize=run_info.config.normalize
    )
    mem_x = train_x_full[run_info.train_indices]
    mem_y = run_info.train_labels
    shuffle = data_cfg.shuffle_train if split == "train" else False
    dataset = MnistMemorizedDataset(
        mem_x, mem_y, batch_size=batch_size, device=device, shuffle=shuffle, seed=(seed or 0)
    )
    return DataLoader(dataset, batch_size=None)


def make_run_batch(target_cfg: MnistTargetConfig) -> RunBatch:
    del target_cfg
    return run_batch_first_element


@dataclass(frozen=True)
class SavedMnistRun:
    """Handle to a completed MNIST PD run on disk or in W&B."""

    cfg: MnistExperimentConfig
    checkpoint_path: Path

    @classmethod
    def from_path(cls, path: ModelPath) -> "SavedMnistRun":
        files = resolve_run_files(
            path, config_filename=EXPERIMENT_CONFIG_FILENAME, checkpoint_prefix="model"
        )
        return cls(
            cfg=MnistExperimentConfig.from_file(files.config_path),
            checkpoint_path=files.checkpoint_path,
        )

    def load_model(self) -> ComponentModel:
        return load_component_model(
            pd_config=self.cfg.pd,
            checkpoint_path=self.checkpoint_path,
            target_model=build_target(self.cfg.target),
            run_batch=make_run_batch(self.cfg.target),
        )


def _build_eval_loop(cfg: MnistExperimentConfig, device: str) -> EvalLoop | None:
    if cfg.eval is None:
        return None
    return EvalLoop(
        loader=build_mnist_loader(
            cfg.target, cfg.data, split="eval", device=device, batch_size=cfg.eval.batch_size
        ),
        metrics=[EVAL_METRIC_CLASSES[m.type](m) for m in cfg.eval.metrics],
        n_steps=cfg.eval.n_steps,
        every=cfg.eval.every,
        slow_every=cfg.eval.slow_every,
        slow_on_first_step=cfg.eval.slow_on_first_step,
    )


def main(
    config_path: str | Path,
    *,
    group: str | None = None,
    tags: str | None = None,
) -> None:
    """Run an MNIST MLP PD experiment end-to-end from a YAML config. `group`/`tags` are
    wandb-only."""
    cfg = MnistExperimentConfig.from_file(config_path)

    set_seed(cfg.pd.seed)
    device = get_device()
    logger.info(f"Using device: {device}")
    cfg = cfg.model_copy(update={"runtime": cfg.runtime.model_copy(update={"device": device})})

    target_model = build_target(cfg.target).to(device)

    train_loader = build_mnist_loader(
        cfg.target,
        cfg.data,
        split="train",
        device=device,
        batch_size=cfg.pd.batch_size,
        seed=cfg.pd.seed,
    )
    eval_loop = _build_eval_loop(cfg, device)

    sink = init_pd_run(cfg, group=group, tags=tags)
    register_wandb_url()

    try:
        trainer = Trainer(
            target_model=target_model,
            run_batch=make_run_batch(cfg.target),
            reconstruction_loss=recon_loss_kl,
            pd_config=cfg.pd,
            runtime_config=cfg.runtime,
        )
        trainer.run(train_loader, sink, cfg.cadence, eval_loop)
    finally:
        sink.finish()


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
