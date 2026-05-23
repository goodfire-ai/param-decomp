"""Residual MLP PD experiment: YAML -> `optimize()` glue.

`ResidMLPReloader` is the single class `SavedRun` resolves via the FQN written into
`run_meta.yaml::reloader_class`. It owns target / loader / run_batch construction so the
same code path is used for "fresh run from YAML" and "reload from disk". Run via
``pd-resid-mlp path/to/config.yaml``.
"""

from pathlib import Path
from typing import Any, ClassVar, Literal, Self

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
from param_decomp_lab.experiments.resid_mlp.data import ResidMLPDataset
from param_decomp_lab.experiments.resid_mlp.models import ResidMLP, ResidMLPTargetRunInfo
from param_decomp_lab.experiments.utils import ExperimentConfig, save_run_meta
from param_decomp_lab.infra.run_files import generate_run_id
from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR
from param_decomp_lab.run_sink import RunSink
from param_decomp_lab.saved_run import RunMeta
from param_decomp_lab.seed import set_seed


class ResidMLPTargetConfig(BaseConfig):
    """Path to the trained ResidMLP target run."""

    run_path: str = Field(..., description="Local or wandb path to a ResidMLP pretrain run.")


class ResidMLPDataConfig(BaseConfig):
    """Synthetic-feature dataset settings for ResidMLP PD."""

    feature_probability: Probability
    data_generation_type: Literal[
        "exactly_one_active", "exactly_two_active", "at_least_zero_active"
    ] = "at_least_zero_active"


class ResidMLPExperimentConfig(ExperimentConfig[ResidMLPTargetConfig, ResidMLPDataConfig]):
    pass


class ResidMLPReloader:
    target_config_type: ClassVar[type[ResidMLPTargetConfig]] = ResidMLPTargetConfig
    data_config_type: ClassVar[type[ResidMLPDataConfig]] = ResidMLPDataConfig

    def __init__(self, target_cfg: ResidMLPTargetConfig, data_cfg: ResidMLPDataConfig):
        self.target_cfg = target_cfg
        self.data_cfg = data_cfg

    @classmethod
    def from_meta(cls, meta: RunMeta) -> Self:
        return cls(
            target_cfg=cls.target_config_type.model_validate(meta.target_dict),
            data_cfg=cls.data_config_type.model_validate(meta.data_dict),
        )

    def build_target(self) -> ResidMLP:
        run_info = ResidMLPTargetRunInfo.from_path(self.target_cfg.run_path)
        target_model = ResidMLP.from_run_info(run_info)
        target_model.eval()
        return target_model

    def build_loader(
        self,
        *,
        split: Literal["train", "eval"],
        device: str,
        batch_size: int,
        dist_state: DistributedState | None = None,
        seed: int | None = None,
    ) -> DataLoader[Any]:
        del split, dist_state, seed
        train_config = ResidMLPTargetRunInfo.from_path(self.target_cfg.run_path).config
        dataset = ResidMLPDataset(
            n_features=train_config.resid_mlp_model_config.n_features,
            feature_probability=self.data_cfg.feature_probability,
            device=device,
            batch_size=batch_size,
            calc_labels=False,
            label_type=None,
            act_fn_name=None,
            label_fn_seed=None,
            label_coeffs=None,
            data_generation_type=self.data_cfg.data_generation_type,
            synced_inputs=train_config.synced_inputs,
        )
        return DataLoader(dataset, batch_size=None)

    def make_run_batch(self) -> RunBatch:
        return run_batch_first_element


def main(config_path: str | Path) -> None:
    cfg = ResidMLPExperimentConfig.from_file(config_path)

    set_seed(cfg.pd.seed)
    device = get_device()
    logger.info(f"Using device: {device}")
    cfg = cfg.model_copy(update={"runtime": cfg.runtime.model_copy(update={"device": device})})

    reloader = ResidMLPReloader(target_cfg=cfg.target, data_cfg=cfg.data)
    target_model = reloader.build_target().to(device)

    train_loader = reloader.build_loader(split="train", device=device, batch_size=cfg.pd.batch_size)
    eval_loop = _build_eval_loop(cfg, reloader, device)

    run_id = generate_run_id("param_decomp")
    out_dir = PARAM_DECOMP_OUT_DIR / "decompositions" / run_id
    sink = RunSink.local(out_dir)
    save_run_meta(out_dir, reloader_class=ResidMLPReloader, cfg=cfg)

    try:
        optimize(
            target_model=target_model,
            train_loader=train_loader,
            run_batch=reloader.make_run_batch(),
            reconstruction_loss=recon_loss_mse,
            pd_config=cfg.pd,
            runtime_config=cfg.runtime,
            sink=sink,
            cadence=cfg.cadence,
            eval_loop=eval_loop,
        )
    finally:
        sink.finish()


def _build_eval_loop(
    cfg: ResidMLPExperimentConfig, reloader: ResidMLPReloader, device: str
) -> EvalLoop | None:
    if cfg.eval is None:
        return None
    return EvalLoop(
        loader=reloader.build_loader(split="eval", device=device, batch_size=cfg.eval.batch_size),
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
