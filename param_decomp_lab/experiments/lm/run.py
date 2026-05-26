"""Language-model PD experiment: YAML -> `optimize()` glue, plus the saved-run reload class.

The fresh-run path (`main`) and the reload path (`SavedLMRun`) both consume the
module-level `build_target` / `build_lm_loader` / `make_run_batch` functions so there's
no duplication between them. Run via ``pd-lm path/to/config.yaml``; multi-process
(DDP) entry via ``torchrun`` of the same module.
"""

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import fire
from pydantic import Discriminator
from torch.utils.data import DataLoader

from param_decomp.base_config import BaseConfig
from param_decomp.batch_and_loss_fns import RunBatch
from param_decomp.component_model import ComponentModel
from param_decomp.distributed import DistributedState, is_main_process
from param_decomp.log import logger
from param_decomp.optimize import EvalLoop, optimize
from param_decomp_lab.batch_and_loss_fns import make_run_batch as _make_run_batch
from param_decomp_lab.batch_and_loss_fns import recon_loss_kl
from param_decomp_lab.component_model_io import load_component_model
from param_decomp_lab.distributed import (
    ensure_cached_and_call,
    get_device,
    init_distributed,
    with_distributed_cleanup,
)
from param_decomp_lab.eval_metrics import EVAL_METRIC_CLASSES
from param_decomp_lab.experiments.lm.data import (
    LMDataConfig,
    collate_fn_for,
    create_lm_data_loader,
    rank_batch_size,
)
from param_decomp_lab.experiments.utils import (
    RUN_META_FILENAME,
    ExperimentConfig,
    init_pd_run,
)
from param_decomp_lab.infra.paths import ModelPath
from param_decomp_lab.infra.run_files import resolve_run_files
from param_decomp_lab.run_sink import RunSink
from param_decomp_lab.seed import set_seed


def _resolve_class(fqn: str) -> type:
    """Load a class from a fully-qualified name, e.g. 'transformers.LlamaForCausalLM'."""
    module_path, _, class_name = fqn.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class HFTarget(BaseConfig):
    """Load a HuggingFace model via ``<class>.from_pretrained(<hub_id>)``.

    Attributes:
        kind: Discriminator literal for `LMTargetSpec`.
        model_class: Fully-qualified class name, e.g. `transformers.GPT2LMHeadModel`.
        model_name: Hugging Face Hub identifier passed to `from_pretrained`.
    """

    kind: Literal["hf"] = "hf"
    model_class: str
    model_name: str


class PretrainedTarget(BaseConfig):
    """Load an in-repo lab-pretrained model from a wandb/local pretrain run.

    Attributes:
        kind: Discriminator literal for `LMTargetSpec`.
        model_class: Fully-qualified class name, e.g. `transformers.LlamaForCausalLM`.
        run_path: Any form `PretrainRunInfo.from_path` accepts — compact W&B
            (``entity/project/runId``), full W&B (``entity/project/runs/runId``), or a
            local checkpoint path.
    """

    kind: Literal["pretrained"] = "pretrained"
    model_class: str
    run_path: ModelPath


LMTargetSpec = Annotated[
    HFTarget | PretrainedTarget,
    Discriminator("kind"),
]
"""Discriminated union over LM target sources, keyed on ``kind`` (`"hf"` vs `"pretrained"`)."""


class LMTargetConfig(BaseConfig):
    """How to load the LM target plus how to extract its prediction tensor.

    Attributes:
        spec: Discriminated union selecting between a HuggingFace model and an in-repo
            lab-pretrained model.
        output_extract: Key/index passed to the `RunBatch` helper to pull the prediction
            tensor out of the model's forward output (defaults to `"logits"`).
    """

    spec: LMTargetSpec
    output_extract: int | str | None = "logits"


class LMExperimentConfig(ExperimentConfig[LMTargetConfig, LMDataConfig]):
    """Full YAML schema for an LM PD run."""

    pass


def build_target(target_cfg: LMTargetConfig) -> Any:
    """Load the LM target model in eval mode, dispatching on `target_cfg.spec.kind`."""
    spec = target_cfg.spec
    cls = _resolve_class(spec.model_class)
    match spec:
        case HFTarget():
            target_model = ensure_cached_and_call(cls.from_pretrained, spec.model_name)
        case PretrainedTarget():
            from param_decomp_lab.experiments.lm.pretrain.run_info import PretrainRunInfo

            run_info = ensure_cached_and_call(PretrainRunInfo.from_path, spec.run_path)
            if "model_type" not in run_info.model_config_dict:
                run_info.model_config_dict["model_type"] = spec.model_class.rsplit(".", 1)[-1]
            target_model = cls.from_run_info(run_info)
    target_model.eval()
    return target_model


def build_lm_loader(
    target_cfg: LMTargetConfig,
    data_cfg: LMDataConfig,
    *,
    split: Literal["train", "eval"],
    device: str,
    batch_size: int,
    dist_state: DistributedState | None = None,
    seed: int | None = None,
) -> DataLoader[Any]:
    """Build the LM `DataLoader` for the requested split.

    The eval seed is offset by 1 so eval shuffles differently from train when both are
    constructed from the same `pd_config.seed`.
    """
    del target_cfg, device
    effective_seed = (seed or 0) + (1 if split == "eval" else 0)
    split_name = data_cfg.eval_split if split == "eval" else data_cfg.train_split
    loader, _ = create_lm_data_loader(
        data_cfg,
        split=split_name,
        batch_size=rank_batch_size(batch_size, dist_state, label=f"{split}_batch_size"),
        seed=effective_seed,
        dist_state=dist_state,
        collate_fn=collate_fn_for(data_cfg),
    )
    return loader


def make_run_batch(target_cfg: LMTargetConfig) -> RunBatch:
    """Return the `RunBatch` callable bound to `target_cfg.output_extract`."""
    return _make_run_batch(target_cfg.output_extract)


@dataclass(frozen=True)
class SavedLMRun:
    """Handle to a completed LM PD run on disk or in W&B.

    Attributes:
        cfg: The resolved `LMExperimentConfig` from ``run_meta.yaml``.
        checkpoint_path: Resolved local path to the chosen ``model_<step>.pth`` file.
    """

    cfg: LMExperimentConfig
    checkpoint_path: Path

    @classmethod
    def from_path(cls, path: ModelPath) -> "SavedLMRun":
        """Resolve a run directory or W&B path into a fully-validated `SavedLMRun`."""
        files = resolve_run_files(
            path, config_filename=RUN_META_FILENAME, checkpoint_prefix="model"
        )
        return cls(
            cfg=LMExperimentConfig.from_file(files.config_path),
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


@with_distributed_cleanup
def main(
    config_path: str | Path,
    *,
    group: str | None = None,
    tags: str | None = None,
) -> None:
    """Run an LM PD experiment end-to-end from a YAML config.

    Parses the YAML into `LMExperimentConfig`, initialises DDP, builds the target /
    loaders / eval loop, writes ``run_meta.yaml`` on the main rank, and calls
    `optimize(...)`. Non-main ranks use a silent sink.

    Args:
        config_path: Path to the experiment YAML config.
        group: Wandb group for "launched together" collapsing. `pd-lm-layerwise`
            sets this automatically per array; pass it by hand to stamp ad-hoc
            multi-launches as one experiment.
        tags: Comma-separated wandb tags (orthogonal to `group`; many per run).
    """
    cfg = LMExperimentConfig.from_file(config_path)

    dist_state = init_distributed()
    if is_main_process():
        logger.info(f"Distributed state: {dist_state}")
    set_seed(cfg.pd.seed)
    device = get_device()
    cfg = cfg.model_copy(update={"runtime": cfg.runtime.model_copy(update={"device": device})})

    target_model = build_target(cfg.target)

    train_loader = build_lm_loader(
        cfg.target,
        cfg.data,
        split="train",
        device=device,
        batch_size=cfg.pd.batch_size,
        dist_state=dist_state,
        seed=cfg.pd.seed,
    )
    eval_loop = _build_eval_loop(cfg, device, dist_state)

    sink = init_pd_run(cfg, group=group, tags=tags) if is_main_process() else RunSink.silent()

    try:
        optimize(
            target_model=target_model,
            train_loader=train_loader,
            run_batch=make_run_batch(cfg.target),
            reconstruction_loss=recon_loss_kl,
            pd_config=cfg.pd,
            runtime_config=cfg.runtime,
            sink=sink,
            cadence=cfg.cadence,
            eval_loop=eval_loop,
        )
    finally:
        sink.finish()


def _build_eval_loop(
    cfg: LMExperimentConfig,
    device: str,
    dist_state: DistributedState | None,
) -> EvalLoop | None:
    """Build the optional `EvalLoop` from `cfg.eval`, returning None when eval is disabled."""
    if cfg.eval is None:
        return None
    eval_loader = build_lm_loader(
        cfg.target,
        cfg.data,
        split="eval",
        device=device,
        batch_size=cfg.eval.batch_size,
        dist_state=dist_state,
        seed=cfg.pd.seed,
    )
    return EvalLoop(
        loader=eval_loader,
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
