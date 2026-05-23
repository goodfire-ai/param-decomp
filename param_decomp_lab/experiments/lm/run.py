"""Language-model PD experiment: YAML -> `optimize()` glue.

Exposes module-level `TARGET_CONFIG_TYPE`, `DATA_CONFIG_TYPE`, `build_target`,
`build_loader`, and `make_run_batch` so `SavedRun` can rebuild a run by dispatching
on `run_meta.yaml::experiment_kind`. The fresh-run path (`main`) calls the same
functions. Run via ``pd-lm path/to/config.yaml``; multi-process (DDP) entry via
``torchrun`` of the same module.
"""

import importlib
from pathlib import Path
from typing import Annotated, Any, Literal

import fire
from pydantic import Discriminator
from torch.utils.data import DataLoader

from param_decomp.base_config import BaseConfig
from param_decomp.batch_and_loss_fns import RunBatch
from param_decomp.distributed import DistributedState, is_main_process
from param_decomp.log import logger
from param_decomp.optimize import EvalLoop, optimize
from param_decomp_lab.batch_and_loss_fns import make_run_batch as _make_run_batch
from param_decomp_lab.batch_and_loss_fns import recon_loss_kl
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
from param_decomp_lab.experiments.utils import ExperimentConfig, save_run_meta
from param_decomp_lab.infra.paths import ModelPath
from param_decomp_lab.infra.run_files import generate_run_id
from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR
from param_decomp_lab.run_sink import RunSink
from param_decomp_lab.seed import set_seed


def _resolve_class(fqn: str) -> type:
    """Load a class from a fully-qualified name, e.g. 'transformers.LlamaForCausalLM'."""
    module_path, _, class_name = fqn.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class HFTarget(BaseConfig):
    """Load a HuggingFace model via `<class>.from_pretrained(<hub_id>)`."""

    kind: Literal["hf"] = "hf"
    model_class: str
    model_name: str


class PretrainedTarget(BaseConfig):
    """Load an in-repo lab-pretrained model from a wandb/local pretrain run.

    `run_path` accepts any form `PretrainRunInfo.from_path` accepts: compact W&B
    (`entity/project/runId`), full W&B (`entity/project/runs/runId`), or a local
    checkpoint path.
    """

    kind: Literal["pretrained"] = "pretrained"
    model_class: str
    run_path: ModelPath


LMTargetSpec = Annotated[
    HFTarget | PretrainedTarget,
    Discriminator("kind"),
]


class LMTargetConfig(BaseConfig):
    """How to load the LM target + how to extract its prediction tensor."""

    spec: LMTargetSpec
    output_extract: int | str | None = "logits"


class LMExperimentConfig(ExperimentConfig[LMTargetConfig, LMDataConfig]):
    pass


TARGET_CONFIG_TYPE = LMTargetConfig
DATA_CONFIG_TYPE = LMDataConfig


def build_target(target_cfg: LMTargetConfig) -> Any:
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


def build_loader(
    target_cfg: LMTargetConfig,
    data_cfg: LMDataConfig,
    *,
    split: Literal["train", "eval"],
    device: str,
    batch_size: int,
    dist_state: DistributedState | None = None,
    seed: int | None = None,
) -> DataLoader[Any]:
    """Eval seed is offset by 1 so eval shuffles differently from train when both
    are constructed from the same `pd_config.seed`."""
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
    return _make_run_batch(target_cfg.output_extract)


@with_distributed_cleanup
def main(config_path: str | Path) -> None:
    cfg = LMExperimentConfig.from_file(config_path)

    dist_state = init_distributed()
    if is_main_process():
        logger.info(f"Distributed state: {dist_state}")
    set_seed(cfg.pd.seed)
    device = get_device()
    cfg = cfg.model_copy(update={"runtime": cfg.runtime.model_copy(update={"device": device})})

    target_model = build_target(cfg.target)

    train_loader = build_loader(
        cfg.target,
        cfg.data,
        split="train",
        device=device,
        batch_size=cfg.pd.batch_size,
        dist_state=dist_state,
        seed=cfg.pd.seed,
    )
    eval_loop = _build_eval_loop(cfg, device, dist_state)

    run_id = generate_run_id("param_decomp")
    out_dir = PARAM_DECOMP_OUT_DIR / "decompositions" / run_id if is_main_process() else None
    sink = RunSink.local(out_dir) if out_dir is not None else RunSink.silent()
    save_run_meta(out_dir, kind="lm", cfg=cfg)

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
    if cfg.eval is None:
        return None
    eval_loader = build_loader(
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
