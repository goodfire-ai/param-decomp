"""LM PD experiment: YAML -> `optimize()` glue, plus the `SavedLMRun` reload class.

Both the fresh-run path (`main`) and the reload path share the module-level
`build_target` / `build_lm_loader` / `make_run_batch`. Run via
`pd-lm path/to/config.yaml`; pass `--dp N` to submit a DDP SLURM job (single-node for
N <= 8, multi-node for N > 8 — N must then be a multiple of 8). For local DDP, invoke
`torchrun --standalone --nproc_per_node=N -m param_decomp_lab.experiments.lm.run` directly.
"""

import importlib
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import fire
import torch.nn as nn
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
from param_decomp_lab.infra.ddp_launch import build_ddp_launch
from param_decomp_lab.infra.git import create_git_snapshot
from param_decomp_lab.infra.paths import ModelPath
from param_decomp_lab.infra.run_files import generate_run_id, resolve_run_files
from param_decomp_lab.infra.settings import DEFAULT_PARTITION_NAME, REPO_ROOT
from param_decomp_lab.infra.slurm import SlurmConfig, generate_script, submit_slurm_job
from param_decomp_lab.infra.wandb import get_wandb_entity
from param_decomp_lab.seed import set_seed


def _resolve_class(fqn: str) -> type:
    """Load a class from a fully-qualified name, e.g. 'transformers.LlamaForCausalLM'."""
    module_path, _, class_name = fqn.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class HFTarget(BaseConfig):
    """Load a HuggingFace model via `<model_class>.from_pretrained(<model_name>)`."""

    kind: Literal["hf"] = "hf"
    model_class: str
    model_name: str


class PretrainedTarget(BaseConfig):
    """Load an in-repo lab-pretrained model.

    `run_path` accepts any form `PretrainRunInfo.from_path` does — compact W&B
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
    """Config for the LM target model and how to extract the prediction tensor.

    `output_extract` (passed to `make_run_batch`) pulls the prediction tensor out of the
    model's forward output (default `"logits"`).
    """

    spec: LMTargetSpec
    output_extract: int | str | None = "logits"


class LMExperimentConfig(ExperimentConfig[LMTargetConfig, LMDataConfig]):
    pass


def build_target(target_cfg: LMTargetConfig) -> nn.Module:
    """Load the LM target model in eval mode, dispatching on `target_cfg.spec.kind`."""
    spec = target_cfg.spec
    cls = _resolve_class(spec.model_class)
    match spec:
        case HFTarget():
            target_model = ensure_cached_and_call(cls.from_pretrained, spec.model_name)
        case PretrainedTarget():
            from param_decomp_lab.experiments.lm.pretrain.run_info import PretrainRunInfo

            run_info = ensure_cached_and_call(PretrainRunInfo.from_path, spec.run_path)
            # Older PretrainRunInfo objects predate model_type; default it from the model class.
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
    """LM `DataLoader` for the requested split.

    The eval seed is offset by 1 so eval shuffles differently from train when both come
    from the same `pd_config.seed`.
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
    return _make_run_batch(target_cfg.output_extract)


@dataclass(frozen=True)
class SavedLMRun:
    """Handle to a completed LM PD run on disk or in W&B."""

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
    dp: int | None = None,
    partition: str | None = DEFAULT_PARTITION_NAME,
    time: str = "72:00:00",
    job_name: str = "pd-lm",
    no_snapshot: bool = False,
    run_id: str | None = None,
) -> None:
    """Run an LM PD experiment end-to-end from a YAML config.

    Parses the YAML, initialises DDP, builds the target / loaders / eval loop, writes
    `run_meta.yaml`, and calls `optimize(...)`. Non-main ranks use a silent sink.
    `group` / `tags` are wandb-only (no-ops without `wandb:`). Passing `--dp N` outside
    torchrun submits a SLURM job: single-node for N <= 8, multi-node for N > 8 (N must
    be a multiple of 8). For local DDP, invoke
    `torchrun --standalone --nproc_per_node=N -m param_decomp_lab.experiments.lm.run`.
    """
    if dp is not None and os.environ.get("WORLD_SIZE") is None:
        _submit_slurm(
            config_path,
            dp=dp,
            group=group,
            tags=tags,
            partition=partition,
            time=time,
            job_name=job_name,
            no_snapshot=no_snapshot,
            run_id=run_id,
        )
        return

    cfg = LMExperimentConfig.from_file(config_path)

    dist_state = init_distributed()
    if is_main_process():
        logger.info(f"Distributed state: {dist_state}")
    set_seed(cfg.pd.seed)
    device = get_device()
    cfg = cfg.model_copy(
        update={
            "runtime": cfg.runtime.model_copy(
                update={
                    "device": device,
                    "dp": dist_state.world_size if dist_state is not None else None,
                }
            )
        }
    )

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

    sink = init_pd_run(cfg, group=group, tags=tags, run_id=run_id)

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
    """Build the `EvalLoop` from `cfg.eval`, or `None` when eval is disabled."""
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


def _submit_slurm(
    config_path: str | Path,
    *,
    dp: int,
    group: str | None,
    tags: str | None,
    partition: str | None,
    time: str,
    job_name: str,
    no_snapshot: bool,
    run_id: str | None,
) -> None:
    run_id = run_id or generate_run_id("param_decomp")
    snapshot_ref: str | None = None
    commit_hash = "no-snapshot"
    if not no_snapshot:
        snapshot_ref, commit_hash = create_git_snapshot(snapshot_id=run_id)
        logger.info(f"Created git snapshot: {snapshot_ref} ({commit_hash[:8]})")

    # If the config is an absolute path inside REPO_ROOT, rewrite to repo-relative so
    # the SLURM job picks up the snapshot's copy rather than the live worktree.
    path = Path(config_path)
    if path.is_absolute() and path.is_relative_to(REPO_ROOT):
        config_arg = path.relative_to(REPO_ROOT).as_posix()
    else:
        config_arg = str(config_path)

    base_parts = ["-m", "param_decomp_lab.experiments.lm.run", config_arg, "--run_id", run_id]
    if group is not None:
        base_parts += ["--group", group]
    if tags is not None:
        base_parts += ["--tags", tags]
    base_command = shlex.join(base_parts)

    launch = build_ddp_launch(
        base_command,
        dp=dp,
        job_name=job_name,
        snapshot_ref=snapshot_ref,
        port_seed=run_id,
    )
    slurm_config = SlurmConfig(
        job_name=job_name,
        partition=partition,
        n_gpus=launch.gpus_per_node,
        n_nodes=launch.n_nodes,
        time=time,
        snapshot_ref=snapshot_ref,
        comment=run_id,
    )
    script = generate_script(slurm_config, launch.command, env=launch.env)
    result = submit_slurm_job(script, "lm")

    wandb_url = _wandb_url_for_config(config_path, run_id)

    logger.section("LM PD job submitted!")
    summary: dict[str, str | None] = {
        "Run ID": run_id,
        "Job ID": result.job_id,
        "Log file": result.log_pattern,
        "Script": str(result.script_path),
        "Snapshot": f"{snapshot_ref} ({commit_hash[:8]})" if snapshot_ref else "(none)",
    }
    if wandb_url is not None:
        summary["WandB run URL"] = wandb_url
    logger.values(summary)


def _wandb_url_for_config(config_path: str | Path, run_id: str) -> str | None:
    cfg = LMExperimentConfig.from_file(config_path)
    if cfg.wandb is None:
        return None
    entity = cfg.wandb.entity or get_wandb_entity()
    return f"https://wandb.ai/{entity}/{cfg.wandb.project}/runs/{run_id}"


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
