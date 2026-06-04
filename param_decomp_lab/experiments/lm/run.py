"""LM PD experiment: YAML -> `Trainer` glue + `SavedLMRun` reload + resumption.

Single-pool only — the 3-pool path is its own composition root
(`experiments.lm.three_pool_run`, entry point `pd-lm-3pool`). This module keeps the
pure, pool-agnostic builders (`build_target`, `build_lm_loader`, `make_run_batch`,
`_build_eval_loop`, `_split_metrics_by_slow`, `_resolve_train_run_id`) that both paths
import.

Both the fresh-run path (`main`) and the reload path (`SavedLMRun`) share the
module-level `build_target` / `build_lm_loader` / `make_run_batch`. The resume
path (`main --resume <yaml>`) reads a parent run's `experiment_config.yaml` plus
`training_<step>.pth`, rebuilds a `Trainer` via `Trainer.from_snapshot`, and
continues training.

Run via `pd-lm path/to/config.yaml` (fresh) or `pd-lm --resume path/to/resume.yaml`
(resume). Pass `--dp N` to submit a DDP SLURM job (single-node for N <= 8,
multi-node for N > 8 — N must then be a multiple of 8). For local DDP, invoke
directly via
`torchrun --standalone --nproc_per_node=N -m param_decomp_lab.experiments.lm.run config.yaml`.
"""

import importlib
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol

import fire
import torch
import torch.nn as nn
from pydantic import Discriminator
from torch.utils.data import DataLoader

from param_decomp.base_config import BaseConfig
from param_decomp.batch_and_loss_fns import RunBatch
from param_decomp.component_model import ComponentModel
from param_decomp.configs import PDConfig
from param_decomp.distributed import DistributedState, is_main_process
from param_decomp.log import logger
from param_decomp.optimize import EvalLoop, Trainer
from param_decomp.training_state import TrainingState
from param_decomp_lab.batch_and_loss_fns import make_run_batch as _make_run_batch
from param_decomp_lab.batch_and_loss_fns import recon_loss_kl
from param_decomp_lab.component_model_io import load_component_model
from param_decomp_lab.distributed import (
    ensure_cached_and_call,
    get_device,
    init_distributed,
    with_distributed_cleanup,
)
from param_decomp_lab.eval_metrics import EVAL_METRIC_CLASSES, build_eval_metrics
from param_decomp_lab.eval_metrics.autointerp_labels import AutointerpRunContext
from param_decomp_lab.experiments.lm.data import (
    LMDataConfig,
    collate_fn_for,
    create_lm_data_loader,
    rank_batch_size,
)
from param_decomp_lab.experiments.utils import (
    EXPERIMENT_CONFIG_FILENAME,
    EvalConfig,
    ExperimentConfig,
    init_pd_run,
)
from param_decomp_lab.infra.ddp_launch import build_ddp_launch
from param_decomp_lab.infra.git import create_git_snapshot
from param_decomp_lab.infra.paths import ModelPath
from param_decomp_lab.infra.run_files import generate_run_id, resolve_run_files
from param_decomp_lab.infra.settings import DEFAULT_PARTITION_NAME, REPO_ROOT
from param_decomp_lab.infra.slurm import SlurmConfig, generate_script, submit_slurm_job
from param_decomp_lab.infra.wandb import get_wandb_entity, parse_wandb_run_path
from param_decomp_lab.resumption import (
    ResumeConfig,
    ResumeProvenance,
    read_training_snapshot,
    resolve_step,
)
from param_decomp_lab.run_sink import OnePoolSink
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


class HFWeightsInVendored(BaseConfig):
    """Load HF pretrained weights into a vendored `param_decomp_lab.experiments.lm.pretrain.models.*`
    architecture via `<class>.from_hf_pretrained(<hub_id>)`.

    Useful when the decomposition target needs structural changes vs HF — e.g.
    `GPT2Simple`'s separate q/k/v projections vs HF's fused `c_attn`.
    """

    kind: Literal["hf_weights_in_vendored"] = "hf_weights_in_vendored"
    model_class: str  # must expose `from_hf_pretrained`
    model_name: str  # HF hub id


LMTargetSpec = Annotated[
    HFTarget | PretrainedTarget | HFWeightsInVendored,
    Discriminator("kind"),
]


class LMTargetConfig(BaseConfig):
    """Config for the LM target model and how to extract the prediction tensor.

    `output_extract` (passed to `make_run_batch`) pulls the prediction tensor out of the
    model's forward output (default `"logits"`).
    """

    spec: LMTargetSpec
    output_extract: int | str | None = "logits"
    activation_checkpointing: bool = False
    """If True and the target exposes `enable_activation_checkpointing()`, turn on
    per-block gradient checkpointing on the frozen target forward. Trades ~33% extra
    compute for ~10–15x less stored activation memory under 3-pool — the main lever for
    raising `b_per_rank` on deep targets."""
    weights_dtype: Literal["float32", "bfloat16"] = "float32"
    """dtype for the FROZEN target weights. `bfloat16` halves the target's resident footprint
    on every pool (the dominant resident term for an 8B target) — for natively-bf16 models the
    matmuls already run bf16 under autocast, so this only changes residual/norm accumulation
    precision (measured ~5e-4 nats KL on Llama-3.1-8B clean logits, negligible vs recon KLs).
    Only the frozen target is cast; trained V/U components stay fp32 (their AdamW master)."""


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
        case HFWeightsInVendored():
            assert hasattr(cls, "from_hf_pretrained"), (
                f"HFWeightsInVendored target requires {spec.model_class!r} to expose a "
                "`from_hf_pretrained` classmethod"
            )
            target_model = ensure_cached_and_call(cls.from_hf_pretrained, spec.model_name)
    if target_cfg.activation_checkpointing:
        assert hasattr(target_model, "enable_activation_checkpointing"), (
            f"activation_checkpointing=True but {type(target_model).__name__} has no "
            "`enable_activation_checkpointing()` method"
        )
        target_model.enable_activation_checkpointing()
    if target_cfg.weights_dtype == "bfloat16":
        # Frozen target only — make_components creates V/U as fp32 nn.Parameters regardless,
        # so componentizing after this keeps the trained components in fp32.
        target_model = target_model.to(torch.bfloat16)
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
            path, config_filename=EXPERIMENT_CONFIG_FILENAME, checkpoint_prefix="model"
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
    config_path: str | Path | None = None,
    *,
    resume: str | Path | None = None,
    group: str | None = None,
    tags: str | None = None,
    dp: int | None = None,
    partition: str | None = DEFAULT_PARTITION_NAME,
    time: str = "72:00:00",
    job_name: str = "pd-lm",
    no_snapshot: bool = False,
    run_id: str | None = None,
) -> None:
    """Run an LM PD experiment end-to-end.

    Args:
        config_path: YAML for a fresh run. Required when not resuming.
        resume: Path to a `ResumeConfig` YAML pointing at a prior run. When set,
            the parent's `experiment_config.yaml` is the source of cfg truth; a new
            `run_id` is allocated and `resume_provenance` (parent dir + step) is stamped
            onto the effective config so it lands in `experiment_config.yaml` +
            `wandb.config`.
        group / tags: wandb-only (no-ops without `wandb:`).
        dp / partition / time / job_name / no_snapshot / run_id: SLURM submission
            knobs. Passing `--dp N` outside torchrun submits a SLURM job: single-node
            for N <= 8, multi-node for N > 8 (N must be a multiple of 8). For local
            DDP, invoke directly via
            `torchrun --standalone --nproc_per_node=N -m param_decomp_lab.experiments.lm.run`.
    """
    if dp is not None and os.environ.get("WORLD_SIZE") is None:
        assert (config_path is not None) != (resume is not None), (
            "--dp SLURM submission requires exactly one of config_path or --resume"
        )
        _submit_slurm(
            config_path,
            resume=resume,
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

    if resume is not None:
        assert config_path is None, "pass either config_path or --resume, not both"
        _resume_main(Path(resume), group=group, tags=tags, run_id=run_id)
    else:
        assert config_path is not None, "must provide either config_path or --resume"
        _fresh_main(Path(config_path), group=group, tags=tags, run_id=run_id)


def _fresh_main(
    config_path: Path,
    *,
    group: str | None,
    tags: str | None,
    run_id: str | None,
) -> None:
    """Fresh-run path: parse YAML, build everything, train from step 0."""
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

    sink = init_pd_run(cfg, sink_class=OnePoolSink, group=group, tags=tags, run_id=run_id)
    eval_loop = _build_eval_loop(cfg, device, dist_state, include_slow=True)
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


def _resume_main(
    resume_cfg_path: Path,
    *,
    group: str | None,
    tags: str | None,
    run_id: str | None,
) -> None:
    """Resume-run path: read parent `experiment_config.yaml` + `training_<step>.pth`,
    rebuild trainer via `from_snapshot`, continue training."""
    resume_cfg = ResumeConfig.from_file(resume_cfg_path)
    parent_cfg = LMExperimentConfig.from_file(resume_cfg.from_run / EXPERIMENT_CONFIG_FILENAME)

    dist_state = init_distributed()
    if is_main_process():
        logger.info(f"Distributed state: {dist_state}")
        logger.info(f"Resuming from {resume_cfg.from_run} @ step {resume_cfg.step}")
    set_seed(parent_cfg.pd.seed)
    device = get_device()

    resolved_step = resolve_step(resume_cfg.from_run, resume_cfg.step)
    effective_cfg = parent_cfg.model_copy(
        update={
            "runtime": parent_cfg.runtime.model_copy(
                update={
                    "device": device,
                    "dp": dist_state.world_size if dist_state is not None else None,
                }
            ),
            "resume_provenance": ResumeProvenance(
                parent_run_dir=resume_cfg.from_run, parent_step=resolved_step
            ),
        }
    )

    snapshot = read_training_snapshot(resume_cfg.from_run, resolved_step)
    assert isinstance(snapshot, TrainingState), (
        f"1-pool resume needs TrainingState; got {type(snapshot).__name__}"
    )
    snapshot.runtime_config["device"] = device

    target_model = build_target(effective_cfg.target)
    train_loader = build_lm_loader(
        effective_cfg.target,
        effective_cfg.data,
        split="train",
        device=device,
        batch_size=effective_cfg.pd.batch_size,
        dist_state=dist_state,
        seed=effective_cfg.pd.seed,
    )
    run_batch = make_run_batch(effective_cfg.target)

    sink = init_pd_run(effective_cfg, sink_class=OnePoolSink, group=group, tags=tags, run_id=run_id)
    eval_loop = _build_eval_loop(effective_cfg, device, dist_state, include_slow=True)
    try:
        trainer = Trainer.from_snapshot(
            snapshot,
            target_model=target_model,
            run_batch=run_batch,
            reconstruction_loss=recon_loss_kl,
        )
        trainer.run(train_loader, sink, effective_cfg.cadence, eval_loop)
    finally:
        sink.finish()


def _split_metrics_by_slow(
    metrics: list[Any],
) -> tuple[list[Any], list[Any]]:
    """Split an `EvalConfig.metrics` list into `(slow, fast)`.

    Slowness is read from the metric class's `slow` class-attr — we look it up via
    ``EVAL_METRIC_CLASSES[m.type]``. Filtering at launch time means the parent
    YAML stays the single source of truth: training filters to fast, async eval
    filters to slow, both reading the same `cfg.eval.metrics`.
    """
    slow_metrics: list[Any] = []
    fast_metrics: list[Any] = []
    for m in metrics:
        cls = EVAL_METRIC_CLASSES[m.type]
        if getattr(cls, "slow", False):
            slow_metrics.append(m)
        else:
            fast_metrics.append(m)
    return slow_metrics, fast_metrics


def _resolve_train_run_id(run_path: str | Path) -> str:  # pyright: ignore[reportUnusedFunction]
    """Extract the parent run id from a `SavedLMRun.from_path`-compatible reference.

    Shared helper imported by `experiments.lm.async_eval` and
    `experiments.lm.three_pool_run` (basedpyright can't see the cross-module uses).

    Accepts wandb URL / `entity/project/runId` / bare `p-xxxxxxxx` / local directory
    whose final name is the run id (i.e. `PARAM_DECOMP_OUT_DIR/runs/<run_id>/`).
    """
    s = str(run_path)
    try:
        _, _, run_id = parse_wandb_run_path(s)
        return run_id
    except ValueError:
        pass
    p = Path(s)
    return (p if p.is_dir() else p.parent).name


class _EvalLoopInputs(Protocol):
    """The slice of an experiment config `_build_eval_loop` reads.

    Lets the single-pool `LMExperimentConfig` and the 3-pool
    `ThreePoolLMExperimentConfig` share one eval-loop builder without a common base.
    """

    @property
    def eval(self) -> EvalConfig | None: ...
    @property
    def target(self) -> LMTargetConfig: ...
    @property
    def data(self) -> LMDataConfig: ...
    @property
    def pd(self) -> PDConfig: ...


def _build_eval_loop(
    cfg: _EvalLoopInputs,
    device: str,
    dist_state: DistributedState | None,
    *,
    include_slow: bool,
) -> EvalLoop | None:
    """Build the `EvalLoop` from `cfg.eval`, or `None` when eval is disabled.

    `include_slow` decides what the in-train eval runs. The single-pool path
    passes `True` (slow metrics fire in-train at `slow_every`, logged under
    `slow_eval/`). The 3-pool path passes `False`: its in-train eval is fast-only,
    and slow metrics are picked up later by the async job (which receives the slow
    subset via a temp `EvalConfig` YAML, see
    `submit_slurm_async_consolidate_and_eval` in `experiments.lm.three_pool_run`).
    """
    if cfg.eval is None:
        return None
    slow_metrics, fast_metrics = _split_metrics_by_slow(cfg.eval.metrics)
    metrics = cfg.eval.metrics if include_slow else fast_metrics
    del slow_metrics
    eval_loader = build_lm_loader(
        cfg.target,
        cfg.data,
        split="eval",
        device=device,
        batch_size=cfg.eval.batch_size,
        dist_state=dist_state,
        seed=cfg.pd.seed,
    )
    autointerp_run_context = AutointerpRunContext(
        model_class=cfg.target.spec.model_class,
        dataset_name=cfg.data.dataset_name,
        seq_len=cfg.data.max_seq_len,
        tokenizer_name=cfg.data.tokenizer_name,
    )
    return EvalLoop(
        loader=eval_loader,
        metrics=build_eval_metrics(metrics, autointerp_run_context=autointerp_run_context),
        n_steps=cfg.eval.n_steps,
        every=cfg.eval.every,
        slow_every=cfg.eval.slow_every,
        slow_on_first_step=cfg.eval.slow_on_first_step,
    )


def _submit_slurm(
    config_path: str | Path | None,
    *,
    resume: str | Path | None,
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

    # If the yaml is an absolute path inside REPO_ROOT, rewrite to repo-relative so
    # the SLURM job picks up the snapshot's copy rather than the live worktree.
    yaml_target = resume if resume is not None else config_path
    assert yaml_target is not None
    path = Path(yaml_target)
    if path.is_absolute() and path.is_relative_to(REPO_ROOT):
        yaml_arg = path.relative_to(REPO_ROOT).as_posix()
    else:
        yaml_arg = str(yaml_target)

    if resume is not None:
        base_parts = [
            "-m",
            "param_decomp_lab.experiments.lm.run",
            "--resume",
            yaml_arg,
            "--run_id",
            run_id,
        ]
    else:
        base_parts = ["-m", "param_decomp_lab.experiments.lm.run", yaml_arg, "--run_id", run_id]
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

    wandb_url = _wandb_url_for_config(config_path, run_id) if config_path is not None else None

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
