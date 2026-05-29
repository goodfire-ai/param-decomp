"""LM PD experiment: YAML -> `Trainer` glue + `SavedLMRun` reload + resumption.

Both the fresh-run path (`main`) and the reload path (`SavedLMRun`) share the
module-level `build_target` / `build_lm_loader` / `make_run_batch`. The resume
path (`main --resume <yaml>`) reads a parent run's `run_meta.yaml` plus
`training_<step>.pth`, rebuilds a `Trainer` via `Trainer.from_snapshot`, and
continues training.

Run via `pd-lm path/to/config.yaml` (fresh) or `pd-lm --resume path/to/resume.yaml`
(resume). Pass `--dp N` to submit a DDP SLURM job (single-node for N <= 8,
multi-node for N > 8 — N must then be a multiple of 8). For local DDP, invoke
directly via
`torchrun --standalone --nproc_per_node=N -m param_decomp_lab.experiments.lm.run config.yaml`.

Async consolidation + slow-eval of 3-pool saves lives in
``param_decomp_lab.experiments.lm.async_eval``; the helper
:func:`submit_slurm_async_consolidate_and_eval` below submits an sbatch job that
assembles the canonical checkpoint from the train loop's per-rank partials and
then runs the parent's slow metrics against it.
"""

import atexit
import datetime
import importlib
import json
import os
import shlex
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import fire
import torch
import torch.distributed as dist
import torch.nn as nn
from pydantic import Discriminator
from torch.utils.data import DataLoader

from param_decomp.base_config import BaseConfig
from param_decomp.batch_and_loss_fns import RunBatch
from param_decomp.component_model import ComponentModel
from param_decomp.distributed import DistributedState, is_main_process
from param_decomp.log import logger
from param_decomp.optimize import EvalLoop, Trainer
from param_decomp.training_state import ThreePoolTrainingState, TrainingState
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
from param_decomp_lab.infra.settings import (
    DEFAULT_PARTITION_NAME,
    PARAM_DECOMP_OUT_DIR,
    REPO_ROOT,
)
from param_decomp_lab.infra.slurm import SlurmConfig, generate_script, submit_slurm_job
from param_decomp_lab.infra.wandb import get_wandb_entity, parse_wandb_run_path
from param_decomp_lab.resumption import (
    ResumeConfig,
    ResumeProvenance,
    read_training_snapshot,
    resolve_step,
    write_provenance,
)
from param_decomp_lab.run_sink import OnePoolSink, ThreePoolSink
from param_decomp_lab.seed import set_seed
from param_decomp_lab.three_pool import ThreePoolConfig, ThreePoolTrainer
from param_decomp_lab.three_pool.consolidate import SNAPSHOT_SCRATCH_DIRNAME


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


class LMExperimentConfig(ExperimentConfig[LMTargetConfig, LMDataConfig]):
    three_pool: ThreePoolConfig | None = None
    """When set, training runs under the 3-pool strategy
    (:class:`param_decomp_lab.three_pool.ThreePoolTrainer`) instead of single-process
    :class:`param_decomp.optimize.Trainer`."""


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
            the parent's `run_meta.yaml` is the source of cfg truth; a new
            `run_id` + sibling `resume_provenance.yaml` are written.
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


def _agree_on_run_id_three_pool(run_id: str | None, dist_state: DistributedState | None) -> str:
    """Broadcast (or generate-then-broadcast) a single run id across all ranks.

    3-pool snapshot uses a file-based gather under
    ``PARAM_DECOMP_OUT_DIR/decompositions/<run_id>/.snapshot_scratch/``; every
    rank must compute the same path.
    """
    if is_main_process() and run_id is None:
        run_id = generate_run_id("param_decomp")
    if dist_state is not None:
        objs: list[str | None] = [run_id]
        dist.broadcast_object_list(objs, src=0)
        run_id = objs[0]
    assert run_id is not None
    return run_id


def _install_first_fail_marker(rank: int) -> None:
    """On uncaught exception, write a structured marker to shared FS so a
    debugger can identify which rank died and what hit it without grepping
    through GB of NCCL log noise across N ranks. Always-on; cost is one
    excepthook registration.

    Writes ``$HOME/pd_first_fail/$SLURM_JOB_ID/rank<R>.json`` containing
    ``{rank, exception_type, exception_message, traceback, timestamp_utc,
    pid}``. Chains to the previous excepthook so behavior is otherwise
    unchanged.

    Open follow-up (#11 in the task list): once any rank writes its marker,
    the rest of the world will block in their next collective for up to
    30 min before monitoredBarrier times out. A future change should
    ``dist.destroy_process_group()`` + ``os._exit(1)`` here so siblings
    fast-fail in seconds via ``TORCH_NCCL_ASYNC_ERROR_HANDLING``. Skipped
    for now because doing that wrong can hang siblings *worse*.
    """
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    out_dir = Path.home() / "pd_first_fail" / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"rank{rank}.json"

    prev_excepthook = sys.excepthook

    def _excepthook(exctype: type[BaseException], value: BaseException, tb: Any) -> None:
        try:
            payload = {
                "rank": rank,
                "exception_type": exctype.__name__,
                "exception_message": str(value),
                "traceback": "".join(traceback.format_exception(exctype, value, tb)),
                "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
                "pid": os.getpid(),
            }
            out_path.write_text(json.dumps(payload, indent=2))
            print(f"[first-fail] rank={rank} wrote {out_path}", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[first-fail] failed to write marker: {e}", file=sys.stderr, flush=True)
        prev_excepthook(exctype, value, tb)

    sys.excepthook = _excepthook


def _maybe_enable_memory_profile(rank: int) -> None:
    """Opt-in CUDA memory-history recorder for offline ``memory_viz`` analysis.

    Activated by env vars:
      * ``PD_MEMORY_PROFILE_RANKS=0,32,96`` — comma-separated ranks to profile.
      * ``PD_MEMORY_PROFILE_OUT=/abs/path/to/dir`` — dump directory.

    Dumps to ``<dir>/mem_rank<R>.pickle`` on normal exit and on uncaught
    exception. Load the pickle at https://pytorch.org/memory_viz.
    """
    prof_ranks_env = os.environ.get("PD_MEMORY_PROFILE_RANKS")
    if not prof_ranks_env:
        return
    if rank not in {int(r) for r in prof_ranks_env.split(",") if r.strip()}:
        return
    out_dir = Path(os.environ["PD_MEMORY_PROFILE_OUT"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"mem_rank{rank}.pickle"
    logger.info(f"[mem-profile] recording rank={rank} → {out_path}")
    torch.cuda.memory._record_memory_history(max_entries=200_000)

    def _dump() -> None:
        torch.cuda.memory._dump_snapshot(str(out_path))
        logger.info(f"[mem-profile] dumped rank={rank} → {out_path}")

    atexit.register(_dump)
    prev_excepthook = sys.excepthook

    def _excepthook(exctype: type[BaseException], value: BaseException, tb: Any) -> None:
        _dump()
        prev_excepthook(exctype, value, tb)

    sys.excepthook = _excepthook


def _maybe_build_torch_profiler(trainer: ThreePoolTrainer) -> torch.profiler.profile | None:
    """Opt-in ``torch.profiler.profile`` for the listed ranks.

    Activated by env vars:
      * ``PD_TORCH_PROFILE_RANKS=0,96,100`` — ranks to profile. Typically one
        per pool (LW block-0 leader, CI leader, PPGD leader).
      * ``PD_TORCH_PROFILE_OUT=/abs/path/to/dir`` — trace dump directory.
      * ``PD_TORCH_PROFILE_SKIP_FIRST`` (default 20) and
        ``PD_TORCH_PROFILE_ACTIVE`` (default 3) — schedule knobs.
      * ``PD_TORCH_PROFILE_MEMORY=0`` — disable profile_memory (CUPTI memory
        instrumentation is the heaviest subsystem; toggle if you suspect
        it's confounding measurements).
      * ``PD_TORCH_PROFILE_STACK=1`` — Python source location per op
        (~25% step-time hit, much larger traces, but huge readability win).
      * ``PD_TORCH_PROFILE_MODULES=1`` — nn.Module hierarchy labels (cheap;
        useful for per-site decomposition labels).
      * ``PD_TORCH_PROFILE_SHAPES=1`` — per-op input tensor shapes.

    Returns the profile context (caller passes it to ``trainer.run(profiler=
    ...)`` which enters it) or ``None`` if this rank isn't profiled.

    Verified at production scale (104 ranks, gpt2-xl, 3 profiled ranks one
    per pool) on 2026-05-28 after commit 497e1542 removed a cosmetic pre-step
    barrier that was the previously-suspected CUPTI deadlock.
    """
    prof_ranks_env = os.environ.get("PD_TORCH_PROFILE_RANKS", "").strip()
    if not prof_ranks_env:
        return None
    prof_ranks = {int(r) for r in prof_ranks_env.split(",") if r.strip()}
    my_rank = trainer.ctx.role.rank
    if my_rank not in prof_ranks:
        return None
    out_dir = Path(os.environ["PD_TORCH_PROFILE_OUT"])
    out_dir.mkdir(parents=True, exist_ok=True)
    skip_first = int(os.environ.get("PD_TORCH_PROFILE_SKIP_FIRST", "20"))
    active = int(os.environ.get("PD_TORCH_PROFILE_ACTIVE", "3"))
    profile_memory = os.environ.get("PD_TORCH_PROFILE_MEMORY", "1") != "0"
    with_stack = os.environ.get("PD_TORCH_PROFILE_STACK", "0") == "1"
    with_modules = os.environ.get("PD_TORCH_PROFILE_MODULES", "0") == "1"
    record_shapes = os.environ.get("PD_TORCH_PROFILE_SHAPES", "0") == "1"

    pool = trainer.ctx.kind
    trace_path = out_dir / f"trace_{pool}_rank{my_rank}.json"
    logger.info(
        f"[torch-profile] rank={my_rank} pool={pool} → {trace_path} "
        f"(skip_first={skip_first}, active={active}, profile_memory={profile_memory}, "
        f"with_stack={with_stack}, with_modules={with_modules}, record_shapes={record_shapes})"
    )

    def on_trace_ready(prof: torch.profiler.profile) -> None:
        prof.export_chrome_trace(str(trace_path))
        logger.info(f"[torch-profile] rank={my_rank} wrote {trace_path}")

    return torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            skip_first=skip_first, wait=0, warmup=1, active=active, repeat=1
        ),
        on_trace_ready=on_trace_ready,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
        with_modules=with_modules,
    )


def _fresh_main(
    config_path: Path,
    *,
    group: str | None,
    tags: str | None,
    run_id: str | None,
) -> None:
    """Fresh-run path: parse YAML, dispatch on pool config, train from step 0."""
    cfg = LMExperimentConfig.from_file(config_path)

    dist_state = init_distributed()
    if is_main_process():
        logger.info(f"Distributed state: {dist_state}")
    _install_first_fail_marker(dist_state.rank if dist_state is not None else 0)
    _maybe_enable_memory_profile(dist_state.rank if dist_state is not None else 0)
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
    is_three_pool = cfg.three_pool is not None
    train_loader = build_lm_loader(
        cfg.target,
        cfg.data,
        split="train",
        device=device,
        batch_size=cfg.pd.batch_size,
        # 3-pool requires the full global batch on every rank — each pool
        # slices it locally. Single-pool uses standard DistributedSampler.
        dist_state=None if is_three_pool else dist_state,
        seed=cfg.pd.seed,
    )

    if cfg.three_pool is not None:
        run_id = _agree_on_run_id_three_pool(run_id, dist_state)
        out_dir = PARAM_DECOMP_OUT_DIR / "decompositions" / run_id
        scratch_dir = out_dir / SNAPSHOT_SCRATCH_DIRNAME
        three_sink = init_pd_run(
            cfg,
            sink_class=ThreePoolSink,
            group=group,
            tags=tags,
            run_id=run_id,
            on_save=lambda step: submit_slurm_async_consolidate_and_eval(
                out_dir, step=step, parent_cfg=cfg
            ),
        )
        # Multi-pool eval mirrors the train data-handling contract: full eval
        # batch on every rank, sliced internally by each pool. So we pass
        # dist_state=None.
        three_eval_loop = _build_eval_loop(cfg, device, dist_state=None)
        try:
            three_trainer = ThreePoolTrainer(
                target_model=target_model,
                run_batch=make_run_batch(cfg.target),
                reconstruction_loss=recon_loss_kl,
                pd_config=cfg.pd,
                runtime_config=cfg.runtime,
                three_pool_config=cfg.three_pool,
            )
            three_trainer.run(
                train_loader,
                three_sink,
                cfg.cadence,
                scratch_dir=scratch_dir,
                eval_loop=three_eval_loop,
                profiler=_maybe_build_torch_profiler(three_trainer),
            )
        finally:
            three_sink.finish()
        return

    one_sink = init_pd_run(cfg, sink_class=OnePoolSink, group=group, tags=tags, run_id=run_id)
    eval_loop = _build_eval_loop(cfg, device, dist_state)
    try:
        trainer = Trainer(
            target_model=target_model,
            run_batch=make_run_batch(cfg.target),
            reconstruction_loss=recon_loss_kl,
            pd_config=cfg.pd,
            runtime_config=cfg.runtime,
        )
        trainer.run(train_loader, one_sink, cfg.cadence, eval_loop)
    finally:
        one_sink.finish()


def _resume_main(
    resume_cfg_path: Path,
    *,
    group: str | None,
    tags: str | None,
    run_id: str | None,
) -> None:
    """Resume-run path: read parent `run_meta.yaml` + `training_<step>.pth`,
    rebuild trainer via `from_snapshot`, continue training."""
    resume_cfg = ResumeConfig.from_file(resume_cfg_path)
    parent_cfg = LMExperimentConfig.from_file(resume_cfg.from_run / RUN_META_FILENAME)

    dist_state = init_distributed()
    if is_main_process():
        logger.info(f"Distributed state: {dist_state}")
        logger.info(f"Resuming from {resume_cfg.from_run} @ step {resume_cfg.step}")
    _install_first_fail_marker(dist_state.rank if dist_state is not None else 0)
    _maybe_enable_memory_profile(dist_state.rank if dist_state is not None else 0)
    set_seed(parent_cfg.pd.seed)
    device = get_device()

    effective_cfg = parent_cfg.model_copy(
        update={
            "runtime": parent_cfg.runtime.model_copy(
                update={
                    "device": device,
                    "dp": dist_state.world_size if dist_state is not None else None,
                }
            ),
        }
    )

    resolved_step = resolve_step(resume_cfg.from_run, resume_cfg.step)
    snapshot = read_training_snapshot(resume_cfg.from_run, resolved_step)
    snapshot.runtime_config["device"] = device

    target_model = build_target(effective_cfg.target)
    is_three_pool = effective_cfg.three_pool is not None
    train_loader = build_lm_loader(
        effective_cfg.target,
        effective_cfg.data,
        split="train",
        device=device,
        batch_size=effective_cfg.pd.batch_size,
        dist_state=None if is_three_pool else dist_state,
        seed=effective_cfg.pd.seed,
    )
    run_batch = make_run_batch(effective_cfg.target)

    if effective_cfg.three_pool is not None:
        run_id = _agree_on_run_id_three_pool(run_id, dist_state)
        out_dir = PARAM_DECOMP_OUT_DIR / "decompositions" / run_id
        scratch_dir = out_dir / SNAPSHOT_SCRATCH_DIRNAME
        three_sink = init_pd_run(
            effective_cfg,
            sink_class=ThreePoolSink,
            group=group,
            tags=tags,
            run_id=run_id,
            on_save=lambda step: submit_slurm_async_consolidate_and_eval(
                out_dir, step=step, parent_cfg=effective_cfg
            ),
        )
        if three_sink.out_dir is not None:
            write_provenance(
                three_sink.out_dir,
                ResumeProvenance(parent_run_dir=resume_cfg.from_run, parent_step=resolved_step),
            )
        three_eval_loop = _build_eval_loop(effective_cfg, device, dist_state=None)
        try:
            assert isinstance(snapshot, ThreePoolTrainingState), (
                f"3-pool resume needs ThreePoolTrainingState; got {type(snapshot).__name__}"
            )
            three_trainer = ThreePoolTrainer.from_snapshot(
                snapshot,
                target_model=target_model,
                run_batch=run_batch,
                reconstruction_loss=recon_loss_kl,
            )
            three_trainer.run(
                train_loader,
                three_sink,
                effective_cfg.cadence,
                scratch_dir=scratch_dir,
                eval_loop=three_eval_loop,
                profiler=_maybe_build_torch_profiler(three_trainer),
            )
        finally:
            three_sink.finish()
        return

    one_sink = init_pd_run(
        effective_cfg, sink_class=OnePoolSink, group=group, tags=tags, run_id=run_id
    )
    if one_sink.out_dir is not None:
        write_provenance(
            one_sink.out_dir,
            ResumeProvenance(parent_run_dir=resume_cfg.from_run, parent_step=resolved_step),
        )
    eval_loop = _build_eval_loop(effective_cfg, device, dist_state)
    try:
        assert isinstance(snapshot, TrainingState), (
            f"1-pool resume needs TrainingState; got {type(snapshot).__name__}"
        )
        trainer = Trainer.from_snapshot(
            snapshot,
            target_model=target_model,
            run_batch=run_batch,
            reconstruction_loss=recon_loss_kl,
        )
        trainer.run(train_loader, one_sink, effective_cfg.cadence, eval_loop)
    finally:
        one_sink.finish()


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


def _resolve_train_run_id(run_path: str | Path) -> str:
    """Extract the parent run id from a `SavedLMRun.from_path`-compatible reference.

    Accepts wandb URL / `entity/project/runId` / bare `p-xxxxxxxx` / local directory
    whose final name is the run id (i.e. `PARAM_DECOMP_OUT_DIR/decompositions/<run_id>/`).
    """
    s = str(run_path)
    try:
        _, _, run_id = parse_wandb_run_path(s)
        return run_id
    except ValueError:
        pass
    p = Path(s)
    return (p if p.is_dir() else p.parent).name


def _build_eval_loop(
    cfg: LMExperimentConfig,
    device: str,
    dist_state: DistributedState | None,
) -> EvalLoop | None:
    """Build the `EvalLoop` from `cfg.eval`, or `None` when eval is disabled.

    Slow metrics (class-attr ``slow=True``) are filtered out — in-train eval is
    fast-only. Slow metrics are picked up later by the async job (which receives
    the slow subset via a temp ``EvalConfig`` YAML, see
    :func:`submit_slurm_async_consolidate_and_eval`).
    """
    if cfg.eval is None:
        return None
    _slow_metrics, fast_metrics = _split_metrics_by_slow(cfg.eval.metrics)
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
        metrics=[EVAL_METRIC_CLASSES[m.type](m) for m in fast_metrics],
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


def submit_slurm_async_consolidate_and_eval(
    run_path: str | Path,
    *,
    step: int,
    parent_cfg: "LMExperimentConfig",
    dp: int = 8,
    time: str = "2:00:00",
    partition: str | None = DEFAULT_PARTITION_NAME,
    job_name: str = "pd-lm-consol-eval",
    group: str | None = None,
    tags: str | None = None,
    no_snapshot: bool = False,
) -> None:
    """Submit the async job that consolidates a 3-pool save and runs slow eval.

    Called from 3-pool training right after a save. The train loop has written
    per-rank partials to the scratch dir; this job (off the critical path)
    assembles ``model_<step>.pth`` + ``training_<step>.pth`` from them, prunes old
    ``training_*.pth``, then runs the parent's *slow* eval metrics against the
    assembled ``model_<step>.pth`` (logging into the parent's wandb run at the
    same step). The job is ALWAYS submitted for 3-pool — consolidation is
    mandatory even when there are no slow metrics; in that case the eval pass is a
    no-op (see ``async_eval``).
    """
    from param_decomp_lab.experiments.utils import EvalConfig

    assert parent_cfg.three_pool is not None, (
        "async consolidate+eval is only for 3-pool runs (consolidation has nothing to do otherwise)"
    )

    # Test hook (never set in production): skip the child-job submission so a
    # smoke can drive consolidation/eval out-of-band and stay within a GPU
    # budget. The train loop still writes its partials regardless.
    if os.environ.get("PD_3POOL_SKIP_ASYNC_ONSAVE", "").strip() in ("1", "true"):
        logger.info(f"PD_3POOL_SKIP_ASYNC_ONSAVE set; skipping async job for step {step}")
        return

    # The consolidate+eval job's GPU count. Defaults to `dp`; overridable so a
    # small-topology smoke can keep train + child within one node's 8 GPUs (and,
    # in production, so a cheap eval doesn't have to match the train width).
    dp_override = os.environ.get("PD_ASYNC_EVAL_DP", "").strip()
    if dp_override:
        dp = int(dp_override)

    slow_metrics = (
        _split_metrics_by_slow(parent_cfg.eval.metrics)[0] if parent_cfg.eval is not None else []
    )
    eval_batch_size = parent_cfg.eval.batch_size if parent_cfg.eval is not None else dp
    eval_n_steps = parent_cfg.eval.n_steps if parent_cfg.eval is not None else 1
    slow_eval_cfg = EvalConfig(
        batch_size=eval_batch_size,
        n_steps=eval_n_steps,
        every=1,  # any value; not consumed by async_eval
        slow_every=1,  # any value; not consumed by async_eval
        slow_on_first_step=True,
        metrics=slow_metrics,
    )
    train_run_id = _resolve_train_run_id(run_path)
    scratch = PARAM_DECOMP_OUT_DIR / "decompositions" / train_run_id / ".async_eval_configs"
    scratch.mkdir(parents=True, exist_ok=True)
    cfg_path = scratch / f"slow_eval_step_{step}.yaml"
    slow_eval_cfg.to_file(cfg_path)

    # Reuse the training run's snapshot ref. The training was launched from
    # $HOME/param-decomp and the snapshot already exists there. Creating a new
    # snapshot here would operate on whatever git repo this process happens to
    # be in — when called from inside a training job, that's the node-local
    # /tmp/.../workspace-* clone, which other nodes can't see. Plus, async eval
    # should run the same code as the training that produced the checkpoint.
    eval_run_id = generate_run_id("param_decomp")
    snapshot_ref: str | None = f"refs/runs/snapshot/{train_run_id}" if not no_snapshot else None

    base_command = shlex.join(
        [
            "-m",
            "param_decomp_lab.experiments.lm.async_eval",
            "--run",
            str(run_path),
            "--step",
            str(step),
            "--eval-config",
            str(cfg_path),
            *(["--group", group] if group is not None else []),
            *(["--tags", tags] if tags is not None else []),
        ]
    )
    launch = build_ddp_launch(
        base_command,
        dp=dp,
        job_name=job_name,
        snapshot_ref=snapshot_ref,
        port_seed=eval_run_id,
    )
    slurm_config = SlurmConfig(
        job_name=job_name,
        partition=partition,
        n_gpus=launch.gpus_per_node,
        n_nodes=launch.n_nodes,
        time=time,
        snapshot_ref=snapshot_ref,
        comment=f"async-slow-eval:{train_run_id}@{step}",
    )
    script = generate_script(slurm_config, launch.command, env=launch.env)
    result = submit_slurm_job(script, "lm")
    logger.info(
        f"Async slow-eval submitted: parent={train_run_id} step={step} "
        f"job_id={result.job_id} log={result.log_pattern}"
    )


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
