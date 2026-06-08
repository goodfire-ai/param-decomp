"""3-pool LM PD experiment: YAML -> `ThreePoolTrainer` glue + reload + resumption.

The 3-pool sibling of `experiments.lm.run`. Its config — `ThreePoolLMExperimentConfig`
— bakes the 3-pool's constraints into the types (`ThreePoolConstrainedPDConfig`'s fixed
scalars + typed `ThreePoolLosses` struct; `ThreePoolRuntimeConfig`'s authored
`topology`), so misconfigurations fail at YAML parse on the login node rather than
minutes into a multi-node launch.

Dispatch is by entry point, not a discriminator: `pd-lm-3pool` selects this composition
root (single-pool `pd-lm` -> `experiments.lm.run`), matching the repo's per-experiment
idiom. The pure, pool-agnostic builders (`build_target`, `build_lm_loader`,
`make_run_batch`, `_build_eval_loop`, `_split_metrics_by_slow`, `_resolve_train_run_id`)
are imported from `experiments.lm.run` — only the 3-pool-specific scaffolding (full-batch
loader, run-id agreement, `ThreePoolSink`, scratch dir, profiler / mem-profile /
first-fail debug hooks, async slow-eval) lives here.

Run via `pd-lm-3pool path/to/config.yaml` (fresh) or
`pd-lm-3pool --resume path/to/resume.yaml` (resume). Pass `--dp N` to submit a DDP SLURM
job (single-node for N <= 8, multi-node for N > 8 — N must then be a multiple of 8). For
local DDP, invoke directly via
`torchrun --standalone --nproc_per_node=N -m param_decomp_lab.experiments.lm.three_pool_run config.yaml`.
"""

import atexit
import datetime
import json
import os
import shlex
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import fire
import torch
import torch.distributed as dist
import torch.profiler
from pydantic import model_validator

from param_decomp.base_config import BaseConfig
from param_decomp.component_model import ComponentModel
from param_decomp.configs import Cadence, RuntimeConfig
from param_decomp.distributed import DistributedState, is_main_process
from param_decomp.log import logger
from param_decomp.training_state import ThreePoolTrainingState
from param_decomp_lab.batch_and_loss_fns import recon_loss_kl
from param_decomp_lab.component_model_io import (
    VendoredHarvestModel,
    detect_checkpoint_format,
    load_component_model,
    load_vendored_component_model,
)
from param_decomp_lab.distributed import (
    get_device,
    init_distributed,
    with_distributed_cleanup,
)
from param_decomp_lab.experiments.lm.data import LMDataConfig
from param_decomp_lab.experiments.lm.run import (
    LMTargetConfig,
    _build_eval_loop,
    _resolve_train_run_id,
    _split_metrics_by_slow,
    build_lm_loader,
    build_target,
    make_run_batch,
)
from param_decomp_lab.experiments.utils import (
    EXPERIMENT_CONFIG_FILENAME,
    EvalConfig,
    WandbConfig,
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
from param_decomp_lab.infra.wandb import get_wandb_entity
from param_decomp_lab.resumption import (
    ResumeConfig,
    ResumeProvenance,
    latest_checkpoint_step,
    read_training_snapshot,
    resolve_step,
)
from param_decomp_lab.run_sink import ThreePoolSink
from param_decomp_lab.seed import set_seed
from param_decomp_lab.three_pool import ThreePoolTopology, ThreePoolTrainer
from param_decomp_lab.three_pool.consolidate import SNAPSHOT_SCRATCH_DIRNAME, ppgd_shard_dirname
from param_decomp_lab.three_pool.pd_config import ThreePoolConstrainedPDConfig

# Cross-pool NCCL p2p deadlock guard. An asymmetric topology (fanout>1 on a cross-pool
# edge) at long sequence lengths wedges when a rendezvous-blocked NCCL send/recv kernel
# co-resides with the caching allocator's cudaFree (a device-wide sync). expandable_segments
# remaps virtual address space instead of calling cudaFree, so that sync — and the
# deadlock — never happen. Scoped to the 3-pool path (single-pool runs do no cross-pool
# p2p). Applies to both the train job and the async consolidate+eval job (eval also runs
# cross-pool collectives).
THREE_POOL_SLURM_ENV: dict[str, str] = {
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "PD_TRACE": "1", "PD_TRACE_RANKS": "0,16,79",
    "TORCH_NCCL_TRACE_BUFFER_SIZE": "20000", "TORCH_NCCL_DUMP_ON_TIMEOUT": "1", "TORCH_NCCL_DESYNC_DEBUG": "1",
}


class ThreePoolRuntimeConfig(RuntimeConfig):
    """Core's substrate scalars (`autocast_bf16` / `device` / `dp`) + the normalized
    `topology`.

    Subclasses `RuntimeConfig` so it's substitutable wherever a `RuntimeConfig` is
    expected (`ThreePoolTrainer.__init__`, snapshot serialization) and reuses the three
    scalars rather than duplicating them. `dp` is the launch-derived world readout
    (overwritten from the torchrun world at launch; not authored in 3-pool YAMLs — the
    world size is derived from the resolved topology and asserted == torchrun world in
    `build_world`). `topology` is the normalized `ThreePoolTopology` (per-rank batches +
    site→chunk split; ranks derived) that pairs with `ThreePoolConstrainedPDConfig`.
    Core's `RuntimeConfig` itself stays pool-blind — the topology is added only here, in lab.
    """

    topology: ThreePoolTopology


class ThreePoolLMExperimentConfig(BaseConfig):
    """Full YAML schema for a 3-pool LM PD run. Standalone sibling of
    `LMExperimentConfig` (not a variant of it).

    The cross-field check that couples `pd` to `runtime.topology` (each per-rank batch
    divides the global batch) runs here at load time — the only place both are visible —
    so it fails at parse rather than inside `ThreePoolTrainer.__init__`. The rank-0
    convention is no longer a check: the canonical resolver makes rank 0 the chunk-0
    leader by construction.
    """

    pd: ThreePoolConstrainedPDConfig
    runtime: ThreePoolRuntimeConfig
    cadence: Cadence
    target: LMTargetConfig
    data: LMDataConfig
    eval: EvalConfig | None = None
    wandb: WandbConfig | None = None
    resume_provenance: ResumeProvenance | None = None
    """Set on resumed runs (parent run dir + step); `None` for fresh runs. Lives on the
    config so it flows into `experiment_config.yaml` and `wandb.config` via `init_pd_run`, making a
    resumed run's lineage visible in the wandb UI."""

    @model_validator(mode="after")
    def validate_pd_against_topology(self) -> Self:
        topology = self.runtime.topology
        bs = self.pd.batch_size
        for name, per_rank_batch in (
            ("ci", topology.ci.per_rank_batch),
            ("ppgd", topology.ppgd.per_rank_batch),
            ("chunkwise", topology.chunkwise.per_rank_batch),
        ):
            assert bs % per_rank_batch == 0, (
                f"pd.batch_size ({bs}) must be divisible by topology.{name}.per_rank_batch "
                f"({per_rank_batch})"
            )
        # Site coverage (every resolved chunk site is in decomposition_targets after
        # pattern expansion) needs the loaded target model and stays in `_build_runtime`.
        return self


@dataclass(frozen=True)
class SavedThreePoolLMRun:
    """Handle to a completed 3-pool LM PD run on disk or in W&B.

    Sibling of `SavedLMRun` for the 3-pool config type. `load_model` consumes only
    inherited `PDConfig` fields, so inference / inspection of a 3-pool decomposition
    works identically to a single-pool one.
    """

    cfg: ThreePoolLMExperimentConfig
    checkpoint_path: Path

    @classmethod
    def from_path(cls, path: ModelPath) -> "SavedThreePoolLMRun":
        files = resolve_run_files(
            path, config_filename=EXPERIMENT_CONFIG_FILENAME, checkpoint_prefix="model"
        )
        return cls(
            cfg=ThreePoolLMExperimentConfig.from_file(files.config_path),
            checkpoint_path=files.checkpoint_path,
        )

    def load_model(self) -> ComponentModel | VendoredHarvestModel:
        """Dispatch on the on-disk checkpoint format: post-`e8ff5a64` runs are the vendored
        `LMComponentModel` (wrapped for the harvest surface); earlier ones are core."""
        match detect_checkpoint_format(self.checkpoint_path):
            case "vendored":
                return VendoredHarvestModel(
                    load_vendored_component_model(
                        pd_config=self.cfg.pd,
                        checkpoint_path=self.checkpoint_path,
                        target_model=build_target(self.cfg.target),
                    )
                )
            case "core":
                return load_component_model(
                    pd_config=self.cfg.pd,
                    checkpoint_path=self.checkpoint_path,
                    target_model=build_target(self.cfg.target),
                    run_batch=make_run_batch(self.cfg.target),
                )
            case other:
                raise AssertionError(f"unknown checkpoint format: {other}")


def _agree_on_run_id(run_id: str | None, dist_state: DistributedState | None) -> str:
    """Broadcast (or generate-then-broadcast) a single run id across all ranks.

    The 3-pool save writes per-rank partials under
    `PARAM_DECOMP_OUT_DIR/runs/<run_id>/.snapshot_scratch/`; every rank must
    compute the same path.
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
    """On uncaught exception, write a structured marker to shared FS so a debugger can
    identify which rank died and what hit it without grepping GB of NCCL log noise.

    Writes `$HOME/pd_first_fail/$SLURM_JOB_ID/rank<R>.json`. Chains to the previous
    excepthook so behavior is otherwise unchanged.
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
    """Opt-in CUDA memory-history recorder for offline `memory_viz` analysis.

    Env: `PD_MEMORY_PROFILE_RANKS=0,32,96` (ranks) + `PD_MEMORY_PROFILE_OUT=/abs/dir`.
    Dumps `<dir>/mem_rank<R>.pickle` on normal exit and on uncaught exception. Load at
    https://pytorch.org/memory_viz.
    """
    prof_ranks_env = os.environ.get("PD_MEMORY_PROFILE_RANKS")
    if not prof_ranks_env:
        return
    if rank not in {int(r) for r in prof_ranks_env.split(",") if r.strip()}:
        return
    out_dir = Path(os.environ["PD_MEMORY_PROFILE_OUT"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"mem_rank{rank}.pickle"
    logger.info(f"[mem-profile] recording rank={rank} -> {out_path}")
    torch.cuda.memory._record_memory_history(max_entries=200_000)

    def _dump() -> None:
        torch.cuda.memory._dump_snapshot(str(out_path))
        logger.info(f"[mem-profile] dumped rank={rank} -> {out_path}")

    atexit.register(_dump)
    prev_excepthook = sys.excepthook

    def _excepthook(exctype: type[BaseException], value: BaseException, tb: Any) -> None:
        _dump()
        prev_excepthook(exctype, value, tb)

    sys.excepthook = _excepthook


def _maybe_build_torch_profiler(trainer: ThreePoolTrainer) -> "torch.profiler.profile | None":
    """Opt-in `torch.profiler.profile` for the listed ranks (typically one per pool).

    Env: `PD_TORCH_PROFILE_RANKS=0,96,100` + `PD_TORCH_PROFILE_OUT=/abs/dir`, plus
    schedule / instrumentation knobs (`PD_TORCH_PROFILE_SKIP_FIRST` default 20,
    `_ACTIVE` default 3, `_MEMORY` default on, `_STACK` / `_MODULES` / `_SHAPES` off).
    Returns the profile context (caller passes it to `trainer.run(profiler=...)`) or
    `None` if this rank isn't profiled. Verified at 104 ranks (gpt2-xl) 2026-05-28.
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
        f"[torch-profile] rank={my_rank} pool={pool} -> {trace_path} "
        f"(skip_first={skip_first}, active={active}, profile_memory={profile_memory}, "
        f"with_stack={with_stack}, with_modules={with_modules}, record_shapes={record_shapes})"
    )

    def on_trace_ready(prof: "torch.profiler.profile") -> None:
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


@with_distributed_cleanup
def _submit_world_size(config_path: Path | None, resume: Path | None) -> int:
    """GPU count for a submission, derived purely from the config — `pool_ranks +
    n_chunks * chunk_dp` (no model). A resume reads its parent run's config."""
    if config_path is not None:
        cfg = ThreePoolLMExperimentConfig.from_file(config_path)
    else:
        assert resume is not None
        from_run = ResumeConfig.from_file(resume).from_run
        cfg = ThreePoolLMExperimentConfig.from_file(from_run / EXPERIMENT_CONFIG_FILENAME)
    topo = cfg.runtime.topology
    return topo.world_size(cfg.pd.batch_size, topo.chunkwise.n_chunks)


def main(
    config_path: str | Path | None = None,
    *,
    resume: str | Path | None = None,
    group: str | None = None,
    tags: str | None = None,
    partition: str | None = DEFAULT_PARTITION_NAME,
    qos: str | None = None,
    time: str = "72:00:00",
    job_name: str = "pd-lm-3pool",
    no_snapshot: bool = False,
    run_id: str | None = None,
) -> None:
    """Run a 3-pool LM PD experiment end-to-end.

    Args:
        config_path: YAML for a fresh run. Required when not resuming.
        resume: Path to a `ResumeConfig` YAML pointing at a prior 3-pool run.
        group / tags: wandb-only (no-ops without `wandb:`).
        partition / qos / time / job_name / no_snapshot / run_id: SLURM submission knobs.
            A direct (login-node) invocation submits a SLURM job; the GPU count is derived
            from the config's topology + batch (single-node for <= 8, multi-node for > 8, a
            multiple of 8) — there is no `--dp`. `qos=None` uses the cluster default; pass
            e.g. `opportunistic` to run off-quota.
    """
    if os.environ.get("WORLD_SIZE") is None:
        # Direct (login-node) invocation → submit a SLURM job. A torchrun worker has
        # WORLD_SIZE set in its env and falls through to the training path below.
        assert (config_path is not None) != (resume is not None), (
            "submission requires exactly one of config_path or --resume"
        )
        _submit_slurm(
            config_path,
            resume=resume,
            dp=_submit_world_size(
                Path(config_path) if config_path is not None else None,
                Path(resume) if resume is not None else None,
            ),
            group=group,
            tags=tags,
            partition=partition,
            qos=qos,
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
        _fresh_or_requeue_main(Path(config_path), group=group, tags=tags, run_id=run_id)


def _fresh_or_requeue_main(
    config_path: Path,
    *,
    group: str | None,
    tags: str | None,
    run_id: str | None,
) -> None:
    """Worker entry for a config launch. Resumes in place from this run's own latest
    consolidated checkpoint if one exists (SLURM requeued the job after a save), else
    trains from step 0.

    The requeue re-runs the identical `... <config> --run_id <id>` command. `run_id` is
    therefore always present on a requeue (set by `_submit_slurm`); a checkpoint under the
    run's own dir is the signal that this is a requeue, not a first launch.
    """
    if run_id is not None:
        out_dir = PARAM_DECOMP_OUT_DIR / "runs" / run_id
        prior_step = latest_checkpoint_step(out_dir)
        if prior_step is not None:
            _resume_in_place(config_path, out_dir, prior_step, group=group, tags=tags)
            return
    _fresh_main(config_path, group=group, tags=tags, run_id=run_id)


def _fresh_main(
    config_path: Path,
    *,
    group: str | None,
    tags: str | None,
    run_id: str | None,
) -> None:
    """Fresh-run path: parse YAML, build everything, train from step 0."""
    cfg = ThreePoolLMExperimentConfig.from_file(config_path)

    dist_state = init_distributed()
    if is_main_process():
        logger.info(f"Distributed state: {dist_state}")
    rank = dist_state.rank if dist_state is not None else 0
    _install_first_fail_marker(rank)
    _maybe_enable_memory_profile(rank)
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
    # 3-pool requires the full global batch on every rank — each pool slices it
    # locally. So the loader is built with dist_state=None (no DistributedSampler).
    train_loader = build_lm_loader(
        cfg.target,
        cfg.data,
        split="train",
        device=device,
        batch_size=cfg.pd.batch_size,
        dist_state=None,
        seed=cfg.pd.seed,
    )

    run_id = _agree_on_run_id(run_id, dist_state)
    out_dir = PARAM_DECOMP_OUT_DIR / "runs" / run_id
    scratch_dir = out_dir / SNAPSHOT_SCRATCH_DIRNAME
    sink = init_pd_run(
        cfg,
        sink_class=ThreePoolSink,
        group=group,
        tags=tags,
        resume_wandb=False,
        run_id=run_id,
        on_save=lambda step: submit_slurm_async_consolidate_and_eval(
            out_dir, step=step, parent_cfg=cfg
        ),
    )
    eval_loop = _build_eval_loop(cfg, device, dist_state=None, include_slow=False)
    try:
        trainer = ThreePoolTrainer(
            target_model=target_model,
            run_batch=make_run_batch(cfg.target),
            reconstruction_loss=recon_loss_kl,
            pd_config=cfg.pd,
            runtime_config=cfg.runtime,
            three_pool_config=cfg.runtime.topology,
        )
        trainer.run(
            train_loader,
            sink,
            cfg.cadence,
            scratch_dir=scratch_dir,
            eval_loop=eval_loop,
            profiler=_maybe_build_torch_profiler(trainer),
        )
    finally:
        sink.finish()


def _resume_main(
    resume_cfg_path: Path,
    *,
    group: str | None,
    tags: str | None,
    run_id: str | None,
) -> None:
    """Cross-run resume: read a `ResumeConfig`, load the parent's `experiment_config.yaml`,
    and continue it under a NEW run id (a fresh wandb run)."""
    resume_cfg = ResumeConfig.from_file(resume_cfg_path)
    parent_cfg = ThreePoolLMExperimentConfig.from_file(
        resume_cfg.from_run / EXPERIMENT_CONFIG_FILENAME
    )
    resolved_step = resolve_step(resume_cfg.from_run, resume_cfg.step)
    _run_resume(
        parent_cfg,
        from_run=resume_cfg.from_run,
        resolved_step=resolved_step,
        run_id=run_id,
        group=group,
        tags=tags,
        resume_wandb=False,
    )


def _resume_in_place(
    config_path: Path,
    out_dir: Path,
    resolved_step: int,
    *,
    group: str | None,
    tags: str | None,
) -> None:
    """SLURM-requeue resume: continue THIS run from its own latest checkpoint, keeping the
    same run id and wandb run. The passed `config_path` equals the run's own
    `experiment_config.yaml` (the requeue re-runs the identical command)."""
    parent_cfg = ThreePoolLMExperimentConfig.from_file(config_path)
    _run_resume(
        parent_cfg,
        from_run=out_dir,
        resolved_step=resolved_step,
        run_id=out_dir.name,
        group=group,
        tags=tags,
        resume_wandb=True,
    )


def _run_resume(
    parent_cfg: ThreePoolLMExperimentConfig,
    *,
    from_run: Path,
    resolved_step: int,
    run_id: str | None,
    group: str | None,
    tags: str | None,
    resume_wandb: bool,
) -> None:
    """Shared resume body: rebuild via `ThreePoolTrainer.from_snapshot` from `from_run`'s
    `training_<resolved_step>.pth` + ppgd shards, and continue training. `resume_wandb`
    continues the existing wandb run (requeue) vs. starting a new one (cross-run)."""
    dist_state = init_distributed()
    if is_main_process():
        logger.info(f"Distributed state: {dist_state}")
        logger.info(f"Resuming from {from_run} @ step {resolved_step}")
    rank = dist_state.rank if dist_state is not None else 0
    _install_first_fail_marker(rank)
    _maybe_enable_memory_profile(rank)
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
            "resume_provenance": ResumeProvenance(
                parent_run_dir=from_run, parent_step=resolved_step
            ),
        }
    )

    snapshot = read_training_snapshot(from_run, resolved_step)
    assert isinstance(snapshot, ThreePoolTrainingState), (
        f"3-pool resume needs ThreePoolTrainingState; got {type(snapshot).__name__}"
    )
    snapshot.runtime_config["device"] = device

    target_model = build_target(effective_cfg.target)
    train_loader = build_lm_loader(
        effective_cfg.target,
        effective_cfg.data,
        split="train",
        device=device,
        batch_size=effective_cfg.pd.batch_size,
        dist_state=None,
        seed=effective_cfg.pd.seed,
    )

    run_id = _agree_on_run_id(run_id, dist_state)
    out_dir = PARAM_DECOMP_OUT_DIR / "runs" / run_id
    scratch_dir = out_dir / SNAPSHOT_SCRATCH_DIRNAME
    sink = init_pd_run(
        effective_cfg,
        sink_class=ThreePoolSink,
        group=group,
        tags=tags,
        resume_wandb=resume_wandb,
        run_id=run_id,
        on_save=lambda step: submit_slurm_async_consolidate_and_eval(
            out_dir, step=step, parent_cfg=effective_cfg
        ),
    )
    eval_loop = _build_eval_loop(effective_cfg, device, dist_state=None, include_slow=False)
    try:
        trainer = ThreePoolTrainer.from_snapshot(
            snapshot,
            target_model=target_model,
            run_batch=make_run_batch(effective_cfg.target),
            reconstruction_loss=recon_loss_kl,
            ppgd_shard_dir=from_run / ppgd_shard_dirname(resolved_step),
        )
        trainer.run(
            train_loader,
            sink,
            effective_cfg.cadence,
            scratch_dir=scratch_dir,
            eval_loop=eval_loop,
            profiler=_maybe_build_torch_profiler(trainer),
        )
    finally:
        sink.finish()


def _submit_slurm(
    config_path: str | Path | None,
    *,
    resume: str | Path | None,
    dp: int,
    group: str | None,
    tags: str | None,
    partition: str | None,
    qos: str | None,
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

    yaml_target = resume if resume is not None else config_path
    assert yaml_target is not None
    path = Path(yaml_target)
    if path.is_absolute() and path.is_relative_to(REPO_ROOT):
        yaml_arg = path.relative_to(REPO_ROOT).as_posix()
    else:
        yaml_arg = str(yaml_target)

    module = "param_decomp_lab.experiments.lm.three_pool_run"
    if resume is not None:
        base_parts = ["-m", module, "--resume", yaml_arg, "--run_id", run_id]
    else:
        base_parts = ["-m", module, yaml_arg, "--run_id", run_id]
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
        qos=qos,
        n_gpus=launch.gpus_per_node,
        n_nodes=launch.n_nodes,
        time=time,
        snapshot_ref=snapshot_ref,
        comment=run_id,
        requeue=True,
    )
    script = generate_script(
        slurm_config, launch.command, env={**launch.env, **THREE_POOL_SLURM_ENV}
    )
    result = submit_slurm_job(script, "lm")

    wandb_url = _wandb_url_for_config(config_path, run_id) if config_path is not None else None

    logger.section("3-pool LM PD job submitted!")
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
    cfg = ThreePoolLMExperimentConfig.from_file(config_path)
    if cfg.wandb is None:
        return None
    entity = cfg.wandb.entity or get_wandb_entity()
    return f"https://wandb.ai/{entity}/{cfg.wandb.project}/runs/{run_id}"


def submit_slurm_async_consolidate_and_eval(
    run_path: str | Path,
    *,
    step: int,
    parent_cfg: ThreePoolLMExperimentConfig,
    dp: int = 8,
    time: str = "2:00:00",
    partition: str | None = DEFAULT_PARTITION_NAME,
    job_name: str = "pd-lm-consol-eval",
    group: str | None = None,
    tags: str | None = None,
    no_snapshot: bool = False,
) -> None:
    """Submit the async job that consolidates a 3-pool save and runs slow eval.

    Called from 3-pool training right after a save. The train loop has written per-rank
    partials to the scratch dir; this job (off the critical path) assembles
    `model_<step>.pth` + `training_<step>.pth` from them, prunes old `training_*.pth`,
    then runs the parent's slow eval metrics against the assembled `model_<step>.pth`
    (logging into the parent's wandb run at the same step). The job is ALWAYS submitted —
    consolidation is mandatory even when there are no slow metrics; in that case the eval
    pass is a no-op (see `async_eval`).
    """
    # Test hook (never set in production): skip the child-job submission so a smoke can
    # drive consolidation/eval out-of-band and stay within a GPU budget. The train loop
    # still writes its partials regardless.
    if os.environ.get("PD_3POOL_SKIP_ASYNC_ONSAVE", "").strip() in ("1", "true"):
        logger.info(f"PD_3POOL_SKIP_ASYNC_ONSAVE set; skipping async job for step {step}")
        return

    # The consolidate+eval job's GPU count. Defaults to `dp`; overridable so a small-
    # topology smoke can keep train + child within one node's 8 GPUs (and, in production,
    # so a cheap eval doesn't have to match the train width).
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
    scratch = PARAM_DECOMP_OUT_DIR / "runs" / train_run_id / ".async_eval_configs"
    scratch.mkdir(parents=True, exist_ok=True)
    cfg_path = scratch / f"slow_eval_step_{step}.yaml"
    slow_eval_cfg.to_file(cfg_path)

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
        comment=f"async-consol-eval:{train_run_id}@{step}",
    )
    script = generate_script(
        slurm_config, launch.command, env={**launch.env, **THREE_POOL_SLURM_ENV}
    )
    result = submit_slurm_job(script, "lm")
    logger.info(
        f"Async consolidate+eval submitted: parent={train_run_id} step={step} "
        f"job_id={result.job_id} log={result.log_pattern}"
    )


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
