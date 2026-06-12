"""Single-pool FSDP LM PD experiment: YAML -> `FsdpLMTrainer` glue + reload + resumption.

The FSDP2 sibling of `experiments.lm.run`. It runs the same single-pool VPD algorithm
(faith + importance-min + stochastic-recon + persistent-PGD authored as ordinary
`pd.loss_metrics`), but scales the **vendored** `LMComponentModel` via FSDP2 (memory) +
torch.compile (speed) instead of DDP. Its config — `FsdpLMExperimentConfig` — carries a
plain `pd: PDConfig` (no 3-pool `ThreePoolConstrainedPDConfig`) and a
`runtime: FsdpRuntimeConfig` (core substrate + the FSDP / compile toggles).

Dispatch is by entry point, not a discriminator: `pd-lm-fsdp` selects this composition
root. The pure, pool-agnostic builders (`build_target`, `build_lm_loader`,
`make_run_batch`, `_build_eval_loop`, `_split_metrics_by_slow`, `_resolve_train_run_id`)
are imported from `experiments.lm.run`.

Two facts shape this file vs the 3-pool sibling:

- **The loader is data-parallel.** FSDP shards parameters, not data — every rank consumes
  a different slice of the global batch. So `build_lm_loader` is built with
  `dist_state=dist_state` (a `DistributedSampler` shard, exactly like `pd-lm`), NOT the
  pools' full-batch-per-rank `dist_state=None`.
- **Saves are sharded DCP, consolidated off-loop.** The trainer writes per-rank DCP shards
  under `<run_dir>/.dcp/step_<S>/` and fires the sink's `on_save`, which submits an async
  job that reads the shards into a full `LMComponentModel`, emits the downstream
  `model_<S>.pth` + `training_<S>.pth`, and runs the slow eval. Requeue detection therefore
  scans `latest_dcp_step` (the shards, written on-loop) rather than `latest_checkpoint_step`
  (the consolidated `training_<S>.pth`, which may lag). Both resume paths load the sharded DCP
  checkpoint directly (independent of consolidation) via `FsdpLMTrainer.from_dcp`: a
  consolidated `training_<S>.pth` carries no PPGD persistent sources, so the shards are the
  only faithful resume source. Requeue-in-place continues this run's own shards; cross-run
  resume reads the parent run's shards into a fresh run.

Run via `pd-lm-fsdp path/to/config.yaml` (fresh) or
`pd-lm-fsdp --resume path/to/resume.yaml` (resume). Pass `--dp N` to submit an FSDP SLURM
job (single-node for N <= 8, multi-node for N > 8 — N must then be a multiple of 8). For
local FSDP, invoke directly via
`torchrun --standalone --nproc_per_node=N -m param_decomp_lab.experiments.lm.fsdp_run config.yaml`.
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
from typing import Any

import fire
import torch

from param_decomp.distributed import is_main_process
from param_decomp.log import logger
from param_decomp_config.base import BaseConfig
from param_decomp_config.experiment import EvalConfig, ResumeProvenance, WandbConfig
from param_decomp_config.lm import LMDataConfig, LMTargetConfig
from param_decomp_config.pd import Cadence, PDConfig
from param_decomp_lab.component_model_io import (
    VendoredHarvestModel,
    load_vendored_component_model,
)
from param_decomp_lab.distributed import (
    get_device,
    init_distributed,
    with_distributed_cleanup,
)
from param_decomp_lab.experiments.lm.run import (
    _build_eval_loop,
    _resolve_train_run_id,
    _split_metrics_by_slow,
    build_lm_loader,
    build_target,
)
from param_decomp_lab.experiments.utils import EXPERIMENT_CONFIG_FILENAME, init_pd_run
from param_decomp_lab.fsdp.checkpoint import DCP_DIRNAME, latest_dcp_step
from param_decomp_lab.fsdp.config import FsdpRuntimeConfig
from param_decomp_lab.fsdp.trainer import FsdpLMTrainer
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
from param_decomp_lab.resumption import ResumeConfig
from param_decomp_lab.run_sink import ThreePoolSink
from param_decomp_lab.seed import set_seed

# The DCP train job streams the fineweb dataset from HF (parquet shards fetched over the
# network on every rank). At N ranks the default 10s read timeout produces stragglers that
# stall the world at the next collective; a generous download timeout lets the contended
# reads complete instead of timing out.
# `expandable_segments` is for FRAGMENTATION here, not the pools' cross-pool deadlock: at
# the memory-tight 12-layer / large-C / PPGD regime the allocator strands ~1 GB of
# reserved-but-unallocated memory and OOMs by a few hundred MiB; expandable segments remap
# the address space and reclaim it (the OOM error itself recommends it).
FSDP_SLURM_ENV: dict[str, str] = {
    "HF_HUB_DOWNLOAD_TIMEOUT": "120",
    "HF_HUB_ETAG_TIMEOUT": "60",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}

# Subdir under the run dir holding async-eval override config YAMLs (mirrors 3-pool).
_ASYNC_EVAL_CONFIGS_DIRNAME = ".async_eval_configs"

# Profiling env vars only reach the compute node if forwarded explicitly — the SLURM script
# exports a curated env dict, not the submitter's whole environment.
_PROFILE_ENV_PREFIXES = ("PD_TORCH_PROFILE_", "PD_MEMORY_PROFILE_", "PD_TRACE")


def profiling_env_passthrough() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k.startswith(_PROFILE_ENV_PREFIXES)}


class FsdpLMExperimentConfig(BaseConfig):
    """Full YAML schema for a single-pool FSDP LM PD run. Standalone sibling of
    `LMExperimentConfig` and `ThreePoolLMExperimentConfig`.

    `pd` is a plain `PDConfig` — faith / imp / stoch / ppgd are authored as ordinary
    `pd.loss_metrics`, not a typed pool struct. `runtime` is an `FsdpRuntimeConfig` (core
    substrate scalars + the FSDP / compile toggles). `runtime.dp` is the launch-derived
    world readout (overwritten from the torchrun world at launch; not authored in YAMLs).
    """

    pd: PDConfig
    runtime: FsdpRuntimeConfig
    cadence: Cadence
    target: LMTargetConfig
    data: LMDataConfig
    eval: EvalConfig | None = None
    wandb: WandbConfig | None = None
    resume_provenance: ResumeProvenance | None = None
    """Set on resumed runs (parent run dir + step); `None` for fresh runs. Lives on the
    config so it flows into `experiment_config.yaml` and `wandb.config` via `init_pd_run`,
    making a resumed run's lineage visible in the wandb UI."""


@dataclass(frozen=True)
class SavedFsdpLMRun:
    """Handle to a completed single-pool FSDP LM PD run on disk or in W&B.

    Sibling of `SavedLMRun` / `SavedThreePoolLMRun`. The FSDP path always consolidates to
    the vendored `LMComponentModel` format, so `load_model` only loads that — there is no
    core-format branch.
    """

    cfg: FsdpLMExperimentConfig
    checkpoint_path: Path

    @classmethod
    def from_path(cls, path: ModelPath) -> "SavedFsdpLMRun":
        files = resolve_run_files(
            path, config_filename=EXPERIMENT_CONFIG_FILENAME, checkpoint_prefix="model"
        )
        return cls(
            cfg=FsdpLMExperimentConfig.from_file(files.config_path),
            checkpoint_path=files.checkpoint_path,
        )

    def load_model(self) -> VendoredHarvestModel:
        return VendoredHarvestModel(
            load_vendored_component_model(
                pd_config=self.cfg.pd,
                checkpoint_path=self.checkpoint_path,
                target_model=build_target(self.cfg.target),
            )
        )


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


@with_distributed_cleanup
def main(
    config_path: str | Path | None = None,
    *,
    resume: str | Path | None = None,
    group: str | None = None,
    tags: str | None = None,
    dp: int | None = None,
    partition: str | None = DEFAULT_PARTITION_NAME,
    qos: str | None = None,
    time: str = "72:00:00",
    job_name: str = "pd-lm-fsdp",
    no_snapshot: bool = False,
    run_id: str | None = None,
) -> None:
    """Run a single-pool FSDP LM PD experiment end-to-end.

    Args:
        config_path: YAML for a fresh run. Required when not resuming.
        resume: Path to a `ResumeConfig` YAML pointing at a prior FSDP run.
        group / tags: wandb-only (no-ops without `wandb:`).
        dp / partition / qos / time / job_name / no_snapshot / run_id: SLURM submission
            knobs. Passing `--dp N` outside torchrun submits an FSDP SLURM job: single-node
            for N <= 8, multi-node for N > 8 (N must be a multiple of 8). FSDP shards over
            the full world, so `dp` is the data-parallel world size. `qos=None` uses the
            cluster default; pass e.g. `opportunistic` to run off-quota.
    """
    if dp is not None and os.environ.get("WORLD_SIZE") is None:
        # Direct (login-node) invocation with --dp → submit a SLURM job. A torchrun worker
        # has WORLD_SIZE set and falls through to the training path below.
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
    """Worker entry for a config launch. Resumes in place from this run's own latest DCP
    shards if any exist (SLURM requeued the job after a save), else trains from step 0.

    Requeue detection scans `latest_dcp_step` (the on-loop sharded checkpoints) rather than
    the consolidated `training_<step>.pth`: consolidation runs off-loop and may lag a save,
    so the shards are the authoritative "did we already checkpoint" signal. The requeue
    re-runs the identical `... <config> --run_id <id>` command, so `run_id` is always present
    on a requeue.
    """
    if run_id is not None:
        out_dir = PARAM_DECOMP_OUT_DIR / "runs" / run_id
        prior_step = latest_dcp_step(out_dir)
        if prior_step is not None:
            _resume_in_place(config_path, out_dir, prior_step, group=group, tags=tags)
            return
    _fresh_main(config_path, group=group, tags=tags, run_id=run_id)


def _build_runtime_cfg(
    cfg: FsdpLMExperimentConfig, device: str, world_size: int | None
) -> FsdpLMExperimentConfig:
    """Stamp the resolved device + torchrun world size onto `cfg.runtime`."""
    return cfg.model_copy(
        update={"runtime": cfg.runtime.model_copy(update={"device": device, "dp": world_size})}
    )


def _fresh_main(
    config_path: Path,
    *,
    group: str | None,
    tags: str | None,
    run_id: str | None,
) -> None:
    """Fresh-run path: parse YAML, build everything, train from step 0."""
    cfg = FsdpLMExperimentConfig.from_file(config_path)

    dist_state = init_distributed()
    if is_main_process():
        logger.info(f"Distributed state: {dist_state}")
    rank = dist_state.rank if dist_state is not None else 0
    _install_first_fail_marker(rank)
    _maybe_enable_memory_profile(rank)
    set_seed(cfg.pd.seed)
    device = get_device()
    cfg = _build_runtime_cfg(cfg, device, dist_state.world_size if dist_state is not None else None)

    target_model = build_target(cfg.target)
    # FSDP shards parameters, not data — every rank consumes a different slice of the global
    # batch, so the loader gets a DistributedSampler shard (like pd-lm), NOT a full batch.
    train_loader = build_lm_loader(
        cfg.target,
        cfg.data,
        split="train",
        device=device,
        batch_size=cfg.pd.batch_size,
        dist_state=dist_state,
        seed=cfg.pd.seed,
    )

    run_id = run_id or _generate_run_id_on_main(dist_state)
    out_dir = PARAM_DECOMP_OUT_DIR / "runs" / run_id
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
    eval_loop = _build_eval_loop(cfg, device, dist_state, include_slow=False)
    try:
        trainer = FsdpLMTrainer(
            target_model=target_model,
            pd_config=cfg.pd,
            runtime_config=cfg.runtime,
        )
        # The trainer writes DCP shards to `run_dir/.dcp/`, which `latest_dcp_step(out_dir)`
        # (requeue detection) and `consolidate(out_dir, step)` (async eval) both read.
        trainer.run(train_loader, sink, cfg.cadence, run_dir=out_dir, eval_loop=eval_loop)
    finally:
        sink.finish()


def _resolve_dcp_step(run_dir: Path, step: int | str) -> int:
    """Resolve a `ResumeConfig.step` against `run_dir`'s DCP shards.

    `"latest"` -> the newest `.dcp/step_<S>/`; an explicit step is asserted to have a shard
    dir. DCP (not `training_*.pth`) is the source of truth for FSDP resume — `keep_last_n`
    may have pruned the consolidated training files while the shards persist.
    """
    if step == "latest":
        latest = latest_dcp_step(run_dir)
        assert latest is not None, f"no DCP shards (`.dcp/step_*/`) under {run_dir}"
        return latest
    assert isinstance(step, int)
    step_dir = run_dir / DCP_DIRNAME / f"step_{step}"
    assert step_dir.is_dir(), f"no DCP shards at {step_dir}"
    return step


def _resume_main(
    resume_cfg_path: Path,
    *,
    group: str | None,
    tags: str | None,
    run_id: str | None,
) -> None:
    """Cross-run resume: read a `ResumeConfig`, load the parent's `experiment_config.yaml`,
    and continue it under a NEW run id (a fresh wandb run) from the parent's sharded DCP
    checkpoint. The step is resolved against the parent's DCP shards (what `from_dcp` reads),
    not its consolidated `training_*.pth` (which `keep_last_n` may have pruned)."""
    resume_cfg = ResumeConfig.from_file(resume_cfg_path)
    parent_cfg = FsdpLMExperimentConfig.from_file(resume_cfg.from_run / EXPERIMENT_CONFIG_FILENAME)
    resolved_step = _resolve_dcp_step(resume_cfg.from_run, resume_cfg.step)
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
    """SLURM-requeue resume: continue THIS run from its own latest DCP shards, keeping the
    same run id and wandb run. The passed `config_path` equals the run's own
    `experiment_config.yaml` (the requeue re-runs the identical command). Loads the sharded
    DCP checkpoint directly — independent of off-loop consolidation."""
    parent_cfg = FsdpLMExperimentConfig.from_file(config_path)
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
    parent_cfg: FsdpLMExperimentConfig,
    *,
    from_run: Path,
    resolved_step: int,
    run_id: str | None,
    group: str | None,
    tags: str | None,
    resume_wandb: bool,
) -> None:
    """Shared resume body: rebuild via `FsdpLMTrainer.from_dcp` from `from_run`'s sharded
    DCP shards (`.dcp/step_<resolved_step>/`) and continue training. `resume_wandb` continues
    the existing wandb run (requeue-in-place) vs. starting a new one (cross-run)."""
    dist_state = init_distributed()
    if is_main_process():
        logger.info(f"Distributed state: {dist_state}")
        logger.info(f"Resuming from {from_run} @ step {resolved_step} (DCP shards)")
    rank = dist_state.rank if dist_state is not None else 0
    _install_first_fail_marker(rank)
    _maybe_enable_memory_profile(rank)
    set_seed(parent_cfg.pd.seed)
    device = get_device()

    # Stamp lineage only when starting a NEW wandb run (cross-run resume). On an in-place
    # requeue (`resume_wandb=True`) the SAME wandb run continues, and wandb rejects changing a
    # config key (`resume_provenance`) that was None at the original launch — a ConfigError.
    provenance_update: dict[str, ResumeProvenance] = {}
    if not resume_wandb:
        provenance_update["resume_provenance"] = ResumeProvenance(
            parent_run_dir=from_run, parent_step=resolved_step
        )
    effective_cfg = _build_runtime_cfg(
        parent_cfg, device, dist_state.world_size if dist_state is not None else None
    ).model_copy(update=provenance_update)

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

    run_id = run_id or _generate_run_id_on_main(dist_state)
    out_dir = PARAM_DECOMP_OUT_DIR / "runs" / run_id
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
    eval_loop = _build_eval_loop(effective_cfg, device, dist_state, include_slow=False)
    try:
        # The trainer's only resume entry is sharded DCP. Resume-in-place loads this run's own
        # shards; cross-run resume loads the source run's `.dcp/step_<step>/` shards (which
        # persist — never pruned) into a fresh trainer at this launch's topology, with
        # `resume_provenance` recording the parent. The faithful resume source is DCP either
        # way: a consolidated `training_<step>.pth` carries no PPGD persistent sources.
        trainer = FsdpLMTrainer.from_dcp(
            target_model=target_model,
            pd_config=effective_cfg.pd,
            runtime_config=effective_cfg.runtime,
            run_dir=from_run,
            step=resolved_step,
        )
        trainer.run(
            train_loader,
            sink,
            effective_cfg.cadence,
            run_dir=out_dir,
            eval_loop=eval_loop,
        )
    finally:
        sink.finish()


def _generate_run_id_on_main(dist_state: Any) -> str:
    """Broadcast a single fresh run id to all ranks so every rank's `out_dir` agrees.

    The FSDP DCP save is a collective writing per-rank shards under
    `<run_id>/.dcp/step_<S>/`; every rank must compute the same path.
    """
    import torch.distributed as dist

    run_id = generate_run_id("param_decomp") if is_main_process() else None
    if dist_state is not None:
        objs: list[str | None] = [run_id]
        dist.broadcast_object_list(objs, src=0)
        run_id = objs[0]
    assert run_id is not None
    return run_id


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

    module = "param_decomp_lab.experiments.lm.fsdp_run"
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
    script = generate_script(slurm_config, launch.command, env={**launch.env, **FSDP_SLURM_ENV})
    result = submit_slurm_job(script, "lm")

    wandb_url = _wandb_url_for_config(config_path, run_id) if config_path is not None else None

    logger.section("FSDP LM PD job submitted!")
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
    cfg = FsdpLMExperimentConfig.from_file(config_path)
    if cfg.wandb is None:
        return None
    entity = cfg.wandb.entity or get_wandb_entity()
    return f"https://wandb.ai/{entity}/{cfg.wandb.project}/runs/{run_id}"


def submit_slurm_async_consolidate_and_eval(
    run_path: str | Path,
    *,
    step: int,
    parent_cfg: FsdpLMExperimentConfig,
    dp: int = 8,
    time: str = "2:00:00",
    partition: str | None = DEFAULT_PARTITION_NAME,
    job_name: str = "pd-lm-fsdp-consol-eval",
    group: str | None = None,
    tags: str | None = None,
    no_snapshot: bool = False,
) -> None:
    """Submit the async job that consolidates an FSDP DCP save and runs slow eval.

    Called from FSDP training right after a sharded save (on-loop, off the critical path).
    The train loop has written per-rank DCP shards under `<run_dir>/.dcp/step_<step>/`; this
    job (`experiments.lm.fsdp_async_eval`) reads those shards into a full `LMComponentModel`,
    writes `model_<step>.pth` + `training_<step>.pth`, prunes old `training_*.pth`, then runs
    the parent's slow eval metrics against the assembled `model_<step>.pth` (logging into the
    parent's wandb run at the same step). The job is ALWAYS submitted — consolidation is
    mandatory even with no slow metrics; in that case the eval pass is a no-op.
    """
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
        every=1,  # any value; not consumed by fsdp_async_eval
        slow_every=1,  # any value; not consumed by fsdp_async_eval
        slow_on_first_step=True,
        metrics=slow_metrics,
    )
    train_run_id = _resolve_train_run_id(run_path)
    scratch = PARAM_DECOMP_OUT_DIR / "runs" / train_run_id / _ASYNC_EVAL_CONFIGS_DIRNAME
    scratch.mkdir(parents=True, exist_ok=True)
    cfg_path = scratch / f"slow_eval_step_{step}.yaml"
    slow_eval_cfg.to_file(cfg_path)

    eval_run_id = generate_run_id("param_decomp")
    snapshot_ref: str | None = f"refs/runs/snapshot/{train_run_id}" if not no_snapshot else None

    base_command = shlex.join(
        [
            "-m",
            "param_decomp_lab.experiments.lm.fsdp_async_eval",
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
        comment=f"fsdp-consol-eval:{train_run_id}@{step}",
    )
    script = generate_script(slurm_config, launch.command, env={**launch.env, **FSDP_SLURM_ENV})
    result = submit_slurm_job(script, "lm")
    logger.info(
        f"Async FSDP consolidate+eval submitted: parent={train_run_id} step={step} "
        f"job_id={result.job_id} log={result.log_pattern}"
    )


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
