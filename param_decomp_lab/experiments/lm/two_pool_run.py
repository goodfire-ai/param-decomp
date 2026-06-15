"""2-pool LM PD experiment: YAML -> ``TwoPoolTrainer`` glue + reload + resumption.

The 2-pool sibling of ``three_pool_run``. Its config — ``TwoPoolLMExperimentConfig`` —
pairs a ``ThreePoolConstrainedPDConfig`` (same four-loss set + frozen algorithm scalars
the 2-pool also honours) with a ``TwoPoolRuntimeConfig`` (core ``RuntimeConfig`` scalars
+ an authored ``TwoPoolTopology``).

Dispatch is by entry point, not a discriminator: ``pd-lm-2pool`` selects this composition
root. The pure, pool-agnostic builders (``build_target``, ``build_lm_loader``,
``make_run_batch``, ``_build_eval_loop``, ``_split_metrics_by_slow``,
``_resolve_train_run_id``) are imported from ``experiments.lm.run``; the 3-pool's launch
scaffolding (git snapshot, DDP SLURM submission, first-fail / profiler hooks,
run-id agreement) is shared verbatim from ``three_pool_run``.

Run via ``pd-lm-2pool path/to/config.yaml`` (fresh) or
``pd-lm-2pool --resume path/to/resume.yaml`` (resume). Pass ``--dp N`` to submit a DDP
SLURM job (single-node for N <= 8, multi-node for N > 8 — N must then be a multiple of 8).
For local DDP, invoke directly via
``torchrun --standalone --nproc_per_node=N -m param_decomp_lab.experiments.lm.two_pool_run config.yaml``.
"""

import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import fire
from pydantic import model_validator

from param_decomp._trace import trace
from param_decomp.component_model import ComponentModel
from param_decomp.distributed import is_main_process
from param_decomp.log import logger
from param_decomp.training_state import ThreePoolTrainingState
from param_decomp_config.base import BaseConfig
from param_decomp_config.experiment import EvalConfig, ResumeProvenance, WandbConfig
from param_decomp_config.lm import LMDataConfig, LMTargetConfig
from param_decomp_config.pd import Cadence
from param_decomp_lab.batch_and_loss_fns import recon_loss_kl
from param_decomp_lab.component_model_io import load_component_model
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
    make_run_batch,
)
from param_decomp_lab.experiments.lm.three_pool_run import (
    THREE_POOL_SLURM_ENV,
    _agree_on_run_id,
    _install_first_fail_marker,
    _maybe_build_torch_profiler,
    _maybe_enable_memory_profile,
    profiling_env_passthrough,
)
from param_decomp_lab.experiments.utils import EXPERIMENT_CONFIG_FILENAME, init_pd_run
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
    latest_checkpoint_step,
    read_training_snapshot,
    resolve_step,
)
from param_decomp_lab.run_sink import ThreePoolSink
from param_decomp_lab.seed import set_seed
from param_decomp_lab.three_pool.config import PooledRuntimeConfig
from param_decomp_lab.three_pool.consolidate import SNAPSHOT_SCRATCH_DIRNAME, ppgd_shard_dirname
from param_decomp_lab.three_pool.pd_config import ThreePoolConstrainedPDConfig
from param_decomp_lab.three_pool.two_pool_config import TwoPoolTopology
from param_decomp_lab.three_pool.two_pool_optimize import TwoPoolTrainer


class TwoPoolRuntimeConfig(PooledRuntimeConfig):
    topology: TwoPoolTopology


class TwoPoolLMExperimentConfig(BaseConfig):
    """Full YAML schema for a 2-pool LM PD run."""

    pd: ThreePoolConstrainedPDConfig
    runtime: TwoPoolRuntimeConfig
    cadence: Cadence
    target: LMTargetConfig
    data: LMDataConfig
    eval: EvalConfig | None = None
    wandb: WandbConfig | None = None
    resume_provenance: ResumeProvenance | None = None
    """Set on resumed runs (parent run dir + step); `None` for fresh runs. Mirrors the
    3-pool field so a resumed run's lineage surfaces in `experiment_config.yaml` and the
    wandb UI."""

    @model_validator(mode="after")
    def validate_pd_against_topology(self) -> Self:
        topology = self.runtime.topology
        bs = self.pd.batch_size
        for name, per_rank_batch in (
            ("pool_a", topology.pool_a.per_rank_batch),
            ("chunkwise", topology.chunkwise.per_rank_batch),
        ):
            assert bs % per_rank_batch == 0, (
                f"pd.batch_size ({bs}) must be divisible by topology.{name}.per_rank_batch "
                f"({per_rank_batch})"
            )
        return self


@dataclass(frozen=True)
class SavedTwoPoolLMRun:
    """Handle to a completed 2-pool LM PD run on disk or in W&B.

    Sibling of ``SavedThreePoolLMRun`` for the 2-pool config type. ``load_model``
    consumes only inherited ``PDConfig`` fields, so inference / inspection of a 2-pool
    decomposition works identically to a 3-pool one.
    """

    cfg: TwoPoolLMExperimentConfig
    checkpoint_path: Path

    @classmethod
    def from_path(cls, path: ModelPath) -> "SavedTwoPoolLMRun":
        files = resolve_run_files(
            path, config_filename=EXPERIMENT_CONFIG_FILENAME, checkpoint_prefix="model"
        )
        return cls(
            cfg=TwoPoolLMExperimentConfig.from_file(files.config_path),
            checkpoint_path=files.checkpoint_path,
        )

    def load_model(self) -> ComponentModel:
        return load_component_model(
            pd_config=self.cfg.pd,
            checkpoint_path=self.checkpoint_path,
            target_model=build_target(self.cfg.target),
            run_batch=make_run_batch(self.cfg.target),
        )


def _submit_world_size(config_path: Path | None, resume: Path | None) -> int:
    """GPU count for a submission, derived purely from the config — `n_a +
    n_chunks * chunk_dp` (no model). A resume reads its parent run's config."""
    if config_path is not None:
        cfg = TwoPoolLMExperimentConfig.from_file(config_path)
    else:
        assert resume is not None
        from_run = ResumeConfig.from_file(resume).from_run
        cfg = TwoPoolLMExperimentConfig.from_file(from_run / EXPERIMENT_CONFIG_FILENAME)
    topo = cfg.runtime.topology
    return topo.world_size(cfg.pd.batch_size, topo.chunkwise.n_chunks)


@with_distributed_cleanup
def main(
    config_path: str | Path | None = None,
    *,
    resume: str | Path | None = None,
    group: str | None = None,
    tags: str | None = None,
    partition: str | None = DEFAULT_PARTITION_NAME,
    qos: str | None = None,
    time: str = "72:00:00",
    job_name: str = "pd-lm-2pool",
    no_snapshot: bool = False,
    run_id: str | None = None,
) -> None:
    """Run a 2-pool LM PD experiment end-to-end.

    Args:
        config_path: YAML for a fresh run. Required when not resuming.
        resume: Path to a `ResumeConfig` YAML pointing at a prior 2-pool run.
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

    match (resume, config_path):
        case (None, str(config_path)):
            _fresh_or_requeue_main(Path(config_path), group=group, tags=tags, run_id=run_id)
        case (str(resume), None):
            _resume_main(Path(resume), group=group, tags=tags, run_id=run_id)
        case _:
            raise ValueError("must provide either config_path or --resume")


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
    cfg = TwoPoolLMExperimentConfig.from_file(config_path)

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
    # 2-pool requires the full global batch on every rank — each pool slices it locally.
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
        trainer = TwoPoolTrainer(
            target_model=target_model,
            run_batch=make_run_batch(cfg.target),
            reconstruction_loss=recon_loss_kl,
            pd_config=cfg.pd,
            runtime_config=cfg.runtime,
            two_pool_config=cfg.runtime.topology,
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
    parent_cfg = TwoPoolLMExperimentConfig.from_file(
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
    parent_cfg = TwoPoolLMExperimentConfig.from_file(config_path)
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
    parent_cfg: TwoPoolLMExperimentConfig,
    *,
    from_run: Path,
    resolved_step: int,
    run_id: str | None,
    group: str | None,
    tags: str | None,
    resume_wandb: bool,
) -> None:
    """Shared resume body: rebuild via `TwoPoolTrainer.from_snapshot` from `from_run`'s
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

    trace(f"resume: read_training_snapshot enter (step {resolved_step})")
    snapshot = read_training_snapshot(from_run, resolved_step)
    trace("resume: read_training_snapshot done (mmap, lazy)")
    assert isinstance(snapshot, ThreePoolTrainingState), (
        f"2-pool resume needs ThreePoolTrainingState; got {type(snapshot).__name__}"
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
        trainer = TwoPoolTrainer.from_snapshot(
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

    module = "param_decomp_lab.experiments.lm.two_pool_run"
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
        slurm_config,
        launch.command,
        env={**launch.env, **THREE_POOL_SLURM_ENV, **profiling_env_passthrough()},
    )
    result = submit_slurm_job(script, "lm")

    wandb_url = _wandb_url_for_config(config_path, run_id) if config_path is not None else None

    logger.section("2-pool LM PD job submitted!")
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
    cfg = TwoPoolLMExperimentConfig.from_file(config_path)
    if cfg.wandb is None:
        return None
    entity = cfg.wandb.entity or get_wandb_entity()
    return f"https://wandb.ai/{entity}/{cfg.wandb.project}/runs/{run_id}"


def submit_slurm_async_consolidate_and_eval(
    run_path: str | Path,
    *,
    step: int,
    parent_cfg: TwoPoolLMExperimentConfig,
    dp: int = 8,
    time: str = "2:00:00",
    partition: str | None = DEFAULT_PARTITION_NAME,
    job_name: str = "pd-lm-consol-eval",
    group: str | None = None,
    tags: str | None = None,
    no_snapshot: bool = False,
) -> None:
    """Submit the async job that consolidates a 2-pool save and runs slow eval.

    Sibling of the 3-pool helper: passes ``--variant two_pool`` so ``async_eval`` validates
    the parent's ``experiment_config.yaml`` as a ``TwoPoolLMExperimentConfig``. Always
    submitted — consolidation is mandatory even with no slow metrics (the eval pass is then
    a no-op).
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
            "--variant",
            "two_pool",
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
        slurm_config,
        launch.command,
        env={**launch.env, **THREE_POOL_SLURM_ENV, **profiling_env_passthrough()},
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
