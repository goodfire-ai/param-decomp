"""SLURM launch helpers for PD experiments.

Internal — invoked by ``pd-run`` (``param_decomp/experiments/runner.py``). Resolves the
sweep generator (if any), validates every generated config against the driver, snapshots
the materialized ``SweepSpec`` to disk, creates a git snapshot of the repo for
reproducibility, and submits a SLURM array where each task invokes
``python -m param_decomp.experiments._worker`` on one config.

For single-machine execution, use ``pd-run <experiment> --local``.
"""

import json
import shlex
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import Any

from param_decomp.log import logger
from param_decomp.settings import PARAM_DECOMP_OUT_DIR
from param_decomp.sweeps import SweepRun, SweepSpec, resolve_sweep
from param_decomp.utils.git_utils import create_git_snapshot
from param_decomp.utils.run_utils import generate_run_id
from param_decomp.utils.slurm import (
    SlurmArrayConfig,
    SlurmConfig,
    generate_array_script,
    generate_git_snapshot_setup,
    generate_script,
    submit_slurm_job,
)
from param_decomp.utils.wandb_utils import get_wandb_run_url

_CUDA_FLAGS = {
    "NCCL_DEBUG": "WARN",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
}
_GPUS_PER_NODE = 8


@dataclass(frozen=True, slots=True)
class _TaskSpec:
    """One SLURM task: a SweepRun + a pre-allocated run_id."""

    sweep_run: SweepRun
    run_id: str


def launch_slurm(
    name: str,
    driver_path: str,
    base_config: dict[str, Any],
    sweep: str | None,
    n_agents: int | None,
    job_suffix: str | None,
    cpu: bool,
    partition: str,
    dp: int | None,
    project: str,
) -> None:
    """Submit a PD experiment to SLURM (with optional sweep)."""
    launch_id = _generate_launch_id()
    logger.info(f"Launch ID: {launch_id}")
    logger.info(f"Experiment: {name}")

    n_gpus = _validate_and_get_n_gpus(cpu=cpu, dp=dp)
    logger.info(f"Running on {_format_compute_info(n_gpus)}")

    sweep_spec = _build_sweep_spec(name=name, sweep=sweep, base_config=base_config)
    if len(sweep_spec.runs) > 1:
        assert n_agents is not None, "n_agents must be provided when sweep is enabled"
    logger.info(f"Sweep '{sweep_spec.description}': {len(sweep_spec.runs)} run(s)")

    # Config validation happens worker-side at _worker.run_experiment. Doing it here too
    # would force the launch node (often a login node without GPUs) to import the driver's
    # full deps (e.g. `transformers` for the lm driver). Skip it.

    sweep_dir = PARAM_DECOMP_OUT_DIR / "sweeps" / launch_id
    sweep_spec.write(sweep_dir / "spec.yaml")
    logger.info(f"Wrote sweep spec to {sweep_dir / 'spec.yaml'}")

    task_specs = [
        _TaskSpec(sweep_run=run, run_id=generate_run_id("param_decomp")) for run in sweep_spec.runs
    ]

    snapshot_ref, commit_hash = create_git_snapshot(snapshot_id=launch_id)
    logger.info(f"Created git snapshot ref: {snapshot_ref} ({commit_hash[:8]})")

    slurm_job_name = f"pd-{job_suffix}" if job_suffix else "pd"
    wandb_urls = [get_wandb_run_url(project, t.run_id) for t in task_specs]
    is_array = len(task_specs) > 1

    script_content = _create_slurm_script(
        slurm_job_name=slurm_job_name,
        launch_id=launch_id,
        driver_path=driver_path,
        task_specs=task_specs,
        project=project,
        snapshot_ref=snapshot_ref,
        n_gpus=n_gpus,
        partition=partition,
        max_concurrent_tasks=n_agents,
        per_task_comments=wandb_urls,
    )

    result = submit_slurm_job(
        script_content,
        f"launch_{launch_id}",
        is_array=is_array,
        n_array_tasks=len(task_specs) if is_array else None,
    )

    logger.section("Job submitted successfully!")
    summary: dict[str, str | int | None] = {
        "Array Job ID" if is_array else "Job ID": result.job_id,
        "Total runs": len(task_specs),
        "Max concurrent tasks": n_agents,
        "View logs in": result.log_pattern,
        "Sweep spec": str(sweep_dir / "spec.yaml"),
        "Script": str(result.script_path),
    }
    if len(wandb_urls) <= 10:
        summary["WandB run URLs"] = (
            wandb_urls[0]
            if len(wandb_urls) == 1
            else "\n" + "\n".join(f"  - {u}" for u in wandb_urls)
        )
    logger.values(summary)


def _generate_launch_id() -> str:
    """Generate a unique launch ID based on timestamp.

    Prefixed with 'launch-' to prevent Python Fire from parsing the numeric timestamp as an int.
    """
    return f"launch-{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _build_sweep_spec(name: str, sweep: str | None, base_config: dict[str, Any]) -> SweepSpec:
    """Resolve the sweep generator (or build a trivial single-run spec)."""
    if sweep is None:
        return SweepSpec(
            description=f"single run: {name}",
            runs=[SweepRun(name=name, config=base_config, view_meta={})],
        )
    generator = resolve_sweep(sweep)
    spec = generator(base_config)
    assert isinstance(spec, SweepSpec), (
        f"sweep generator {generator!r} returned {type(spec).__name__}, expected SweepSpec"
    )
    assert spec.runs, f"sweep generator {generator!r} produced zero runs"
    return spec


def _validate_and_get_n_gpus(cpu: bool, dp: int | None) -> int | None:
    """Resolve final GPU count. dp value/shape already validated by RuntimeConfig upstream."""
    if cpu:
        assert dp is None, "dp should not be specified when running on cpu"
        return None
    return dp


def _format_compute_info(n_gpus: int | None) -> str:
    if n_gpus is None:
        return "single GPU"
    if n_gpus <= _GPUS_PER_NODE:
        return f"{n_gpus} GPUs (single node)"
    n_nodes = n_gpus // _GPUS_PER_NODE
    return f"{n_gpus} GPUs ({n_nodes} nodes x {_GPUS_PER_NODE} GPUs)"


def _choose_master_port(run_id_local: str, idx: int) -> int:
    """Choose a unique port per command.

    Uses a stable hash of (run_id, idx) mapped into a high, unprivileged port range so that we can
    run multiple DDP processes on the same machine.
    """
    base: int = 20000
    span: int = 20000  # ports in [20000, 40000)
    h: int = int(sha256(f"{run_id_local}:{idx}".encode()).hexdigest(), 16)
    return base + (h % span)


def _build_worker_args(
    launch_id: str,
    driver_path: str,
    task_spec: _TaskSpec,
    project: str,
) -> str:
    """Build the ``_worker`` CLI arguments for one SLURM task."""
    sweep_run = task_spec.sweep_run
    json_tagged_config = f"json:{json.dumps(sweep_run.config)}"
    parts = [
        f"--config_json {shlex.quote(json_tagged_config)}",
        f"--driver {shlex.quote(driver_path)}",
        f"--launch_id {launch_id}",
        f"--run_id {task_spec.run_id}",
        f"--wandb_project {shlex.quote(project)}",
        f"--wandb_run_name {shlex.quote(sweep_run.name)}",
    ]
    if sweep_run.view_meta:
        json_tagged_view_meta = f"json:{json.dumps(sweep_run.view_meta)}"
        parts.append(f"--view_meta_json {shlex.quote(json_tagged_view_meta)}")
    return " ".join(parts)


def _get_command(
    launch_id: str,
    driver_path: str,
    task_spec: _TaskSpec,
    project: str,
    spec_idx: int,
    n_gpus: int | None,
    snapshot_ref: str,
    is_array: bool,
) -> str:
    """Build the command to run one task spec.

    Args:
        n_gpus: None or 1 means single GPU/CPU. 2-8 means single-node DDP. >8 means multi-node
            DDP (must be divisible by 8).
    """
    port = _choose_master_port(launch_id, spec_idx)
    script_args = _build_worker_args(launch_id, driver_path, task_spec, project)

    worker_module = "param_decomp.experiments._worker"
    match n_gpus:
        case None | 1:
            return f"python -m {worker_module} {script_args}"

        case n if n <= _GPUS_PER_NODE:
            return (
                f"torchrun --standalone --nproc_per_node={n} --master_port={port} "
                f"-m {worker_module} {script_args}"
            )

        case _:
            # Multi-node DDP via srun + torchrun
            # $SLURM_PROCID is the node rank (0, 1, ..., n-1), evaluated on each node by bash -c
            n_nodes = n_gpus // _GPUS_PER_NODE
            torchrun_cmd = (
                f"torchrun "
                f"--nnodes={n_nodes} "
                f"--node_rank=$SLURM_PROCID "
                f"--nproc_per_node={_GPUS_PER_NODE} "
                f'--master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1) '
                f"--master_port={port} "
                f"-m {worker_module} {script_args}"
            )

            # Each node needs its own /tmp workspace since /tmp is node-local
            if is_array:
                job_id_suffix = "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
            else:
                job_id_suffix = "$SLURM_JOB_ID"
            work_dir = f"/tmp/param-decomp/workspace-{job_id_suffix}-node$SLURM_PROCID"
            setup = generate_git_snapshot_setup(work_dir, snapshot_ref)
            # Explicit srun flags ensure one task per node across all allocated nodes
            srun_flags = f"--nodes={n_nodes} --ntasks={n_nodes} --ntasks-per-node=1"
            return f"srun {srun_flags} bash -c {shlex.quote(f'{setup}\n{torchrun_cmd}')}"


def _create_slurm_script(
    slurm_job_name: str,
    launch_id: str,
    driver_path: str,
    task_specs: list[_TaskSpec],
    project: str,
    snapshot_ref: str,
    n_gpus: int | None,
    partition: str,
    max_concurrent_tasks: int | None = None,
    per_task_comments: list[str] | None = None,
) -> str:
    """Create a SLURM script for one or more task specs."""
    is_array = len(task_specs) > 1

    commands = [
        _get_command(
            launch_id=launch_id,
            driver_path=driver_path,
            task_spec=task_spec,
            project=project,
            spec_idx=i,
            n_gpus=n_gpus,
            snapshot_ref=snapshot_ref,
            is_array=is_array,
        )
        for i, task_spec in enumerate(task_specs)
    ]

    match n_gpus:
        case None | 1:
            n_nodes, gpus_per_node = 1, 1
        case n if n <= _GPUS_PER_NODE:
            n_nodes, gpus_per_node = 1, n
        case _:
            n_nodes = n_gpus // _GPUS_PER_NODE
            gpus_per_node = _GPUS_PER_NODE

    if is_array:
        array_config = SlurmArrayConfig(
            job_name=slurm_job_name,
            partition=partition,
            n_gpus=gpus_per_node,
            n_nodes=n_nodes,
            snapshot_ref=snapshot_ref,
            max_concurrent_tasks=max_concurrent_tasks,
        )
        return generate_array_script(
            array_config, commands, env=_CUDA_FLAGS, per_task_comments=per_task_comments
        )
    else:
        comment = per_task_comments[0] if per_task_comments is not None else None
        single_config = SlurmConfig(
            job_name=slurm_job_name,
            partition=partition,
            n_gpus=gpus_per_node,
            n_nodes=n_nodes,
            snapshot_ref=snapshot_ref,
            comment=comment,
        )
        return generate_script(single_config, commands[0], env=_CUDA_FLAGS)
