"""SLURM launch helpers for PD experiments.

Internal — invoked by ``pd-run`` (``param_decomp/experiments/runner.py``). Takes a
``Run`` (single launch) or ``SweepSpec`` (many runs sharing one driver and
substrate). For sweeps, snapshots the spec to disk for reproducibility. Creates
a git snapshot of the repo and submits SLURM: a plain job for ``Run``, an
array (one task per run) for ``SweepSpec``. Each task invokes
``python -m param_decomp.experiments._worker``.

For single-machine execution, use ``pd-run <experiment> --local``.
"""

import json
import shlex
from datetime import datetime
from hashlib import sha256

from param_decomp.configs import RuntimeConfig
from param_decomp.log import logger
from param_decomp.run import Run
from param_decomp.settings import GPUS_PER_NODE, PARAM_DECOMP_OUT_DIR
from param_decomp.sweeps import SweepSpec
from param_decomp.utils.git_utils import create_git_snapshot
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


def launch_slurm(
    launchable: Run | SweepSpec,
    runtime: RuntimeConfig,
    n_agents: int | None,
    job_suffix: str | None,
    partition: str,
    project: str,
) -> None:
    """Submit a PD experiment to SLURM.

    Callers (``pd-run``) resolve their input (built-in experiment, custom
    config, rerun, or sweep generator) into either a single ``Run`` or
    a ``SweepSpec`` and pass it in here. Single runs submit a plain SLURM job;
    sweeps submit an array (one task per run) and snapshot the spec to
    ``PARAM_DECOMP_OUT_DIR/sweeps/<launch_id>/spec.yaml``.

    ``project`` is the W&B project to log every run to. It's a deploy-time
    parameter, not part of the ``Run`` config, so it's passed alongside.
    """
    launch_id = _generate_launch_id()
    is_sweep = isinstance(launchable, SweepSpec)
    runs = launchable.runs if isinstance(launchable, SweepSpec) else [launchable]
    for r in runs:
        assert r.driver_path is not None, "launchable Run must declare a driver_path"

    logger.info(f"Launch ID: {launch_id}")

    n_gpus = _n_gpus_for(runtime)
    logger.info(f"Running on {_format_compute_info(n_gpus)}")

    if is_sweep:
        assert n_agents is not None, "n_agents must be provided for a SweepSpec"
        logger.info(f"Sweep '{launchable.description}': {len(runs)} run(s)")
        sweep_dir = PARAM_DECOMP_OUT_DIR / "sweeps" / launch_id
        launchable.write(sweep_dir / "spec.yaml")
        logger.info(f"Wrote sweep spec to {sweep_dir / 'spec.yaml'}")
        sweep_spec_path: str | None = str(sweep_dir / "spec.yaml")
    else:
        logger.info(f"Single run: {runs[0].logging.wandb_run_name}")
        sweep_spec_path = None

    snapshot_ref, commit_hash = create_git_snapshot(snapshot_id=launch_id)
    logger.info(f"Created git snapshot ref: {snapshot_ref} ({commit_hash[:8]})")

    slurm_job_name = f"pd-{job_suffix}" if job_suffix else "pd"
    wandb_urls = [get_wandb_run_url(project, run.run_id) for run in runs]
    is_array = is_sweep

    script_content = _create_slurm_script(
        slurm_job_name=slurm_job_name,
        launch_id=launch_id,
        runs=runs,
        snapshot_ref=snapshot_ref,
        n_gpus=n_gpus,
        partition=partition,
        is_array=is_array,
        project=project,
        max_concurrent_tasks=n_agents if is_array else None,
        per_task_comments=wandb_urls,
    )

    result = submit_slurm_job(
        script_content,
        f"launch_{launch_id}",
        is_array=is_array,
        n_array_tasks=len(runs) if is_array else None,
    )

    logger.section("Job submitted successfully!")
    summary: dict[str, str | int | None] = {
        "Array Job ID" if is_array else "Job ID": result.job_id,
        "Total runs": len(runs),
        "View logs in": result.log_pattern,
        "Script": str(result.script_path),
    }
    if is_array:
        summary["Max concurrent tasks"] = n_agents
    if sweep_spec_path is not None:
        summary["Sweep spec"] = sweep_spec_path
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


def _n_gpus_for(runtime: RuntimeConfig) -> int | None:
    """Resolve final GPU count for SLURM resource request. RuntimeConfig has already
    validated dp shape and the device/dp interaction."""
    if runtime.device == "cpu":
        return None
    return runtime.dp


def _format_compute_info(n_gpus: int | None) -> str:
    if n_gpus is None:
        return "single GPU"
    if n_gpus <= GPUS_PER_NODE:
        return f"{n_gpus} GPUs (single node)"
    n_nodes = n_gpus // GPUS_PER_NODE
    return f"{n_gpus} GPUs ({n_nodes} nodes x {GPUS_PER_NODE} GPUs)"


def _choose_master_port(run_id_local: str, idx: int) -> int:
    """Choose a unique port per command.

    Uses a stable hash of (run_id, idx) mapped into a high, unprivileged port range so that we can
    run multiple DDP processes on the same machine.
    """
    base: int = 20000
    span: int = 20000  # ports in [20000, 40000)
    h: int = int(sha256(f"{run_id_local}:{idx}".encode()).hexdigest(), 16)
    return base + (h % span)


def _build_worker_args(launch_id: str, run: Run, project: str) -> str:
    """Build the ``_worker`` CLI arguments for one SLURM task."""
    run_json = json.dumps(run.model_dump(mode="json"))
    return " ".join(
        [
            f"--run_json {shlex.quote(run_json)}",
            f"--launch_id {launch_id}",
            f"--wandb_project {shlex.quote(project)}",
        ]
    )


def _get_command(
    launch_id: str,
    run: Run,
    spec_idx: int,
    n_gpus: int | None,
    snapshot_ref: str,
    is_array: bool,
    project: str,
) -> str:
    """Build the command to run one ``Run``.

    Args:
        n_gpus: None or 1 means single GPU/CPU. 2-8 means single-node DDP. >8 means multi-node
            DDP (must be divisible by 8).
    """
    port = _choose_master_port(run.run_id, spec_idx)
    script_args = _build_worker_args(launch_id, run, project)

    worker_module = "param_decomp.experiments._worker"
    match n_gpus:
        case None | 1:
            return f"python -m {worker_module} {script_args}"

        case n if n <= GPUS_PER_NODE:
            return (
                f"torchrun --standalone --nproc_per_node={n} --master_port={port} "
                f"-m {worker_module} {script_args}"
            )

        case _:
            # Multi-node DDP via srun + torchrun
            # $SLURM_PROCID is the node rank (0, 1, ..., n-1), evaluated on each node by bash -c
            n_nodes = n_gpus // GPUS_PER_NODE
            torchrun_cmd = (
                f"torchrun "
                f"--nnodes={n_nodes} "
                f"--node_rank=$SLURM_PROCID "
                f"--nproc_per_node={GPUS_PER_NODE} "
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
    runs: list[Run],
    snapshot_ref: str,
    n_gpus: int | None,
    partition: str,
    is_array: bool,
    project: str,
    max_concurrent_tasks: int | None = None,
    per_task_comments: list[str] | None = None,
) -> str:
    """Create a SLURM script for one or more runs."""
    commands = [
        _get_command(
            launch_id=launch_id,
            run=run,
            spec_idx=i,
            n_gpus=n_gpus,
            snapshot_ref=snapshot_ref,
            is_array=is_array,
            project=project,
        )
        for i, run in enumerate(runs)
    ]

    match n_gpus:
        case None | 1:
            n_nodes, gpus_per_node = 1, 1
        case n if n <= GPUS_PER_NODE:
            n_nodes, gpus_per_node = 1, n
        case _:
            n_nodes = n_gpus // GPUS_PER_NODE
            gpus_per_node = GPUS_PER_NODE

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
        assert len(runs) == 1, "non-array launch must have exactly one run"
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
