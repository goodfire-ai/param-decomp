"""SLURM launch helpers for PD experiments.

Internal — invoked by ``pd-run`` (``param_decomp/experiments/runner.py``).
``launch_run_slurm`` takes a single ``RunConfig`` and submits a plain SLURM
job; ``launch_sweep_slurm`` takes a ``SweepSpec`` (many runs sharing one
driver and substrate), snapshots the spec to disk for reproducibility, and
submits a SLURM array (one task per run). Each task invokes
``python -m param_decomp.experiments._worker``.

For single-machine execution, use ``pd-run <experiment> --local``.
"""

import json
import shlex
from datetime import datetime
from hashlib import sha256

from param_decomp.configs import RuntimeConfig
from param_decomp.log import logger
from param_decomp.run import RunConfig
from param_decomp.settings import GPUS_PER_NODE, PARAM_DECOMP_OUT_DIR
from param_decomp.sweeps import SweepSpec
from param_decomp.utils.git_utils import create_git_snapshot
from param_decomp.utils.slurm import (
    ARRAY_JOB_ID_BASH,
    SINGLETON_JOB_ID_BASH,
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


def launch_run_slurm(
    run_cfg: RunConfig,
    partition: str,
    project: str,
    job_suffix: str | None,
) -> None:
    launch_id = _generate_launch_id()
    logger.info(f"Launch ID: {launch_id}")

    snapshot_ref, commit_hash = create_git_snapshot(snapshot_id=launch_id)
    logger.info(f"Created git snapshot ref: {snapshot_ref} ({commit_hash[:8]})")

    slurm_job_name = f"pd-{job_suffix}" if job_suffix else "pd"

    script_content = _create_singleton_slurm_script(
        slurm_job_name=slurm_job_name,
        launch_id=launch_id,
        run_cfg=run_cfg,
        snapshot_ref=snapshot_ref,
        n_gpus=_n_gpus_for(run_cfg.runtime),
        partition=partition,
        project=project,
        comment=get_wandb_run_url(project, run_cfg.run_id),
    )

    result = submit_slurm_job(
        script_content,
        f"launch_{launch_id}",
        n_array_tasks=None,
    )

    logger.section("Job submitted successfully!")

    summary: dict[str, str | int | None] = {
        "Job ID": result.job_id,
        "View logs in": result.log_pattern,
        "Script": str(result.script_path),
    }

    logger.values(summary)


def launch_sweep_slurm(
    sweep: SweepSpec,
    job_suffix: str | None,
    partition: str,
    project: str,
) -> None:
    launch_id = _generate_launch_id()
    logger.info(f"Launch ID: {launch_id}")

    run_cfgs = sweep.run_cfgs()
    logger.info(f"Sweep '{sweep.description}': {len(run_cfgs)} run(s)")
    sweep_dir = PARAM_DECOMP_OUT_DIR / "sweeps" / launch_id

    sweep.write(sweep_dir / "spec.yaml")
    sweep_spec_path = str(sweep_dir / "spec.yaml")
    logger.info(f"Wrote sweep spec to {sweep_spec_path}")

    snapshot_ref, commit_hash = create_git_snapshot(snapshot_id=launch_id)
    logger.info(f"Created git snapshot ref: {snapshot_ref} ({commit_hash[:8]})")

    slurm_job_name = f"pd-{job_suffix}" if job_suffix else "pd"
    wandb_urls = [get_wandb_run_url(project, run_cfg.run_id) for run_cfg in run_cfgs]

    script_content = _create_array_slurm_script(
        slurm_job_name=slurm_job_name,
        launch_id=launch_id,
        run_cfgs=run_cfgs,
        snapshot_ref=snapshot_ref,
        partition=partition,
        project=project,
        max_concurrent_tasks=sweep.n_agents,
    )

    result = submit_slurm_job(
        script_content,
        f"launch_{launch_id}",
        n_array_tasks=len(run_cfgs),
    )

    logger.section("Job submitted successfully!")
    summary: dict[str, str | int | None] = {
        "Array Job ID": result.job_id,
        "Total runs": len(run_cfgs),
        "View logs in": result.log_pattern,
        "Script": str(result.script_path),
        "Max concurrent tasks": sweep.n_agents,
        "Sweep spec": sweep_spec_path,
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


def _n_gpus_for(runtime: RuntimeConfig) -> int | None:
    """Resolve final GPU count for SLURM resource request. RuntimeConfig has already
    validated dp shape and the device/dp interaction."""
    if runtime.device == "cpu":
        return None
    return runtime.dp


# def _format_compute_info(n_gpus: int | None) -> str:
#     if n_gpus is None:
#         return "single GPU"
#     if n_gpus <= GPUS_PER_NODE:
#         return f"{n_gpus} GPUs (single node)"
#     n_nodes = n_gpus // GPUS_PER_NODE
#     return f"{n_gpus} GPUs ({n_nodes} nodes x {GPUS_PER_NODE} GPUs)"


def _choose_master_port(run_id_local: str, idx: int) -> int:
    """Choose a unique port per command.

    Uses a stable hash of (run_id, idx) mapped into a high, unprivileged port range so that we can
    run multiple DDP processes on the same machine.
    """
    base: int = 20000
    span: int = 20000  # ports in [20000, 40000)
    h: int = int(sha256(f"{run_id_local}:{idx}".encode()).hexdigest(), 16)
    return base + (h % span)


def _build_worker_args(launch_id: str, run: RunConfig, project: str) -> str:
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
    run_cfg: RunConfig,
    spec_idx: int,
    n_gpus: int | None,
    snapshot_ref: str,
    workspace_job_id_bash: str,
    project: str,
) -> str:
    """Build the command to run one ``RunConfig``.

    Args:
        n_gpus: None or 1 means single GPU/CPU. 2-8 means single-node DDP. >8 means multi-node
            DDP (must be divisible by 8).
        workspace_job_id_bash: Bash expression uniquely identifying this job invocation,
            used to name per-node /tmp workspaces in the multi-node DDP path. Pass
            ``SINGLETON_JOB_ID_BASH`` or ``ARRAY_JOB_ID_BASH`` from ``utils.slurm``.
    """
    port = _choose_master_port(run_cfg.run_id, spec_idx)
    script_args = _build_worker_args(launch_id, run_cfg, project)

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
            work_dir = f"/tmp/param-decomp/workspace-{workspace_job_id_bash}-node$SLURM_PROCID"
            setup = generate_git_snapshot_setup(work_dir, snapshot_ref)
            # Explicit srun flags ensure one task per node across all allocated nodes
            srun_flags = f"--nodes={n_nodes} --ntasks={n_nodes} --ntasks-per-node=1"
            return f"srun {srun_flags} bash -c {shlex.quote(f'{setup}\n{torchrun_cmd}')}"


def _n_nodes_and_gpus_per_node(n_gpus: int | None) -> tuple[int, int]:
    match n_gpus:
        case None | 1:
            n_nodes, gpus_per_node = 1, 1
        case n if n <= GPUS_PER_NODE:
            n_nodes, gpus_per_node = 1, n
        case _:
            n_nodes = n_gpus // GPUS_PER_NODE
            gpus_per_node = GPUS_PER_NODE
    return n_nodes, gpus_per_node


def _create_singleton_slurm_script(
    slurm_job_name: str,
    launch_id: str,
    run_cfg: RunConfig,
    snapshot_ref: str,
    n_gpus: int | None,
    partition: str,
    project: str,
    comment: str | None = None,
) -> str:
    """Create a SLURM script for one or more runs."""
    command = _get_command(
        launch_id=launch_id,
        run_cfg=run_cfg,
        spec_idx=0,
        n_gpus=n_gpus,
        snapshot_ref=snapshot_ref,
        workspace_job_id_bash=SINGLETON_JOB_ID_BASH,
        project=project,
    )

    n_nodes, gpus_per_node = _n_nodes_and_gpus_per_node(n_gpus)

    single_config = SlurmConfig(
        job_name=slurm_job_name,
        partition=partition,
        n_gpus=gpus_per_node,
        n_nodes=n_nodes,
        snapshot_ref=snapshot_ref,
        comment=comment,
    )

    return generate_script(single_config, command, env=_CUDA_FLAGS)


def _create_array_slurm_script(
    slurm_job_name: str,
    launch_id: str,
    run_cfgs: list[RunConfig],
    snapshot_ref: str,
    partition: str,
    project: str,
    max_concurrent_tasks: int | None = None,
) -> str:
    """Create a SLURM script for one or more runs."""
    n_gpus_each = [_n_gpus_for(run_cfg.runtime) for run_cfg in run_cfgs]
    assert all(n == n_gpus_each[0] for n in n_gpus_each), (
        "all runs must have the same number of GPUs"
    )
    n_gpus = n_gpus_each[0]

    commands = [
        _get_command(
            launch_id=launch_id,
            run_cfg=run_cfg,
            spec_idx=i,
            n_gpus=n_gpus,
            snapshot_ref=snapshot_ref,
            workspace_job_id_bash=ARRAY_JOB_ID_BASH,
            project=project,
        )
        for i, run_cfg in enumerate(run_cfgs)
    ]

    n_nodes, gpus_per_node = _n_nodes_and_gpus_per_node(n_gpus)

    array_config = SlurmArrayConfig(
        job_name=slurm_job_name,
        partition=partition,
        n_gpus=gpus_per_node,
        n_nodes=n_nodes,
        snapshot_ref=snapshot_ref,
        max_concurrent_tasks=max_concurrent_tasks,
    )

    return generate_array_script(
        array_config,
        commands,
        env=_CUDA_FLAGS,
        per_task_comments=[get_wandb_run_url(project, run_cfg.run_id) for run_cfg in run_cfgs],
    )
