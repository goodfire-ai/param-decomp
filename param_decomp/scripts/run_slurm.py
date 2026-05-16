"""SLURM launch helpers for PD experiments.

Internal — invoked by ``pd-run`` (``param_decomp/experiments/runner.py``). Discovers the
named experiment, optionally expands a parameter grid (``--sweep``), creates a git snapshot
for reproducibility, builds in-memory config dicts, and submits a SLURM array where each task
invokes ``python -m param_decomp.experiments._worker`` on one config.

For single-machine execution, use ``pd-run <experiment> --local``.
"""

import copy
import json
import shlex
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

from param_decomp.log import logger
from param_decomp.settings import REPO_ROOT
from param_decomp.utils.git_utils import create_git_snapshot
from param_decomp.utils.run_utils import (
    apply_nested_updates,
    generate_grid_combinations,
    generate_run_id,
)
from param_decomp.utils.slurm import (
    SlurmArrayConfig,
    SlurmConfig,
    generate_array_script,
    generate_git_snapshot_setup,
    generate_script,
    submit_slurm_job,
)
from param_decomp.utils.wandb_utils import (
    generate_wandb_run_name,
    get_wandb_run_url,
)

_CUDA_FLAGS = {
    "NCCL_DEBUG": "WARN",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
}
_GPUS_PER_NODE = 8


@dataclass(frozen=True, slots=True)
class _RunSpec:
    """Everything needed to launch one PD run: a driver, a config, and a pre-allocated run_id.

    A list of these is what a launcher submits as a SLURM array (one task per spec).
    """

    driver_path: str
    config_dict: dict[str, Any]
    """The experiment config as a JSON-serializable mapping. Passed to the worker via
    `--config_json` and validated by the driver's Pydantic config_type."""
    run_id: str
    """Pre-generated unique run identifier (e.g. "s-a1b2c3d4"). The worker passes this
    to `run_pd`, which writes outputs to PARAM_DECOMP_OUT_DIR/decompositions/<run_id>/."""


def launch_slurm(
    name: str,
    driver_path: str,
    base_config: dict[str, Any],
    sweep: str | bool,
    n_agents: int | None,
    job_suffix: str | None,
    cpu: bool,
    partition: str,
    dp: int | None,
    project: str,
) -> None:
    """Submit a PD experiment to SLURM (with optional sweep).

    Args:
        name: Display name (used as the sweep-params lookup key and in run-name prefixes).
        driver_path: Driver import path ``pkg.module:ClassName``.
        base_config: Parsed experiment config dict (the YAML loaded into a mapping).
        sweep: Enable parameter sweep. Pass True for default params or a YAML path.
        n_agents: Number of concurrent SLURM tasks (required for sweeps).
        job_suffix: Suffix for SLURM job names.
        cpu: Run on CPU instead of GPU.
        partition: SLURM partition name.
        dp: Number of GPUs for data parallelism. For multi-node, dp > 8 (must be divisible by 8).
        project: W&B project name.
    """

    launch_id = _generate_launch_id()
    logger.info(f"Launch ID: {launch_id}")
    logger.info(f"Experiment: {name}")

    n_gpus = _validate_and_get_n_gpus(cpu=cpu, dp=dp)
    logger.info(f"Running on {_format_compute_info(n_gpus)}")

    sweep_params = _get_sweep_params(sweep)
    if sweep_params is not None:
        assert n_agents is not None, "n_agents must be provided when sweep is enabled"

    run_specs = _create_run_specs(
        name=name,
        driver_path=driver_path,
        base_config=base_config,
        project=project,
        sweep_params=sweep_params,
    )

    snapshot_ref, commit_hash = create_git_snapshot(snapshot_id=launch_id)
    logger.info(f"Created git snapshot ref: {snapshot_ref} ({commit_hash[:8]})")

    slurm_job_name = f"pd-{job_suffix}" if job_suffix else "pd"

    wandb_urls = [get_wandb_run_url(project, spec.run_id) for spec in run_specs]

    is_array = len(run_specs) > 1

    script_content = _create_slurm_script(
        slurm_job_name=slurm_job_name,
        launch_id=launch_id,
        run_specs=run_specs,
        sweep_params=sweep_params,
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
        n_array_tasks=len(run_specs) if is_array else None,
    )

    logger.section("Job submitted successfully!")
    summary: dict[str, str | int | None] = {
        "Array Job ID" if is_array else "Job ID": result.job_id,
        "Total runs": len(run_specs),
        "Max concurrent tasks": n_agents,
        "View logs in": result.log_pattern,
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


def _create_run_specs(
    name: str,
    driver_path: str,
    base_config: dict[str, Any],
    project: str,
    sweep_params: dict[str, Any] | None,
) -> list[_RunSpec]:
    """Build the run specs for one experiment.

    Returns one spec for a fixed run, or one per grid combination for a sweep.
    """
    if sweep_params is None:
        config_dict = copy.deepcopy(base_config)
        config_dict.setdefault("pd", {})["wandb_project"] = project
        return [
            _RunSpec(
                driver_path=driver_path,
                config_dict=config_dict,
                run_id=generate_run_id("param_decomp"),
            )
        ]

    exp_sweep_params = _get_experiment_sweep_params(name, sweep_params)
    combinations = generate_grid_combinations(exp_sweep_params)
    logger.info(f"Sweep: {len(combinations)} runs")
    logger.info(f"  Example param overrides: {combinations[0]}")

    run_specs: list[_RunSpec] = []
    for param_combo in combinations:
        config_dict = apply_nested_updates(base_config, param_combo)
        config_dict.setdefault("pd", {})["wandb_project"] = project
        config_dict["pd"]["wandb_run_name"] = f"{name}-{generate_wandb_run_name(param_combo)}"
        run_specs.append(
            _RunSpec(
                driver_path=driver_path,
                config_dict=config_dict,
                run_id=generate_run_id("param_decomp"),
            )
        )
    return run_specs


def _get_experiment_sweep_params(
    experiment_name: str, sweep_params: dict[str, Any]
) -> dict[str, Any]:
    assert experiment_name != "global"

    params = copy.deepcopy(sweep_params["global"]) if "global" in sweep_params else {}

    if experiment_name in sweep_params:
        _merge_sweep_params(params, sweep_params[experiment_name])

    if not params:
        raise ValueError(f"No sweep parameters found for experiment '{experiment_name}'")

    return params


def _merge_sweep_params(base: dict[str, Any], override: dict[str, Any]) -> None:
    """Recursively merge override parameters into base parameters."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _merge_sweep_params(base[key], value)
        else:
            base[key] = value


def _validate_and_get_n_gpus(cpu: bool, dp: int | None) -> int | None:
    """Validate dp argument and return the number of GPUs to use.

    Returns None for CPU-only runs, otherwise returns the validated number of GPUs.
    """
    if cpu:
        assert dp is None, "dp should not be specified when running on cpu"
        return None

    if dp is None:
        return None

    assert dp >= 2, "if given, dp must be at least 2. pass dp=None to use a single GPU."
    assert dp <= _GPUS_PER_NODE or dp % _GPUS_PER_NODE == 0, (
        f"dp must be <= {_GPUS_PER_NODE} (single node) or divisible by {_GPUS_PER_NODE} (multi-node), "
        f"got {dp}"
    )
    return dp


def _format_compute_info(n_gpus: int | None) -> str:
    if n_gpus is None:
        return "single GPU"
    if n_gpus <= _GPUS_PER_NODE:
        return f"{n_gpus} GPUs (single node)"
    n_nodes = n_gpus // _GPUS_PER_NODE
    return f"{n_gpus} GPUs ({n_nodes} nodes x {_GPUS_PER_NODE} GPUs)"


def _get_sweep_params(sweep: str | bool) -> dict[str, Any] | None:
    if sweep is False:
        return None
    sweep_params_file = "sweep_params.yaml" if sweep is True else sweep
    sweep_params_path = _resolve_sweep_params_path(sweep_params_file)
    with open(sweep_params_path) as f:
        return yaml.safe_load(f)


def _resolve_sweep_params_path(sweep_params_file: str) -> Path:
    if "/" not in sweep_params_file:
        return REPO_ROOT / "param_decomp/scripts" / sweep_params_file
    return REPO_ROOT / sweep_params_file


def _choose_master_port(run_id_local: str, idx: int) -> int:
    """Choose a unique port per command.

    Uses a stable hash of (run_id, idx) mapped into a high, unprivileged port range so that we can
    run multiple DDP processes on the same machine.
    """
    base: int = 20000
    span: int = 20000  # ports in [20000, 40000)
    h: int = int(sha256(f"{run_id_local}:{idx}".encode()).hexdigest(), 16)
    return base + (h % span)


def _build_script_args(
    launch_id: str,
    run_spec: _RunSpec,
    sweep_params: dict[str, Any] | None,
) -> str:
    """Build the worker-CLI arguments for one SLURM task."""
    json_tagged_config = f"json:{json.dumps(run_spec.config_dict)}"
    args = (
        f"--config_json {shlex.quote(json_tagged_config)} "
        f"--driver {shlex.quote(run_spec.driver_path)} "
        f"--launch_id {launch_id} "
        f"--run_id {run_spec.run_id}"
    )
    if sweep_params is not None:
        json_tagged_sweep_params = f"json:{json.dumps(sweep_params)}"
        args += f" --sweep_params_json {shlex.quote(json_tagged_sweep_params)}"
    return args


def _get_command(
    launch_id: str,
    run_spec: _RunSpec,
    spec_idx: int,
    n_gpus: int | None,
    sweep_params: dict[str, Any] | None,
    snapshot_ref: str,
    is_array: bool,
) -> str:
    """Build the command to run one PD run spec.

    Args:
        launch_id: Launch identifier for this group of runs.
        run_spec: The run spec to execute.
        spec_idx: Index of the run spec within the launch.
        n_gpus: Number of GPUs. None or 1 means single GPU/CPU. 2-8 means single-node DDP.
                >8 means multi-node DDP (must be divisible by 8).
        sweep_params: Optional sweep parameters to pass to the worker.
        snapshot_ref: Git ref to checkout (used for multi-node workspace setup).
        is_array: Whether this command is part of a SLURM array.
    """
    port = _choose_master_port(launch_id, spec_idx)
    script_args = _build_script_args(launch_id, run_spec, sweep_params)

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
    run_specs: list[_RunSpec],
    sweep_params: dict[str, Any] | None,
    snapshot_ref: str,
    n_gpus: int | None,
    partition: str,
    max_concurrent_tasks: int | None = None,
    per_task_comments: list[str] | None = None,
) -> str:
    """Create a SLURM script for one or more run specs (with a git-snapshot checkout step).

    For a single spec, generates a regular SLURM script. For multiple, generates a SLURM
    array script with a case statement (one task per spec).

    Args:
        slurm_job_name: Name for the SLURM job.
        launch_id: Launch identifier for this group of runs.
        run_specs: Run specs to execute (one task per spec).
        sweep_params: Optional sweep parameters to pass to the worker.
        snapshot_ref: Git ref to checkout.
        n_gpus: Number of GPUs. None or 1 means single GPU. 2-8 means single-node DDP.
                >8 means multi-node DDP (must be divisible by 8).
        partition: SLURM partition to use.
        max_concurrent_tasks: Maximum number of array tasks to run concurrently. If None, no limit.
        per_task_comments: If provided, each task sets its own SLURM comment (e.g. wandb URL).
    """
    is_array = len(run_specs) > 1

    commands: list[str] = []
    for i, run_spec in enumerate(run_specs):
        cmd = _get_command(
            launch_id,
            run_spec,
            i,
            n_gpus,
            sweep_params,
            snapshot_ref=snapshot_ref,
            is_array=is_array,
        )
        commands.append(cmd)

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
