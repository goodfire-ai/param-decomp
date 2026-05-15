"""SLURM launch helpers for PD experiments.

Internal — invoked by ``pd-run`` (``param_decomp/experiments/runner.py``). Discovers the
named experiment, optionally expands a parameter grid (``--sweep``), creates a git snapshot
for reproducibility, builds in-memory config dicts, and submits a SLURM array where each task
invokes ``python -m param_decomp.experiments._worker`` on one config.

For single-machine execution, use ``pd-run <experiment> --local``.
"""

import copy
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from param_decomp.log import logger
from param_decomp.settings import REPO_ROOT
from param_decomp.utils.compute_utils import (
    GPUS_PER_NODE,
    RunSpec,
    create_slurm_script,
)
from param_decomp.utils.git_utils import create_git_snapshot
from param_decomp.utils.run_utils import (
    apply_nested_updates,
    generate_grid_combinations,
    generate_run_id,
)
from param_decomp.utils.slurm import submit_slurm_job
from param_decomp.utils.wandb_utils import (
    generate_wandb_run_name,
    get_wandb_run_url,
)


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

    snapshot_branch, commit_hash = create_git_snapshot(snapshot_id=launch_id)
    logger.info(f"Created git snapshot branch: {snapshot_branch} ({commit_hash[:8]})")

    slurm_job_name = f"pd-{job_suffix}" if job_suffix else "pd"

    wandb_urls = [get_wandb_run_url(project, spec.run_id) for spec in run_specs]

    is_array = len(run_specs) > 1

    script_content = create_slurm_script(
        slurm_job_name=slurm_job_name,
        launch_id=launch_id,
        run_specs=run_specs,
        sweep_params=sweep_params,
        snapshot_branch=snapshot_branch,
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
) -> list[RunSpec]:
    """Build the run specs for one experiment.

    Returns one spec for a fixed run, or one per grid combination for a sweep.
    """
    if sweep_params is None:
        config_dict = copy.deepcopy(base_config)
        config_dict.setdefault("pd", {})["wandb_project"] = project
        return [
            RunSpec(
                driver_path=driver_path,
                config_dict=config_dict,
                run_id=generate_run_id("param_decomp"),
            )
        ]

    exp_sweep_params = _get_experiment_sweep_params(name, sweep_params)
    combinations = generate_grid_combinations(exp_sweep_params)
    logger.info(f"Sweep: {len(combinations)} runs")
    logger.info(f"  Example param overrides: {combinations[0]}")

    run_specs: list[RunSpec] = []
    for param_combo in combinations:
        config_dict = apply_nested_updates(base_config, param_combo)
        config_dict.setdefault("pd", {})["wandb_project"] = project
        config_dict["pd"]["wandb_run_name"] = f"{name}-{generate_wandb_run_name(param_combo)}"
        run_specs.append(
            RunSpec(
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
    assert dp <= GPUS_PER_NODE or dp % GPUS_PER_NODE == 0, (
        f"dp must be <= {GPUS_PER_NODE} (single node) or divisible by {GPUS_PER_NODE} (multi-node), "
        f"got {dp}"
    )
    return dp


def _format_compute_info(n_gpus: int | None) -> str:
    if n_gpus is None:
        return "single GPU"
    if n_gpus <= GPUS_PER_NODE:
        return f"{n_gpus} GPUs (single node)"
    n_nodes = n_gpus // GPUS_PER_NODE
    return f"{n_gpus} GPUs ({n_nodes} nodes x {GPUS_PER_NODE} GPUs)"


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
