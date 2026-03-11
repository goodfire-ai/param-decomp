"""SPD launcher for experiments with sweeps and SLURM orchestration.

Provides a full-featured entry point for launching SPD experiments on the cluster, supporting
parameter sweeps, multi-node training, git snapshots, and W&B workspace views/reports.

For simpler local execution without SLURM, use simple.py instead.

The actual cli entry point is in run_cli.py. this is to speed up --help.
"""

import copy
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from spd.configs import Config
from spd.log import logger
from spd.registry import EXPERIMENT_REGISTRY, get_max_expected_runtime
from spd.settings import REPO_ROOT
from spd.utils.compute_utils import (
    GPUS_PER_NODE,
    TrainingJob,
    create_slurm_script,
)
from spd.utils.git_utils import create_git_snapshot
from spd.utils.run_utils import apply_nested_updates, generate_grid_combinations, generate_run_id
from spd.utils.slurm import submit_slurm_job
from spd.utils.wandb_utils import (
    ReportCfg,
    create_view_and_report,
    generate_wandb_run_name,
    get_wandb_run_url,
)


def launch_slurm_run(
    experiments: str | tuple[str, ...] | None,
    sweep: str | bool,
    n_agents: int | None,
    create_report: bool,
    report_title: str | None,
    job_suffix: str | None,
    cpu: bool,
    partition: str,
    dp: int | None,
    project: str,
) -> None:
    """Run SPD experiments on SLURM cluster with optional sweeps.

    Args:
        experiments: Comma-separated experiment names (default: all experiments)
        sweep: Enable parameter sweep. Pass True for default params or a YAML path.
        n_agents: Number of concurrent SLURM tasks (required for sweeps)
        create_report: Create a W&B report in addition to workspace view
        report_title: Title for the W&B report (requires create_report=True)
        job_suffix: Suffix for SLURM job names
        cpu: Run on CPU instead of GPU
        partition: SLURM partition name (default: h200-reserved)
        dp: Number of GPUs for data parallelism. For multi-node, dp > 8 (must be divisible by 8).
        project: W&B project name
    """

    launch_id = _generate_launch_id()
    logger.info(f"Launch ID: {launch_id}")

    experiments_list = _get_experiments(experiments)
    logger.info(f"Experiments: {', '.join(experiments_list)}")

    n_gpus = _validate_and_get_n_gpus(cpu=cpu, dp=dp)
    logger.info(f"Running on {_format_compute_info(n_gpus)}")

    sweep_params = _get_sweep_params(sweep)
    if sweep_params is not None:
        assert n_agents is not None, "n_agents must be provided when sweep is enabled"

    training_jobs = _create_training_jobs(
        experiments=experiments_list,
        project=project,
        sweep_params=sweep_params,
    )

    snapshot_branch, commit_hash = create_git_snapshot(snapshot_id=launch_id)
    logger.info(f"Created git snapshot branch: {snapshot_branch} ({commit_hash[:8]})")

    if len(training_jobs) > 1:
        _create_wandb_views_and_report(
            create_report=create_report,
            report_title=report_title,
            project=project,
            launch_id=launch_id,
            experiments_list=experiments_list,
            snapshot_branch=snapshot_branch,
            commit_hash=commit_hash,
        )

    slurm_job_name = f"spd-{job_suffix or get_max_expected_runtime(experiments_list)}"

    wandb_urls = [get_wandb_run_url(project, job.run_id) for job in training_jobs]

    is_array = len(training_jobs) > 1

    script_content = create_slurm_script(
        slurm_job_name=slurm_job_name,
        launch_id=launch_id,
        training_jobs=training_jobs,
        sweep_params=sweep_params,
        snapshot_branch=snapshot_branch,
        n_gpus=n_gpus,
        partition=partition,
        max_concurrent_tasks=n_agents,
        per_task_comments=wandb_urls,
    )

    # Submit script (handles file writing, submission, renaming, and log file creation)
    result = submit_slurm_job(
        script_content,
        f"launch_{launch_id}",
        is_array=is_array,
        n_array_tasks=len(training_jobs) if is_array else None,
    )

    logger.section("Job submitted successfully!")
    summary: dict[str, str | int | None] = {
        "Array Job ID" if is_array else "Job ID": result.job_id,
        "Total training jobs": len(training_jobs),
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


def _create_training_jobs(
    experiments: list[str],
    project: str,
    sweep_params: dict[str, Any] | None,
) -> list[TrainingJob]:
    """Build a Run containing jobs for all experiments.

    NOTE: When we convert parameter settings into JSON strings to pass to our decomposition scripts,
    we add a prefix to prevent Fire parsing with ast.literal_eval
    (https://github.com/google/python-fire/issues/332)
    """

    training_jobs: list[TrainingJob] = []

    logger.info("Task breakdown by experiment:")
    task_breakdown: dict[str, str] = {}

    for experiment in experiments:
        exp_config = EXPERIMENT_REGISTRY[experiment]

        # Load base config
        base_config = Config.from_file(exp_config.config_path)

        if sweep_params is None:
            # Fixed configuration run - still use JSON to ensure project override works
            base_config_dict = base_config.model_dump(mode="json")
            base_config_dict["wandb_project"] = project
            config_with_overrides = Config(**base_config_dict)

            training_jobs.append(
                TrainingJob(
                    experiment=experiment,
                    script_path=exp_config.decomp_script,
                    config=config_with_overrides,
                    run_id=generate_run_id("spd"),
                )
            )
            task_breakdown[experiment] = "1 job"

        else:
            # Parameter sweep run
            exp_sweep_params = _get_experiment_sweep_params(experiment, sweep_params)

            combinations = generate_grid_combinations(exp_sweep_params)

            for i, param_combo in enumerate(combinations):
                # Apply parameter overrides
                base_config_dict = base_config.model_dump(mode="json")
                config_dict_with_overrides = apply_nested_updates(base_config_dict, param_combo)
                config_dict_with_overrides["wandb_project"] = project
                wandb_run_name = f"{experiment}-{generate_wandb_run_name(param_combo)}"
                config_dict_with_overrides["wandb_run_name"] = wandb_run_name
                config_with_overrides = Config(**config_dict_with_overrides)

                training_jobs.append(
                    TrainingJob(
                        experiment=experiment,
                        script_path=exp_config.decomp_script,
                        config=config_with_overrides,
                        run_id=generate_run_id("spd"),
                    )
                )

                # Print first combination as example
                if i == 0:
                    logger.info(f"  {experiment}: {len(combinations)} jobs")
                    logger.info(f"    Example param overrides: {param_combo}")

    if task_breakdown:
        logger.values(task_breakdown)

    return training_jobs


def _get_experiment_sweep_params(
    experiment_name: str, sweep_params: dict[str, Any]
) -> dict[str, Any]:
    assert experiment_name != "global"

    # Start with global parameters if they exist
    params = copy.deepcopy(sweep_params["global"]) if "global" in sweep_params else {}

    # Merge experiment-specific parameters if they exist
    if experiment_name in sweep_params:
        experiment_params = sweep_params[experiment_name]
        _merge_sweep_params(params, experiment_params)

    if not params:
        raise ValueError(f"No sweep parameters found for experiment '{experiment_name}'")

    return params


def _merge_sweep_params(base: dict[str, Any], override: dict[str, Any]) -> None:
    """Recursively merge override parameters into base parameters.

    Handles nested parameter structures and overwrites values from base with override.
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            # Both are dicts, merge recursively
            _merge_sweep_params(base[key], value)
        else:
            # Override the value
            base[key] = value


def _get_experiments(
    experiments_input: str | tuple[str, ...] | None = None,
) -> list[str]:
    """Get and validate the list of experiments to run.

    Args:
        experiments_input: Experiment names as comma-separated string or tuple.
            If None, runs all experiments.

    Returns:
        List of experiment names to run.
    """

    # Determine experiment list
    if experiments_input is None:
        experiments = list(EXPERIMENT_REGISTRY.keys())
    elif isinstance(experiments_input, tuple):
        experiments = [exp.strip() for exp in experiments_input]
    else:
        experiments = [exp.strip() for exp in experiments_input.split(",")]

    # Validate experiment names
    invalid_experiments = [exp for exp in experiments if exp not in EXPERIMENT_REGISTRY]
    if invalid_experiments:
        raise ValueError(f"Invalid experiments: {invalid_experiments}")

    return experiments


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
    """Format compute configuration for logging."""
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
        sweep_params = yaml.safe_load(f)
    return sweep_params


def _resolve_sweep_params_path(sweep_params_file: str) -> Path:
    """Resolve the full path to the sweep parameters file."""
    if "/" not in sweep_params_file:
        # Look in scripts directory by default
        return REPO_ROOT / "spd/scripts" / sweep_params_file
    else:
        return REPO_ROOT / sweep_params_file


def _create_wandb_views_and_report(
    create_report: bool,
    report_title: str | None,
    project: str,
    launch_id: str,
    experiments_list: list[str],
    snapshot_branch: str,
    commit_hash: str,
) -> None:
    """Set up W&B workspace view and optionally a report."""

    match create_report, report_title:
        case True, None:
            report_cfg = ReportCfg(
                report_title=report_title,
                branch=snapshot_branch,
                commit_hash=commit_hash,
            )
        case True, title:
            report_cfg = ReportCfg(
                report_title=title,
                branch=snapshot_branch,
                commit_hash=commit_hash,
            )
        case False, _:
            report_cfg = None

    create_view_and_report(
        project=project,
        launch_id=launch_id,
        experiments=experiments_list,
        report_cfg=report_cfg,
    )
