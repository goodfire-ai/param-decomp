"""``pd-run``: the single user-facing CLI for running PD experiments.

By default ``pd-run`` submits a SLURM job. Pass ``--local`` to run in-process instead
(no SLURM, no git snapshot). The launcher's internal worker entrypoint lives in
``param_decomp/experiments/_worker.py`` and is not user-facing.
"""

import os
import shutil
from pathlib import Path
from typing import Any

import fire
import yaml

from param_decomp.experiments._worker import run_experiment
from param_decomp.experiments.discovery import discover_experiments
from param_decomp.settings import (
    DEFAULT_PARTITION_NAME,
    DEFAULT_PROJECT_NAME,
    REPO_ROOT,
)


def _resolve_source(
    experiment: str | None,
    config_path: str | Path | None,
    driver: str | None,
    rerun: str | None,
) -> tuple[str, str, dict[str, Any]]:
    """Resolve CLI inputs into ``(name, driver_path, base_config)``.

    Exactly one of ``experiment``, ``config_path``, or ``rerun`` must be set.
    ``driver`` is required iff ``config_path`` is set.
    """
    sources_set = sum(x is not None for x in (experiment, config_path, rerun))
    assert sources_set == 1, (
        "Pass exactly one of: positional <experiment>, --config_path, or --rerun. "
        "Run `pd-run --help` for examples."
    )

    if experiment is not None:
        assert driver is None, "--driver is only used with --config_path"
        discovered = discover_experiments()
        assert experiment in discovered, (
            f"Unknown experiment '{experiment}'. Available: {', '.join(sorted(discovered))}"
        )
        exp = discovered[experiment]
        with open(REPO_ROOT / exp.config_path) as f:
            return experiment, exp.driver_path, yaml.safe_load(f)

    if config_path is not None:
        assert driver is not None, "--config_path requires --driver <module:Driver>"
        with open(Path(config_path)) as f:
            config_data = yaml.safe_load(f)
        assert isinstance(config_data, dict), "config must be a YAML mapping"
        return Path(config_path).stem, driver, config_data

    assert rerun is not None  # by `sources_set == 1`
    assert driver is None, "--driver is implied by --rerun (read from saved metadata)"
    from param_decomp.saved_run import PDRun

    metadata = PDRun.metadata_from_path(rerun)
    assert metadata.driver is not None, (
        f"Cannot rerun {rerun!r}: saved run has no driver. Reruns require a driver-managed run."
    )
    return "rerun", metadata.driver, metadata.config


def _resolve_project(project: str | None, rerun: str | None) -> str:
    """If --project is unset and we're rerunning, inherit from the saved metadata."""
    if project is not None:
        return project
    if rerun is not None:
        from param_decomp.saved_run import PDRun

        metadata = PDRun.metadata_from_path(rerun)
        if metadata.wandb_project is not None:
            return metadata.wandb_project
    return DEFAULT_PROJECT_NAME


def _resolve_dp(cli_dp: int | None, base_config: dict[str, Any]) -> int | None:
    """CLI --dp overrides; otherwise use ``runtime.dp`` from the experiment config (if set).

    Validates via RuntimeConfig (pydantic) on whichever value we pick, then writes the
    resolved value back into ``base_config`` so the worker — and the saved RunMetadata —
    record what actually ran.
    """
    from param_decomp.configs import RuntimeConfig

    runtime_dict = dict(base_config.get("runtime", {}))
    effective = cli_dp if cli_dp is not None else runtime_dict.get("dp")
    runtime_dict["dp"] = effective
    RuntimeConfig.model_validate(runtime_dict)
    base_config.setdefault("runtime", {})["dp"] = effective
    return effective


def main(
    experiment: str | None = None,
    *,
    config_path: str | Path | None = None,
    driver: str | None = None,
    rerun: str | None = None,
    local: bool = False,
    sweep: str | None = None,
    n_agents: int | None = None,
    job_suffix: str | None = None,
    cpu: bool = False,
    partition: str = DEFAULT_PARTITION_NAME,
    dp: int | None = None,
    project: str | None = None,
) -> None:
    """Run a PD experiment, on SLURM by default.

    Args:
        experiment: Built-in experiment name (e.g. ``tms_5-2``). Run with no args to
            see the discovered list.
        config_path: Path to an experiment YAML. Requires --driver.
        driver: Driver import path ``pkg.module:ClassName``. Used with --config_path.
        rerun: Path or wandb URL of a saved run to rerun. Loads driver + config from
            the run's ``run_metadata.yaml``.
        local: Run in this process; skip SLURM, git snapshot, etc. Useful for quick
            checks. Disables --sweep and --dp.
        sweep: Sweep spec. Either a yaml path (e.g. ``my_grid.yaml`` — shorthand for
            the built-in cartesian generator), a registered generator name
            (``cartesian:my_grid.yaml``, ``my_sweep_name``), or a custom import path
            (``pkg.module:MyGenerator`` or ``pkg.module:MyGenerator:<arg>``).
        n_agents: Max concurrent SLURM tasks for sweeps.
        job_suffix: Suffix for the SLURM job name.
        cpu: Run on CPU.
        partition: SLURM partition.
        dp: GPUs for DDP. Overrides ``runtime.dp`` in the experiment config if set.
            ``<= 8`` is single-node; multiples of 8 above 8 multi-node.
        project: W&B project name. Defaults to the project recorded on the rerun's
            saved metadata (if any), otherwise ``DEFAULT_PROJECT_NAME``.

    Examples:
        pd-run tms_5-2                                       # one SLURM job
        pd-run tms_5-2 --sweep my_grid.yaml --n_agents 4     # cartesian grid sweep
        pd-run tms_5-2 --sweep pkg.module:MySweep            # custom generator
        pd-run tms_5-2 --dp 4                                # multi-GPU DDP
        pd-run tms_5-2 --cpu                                 # CPU job
        pd-run --driver pkg:D --config_path my.yaml          # custom driver
        pd-run --rerun s-a1b2c3d4                            # rerun from saved metadata
        pd-run tms_5-2 --local                               # in-process; no SLURM
    """
    if experiment is None and config_path is None and rerun is None:
        discovered = discover_experiments()
        print("Available experiments:")
        for name in sorted(discovered):
            print(f"  {name}")
        print("\nUse `pd-run <experiment>` or `pd-run --help` for details.")
        return

    name, driver_path, base_config = _resolve_source(experiment, config_path, driver, rerun)
    project = _resolve_project(project, rerun)
    dp = _resolve_dp(dp, base_config)

    if local:
        assert sweep is None, "--sweep is not supported with --local"
        assert dp is None, "--dp is not supported with --local"
        if cpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        from param_decomp.utils.distributed_utils import with_distributed_cleanup

        with_distributed_cleanup(run_experiment)(
            driver_path,
            base_config,
            wandb_project=project,
        )
        return

    assert shutil.which("sbatch") is not None, (
        "`sbatch` not found on PATH. Off-cluster, use `pd-run ... --local`."
    )
    assert not (sweep is not None and rerun is not None), "--sweep is not valid with --rerun"
    assert not (sweep is not None and config_path is not None), (
        "--sweep is only supported for built-in experiments"
    )

    from param_decomp.scripts.run_slurm import launch_slurm

    launch_slurm(
        name=name,
        driver_path=driver_path,
        base_config=base_config,
        sweep=sweep,
        n_agents=n_agents,
        job_suffix=job_suffix,
        cpu=cpu,
        partition=partition,
        dp=dp,
        project=project,
    )


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
