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
    from param_decomp.pd_run import PDRun

    metadata = PDRun.metadata_from_path(rerun)
    assert metadata.driver is not None, (
        f"Cannot rerun {rerun!r}: saved run has no driver. Reruns require a driver-managed run."
    )
    return "rerun", metadata.driver, metadata.config


def main(
    experiment: str | None = None,
    *,
    config_path: str | Path | None = None,
    driver: str | None = None,
    rerun: str | None = None,
    local: bool = False,
    sweep: str | bool = False,
    n_agents: int | None = None,
    job_suffix: str | None = None,
    cpu: bool = False,
    partition: str = DEFAULT_PARTITION_NAME,
    dp: int | None = None,
    project: str = DEFAULT_PROJECT_NAME,
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
        sweep: Enable parameter sweep. ``True`` for the default grid file, or a YAML
            path. Built-in experiments only.
        n_agents: Max concurrent SLURM tasks for sweeps.
        job_suffix: Suffix for the SLURM job name.
        cpu: Run on CPU.
        partition: SLURM partition.
        dp: GPUs for DDP. ``<= 8`` is single-node; multiples of 8 above 8 multi-node.
        project: W&B project name.

    Examples:
        pd-run tms_5-2                                # one SLURM job
        pd-run tms_5-2 --sweep --n_agents 4           # SLURM array sweep
        pd-run tms_5-2 --dp 4                         # multi-GPU DDP
        pd-run tms_5-2 --cpu                          # CPU job
        pd-run --driver pkg:D --config_path my.yaml   # custom driver
        pd-run --rerun s-a1b2c3d4                     # rerun from saved metadata
        pd-run tms_5-2 --local                        # in-process; no SLURM
    """
    if experiment is None and config_path is None and rerun is None:
        discovered = discover_experiments()
        print("Available experiments:")
        for name in sorted(discovered):
            print(f"  {name}")
        print("\nUse `pd-run <experiment>` or `pd-run --help` for details.")
        return

    name, driver_path, base_config = _resolve_source(experiment, config_path, driver, rerun)

    if local:
        assert sweep is False, "--sweep is not supported with --local"
        assert dp is None, "--dp is not supported with --local"
        if cpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        base_config.setdefault("pd", {})["wandb_project"] = project
        from param_decomp.utils.distributed_utils import with_distributed_cleanup

        with_distributed_cleanup(run_experiment)(driver_path, base_config)
        return

    assert shutil.which("sbatch") is not None, (
        "`sbatch` not found on PATH. Off-cluster, use `pd-run ... --local`."
    )
    assert not (sweep is not False and rerun is not None), "--sweep is not valid with --rerun"
    assert not (sweep is not False and config_path is not None), (
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
