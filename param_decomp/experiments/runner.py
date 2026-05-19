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

from param_decomp.configs import RuntimeConfig
from param_decomp.experiments._worker import run_experiment
from param_decomp.experiments.discovery import discover_experiments
from param_decomp.experiments.driver import load_driver
from param_decomp.run import Run
from param_decomp.settings import (
    DEFAULT_PARTITION_NAME,
    DEFAULT_PROJECT_NAME,
    REPO_ROOT,
)
from param_decomp.sweeps import SweepSpec, load_sweep_generator


def _resolve_source(
    experiment: str | None,
    config_path: str | Path | None,
    driver: str | None,
    rerun: str | None,
) -> tuple[str, str, dict[str, Any]]:
    """Resolve CLI inputs into ``(name, driver_path, config_data)``.

    Exactly one of ``experiment``, ``config_path``, or ``rerun`` must be set.
    ``driver`` is required iff ``config_path`` is set.

    Returns the parsed YAML dict (not yet merged with driver_path/wandb_*) so
    the launcher can splice in launcher-stamped fields before validation.
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
            config_data = yaml.safe_load(f)
        assert isinstance(config_data, dict), "config must be a YAML mapping"
        return experiment, exp.driver_path, config_data

    if config_path is not None:
        assert driver is not None, "--config_path requires --driver <module:Driver>"
        with open(Path(config_path)) as f:
            config_data = yaml.safe_load(f)
        assert isinstance(config_data, dict), "config must be a YAML mapping"
        return Path(config_path).stem, driver, config_data

    assert rerun is not None  # by `sources_set == 1`
    assert driver is None, "--driver is implied by --rerun (read from saved run config)"
    from param_decomp.saved_run import PDRun

    run = PDRun.run_from_path(rerun)
    assert run.driver_path is not None, (
        f"Cannot rerun {rerun!r}: saved run has no driver. Reruns require a driver-managed run."
    )
    return "rerun", run.driver_path, run.model_dump(mode="json")


def _resolve_sweep_spec(sweep_generator_path: str) -> SweepSpec:
    """Load and invoke a sweep generator.

    ``SweepSpec.__post_init__`` enforces shared driver + shared ``runtime:``
    block across all runs.
    """
    generator = load_sweep_generator(sweep_generator_path)
    spec = generator()
    assert isinstance(spec, SweepSpec), (
        f"sweep generator {generator!r} returned {type(spec).__name__}, expected SweepSpec"
    )
    return spec


def _build_run(
    *,
    driver_path: str,
    config_data: dict[str, Any],
    wandb_run_name: str,
    project: str,
) -> Run:
    """Splice ``driver_path``, ``wandb_run_name``, and ``wandb_project`` into
    ``config_data`` and validate."""
    logging_data = {
        **config_data.get("logging", {}),
        "wandb_run_name": wandb_run_name,
        "wandb_project": project,
    }
    return load_driver(driver_path).config_type.model_validate(
        {**config_data, "driver_path": driver_path, "logging": logging_data}
    )


def _stamp_project(spec: SweepSpec, project: str) -> SweepSpec:
    """Return a copy of ``spec`` with ``wandb_project`` set on every run.

    Sweep generators don't know the CLI ``--project`` value, so we stamp it
    here before the launcher sees the spec.
    """
    runs = [
        r.model_copy(update={"logging": r.logging.model_copy(update={"wandb_project": project})})
        for r in spec.runs
    ]
    return SweepSpec(description=spec.description, runs=runs)


def _parse_runtime(run: Run) -> RuntimeConfig:
    """Parse the ``runtime:`` block — substrate is config-only, no CLI overrides.

    Want a CPU smoke test of a GPU experiment? Edit the YAML or copy it first. Keeping the
    experiment's declared substrate as the single source of truth avoids silently running
    "the same experiment" on different substrates.
    """
    return run.runtime


def main(
    experiment: str | None = None,
    *,
    config_path: str | Path | None = None,
    driver: str | None = None,
    rerun: str | None = None,
    local: bool = False,
    sweep_generator_path: str | None = None,
    n_agents: int | None = None,
    job_suffix: str | None = None,
    partition: str = DEFAULT_PARTITION_NAME,
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
            checks. Incompatible with --sweep_generator_path and runtime.dp.
        sweep_generator_path: Absolute path to a sweep generator function in the
            form ``/abs/path/file.py:func_name``. The function takes no arguments
            and returns a ``SweepSpec`` (which carries its own driver_path and
            per-run configs). XOR with <experiment>, --config_path, --rerun.
        n_agents: Max concurrent SLURM tasks for sweeps.
        job_suffix: Suffix for the SLURM job name.
        partition: SLURM partition.
        project: W&B project name. Defaults to ``DEFAULT_PROJECT_NAME``.

    Substrate (device, dp, autocast_bf16) is declared in the experiment YAML's
    ``runtime:`` block — there are no CLI overrides. Edit the YAML to change it.
    For sweeps, the substrate comes from each generated config's ``runtime:`` block;
    all runs in one sweep must share the same substrate.

    Examples:
        pd-run tms_5-2                                                              # one SLURM job
        pd-run --sweep_generator_path /abs/path/my_sweep.py:my_sweep --n_agents 4   # sweep
        pd-run --driver pkg:D --config_path my.yaml                                 # custom driver
        pd-run --rerun s-a1b2c3d4                                                   # rerun from saved run
        pd-run tms_5-2 --local                                                      # in-process; no SLURM
    """
    if (
        experiment is None
        and config_path is None
        and rerun is None
        and sweep_generator_path is None
    ):
        discovered = discover_experiments()
        print("Available experiments:")
        for name in sorted(discovered):
            print(f"  {name}")
        print("\nUse `pd-run <experiment>` or `pd-run --help` for details.")
        return

    project = project if project is not None else DEFAULT_PROJECT_NAME

    if sweep_generator_path is not None:
        assert experiment is None and config_path is None and rerun is None and driver is None, (
            "--sweep_generator_path is mutually exclusive with <experiment>, "
            "--config_path, --driver, and --rerun"
        )
        assert not local, "--sweep_generator_path is not supported with --local"
        assert shutil.which("sbatch") is not None, (
            "`sbatch` not found on PATH. Off-cluster, use `pd-run ... --local` (no sweep)."
        )
        sweep_spec = _stamp_project(_resolve_sweep_spec(sweep_generator_path), project)
        runtime = _parse_runtime(sweep_spec.runs[0])
        from param_decomp.scripts.run_slurm import launch_slurm

        launch_slurm(
            launchable=sweep_spec,
            runtime=runtime,
            n_agents=n_agents,
            job_suffix=job_suffix,
            partition=partition,
        )
        return

    name, driver_path, config_data = _resolve_source(experiment, config_path, driver, rerun)
    run = _build_run(
        driver_path=driver_path, config_data=config_data, wandb_run_name=name, project=project
    )
    runtime = _parse_runtime(run)

    if local:
        assert runtime.dp is None, "runtime.dp is not supported with --local"
        if runtime.device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        from param_decomp.utils.distributed_utils import with_distributed_cleanup

        with_distributed_cleanup(run_experiment)(run)
        return

    assert shutil.which("sbatch") is not None, (
        "`sbatch` not found on PATH. Off-cluster, use `pd-run ... --local`."
    )

    from param_decomp.scripts.run_slurm import launch_slurm

    launch_slurm(
        launchable=run,
        runtime=runtime,
        n_agents=n_agents,
        job_suffix=job_suffix,
        partition=partition,
    )


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
