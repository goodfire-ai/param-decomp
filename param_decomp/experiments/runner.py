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
    rerun: str | None,
) -> dict[str, Any]:
    """Resolve the chosen input source into a dict ready for ``Run.from_dict``.

    Exactly one of ``experiment``, ``config_path``, or ``rerun`` must be set.
    Every source is expected to provide ``driver_path`` as a top-level field
    (built-in YAMLs, user YAMLs, and saved ``run_metadata.yaml`` all declare it).

    Stamps ``logging.wandb_run_name`` (the experiment slug, config filename
    stem, or ``"rerun"``) so each YAML doesn't have to set one.
    """
    sources_set = sum(x is not None for x in (experiment, config_path, rerun))
    assert sources_set == 1, (
        "Pass exactly one of: positional <experiment>, --config_path, or --rerun. "
        "Run `pd-run --help` for examples."
    )

    if experiment is not None:
        discovered = discover_experiments()
        assert experiment in discovered, (
            f"Unknown experiment '{experiment}'. Available: {', '.join(sorted(discovered))}"
        )
        config_data = _load_yaml(REPO_ROOT / discovered[experiment].config_path)
        name = experiment
    elif config_path is not None:
        config_data = _load_yaml(Path(config_path))
        name = Path(config_path).stem
    else:
        assert rerun is not None  # by `sources_set == 1`
        from param_decomp.saved_run import PDRun

        config_data = PDRun.run_from_path(rerun).model_dump(mode="json")
        name = "rerun"

    assert config_data.get("driver_path"), (
        "Config is missing a top-level `driver_path:` field. "
        "Every PD config must declare its driver (e.g. "
        "`driver_path: param_decomp.experiments.tms.experiment:Driver`)."
    )
    config_data["logging"] = {**config_data.get("logging", {}), "wandb_run_name": name}
    return config_data


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict), f"config must be a YAML mapping: {path}"
    return data


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


def main(
    experiment: str | None = None,
    *,
    config_path: str | Path | None = None,
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
        config_path: Path to an experiment YAML. The YAML must declare its driver via
            a top-level ``driver_path:`` field.
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
        pd-run --config_path my.yaml                                                # custom config
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
        assert experiment is None and config_path is None and rerun is None, (
            "--sweep_generator_path is mutually exclusive with <experiment>, "
            "--config_path, and --rerun"
        )
        assert not local, "--sweep_generator_path is not supported with --local"
        assert shutil.which("sbatch") is not None, (
            "`sbatch` not found on PATH. Off-cluster, use `pd-run ... --local` (no sweep)."
        )
        sweep_spec = _resolve_sweep_spec(sweep_generator_path)
        from param_decomp.scripts.run_slurm import launch_slurm

        launch_slurm(
            launchable=sweep_spec,
            runtime=sweep_spec.runs[0].runtime,
            n_agents=n_agents,
            job_suffix=job_suffix,
            partition=partition,
            project=project,
        )
        return

    run = Run.from_dict(_resolve_source(experiment, config_path, rerun))

    if local:
        assert run.runtime.dp is None, "runtime.dp is not supported with --local"
        if run.runtime.device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        from param_decomp.utils.distributed_utils import with_distributed_cleanup

        with_distributed_cleanup(run_experiment)(run, wandb_project=project)
        return

    assert shutil.which("sbatch") is not None, (
        "`sbatch` not found on PATH. Off-cluster, use `pd-run ... --local`."
    )

    from param_decomp.scripts.run_slurm import launch_slurm

    launch_slurm(
        launchable=run,
        runtime=run.runtime,
        n_agents=n_agents,
        job_suffix=job_suffix,
        partition=partition,
        project=project,
    )


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
