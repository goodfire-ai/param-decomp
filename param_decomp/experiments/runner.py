"""Single-process PD experiment worker.

Runs one PD training job from either a built-in experiment name, a YAML config path,
or an inline JSON config blob. Invoked directly by users as ``pd-run`` and from the
SLURM launcher as ``pd-run --config_json '...'  --driver '...' ...``.
"""

import json
import os
from pathlib import Path
from typing import Any

import fire
import yaml

from param_decomp import run_pd
from param_decomp.experiments.discovery import discover_experiments
from param_decomp.experiments.driver import load_driver
from param_decomp.log import logger
from param_decomp.run_metadata import RunMetadata
from param_decomp.settings import REPO_ROOT
from param_decomp.utils.distributed_utils import (
    get_device,
    init_distributed,
    is_main_process,
    with_distributed_cleanup,
)
from param_decomp.utils.general_utils import set_seed
from param_decomp.utils.run_utils import parse_sweep_params


def _load_run_inputs(
    config_path: Path | str | None, config_json: str | None
) -> tuple[str | None, dict[str, Any]]:
    """Load the YAML/JSON source and split it into (driver_from_metadata, experiment_config_dict).

    The source is either a pure experiment config (driver_from_metadata is None) or a saved
    ``run_metadata.yaml`` from a run directory (driver_from_metadata is the recorded driver
    import path). Letting users point ``--config_path`` at saved metadata enables one-flag
    reruns of a finished experiment.
    """
    if (config_path is None) == (config_json is None):
        raise ValueError("Pass exactly one of --config_path or --config_json.")

    if config_path is not None:
        with open(Path(config_path)) as f:
            data = yaml.safe_load(f)
    else:
        if config_json is None:
            raise ValueError("Pass exactly one of --config_path or --config_json.")
        data = json.loads(config_json.removeprefix("json:"))

    if not isinstance(data, dict):
        raise ValueError("Config source must contain a YAML/JSON mapping.")

    if "driver" in data and "config" in data:
        metadata = RunMetadata.from_dict(data)
        return metadata.driver, metadata.config
    return None, data


def _resolve_inputs(
    experiment: str | None,
    config_path: Path | str | None,
    config_json: str | None,
    driver: str | None,
) -> tuple[str, dict[str, Any]]:
    """Resolve CLI inputs into (driver_path, config_dict)."""
    if experiment is not None:
        if config_path is not None or config_json is not None or driver is not None:
            raise ValueError(
                "Choose one pd-run input mode: `pd-run <experiment>` for built-ins, "
                "`pd-run --config_path <yaml-or-run_metadata.yaml> "
                "[--driver <module:Driver>]` for config files, or launcher/internal "
                "`pd-run --config_json <json> --driver <module:Driver>`."
            )
        discovered = discover_experiments()
        if experiment not in discovered:
            available = ", ".join(sorted(discovered.keys()))
            raise ValueError(f"Unknown experiment '{experiment}'. Available: {available}")
        exp = discovered[experiment]
        with open(REPO_ROOT / exp.config_path) as f:
            config_data = yaml.safe_load(f)
        return exp.driver_path, config_data

    if config_path is None and config_json is None:
        raise ValueError(
            "No run input provided. Use `pd-run <experiment>`, "
            "`pd-run --config_path <yaml-or-run_metadata.yaml> [--driver <module:Driver>]`, "
            "or launcher/internal `pd-run --config_json <json> --driver <module:Driver>`."
        )

    driver_from_metadata, config_data = _load_run_inputs(config_path, config_json)
    resolved_driver = driver if driver is not None else driver_from_metadata
    if resolved_driver is None:
        raise ValueError(
            "Raw experiment configs require --driver <module:Driver>. "
            "Saved run_metadata.yaml files include their own driver."
        )
    return resolved_driver, config_data


def run_experiment(
    driver_path: str,
    config_data: dict[str, Any],
    *,
    evals_id: str | None = None,
    launch_id: str | None = None,
    sweep_params_json: str | None = None,
    run_id: str | None = None,
) -> None:
    dist_state = init_distributed()
    logger.info(f"Distributed state: {dist_state}")

    driver = load_driver(driver_path)
    experiment_config = driver.config_type.model_validate(config_data)
    set_seed(experiment_config.pd.seed)
    device = get_device()

    if is_main_process():
        logger.info(f"Driver: {driver.name}")
        logger.info(f"Using device: {device}")

    target = driver.build_target(experiment_config)
    target.model.to(device)
    train_loader, eval_loader = driver.build_dataloaders(
        experiment_config,
        train_batch_size=experiment_config.pd.batch_size,
        eval_batch_size=experiment_config.pd.eval_batch_size,
        dist_state=dist_state,
        device=device,
    )
    artifacts = driver.artifacts(experiment_config, target)

    wandb_tags = [driver.name, *(t for t in [evals_id, launch_id] if t is not None)]
    metadata = RunMetadata(
        driver=driver_path,
        config=experiment_config.model_dump(mode="json"),
        artifact_filenames=list(artifacts),
    )
    run_pd(
        config=experiment_config.pd,
        target=target,
        train_loader=train_loader,
        eval_loader=eval_loader,
        device=device,
        run_id=run_id,
        sweep_params=parse_sweep_params(sweep_params_json),
        metadata=metadata,
        artifacts=artifacts,
        wandb_tags=wandb_tags,
    )


@with_distributed_cleanup
def main(
    experiment: str | None = None,
    config_path: Path | str | None = None,
    config_json: str | None = None,
    driver: str | None = None,
    cpu: bool = False,
    evals_id: str | None = None,
    launch_id: str | None = None,
    sweep_params_json: str | None = None,
    run_id: str | None = None,
) -> None:
    """Run a single PD experiment in this process.

    Args:
        experiment: Built-in experiment name (e.g. 'tms_5-2'). Resolves the driver and YAML
            config via discover_experiments(). Mutually exclusive with --config_path,
            --config_json, and --driver.
        config_path: Path to an experiment YAML or a saved run_metadata.yaml.
        config_json: JSON-encoded config dict (may be ``json:``-prefixed). Used by the SLURM
            launcher to pass an in-memory config without writing to disk.
        driver: Driver import path ``pkg.module:ClassName``. Required with --config_path
            unless the config is a saved run_metadata.yaml, which carries its own driver.
        cpu: Force CPU execution by hiding all CUDA devices from this process.
        evals_id, launch_id, run_id, sweep_params_json: Set by the launcher; you generally
            don't need to pass these manually.

    Examples:
        pd-run tms_5-2                                # built-in by name
        pd-run tms_5-2 --cpu                          # built-in on CPU
        pd-run --config_path my.yaml --driver pkg:Driver   # custom driver
        pd-run --config_path run_metadata.yaml         # rerun from saved metadata
    """
    if cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    resolved_driver, config_data = _resolve_inputs(experiment, config_path, config_json, driver)
    run_experiment(
        resolved_driver,
        config_data,
        evals_id=evals_id,
        launch_id=launch_id,
        sweep_params_json=sweep_params_json,
        run_id=run_id,
    )


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
