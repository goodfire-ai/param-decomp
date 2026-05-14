"""Generic experiment runner used by built-in and custom drivers."""

import json
from pathlib import Path
from typing import Any

import fire
import yaml

from param_decomp import run_pd
from param_decomp.experiments.driver import load_driver
from param_decomp.log import logger
from param_decomp.run_metadata import RunMetadata
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
    assert (config_path is None) != (config_json is None), (
        "Exactly one of config_path or config_json must be provided"
    )
    if config_path is not None:
        with open(Path(config_path)) as f:
            data = yaml.safe_load(f)
    else:
        assert config_json is not None
        data = json.loads(config_json.removeprefix("json:"))
    if "driver" in data and "config" in data:
        return data["driver"], data["config"]
    return None, data


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
    config_path: Path | str | None = None,
    config_json: str | None = None,
    driver: str | None = None,
    evals_id: str | None = None,
    launch_id: str | None = None,
    sweep_params_json: str | None = None,
    run_id: str | None = None,
) -> None:
    driver_from_metadata, config_data = _load_run_inputs(config_path, config_json)
    resolved_driver = driver if driver is not None else driver_from_metadata
    assert resolved_driver is not None, (
        "No driver provided and config has no driver field; pass --driver"
    )
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
