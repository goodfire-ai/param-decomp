"""Internal SLURM worker entrypoint for PD experiments.

Each SLURM array task invokes this module via
``python -m param_decomp.experiments._worker --config_json ... --driver ...``.
Not a console script and not part of the user-facing CLI; the launcher
in ``param_decomp/scripts/run_slurm.py`` is the only caller.

Users wanting to run an experiment in-process should call
``pd-run <experiment> --local`` (see ``param_decomp/experiments/runner.py``).
"""

import json
from typing import Any

import fire

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


def _decode_json_arg(value: str | None) -> Any:
    """Decode a ``json:...`` prefixed CLI string. Returns None for None inputs."""
    if value is None:
        return None
    return json.loads(value.removeprefix("json:"))


def run_experiment(
    driver_path: str,
    config_data: dict[str, Any],
    *,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
    view_meta: dict[str, Any] | None = None,
    launch_id: str | None = None,
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
        eval_batch_size=experiment_config.logging.eval_batch_size,
        dist_state=dist_state,
        device=device,
    )

    wandb_tags = [driver.name, *([launch_id] if launch_id is not None else [])]
    metadata = RunMetadata(
        driver=driver_path,
        config=experiment_config.model_dump(mode="json"),
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        view_meta=view_meta or {},
    )
    run_pd(
        config=experiment_config.pd,
        logging_config=experiment_config.logging,
        runtime_config=experiment_config.runtime,
        target=target,
        train_loader=train_loader,
        eval_loader=eval_loader,
        device=device,
        run_id=run_id,
        metadata=metadata,
        wandb_tags=wandb_tags,
    )


@with_distributed_cleanup
def main(
    config_json: str,
    driver: str,
    run_id: str,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
    view_meta_json: str | None = None,
    launch_id: str | None = None,
) -> None:
    """SLURM task entrypoint."""
    config_data = _decode_json_arg(config_json)
    assert isinstance(config_data, dict), "config_json must decode to a mapping"
    view_meta = _decode_json_arg(view_meta_json)
    assert view_meta is None or isinstance(view_meta, dict), (
        "view_meta_json must decode to a mapping"
    )
    run_experiment(
        driver,
        config_data,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        view_meta=view_meta,
        launch_id=launch_id,
        run_id=run_id,
    )


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
