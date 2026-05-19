"""Internal SLURM worker entrypoint for PD experiments.

Each SLURM array task invokes this module via
``python -m param_decomp.experiments._worker --run_spec_json ...``.
Not a console script and not part of the user-facing CLI; the launcher
in ``param_decomp/scripts/run_slurm.py`` is the only subprocess caller.
``run_experiment`` is also called in-process by ``pd-run --local``.

Users wanting to run an experiment in-process should call
``pd-run <experiment> --local`` (see ``param_decomp/experiments/runner.py``).
"""

import json

import fire

from param_decomp import run_pd
from param_decomp.experiments.driver import load_driver
from param_decomp.log import logger
from param_decomp.run_spec import RunSpec
from param_decomp.utils.distributed_utils import (
    get_device,
    init_distributed,
    is_main_process,
    with_distributed_cleanup,
)
from param_decomp.utils.general_utils import set_seed


def run_experiment(
    spec: RunSpec,
    *,
    run_id: str | None = None,
    launch_id: str | None = None,
) -> None:
    assert spec.driver is not None, "run_experiment requires spec.driver to be set"

    dist_state = init_distributed()
    logger.info(f"Distributed state: {dist_state}")

    driver = load_driver(spec.driver)
    experiment_config = driver.config_type.model_validate(spec.config)
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
    run_pd(
        config=experiment_config.pd,
        logging_config=experiment_config.logging,
        runtime_config=experiment_config.runtime,
        target=target,
        train_loader=train_loader,
        eval_loader=eval_loader,
        device=device,
        run_id=run_id,
        spec=spec,
        wandb_tags=wandb_tags,
    )


@with_distributed_cleanup
def main(
    run_spec_json: str,
    run_id: str,
    launch_id: str | None = None,
) -> None:
    """SLURM task entrypoint."""
    spec = RunSpec.from_dict(json.loads(run_spec_json))
    run_experiment(spec, run_id=run_id, launch_id=launch_id)


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
