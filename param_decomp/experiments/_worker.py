"""Internal SLURM worker entrypoint for PD experiments.

Each SLURM array task invokes this module via
``python -m param_decomp.experiments._worker --run_json ...``.
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
from param_decomp.run import RunConfig
from param_decomp.utils.distributed_utils import (
    get_device,
    init_distributed,
    is_main_process,
    with_distributed_cleanup,
)
from param_decomp.utils.general_utils import set_seed


def run_experiment(
    run_cfg: RunConfig,
    *,
    launch_id: str | None = None,
    wandb_project: str | None = None,
) -> None:
    assert run_cfg.driver_path is not None, "run_experiment requires run.driver_path to be set"
    driver = load_driver(run_cfg.driver_path)
    assert isinstance(run_cfg, driver.config_type), (
        f"Run has type {type(run_cfg).__name__}, expected {driver.config_type.__name__}"
    )

    dist_state = init_distributed()
    logger.info(f"Distributed state: {dist_state}")

    set_seed(run_cfg.pd.seed)

    device = get_device()

    if is_main_process():
        logger.info(f"Driver: {driver.name}")
        logger.info(f"Using device: {device}")

    target = driver.build_target(run_cfg)
    target.model.to(device)
    train_loader = driver.build_train_loader(run_cfg, dist_state=dist_state, device=device)
    eval_loader = driver.build_eval_loader(run_cfg, dist_state=dist_state, device=device)

    wandb_tags = [driver.name, *([launch_id] if launch_id is not None else [])]

    run_pd(
        run_cfg,
        target=target,
        train_loader=train_loader,
        eval_loader=eval_loader,
        device=device,
        wandb_project=wandb_project,
        wandb_tags=wandb_tags,
    )


@with_distributed_cleanup
def main(
    run_json: str,
    launch_id: str | None = None,
    wandb_project: str | None = None,
) -> None:
    """SLURM task entrypoint."""
    run = RunConfig.from_dict(json.loads(run_json))
    run_experiment(run, launch_id=launch_id, wandb_project=wandb_project)


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
