"""Internal SLURM worker entrypoint for PD experiments.

Each SLURM array task invokes this module via
``python -m param_decomp.experiments._worker --run_json ...``.
Not a console script and not part of the user-facing CLI; the launcher in
``param_decomp/scripts/run_slurm.py`` is the only subprocess caller.
``run_experiment`` is also called in-process by ``pd-run --local``.
"""

import json

import fire

from param_decomp import run_pd
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
    """Set up the distributed env + per-task seed, then hand off to `run_pd`.

    W&B tagging (driver name + launch_id + SLURM env) lives in `run_pd` — see
    `_wandb_tags` there.
    """
    dist_state = init_distributed()
    logger.info(f"Distributed state: {dist_state}")
    set_seed(run_cfg.pd.seed)
    device = get_device()
    if is_main_process():
        logger.info(f"Using device: {device}")

    run_pd(
        run_cfg,
        device=device,
        dist_state=dist_state,
        wandb_project=wandb_project,
        launch_id=launch_id,
    )


@with_distributed_cleanup
def main(
    run_json: str,
    launch_id: str | None = None,
    wandb_project: str | None = None,
) -> None:
    """SLURM task entrypoint."""
    run = RunConfig.model_validate(json.loads(run_json))
    run_experiment(run, launch_id=launch_id, wandb_project=wandb_project)


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
