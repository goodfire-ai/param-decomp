"""Run PD on a TMS model.

Note that the first instance index is fixed to the identity matrix. This is done so we can compare
the losses of the "correct" solution during training.
"""

from pathlib import Path

import fire

from param_decomp import run_pd
from param_decomp.experiments.tms.experiment import TMSExperimentConfig
from param_decomp.log import logger
from param_decomp.utils.distributed_utils import get_device
from param_decomp.utils.general_utils import set_seed
from param_decomp.utils.run_utils import parse_sweep_params


def _parse_tms_config(
    config_path: Path | str | None, config_json: str | None
) -> TMSExperimentConfig:
    import json

    import yaml

    assert (config_path is None) != (config_json is None), (
        "Exactly one of config_path or config_json must be provided"
    )
    if config_path is not None:
        with open(Path(config_path)) as f:
            data = yaml.safe_load(f)
    else:
        assert config_json is not None
        data = json.loads(config_json.removeprefix("json:"))
    return TMSExperimentConfig.model_validate(data)


def main(
    config_path: Path | str | None = None,
    config_json: str | None = None,
    evals_id: str | None = None,
    launch_id: str | None = None,
    sweep_params_json: str | None = None,
    run_id: str | None = None,
) -> None:
    exp = _parse_tms_config(config_path, config_json)

    device = get_device()
    logger.info(f"Using device: {device}")

    set_seed(exp.pd.seed)

    loaded = exp.load_target()
    loaded.target.model.to(device)

    train_loader, eval_loader = exp.build_dataloaders(
        seed=exp.pd.seed,
        train_batch_size=exp.pd.batch_size,
        eval_batch_size=exp.pd.eval_batch_size,
        device=device,
    )

    wandb_tags = [t for t in [evals_id, launch_id] if t is not None]

    run_pd(
        config=exp.pd,
        target=loaded.target,
        train_loader=train_loader,
        eval_loader=eval_loader,
        device=device,
        run_id=run_id,
        sweep_params=parse_sweep_params(sweep_params_json),
        experiment_config=exp,
        experiment_tag="tms",
        wandb_tags=wandb_tags,
        target_train_config=loaded.target_train_config,
    )


if __name__ == "__main__":
    fire.Fire(main)
