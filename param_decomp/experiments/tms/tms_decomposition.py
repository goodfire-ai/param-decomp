"""Run PD on a TMS model.

Note that the first instance index is fixed to the identity matrix. This is done so we can compare
the losses of the "correct" solution during training.
"""

from pathlib import Path

import fire

from param_decomp.configs import TMSTaskConfig
from param_decomp.experiments.tms.models import TMSModel, TMSTargetRunInfo
from param_decomp.log import logger
from param_decomp.models.batch_and_loss_fns import recon_loss_mse, run_batch_first_element
from param_decomp.run_param_decomp import run_experiment
from param_decomp.utils.data_utils import DatasetGeneratedDataLoader, SparseFeatureDataset
from param_decomp.utils.distributed_utils import get_device
from param_decomp.utils.general_utils import set_seed
from param_decomp.utils.run_utils import parse_config, parse_sweep_params


def main(
    config_path: Path | str | None = None,
    config_json: str | None = None,
    evals_id: str | None = None,
    launch_id: str | None = None,
    sweep_params_json: str | None = None,
    run_id: str | None = None,
) -> None:
    config = parse_config(config_path, config_json)

    device = get_device()
    logger.info(f"Using device: {device}")

    set_seed(config.seed)

    task_config = config.task_config
    assert isinstance(task_config, TMSTaskConfig)

    assert config.pretrained_model_path, "pretrained_model_path must be set"
    target_run_info = TMSTargetRunInfo.from_path(config.pretrained_model_path)
    target_model = TMSModel.from_run_info(target_run_info)
    target_model = target_model.to(device)
    target_model.eval()

    synced_inputs = target_run_info.config.synced_inputs
    dataset = SparseFeatureDataset(
        n_features=target_model.config.n_features,
        feature_probability=task_config.feature_probability,
        device=device,
        data_generation_type=task_config.data_generation_type,
        value_range=(0.0, 1.0),
        synced_inputs=synced_inputs,
    )
    train_loader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    eval_loader = DatasetGeneratedDataLoader(
        dataset, batch_size=config.eval_batch_size, shuffle=False
    )

    tied_weights = None
    if target_model.config.tied_weights:
        tied_weights = [("linear1", "linear2")]

    run_experiment(
        target_model=target_model,
        config=config,
        device=device,
        train_loader=train_loader,
        eval_loader=eval_loader,
        run_batch=run_batch_first_element,
        reconstruction_loss=recon_loss_mse,
        experiment_tag="tms",
        run_id=run_id,
        launch_id=launch_id,
        evals_id=evals_id,
        sweep_params=parse_sweep_params(sweep_params_json),
        target_model_train_config=target_model.config,
        tied_weights=tied_weights,
    )


if __name__ == "__main__":
    fire.Fire(main)
