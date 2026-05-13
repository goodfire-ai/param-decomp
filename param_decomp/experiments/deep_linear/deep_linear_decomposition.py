"""Deep linear decomposition script."""

from pathlib import Path

import fire

from param_decomp.configs import DeepLinearTaskConfig
from param_decomp.experiments.deep_linear.dataset import DeepLinearDataset
from param_decomp.experiments.deep_linear.models import DeepLinearModel
from param_decomp.log import logger
from param_decomp.models.batch_and_loss_fns import recon_loss_kl_probs, run_batch_first_element
from param_decomp.run_param_decomp import run_experiment
from param_decomp.utils.data_utils import DatasetGeneratedDataLoader
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

    set_seed(config.seed)

    device = get_device()
    logger.info(f"Using device: {device}")

    task_config = config.task_config
    assert isinstance(task_config, DeepLinearTaskConfig)

    target_model = DeepLinearModel(D=task_config.D, L=task_config.L, beta=task_config.beta)
    target_model = target_model.to(device)
    target_model.eval()

    dataset = DeepLinearDataset(D=task_config.D, k=task_config.k, device=device)
    train_loader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    eval_loader = DatasetGeneratedDataLoader(
        dataset, batch_size=config.eval_batch_size, shuffle=False
    )

    run_experiment(
        target_model=target_model,
        config=config,
        device=device,
        train_loader=train_loader,
        eval_loader=eval_loader,
        run_batch=run_batch_first_element,
        reconstruction_loss=recon_loss_kl_probs,
        experiment_tag="deep_linear",
        run_id=run_id,
        launch_id=launch_id,
        evals_id=evals_id,
        sweep_params=parse_sweep_params(sweep_params_json),
    )


if __name__ == "__main__":
    fire.Fire(main)
