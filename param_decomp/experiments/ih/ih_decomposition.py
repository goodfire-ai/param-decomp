"""Induction head decomposition script."""

from pathlib import Path

import fire

from param_decomp.configs import IHTaskConfig
from param_decomp.experiments.ih.model import InductionModelTargetRunInfo, InductionTransformer
from param_decomp.log import logger
from param_decomp.models.batch_and_loss_fns import recon_loss_kl, run_batch_first_element
from param_decomp.run_param_decomp import run_experiment
from param_decomp.utils.data_utils import DatasetGeneratedDataLoader, InductionDataset
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
    assert isinstance(task_config, IHTaskConfig)

    assert config.pretrained_model_path, "pretrained_model_path must be set"
    target_run_info = InductionModelTargetRunInfo.from_path(config.pretrained_model_path)
    target_model = InductionTransformer.from_run_info(target_run_info)
    target_model = target_model.to(device)
    target_model.eval()

    prefix_window = task_config.prefix_window or target_model.config.seq_len - 3

    dataset = InductionDataset(
        vocab_size=target_model.config.vocab_size,
        seq_len=target_model.config.seq_len,
        prefix_window=prefix_window,
        device=device,
    )
    train_loader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    eval_loader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    run_experiment(
        target_model=target_model,
        config=config,
        device=device,
        train_loader=train_loader,
        eval_loader=eval_loader,
        experiment_tag="ih",
        run_id=run_id,
        launch_id=launch_id,
        evals_id=evals_id,
        sweep_params=parse_sweep_params(sweep_params_json),
        target_model_train_config=target_run_info.config,
        run_batch=run_batch_first_element,
        reconstruction_loss=recon_loss_kl,
    )


if __name__ == "__main__":
    fire.Fire(main)
