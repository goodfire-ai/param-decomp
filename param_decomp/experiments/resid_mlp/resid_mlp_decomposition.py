"""Residual MLP decomposition script."""

from pathlib import Path

import fire

from param_decomp.configs import ResidMLPTaskConfig
from param_decomp.experiments.resid_mlp.models import (
    ResidMLP,
    ResidMLPTargetRunInfo,
)
from param_decomp.experiments.resid_mlp.resid_mlp_dataset import ResidMLPDataset
from param_decomp.log import logger
from param_decomp.models.batch_and_loss_fns import recon_loss_mse, run_batch_first_element
from param_decomp.run_param_decomp import run_experiment
from param_decomp.settings import PARAM_DECOMP_OUT_DIR
from param_decomp.utils.data_utils import DatasetGeneratedDataLoader
from param_decomp.utils.distributed_utils import get_device
from param_decomp.utils.general_utils import set_seed
from param_decomp.utils.run_utils import (
    generate_run_id,
    parse_config,
    parse_sweep_params,
    save_file,
)


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
    assert isinstance(config.task_config, ResidMLPTaskConfig)

    assert config.pretrained_model_path, "pretrained_model_path must be set"
    target_run_info = ResidMLPTargetRunInfo.from_path(config.pretrained_model_path)
    target_model = ResidMLP.from_run_info(target_run_info)
    target_model = target_model.to(device)
    target_model.eval()

    # Domain-specific: save label coefficients to out_dir
    run_id = run_id or generate_run_id("param_decomp")
    out_dir = PARAM_DECOMP_OUT_DIR / "decompositions" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(target_run_info.label_coeffs.detach().cpu().tolist(), out_dir / "label_coeffs.json")

    synced_inputs = target_run_info.config.synced_inputs
    dataset = ResidMLPDataset(
        n_features=target_model.config.n_features,
        feature_probability=config.task_config.feature_probability,
        device=device,
        calc_labels=False,  # Our labels will be the output of the target model
        label_type=None,
        act_fn_name=None,
        label_fn_seed=None,
        label_coeffs=None,
        data_generation_type=config.task_config.data_generation_type,
        synced_inputs=synced_inputs,
    )

    train_loader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    eval_loader = DatasetGeneratedDataLoader(
        dataset, batch_size=config.eval_batch_size, shuffle=False
    )

    assert config.n_eval_steps is not None, "n_eval_steps must be set"
    run_experiment(
        target_model=target_model,
        config=config,
        device=device,
        train_loader=train_loader,
        eval_loader=eval_loader,
        run_batch=run_batch_first_element,
        reconstruction_loss=recon_loss_mse,
        experiment_tag="resid_mlp",
        run_id=run_id,
        launch_id=launch_id,
        evals_id=evals_id,
        sweep_params=parse_sweep_params(sweep_params_json),
        target_model_train_config=target_run_info.config,
    )


if __name__ == "__main__":
    fire.Fire(main)
