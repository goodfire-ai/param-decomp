"""Residual MLP decomposition entrypoint."""

from pathlib import Path

import fire

from param_decomp import run_pd
from param_decomp.experiments.resid_mlp.experiment import ResidMLPExperimentConfig
from param_decomp.experiments.resid_mlp.models import ResidMLPTargetRunInfo
from param_decomp.log import logger
from param_decomp.settings import PARAM_DECOMP_OUT_DIR
from param_decomp.utils.distributed_utils import get_device
from param_decomp.utils.general_utils import set_seed
from param_decomp.utils.run_utils import generate_run_id, parse_sweep_params, save_file


def _parse_resid_mlp_config(
    config_path: Path | str | None, config_json: str | None
) -> ResidMLPExperimentConfig:
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
    return ResidMLPExperimentConfig.model_validate(data)


def main(
    config_path: Path | str | None = None,
    config_json: str | None = None,
    evals_id: str | None = None,
    launch_id: str | None = None,
    sweep_params_json: str | None = None,
    run_id: str | None = None,
) -> None:
    exp = _parse_resid_mlp_config(config_path, config_json)

    set_seed(exp.pd.seed)

    device = get_device()
    logger.info(f"Using device: {device}")

    loaded = exp.load_target()
    loaded.target.model.to(device)

    # Pre-create the run dir so we can save the domain-specific label_coeffs alongside the run.
    run_id = run_id or generate_run_id("param_decomp")
    out_dir = PARAM_DECOMP_OUT_DIR / "decompositions" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    target_run_info = ResidMLPTargetRunInfo.from_path(exp.target.run_path)
    save_file(target_run_info.label_coeffs.detach().cpu().tolist(), out_dir / "label_coeffs.json")

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
        experiment_tag="resid_mlp",
        wandb_tags=wandb_tags,
        target_train_config=loaded.target_train_config,
    )


if __name__ == "__main__":
    fire.Fire(main)
