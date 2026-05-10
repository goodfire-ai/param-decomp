"""Language Model decomposition entrypoint."""

from pathlib import Path

import fire

from param_decomp import run_pd
from param_decomp.configs import (
    PersistentPGDReconLossConfig,
    PersistentPGDReconSubsetLossConfig,
    RepeatAcrossBatchScope,
)
from param_decomp.experiments.lm.experiment import LMExperimentConfig
from param_decomp.log import logger
from param_decomp.utils.distributed_utils import (
    DistributedState,
    get_device,
    init_distributed,
    is_main_process,
    with_distributed_cleanup,
)
from param_decomp.utils.general_utils import set_seed
from param_decomp.utils.run_utils import parse_sweep_params


def _parse_lm_config(config_path: Path | str | None, config_json: str | None) -> LMExperimentConfig:
    import json

    import yaml

    assert (config_path is None) != (config_json is None), (
        "Exactly one of config_path or config_json must be provided"
    )
    if config_path is not None:
        path = Path(config_path)
        with open(path) as f:
            data = yaml.safe_load(f)
    else:
        assert config_json is not None
        data = json.loads(config_json.removeprefix("json:"))
    return LMExperimentConfig.model_validate(data)


@with_distributed_cleanup
def main(
    config_path: Path | str | None = None,
    config_json: str | None = None,
    evals_id: str | None = None,
    launch_id: str | None = None,
    sweep_params_json: str | None = None,
    run_id: str | None = None,
) -> None:
    exp = _parse_lm_config(config_path, config_json)

    dist_state = init_distributed()
    logger.info(f"Distributed state: {dist_state}")

    set_seed(exp.pd.seed)
    device = get_device()

    if is_main_process():
        logger.info("Loading target model and dataset...")

    loaded = exp.load_target()

    # Validate PersistentPGD scope compatibility with per-rank batch size.
    match dist_state:
        case DistributedState(world_size=world_size):
            train_rank_bs = exp.pd.batch_size // world_size
        case None:
            train_rank_bs = exp.pd.batch_size

    for cfg in exp.pd.loss_metric_configs:
        if isinstance(
            cfg, PersistentPGDReconLossConfig | PersistentPGDReconSubsetLossConfig
        ) and isinstance(cfg.scope, RepeatAcrossBatchScope):
            n = cfg.scope.n_sources
            assert train_rank_bs % n == 0, (
                f"repeat_across_batch n_sources={n} must divide per-rank batch_size={train_rank_bs}"
            )

    train_loader, eval_loader = exp.build_dataloaders(
        seed=exp.pd.seed,
        train_batch_size=exp.pd.batch_size,
        eval_batch_size=exp.pd.eval_batch_size,
        dist_state=dist_state,
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
        experiment_tag="lm",
        wandb_tags=wandb_tags,
        target_train_config=loaded.target_train_config,
    )


if __name__ == "__main__":
    fire.Fire(main)
