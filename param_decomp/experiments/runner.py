"""Generic experiment runner used by built-in and custom drivers."""

import json
from pathlib import Path
from typing import Any

import fire
import yaml

from param_decomp import run_pd
from param_decomp.configs import RepeatAcrossBatchScope
from param_decomp.experiments.driver import (
    ExperimentConfig,
    ExperimentDriver,
    ExperimentManifest,
    PreparedExperiment,
    load_driver,
)
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


def _load_config_data(config_path: Path | str | None, config_json: str | None) -> dict[str, Any]:
    assert (config_path is None) != (config_json is None), (
        "Exactly one of config_path or config_json must be provided"
    )
    if config_path is not None:
        with open(Path(config_path)) as f:
            return yaml.safe_load(f)
    assert config_json is not None
    return json.loads(config_json.removeprefix("json:"))


def _load_experiment_config(
    driver: ExperimentDriver[Any],
    config_path: Path | str | None,
    config_json: str | None,
) -> ExperimentConfig:
    data = _load_config_data(config_path, config_json)
    if "experiment_config" in data and "kind" in data:
        manifest = ExperimentManifest.model_validate(data)
        return driver.config_model.model_validate(manifest.experiment_config)
    return driver.config_model.model_validate(data)


def _per_rank_batch_size(total_batch_size: int, dist_state: DistributedState | None) -> int:
    if dist_state is None:
        return total_batch_size
    assert total_batch_size % dist_state.world_size == 0, (
        f"batch_size {total_batch_size} not divisible by world size {dist_state.world_size}"
    )
    return total_batch_size // dist_state.world_size


def _validate_prepared_experiment(
    prepared: PreparedExperiment,
    dist_state: DistributedState | None,
) -> None:
    train_rank_bs = _per_rank_batch_size(prepared.pd.batch_size, dist_state)
    for cfg in (
        prepared.pd.loss_metrics.persistent_pgd_recon,
        prepared.pd.loss_metrics.persistent_pgd_recon_subset,
    ):
        if cfg is not None and isinstance(cfg.scope, RepeatAcrossBatchScope):
            n = cfg.scope.n_sources
            assert train_rank_bs % n == 0, (
                f"repeat_across_batch n_sources={n} must divide per-rank batch_size={train_rank_bs}"
            )


def run_experiment(
    driver_path: str,
    *,
    config_path: Path | str | None = None,
    config_json: str | None = None,
    evals_id: str | None = None,
    launch_id: str | None = None,
    sweep_params_json: str | None = None,
    run_id: str | None = None,
) -> None:
    dist_state = init_distributed()
    logger.info(f"Distributed state: {dist_state}")

    driver = load_driver(driver_path)
    experiment_config = _load_experiment_config(driver, config_path, config_json)
    set_seed(experiment_config.pd.seed)
    device = get_device()

    if is_main_process():
        logger.info(f"Preparing experiment: {driver.display_name(experiment_config)}")
        logger.info(f"Using device: {device}")

    prepared = driver.prepare(experiment_config, device=device, dist_state=dist_state)
    _validate_prepared_experiment(prepared, dist_state)

    extra_tags = [t for t in [evals_id, launch_id] if t is not None]
    manifest = ExperimentManifest(
        kind=experiment_config.kind,
        driver=driver_path,
        experiment_config=experiment_config.model_dump(mode="json"),
    )
    run_pd(
        config=prepared.pd,
        target=prepared.target,
        train_loader=prepared.train_loader,
        eval_loader=prepared.eval_loader,
        device=device,
        run_id=run_id,
        sweep_params=parse_sweep_params(sweep_params_json),
        manifest=manifest,
        artifacts=prepared.artifacts,
        experiment_tag=prepared.tags[0] if prepared.tags else prepared.target.name,
        wandb_tags=[*prepared.tags[1:], *extra_tags],
    )


@with_distributed_cleanup
def main(
    config_path: Path | str | None = None,
    config_json: str | None = None,
    driver: str | None = None,
    evals_id: str | None = None,
    launch_id: str | None = None,
    sweep_params_json: str | None = None,
    run_id: str | None = None,
) -> None:
    data_driver = driver
    if data_driver is None:
        data = _load_config_data(config_path, config_json)
        manifest = ExperimentManifest.model_validate(data)
        assert manifest.driver is not None, "Experiment manifest has no driver; pass --driver"
        data_driver = manifest.driver
    run_experiment(
        data_driver,
        config_path=config_path,
        config_json=config_json,
        evals_id=evals_id,
        launch_id=launch_id,
        sweep_params_json=sweep_params_json,
        run_id=run_id,
    )


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
