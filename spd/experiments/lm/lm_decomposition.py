"""Language Model decomposition script."""

from pathlib import Path

import fire

from spd.configs import (
    LMTaskConfig,
    PersistentPGDReconLossConfig,
    PersistentPGDReconSubsetLossConfig,
    RepeatAcrossBatchScope,
)
from spd.data import DatasetConfig, create_data_loader
from spd.log import logger
from spd.pretrain.run_info import PretrainRunInfo
from spd.run_spd import run_experiment
from spd.utils.distributed_utils import (
    DistributedState,
    ensure_cached_and_call,
    get_device,
    init_distributed,
    is_main_process,
    with_distributed_cleanup,
)
from spd.utils.general_utils import resolve_class, set_seed
from spd.utils.run_utils import parse_config, parse_sweep_params


@with_distributed_cleanup
def main(
    config_path: Path | str | None = None,
    config_json: str | None = None,
    evals_id: str | None = None,
    launch_id: str | None = None,
    sweep_params_json: str | None = None,
    run_id: str | None = None,
) -> None:
    config = parse_config(config_path, config_json)

    dist_state = init_distributed()
    logger.info(f"Distributed state: {dist_state}")

    # Use the same seed across all ranks for deterministic data loading
    set_seed(config.seed)

    device = get_device()
    assert isinstance(config.task_config, LMTaskConfig), "task_config not LMTaskConfig"

    pretrained_model_class = resolve_class(config.pretrained_model_class)
    assert hasattr(pretrained_model_class, "from_pretrained"), (
        f"Model class {pretrained_model_class} should have a `from_pretrained` method"
    )
    assert config.pretrained_model_name is not None

    if config.pretrained_model_class.startswith("spd.pretrain"):
        # Ensure local_rank 0 on each node caches the model, then all ranks load from local cache
        # (In multi-node setups, /tmp is node-local so we can't broadcast paths across nodes)
        run_info = ensure_cached_and_call(PretrainRunInfo.from_path, config.pretrained_model_name)

        # Handle old training runs not having a model_type in the model_config_dict
        if "model_type" not in run_info.model_config_dict:
            run_info.model_config_dict["model_type"] = config.pretrained_model_class.split(".")[-1]

        assert hasattr(pretrained_model_class, "from_run_info")
        # Just loads from local file
        target_model = pretrained_model_class.from_run_info(run_info)  # pyright: ignore[reportAttributeAccessIssue]
    else:
        # Avoid concurrent wandb API requests by first calling from_pretrained on rank 0 only
        target_model = ensure_cached_and_call(
            pretrained_model_class.from_pretrained,  # pyright: ignore[reportAttributeAccessIssue]
            config.pretrained_model_name,
        )
    target_model.eval()

    # --- Load Data --- #
    if is_main_process():
        logger.info("Loading dataset...")
    train_data_config = DatasetConfig(
        name=config.task_config.dataset_name,
        hf_tokenizer_path=config.tokenizer_name,
        split=config.task_config.train_data_split,
        n_ctx=config.task_config.max_seq_len,
        is_tokenized=config.task_config.is_tokenized,
        streaming=config.task_config.streaming,
        column_name=config.task_config.column_name,
        shuffle_each_epoch=config.task_config.shuffle_each_epoch,
        seed=config.task_config.dataset_seed,
    )

    match dist_state:
        case DistributedState(world_size=world_size):
            # Keep per-process batch size constant to maintain scale of all metrics so we can simply average
            # them across processes.
            assert config.batch_size % world_size == 0 and config.batch_size > 0, (
                f"Batch size {config.batch_size} is not divisible by world size {world_size}. "
            )
            train_rank_batch_size = config.batch_size // world_size
        case None:
            train_rank_batch_size = config.batch_size

    for cfg in config.loss_metric_configs:
        if isinstance(
            cfg, PersistentPGDReconLossConfig | PersistentPGDReconSubsetLossConfig
        ) and isinstance(cfg.scope, RepeatAcrossBatchScope):
            n = cfg.scope.n_sources
            assert train_rank_batch_size % n == 0, (
                f"repeat_across_batch n_sources={n} must divide per-rank batch_size="
                f"{train_rank_batch_size}"
            )

    train_loader, _tokenizer = create_data_loader(
        dataset_config=train_data_config,
        batch_size=train_rank_batch_size,
        buffer_size=config.task_config.buffer_size,
        global_seed=config.seed,
        dist_state=dist_state,
    )

    eval_data_config = DatasetConfig(
        name=config.task_config.dataset_name,
        hf_tokenizer_path=config.tokenizer_name,
        split=config.task_config.eval_data_split,
        n_ctx=config.task_config.max_seq_len,
        is_tokenized=config.task_config.is_tokenized,
        streaming=config.task_config.streaming,
        column_name=config.task_config.column_name,
        shuffle_each_epoch=config.task_config.shuffle_each_epoch,
        seed=config.task_config.dataset_seed,
    )

    match dist_state:
        case DistributedState(world_size=world_size):
            assert config.eval_batch_size % world_size == 0 and config.eval_batch_size > 0, (
                f"Eval batch size {config.eval_batch_size} is not divisible by world size {world_size}. "
            )
            eval_rank_batch_size = config.eval_batch_size // world_size
        case None:
            eval_rank_batch_size = config.eval_batch_size

    eval_loader, _ = create_data_loader(
        dataset_config=eval_data_config,
        batch_size=eval_rank_batch_size,
        buffer_size=config.task_config.buffer_size,
        global_seed=config.seed + 1,
        dist_state=dist_state,
    )

    run_experiment(
        target_model=target_model,
        config=config,
        device=device,
        train_loader=train_loader,
        eval_loader=eval_loader,
        experiment_tag="lm",
        run_id=run_id,
        launch_id=launch_id,
        evals_id=evals_id,
        sweep_params=parse_sweep_params(sweep_params_json),
    )


if __name__ == "__main__":
    fire.Fire(main)
