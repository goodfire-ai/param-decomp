"""OthelloGPT decomposition script.

Thin wrapper around the LM decomposition flow that handles the column rename
(taufeeque/othellogpt uses 'tokens', SPD expects 'input_ids').
"""

from pathlib import Path

import fire
import torch
from datasets import IterableDataset, load_dataset
from jaxtyping import Int
from torch import Tensor
from torch.utils.data import DataLoader

from spd.configs import LMTaskConfig
from spd.log import logger
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


def _make_loader(
    task_config: LMTaskConfig,
    split: str,
    batch_size: int,
    seed: int,
    dist_state: DistributedState | None,
) -> DataLoader[Int[Tensor, "..."]]:
    dataset = load_dataset(
        task_config.dataset_name,
        streaming=True,
        split=split,
        trust_remote_code=False,
    )
    assert isinstance(dataset, IterableDataset)

    if dist_state is not None:
        ds_num_shards = getattr(dataset, "num_shards", None)
        if isinstance(ds_num_shards, int) and ds_num_shards >= dist_state.world_size:
            dataset = dataset.shard(num_shards=dist_state.world_size, index=dist_state.rank)
        else:
            dataset = dataset.filter(
                lambda _ex, idx: idx % dist_state.world_size == dist_state.rank,
                with_indices=True,
            )

    n_ctx = task_config.max_seq_len
    col = task_config.column_name

    dataset = dataset.shuffle(seed=seed, buffer_size=task_config.buffer_size)
    dataset = dataset.map(lambda x: {"input_ids": x[col][:n_ctx]})
    dataset = dataset.remove_columns([col])
    dataset = dataset.with_format("torch")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    return DataLoader[Int[Tensor, "..."]](
        dataset,  # pyright: ignore[reportArgumentType]
        batch_size=batch_size,
        drop_last=True,
        generator=generator,
    )


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
    set_seed(config.seed)
    device = get_device()
    assert isinstance(config.task_config, LMTaskConfig)
    assert config.pretrained_model_name is not None

    pretrained_model_class = resolve_class(config.pretrained_model_class)
    target_model = ensure_cached_and_call(
        pretrained_model_class.from_pretrained,  # pyright: ignore[reportAttributeAccessIssue]
        config.pretrained_model_name,
    )
    target_model.eval()

    match dist_state:
        case DistributedState(world_size=world_size):
            assert config.batch_size % world_size == 0
            train_bs = config.batch_size // world_size
            assert config.eval_batch_size % world_size == 0
            eval_bs = config.eval_batch_size // world_size
        case None:
            train_bs = config.batch_size
            eval_bs = config.eval_batch_size

    if is_main_process():
        logger.info("Loading dataset...")

    train_loader = _make_loader(
        config.task_config,
        config.task_config.train_data_split,
        train_bs,
        config.seed,
        dist_state,
    )
    eval_loader = _make_loader(
        config.task_config,
        config.task_config.eval_data_split,
        eval_bs,
        config.seed + 1,
        dist_state,
    )

    run_experiment(
        target_model=target_model,
        config=config,
        device=device,
        train_loader=train_loader,
        eval_loader=eval_loader,
        experiment_tag="othello",
        run_id=run_id,
        launch_id=launch_id,
        evals_id=evals_id,
        sweep_params=parse_sweep_params(sweep_params_json),
    )


if __name__ == "__main__":
    fire.Fire(main)
