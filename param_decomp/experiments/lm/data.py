"""Build LM dataloaders from `LMDataConfig`."""

from typing import Any

from torch.utils.data import DataLoader

from param_decomp.data import DatasetConfig, create_data_loader, input_ids_collate_fn
from param_decomp.experiments.lm.configs import LMDataConfig
from param_decomp.utils.distributed_utils import DistributedState


def _dataset_config(data_cfg: LMDataConfig, *, split: str, seed: int) -> DatasetConfig:
    return DatasetConfig(
        name=data_cfg.dataset_name,
        hf_tokenizer_path=data_cfg.tokenizer_name,
        split=split,
        n_ctx=data_cfg.max_seq_len,
        is_tokenized=data_cfg.is_tokenized,
        streaming=data_cfg.streaming,
        column_name=data_cfg.column_name,
        shuffle_each_epoch=data_cfg.shuffle_each_epoch,
        seed=seed,
    )


def build_lm_dataloaders(
    data_cfg: LMDataConfig,
    *,
    seed: int,
    train_batch_size: int,
    eval_batch_size: int,
    dist_state: DistributedState | None,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """Build (train, eval) dataloaders for an LM experiment.

    Per-rank batch sizes are derived from world size when `dist_state` is set, so callers
    pass total batch sizes (matching `pd.batch_size`/`pd.eval_batch_size`).
    """
    if dist_state is not None:
        world_size = dist_state.world_size
        assert train_batch_size % world_size == 0 and train_batch_size > 0, (
            f"train_batch_size {train_batch_size} not divisible by world size {world_size}"
        )
        assert eval_batch_size % world_size == 0 and eval_batch_size > 0, (
            f"eval_batch_size {eval_batch_size} not divisible by world size {world_size}"
        )
        train_rank_bs = train_batch_size // world_size
        eval_rank_bs = eval_batch_size // world_size
    else:
        train_rank_bs = train_batch_size
        eval_rank_bs = eval_batch_size

    train_loader, _ = create_data_loader(
        dataset_config=_dataset_config(data_cfg, split=data_cfg.train_split, seed=seed),
        batch_size=train_rank_bs,
        buffer_size=data_cfg.buffer_size,
        global_seed=seed,
        dist_state=dist_state,
        collate_fn=input_ids_collate_fn,
    )

    eval_loader, _ = create_data_loader(
        dataset_config=_dataset_config(data_cfg, split=data_cfg.eval_split, seed=seed + 1),
        batch_size=eval_rank_bs,
        buffer_size=data_cfg.buffer_size,
        global_seed=seed + 1,
        dist_state=dist_state,
        collate_fn=input_ids_collate_fn,
    )

    return train_loader, eval_loader
