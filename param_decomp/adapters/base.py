from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from param_decomp.autointerp.schemas import ModelMetadata
from param_decomp.pretrain.run_info import PretrainRunInfo


class DecompositionAdapter(ABC):
    @property
    @abstractmethod
    def decomposition_id(self) -> str: ...

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @property
    @abstractmethod
    def layer_activation_sizes(self) -> list[tuple[str, int]]: ...

    @property
    @abstractmethod
    def tokenizer_name(self) -> str: ...

    @property
    @abstractmethod
    def model_metadata(self) -> ModelMetadata: ...

    @abstractmethod
    def dataloader(self, batch_size: int) -> DataLoader[Any]: ...


def pretrain_dataloader(run_info: PretrainRunInfo, batch_size: int) -> DataLoader[Tensor]:
    """Build a streaming LM dataloader from a pretrain run's dataset config.

    Currently assumes the pretrain dataset is a HuggingFace tokenized dataset yielding
    ``{"input_ids": Tensor}`` items (as produced by
    `param_decomp.experiments.lm.data.create_lm_data_loader` for LM pretraining)
    and collates them into stacked token tensors. For non-LM
    pretrain runs, build the dataloader directly with `create_lm_data_loader` and an
    appropriate collate_fn.
    """
    from param_decomp.experiments.lm.data import (
        LMDataLoaderConfig,
        create_lm_data_loader,
    )

    ds_cfg = run_info.config_dict["train_dataset_config"]
    block_size = run_info.model_config_dict["block_size"]
    dataset_config = LMDataLoaderConfig.model_validate(
        {**ds_cfg, "streaming": True, "n_ctx": block_size}
    )
    seed = dataset_config.seed if dataset_config.seed is not None else 0

    def collate_input_ids(batch: list[dict[str, Tensor]]) -> Tensor:
        return torch.stack([item["input_ids"] for item in batch])

    loader, _ = create_lm_data_loader(
        dataset_name=dataset_config.name,
        tokenizer_name=dataset_config.hf_tokenizer_path,
        split=dataset_config.split,
        max_seq_len=dataset_config.n_ctx,
        is_tokenized=dataset_config.is_tokenized,
        streaming=dataset_config.streaming,
        column_name=dataset_config.column_name,
        batch_size=batch_size,
        buffer_size=1000,
        seed=seed,
        shuffle_each_epoch=dataset_config.shuffle_each_epoch,
        collate_fn=collate_input_ids,
    )
    return loader
