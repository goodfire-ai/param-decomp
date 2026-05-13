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
    from param_decomp.experiments.lm.data import LMDataConfig, create_lm_data_loader

    data_cfg = LMDataConfig.model_validate({**run_info.config_dict["data"], "streaming": True})

    def collate_input_ids(batch: list[dict[str, Tensor]]) -> Tensor:
        return torch.stack([item["input_ids"] for item in batch])

    loader, _ = create_lm_data_loader(
        data_cfg,
        split=data_cfg.train_split,
        batch_size=batch_size,
        seed=data_cfg.dataset_shuffle_seed,
        collate_fn=collate_input_ids,
    )
    return loader
