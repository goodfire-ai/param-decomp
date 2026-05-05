from abc import ABC, abstractmethod
from typing import Any

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
    ``{"input_ids": Tensor}`` items (as produced by `param_decomp.data.create_data_loader`
    for LM pretraining) and collates them into stacked token tensors. For non-LM
    pretrain runs, build the dataloader directly with `create_data_loader` and an
    appropriate collate_fn.
    """
    from param_decomp.data import DatasetConfig, create_data_loader, input_ids_collate_fn

    ds_cfg = run_info.config_dict["train_dataset_config"]
    block_size = run_info.model_config_dict["block_size"]
    dataset_config = DatasetConfig.model_validate(
        {**ds_cfg, "streaming": True, "n_ctx": block_size}
    )
    loader, _ = create_data_loader(
        dataset_config=dataset_config,
        batch_size=batch_size,
        buffer_size=1000,
        collate_fn=input_ids_collate_fn,
    )
    return loader
