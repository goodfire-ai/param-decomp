from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader

from spd.autointerp.schemas import ModelMetadata
from spd.pretrain.run_info import PretrainRunInfo


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
    def dataloader(self, batch_size: int) -> DataLoader[torch.Tensor]: ...


def pretrain_dataloader(run_info: PretrainRunInfo, batch_size: int) -> DataLoader[torch.Tensor]:
    """Build a streaming dataloader from a pretrain run's dataset config."""
    from spd.data import DatasetConfig, create_data_loader

    ds_cfg = run_info.config_dict["train_dataset_config"]
    block_size = run_info.model_config_dict["block_size"]
    dataset_config = DatasetConfig.model_validate(
        {**ds_cfg, "streaming": True, "n_ctx": block_size}
    )
    loader, _ = create_data_loader(
        dataset_config=dataset_config, batch_size=batch_size, buffer_size=1000
    )
    return loader
