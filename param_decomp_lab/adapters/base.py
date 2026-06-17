from abc import ABC, abstractmethod

from param_decomp_lab.autointerp.schemas import ModelMetadata


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
