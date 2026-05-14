from abc import ABC, abstractmethod

import torch.nn as nn

from param_decomp.types import ModelPath


class LoadableModule(nn.Module, ABC):
    """Base class for nn.Modules that can be loaded from a local path or wandb run id."""

    @classmethod
    @abstractmethod
    def from_pretrained(cls, _path: ModelPath) -> "LoadableModule":
        """Load a pretrained model from a local path or wandb run id."""
        raise NotImplementedError("Subclasses must implement from_pretrained method.")
