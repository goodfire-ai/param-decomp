"""Constants and shared abstractions for clustering pipeline."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, NewType

import numpy as np
from jaxtyping import Bool, Float, Int
from torch import Tensor

# Merge arrays and distances (numpy-based for storage/analysis)
MergesAtIterArray = Int[np.ndarray, "n_ens n_components"]
MergesArray = Int[np.ndarray, "n_ens n_iters n_components"]
DistancesMethod = Literal["perm_invariant_hamming", "matching_dist", "matching_dist_vec"]
DistancesArray = Float[np.ndarray, "n_iters n_ens n_ens"]

# Component and label types (NewType for stronger type safety)
ComponentLabel = NewType("ComponentLabel", str)  # Format: "module_name:component_index"
ComponentLabels = NewType("ComponentLabels", list[str])
BatchId = NewType("BatchId", str)

# Path types
WandBPath = NewType("WandBPath", str)  # Format: "wandb:entity/project/run_id"

# Merge types
MergePair = NewType("MergePair", tuple[int, int])

# Tensor type aliases (torch-based for computation - TypeAlias for jaxtyping compatibility)
ActivationsTensor = Float[Tensor, "samples n_components"]
BoolActivationsTensor = Bool[Tensor, "samples n_components"]
ClusterCoactivationShaped = Float[Tensor, "k_groups k_groups"]
GroupIdxsTensor = Int[Tensor, " n_components"]
BatchTensor = Int[Tensor, "batch_size seq_len"]


class SaveableObject(ABC):
    """Abstract base class for objects that can be saved to and loaded from disk."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save the object to disk at the given path."""
        ...

    @classmethod
    @abstractmethod
    def read(cls, path: Path) -> "SaveableObject":
        """Load the object from disk at the given path."""
        ...
