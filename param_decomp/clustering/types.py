"""Constants and shared abstractions for clustering pipeline."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, NewType

import numpy as np
from jaxtyping import Bool, Float, Int

MergesArray = Int[np.ndarray, "n_ens n_iters n_components"]
DistancesMethod = Literal["perm_invariant_hamming", "matching_dist", "matching_dist_vec"]
DistancesArray = Float[np.ndarray, "n_iters n_ens n_ens"]

ComponentLabel = NewType("ComponentLabel", str)  # Format: "module_name:component_index"
ComponentLabels = NewType("ComponentLabels", list[str])
BatchId = NewType("BatchId", str)

WandBPath = NewType(
    "WandBPath", str
)  # Format: "entity/project/run_id" or "entity/project/runs/run_id"

MergePair = NewType("MergePair", tuple[int, int])

ActivationsArray = Float[np.ndarray, "samples n_components"]
BoolActivationsArray = Bool[np.ndarray, "samples n_components"]
ClusterCoactivationShaped = Float[np.ndarray, "k_groups k_groups"]
GroupIdxsArray = Int[np.ndarray, " n_components"]


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
