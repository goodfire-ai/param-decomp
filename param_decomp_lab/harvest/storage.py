"""Raw storage classes for harvest data.

These are simple data containers with save/load methods backed by `.npz` files.
For query functionality, see harvest/analysis.py.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float, Int

from param_decomp.log import logger


@dataclass
class CorrelationStorage:
    """Raw correlation data between components."""

    component_keys: list[str]
    count_i: Int[np.ndarray, " n_components"]
    """Firing count per component"""
    count_ij: Int[np.ndarray, "n_components n_components"]
    """Co-occurrence matrix: count_ij[i, j] = count of tokens where both fired"""
    count_total: int
    """Total tokens seen"""

    _key_to_idx: dict[str, int] | None = None

    @property
    def key_to_idx(self) -> dict[str, int]:
        """Cached mapping from component key to index."""
        if self._key_to_idx is None:
            self._key_to_idx = {k: i for i, k in enumerate(self.component_keys)}
        return self._key_to_idx

    def pmi(self, key_a: str, key_b: str) -> float | None:
        """Point-wise mutual information between two components.

        Returns None if either component is missing or they never co-fire.
        """
        if key_a not in self.key_to_idx or key_b not in self.key_to_idx:
            return None
        i, j = self.key_to_idx[key_a], self.key_to_idx[key_b]
        count_ij = int(self.count_ij[i][j])
        if count_ij == 0:
            return None
        count_i = int(self.count_i[i])
        count_j = int(self.count_i[j])
        if count_i == 0 or count_j == 0:
            return None
        return math.log(count_ij * self.count_total / (count_i * count_j))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            component_keys=np.array(self.component_keys, dtype=object),
            count_i=self.count_i,
            count_ij=self.count_ij,
            count_total=np.int64(self.count_total),
        )
        logger.info(f"Saved component correlations to {path}")

    @classmethod
    def load(cls, path: Path) -> "CorrelationStorage":
        data = np.load(path, allow_pickle=True)
        return cls(
            component_keys=list(data["component_keys"]),
            count_i=data["count_i"],
            count_ij=data["count_ij"],
            count_total=int(data["count_total"]),
        )


@dataclass
class TokenStatsStorage:
    """Raw token statistics for all components.

    Input stats are hard counts (token appeared when component fired).
    Output stats are probability mass (sum of probs assigned to token when component fired).
    Both are used identically in analysis (precision/recall/PMI computations).
    """

    component_keys: list[str]
    vocab_size: int
    n_tokens: int

    input_counts: Float[np.ndarray, "n_components vocab"]
    input_totals: Float[np.ndarray, " vocab"]
    output_counts: Float[np.ndarray, "n_components vocab"]
    """Probability mass, not hard counts - but used the same way in analysis."""
    output_totals: Float[np.ndarray, " vocab"]
    firing_counts: Float[np.ndarray, " n_components"]

    _key_to_idx: dict[str, int] | None = None

    @property
    def key_to_idx(self) -> dict[str, int]:
        """Cached mapping from component key to index."""
        if self._key_to_idx is None:
            self._key_to_idx = {k: i for i, k in enumerate(self.component_keys)}
        return self._key_to_idx

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            component_keys=np.array(self.component_keys, dtype=object),
            vocab_size=np.int64(self.vocab_size),
            n_tokens=np.int64(self.n_tokens),
            input_counts=self.input_counts,
            input_totals=self.input_totals,
            output_counts=self.output_counts,
            output_totals=self.output_totals,
            firing_counts=self.firing_counts,
        )
        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info(f"Saved token stats to {path} ({size_mb:.1f} MB)")

    @classmethod
    def load(cls, path: Path) -> "TokenStatsStorage":
        data = np.load(path, allow_pickle=True)
        return cls(
            component_keys=list(data["component_keys"]),
            vocab_size=int(data["vocab_size"]),
            n_tokens=int(data["n_tokens"]),
            input_counts=data["input_counts"],
            input_totals=data["input_totals"],
            output_counts=data["output_counts"],
            output_totals=data["output_totals"],
            firing_counts=data["firing_counts"],
        )
