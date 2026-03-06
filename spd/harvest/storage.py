"""Raw storage classes for harvest data.

These are simple data containers with save/load methods.
For query functionality, see harvest/analysis.py.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from jaxtyping import Float, Int
from torch import Tensor

from spd.log import logger


@dataclass
class CorrelationStorage:
    """Raw correlation data between components."""

    component_keys: list[str]
    count_i: Int[Tensor, " n_components"]
    """Firing count per component"""
    count_ij: Int[Tensor, "n_components n_components"]
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
        count_ij = self.count_ij[i][j].item()
        if count_ij == 0:
            return None
        count_i = self.count_i[i].item()
        count_j = self.count_i[j].item()
        if count_i == 0 or count_j == 0:
            return None
        return math.log(count_ij * self.count_total / (count_i * count_j))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "component_keys": self.component_keys,
                "count_i": self.count_i.cpu(),
                "count_ij": self.count_ij.cpu(),
                "count_total": self.count_total,
            },
            path,
        )
        logger.info(f"Saved component correlations to {path}")

    @classmethod
    def load(cls, path: Path) -> "CorrelationStorage":
        data = torch.load(path, weights_only=True, mmap=True)
        return cls(
            component_keys=data["component_keys"],
            count_i=data["count_i"],
            count_ij=data["count_ij"],
            count_total=data["count_total"],
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

    input_counts: Float[Tensor, "n_components vocab"]
    input_totals: Float[Tensor, " vocab"]
    output_counts: Float[Tensor, "n_components vocab"]
    """Probability mass, not hard counts - but used the same way in analysis."""
    output_totals: Float[Tensor, " vocab"]
    firing_counts: Float[Tensor, " n_components"]

    _key_to_idx: dict[str, int] | None = None

    @property
    def key_to_idx(self) -> dict[str, int]:
        """Cached mapping from component key to index."""
        if self._key_to_idx is None:
            self._key_to_idx = {k: i for i, k in enumerate(self.component_keys)}
        return self._key_to_idx

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "component_keys": self.component_keys,
                "vocab_size": self.vocab_size,
                "n_tokens": self.n_tokens,
                "input_counts": self.input_counts.cpu(),
                "input_totals": self.input_totals.cpu(),
                "output_counts": self.output_counts.cpu(),
                "output_totals": self.output_totals.cpu(),
                "firing_counts": self.firing_counts.cpu(),
            },
            path,
        )
        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info(f"Saved token stats to {path} ({size_mb:.1f} MB)")

    @classmethod
    def load(cls, path: Path) -> "TokenStatsStorage":
        data = torch.load(path, weights_only=True, mmap=True)
        return cls(
            component_keys=data["component_keys"],
            vocab_size=data["vocab_size"],
            n_tokens=data["n_tokens"],
            input_counts=data["input_counts"],
            input_totals=data["input_totals"],
            output_counts=data["output_counts"],
            output_totals=data["output_totals"],
            firing_counts=data["firing_counts"],
        )
