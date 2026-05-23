"""Synthetic sparse-feature dataset used by TMS (and inherited by ResidMLP).

The dataset is infinite: each `__iter__` step yields a freshly generated batch of size
`batch_size`. Wrap in `DataLoader(dataset, batch_size=None)` so the loader passes batches
through unchanged.
"""

from collections.abc import Iterator
from typing import Literal, override

import torch
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import IterableDataset

DataGenerationType = Literal[
    "exactly_one_active",
    "exactly_two_active",
    "exactly_three_active",
    "exactly_four_active",
    "exactly_five_active",
    "at_least_zero_active",
]

_N_ACTIVE_MAP: dict[str, int] = {
    "exactly_one_active": 1,
    "exactly_two_active": 2,
    "exactly_three_active": 3,
    "exactly_four_active": 4,
    "exactly_five_active": 5,
}


class SparseFeatureDataset(
    IterableDataset[
        tuple[
            Float[Tensor, "batch n_features"],
            Float[Tensor, "batch n_features"],
        ]
    ]
):
    """Infinite iterable of sparse-feature batches.

    Each iteration step calls `generate_batch(self.batch_size)`. Iteration never stops;
    the trainer drives termination through its own step counter.
    """

    def __init__(
        self,
        n_features: int,
        feature_probability: float,
        device: str,
        batch_size: int,
        data_generation_type: DataGenerationType = "at_least_zero_active",
        value_range: tuple[float, float] = (0.0, 1.0),
        synced_inputs: list[list[int]] | None = None,
    ):
        self.n_features: int = n_features
        self.feature_probability: float = feature_probability
        self.device: str = device
        self.batch_size: int = batch_size
        self.data_generation_type: DataGenerationType = data_generation_type
        self.value_range: tuple[float, float] = value_range
        self.synced_inputs: list[list[int]] | None = synced_inputs

    @override
    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        while True:
            yield self.generate_batch(self.batch_size)

    def sync_inputs(
        self, batch: Float[Tensor, "batch n_features"]
    ) -> Float[Tensor, "batch n_features"]:
        assert self.synced_inputs is not None
        all_indices = [item for sublist in self.synced_inputs for item in sublist]
        assert len(all_indices) == len(set(all_indices)), "Synced inputs must be non-overlapping"
        for indices in self.synced_inputs:
            mask = torch.zeros_like(batch, dtype=torch.bool)
            non_zero_samples = (batch[..., indices] != 0.0).any(dim=-1)
            for idx in indices:
                mask[..., idx] = non_zero_samples
            max_val, min_val = self.value_range
            random_values = torch.rand(batch.shape[0], self.n_features, device=self.device)
            random_values = random_values * (max_val - min_val) + min_val
            batch = torch.where(mask, random_values, batch)
        return batch

    def generate_batch(
        self, batch_size: int
    ) -> tuple[Float[Tensor, "batch n_features"], Float[Tensor, "batch n_features"]]:
        if self.data_generation_type in _N_ACTIVE_MAP:
            n = _N_ACTIVE_MAP[self.data_generation_type]
            batch = self._generate_n_feature_active_batch(batch_size, n=n)
        elif self.data_generation_type == "at_least_zero_active":
            batch = self._masked_batch_generator(batch_size)
            if self.synced_inputs is not None:
                batch = self.sync_inputs(batch)
        else:
            raise ValueError(f"Invalid generation type: {self.data_generation_type}")

        return batch, batch.clone().detach()

    def _generate_n_feature_active_batch(
        self, batch_size: int, n: int
    ) -> Float[Tensor, "batch n_features"]:
        """Generate a batch with exactly n features active per sample."""
        if n > self.n_features:
            raise ValueError(
                f"Cannot activate {n} features when only {self.n_features} features exist"
            )

        batch = torch.zeros(batch_size, self.n_features, device=self.device)

        feature_indices = torch.arange(self.n_features, device=self.device)
        feature_indices = feature_indices.expand(batch_size, self.n_features)

        perm = torch.rand_like(feature_indices.float()).argsort(dim=-1)
        permuted_features = feature_indices.gather(dim=-1, index=perm)
        active_features = permuted_features[..., :n]

        min_val, max_val = self.value_range
        random_values = torch.rand(batch_size, n, device=self.device)
        random_values = random_values * (max_val - min_val) + min_val

        for i in range(n):
            batch.scatter_(
                dim=1, index=active_features[..., i : i + 1], src=random_values[..., i : i + 1]
            )

        return batch

    def _masked_batch_generator(self, batch_size: int) -> Float[Tensor, "batch_size n_features"]:
        """Generate a batch where each feature activates independently with probability
        `feature_probability`."""
        min_val, max_val = self.value_range
        batch = (
            torch.rand((batch_size, self.n_features), device=self.device) * (max_val - min_val)
            + min_val
        )
        mask = torch.rand_like(batch) < self.feature_probability
        return batch * mask

    def _generate_multi_feature_batch_no_zero_samples(
        self, batch_size: int, buffer_ratio: float
    ) -> Float[Tensor, "batch n_features"]:
        """Generate a batch where each feature activates independently with probability
        `feature_probability`, rejecting samples with all zeros."""
        buffer_size = int(batch_size * buffer_ratio)
        batch = torch.empty(0, device=self.device, dtype=torch.float32)
        n_samples_needed = batch_size
        while True:
            buffer = self._masked_batch_generator(buffer_size)
            valid_indices = buffer.sum(dim=-1) != 0
            batch = torch.cat((batch, buffer[valid_indices][:n_samples_needed]))
            if len(batch) == batch_size:
                break
            n_samples_needed = batch_size - len(batch)
            buffer_size = int(n_samples_needed * buffer_ratio)
        return batch
