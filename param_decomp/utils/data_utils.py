from collections.abc import Iterator
from typing import Literal, override

import torch
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class DatasetGeneratedDataLoader[Q](DataLoader[Q]):
    """DataLoader that generates batches by calling the dataset's `generate_batch` method."""

    def __init__(
        self,
        dataset: Dataset[Q],
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
    ):
        # assert that dataset has a generate_batch method
        assert hasattr(dataset, "generate_batch")
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    @override
    def __iter__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
    ) -> Iterator[Q]:
        for _ in range(len(self)):
            yield self.dataset.generate_batch(self.batch_size)  # pyright: ignore[reportAttributeAccessIssue]


class BatchedDataLoader[Q](DataLoader[Q]):
    """DataLoader that unpacks the batch in __getitem__.

    This is used for datasets which generate a whole batch in one call to __getitem__.
    """

    def __init__(
        self,
        dataset: Dataset[Q],
        num_workers: int = 0,
    ):
        super().__init__(dataset, num_workers=num_workers)

    @override
    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:  # pyright: ignore[reportIncompatibleMethodOverride]
        for batch, label in super().__iter__():
            yield batch[0], label[0]


class InductionDataset(
    Dataset[
        tuple[
            Float[Tensor, "batch seq_len"],
            Float[Tensor, "batch 1"],
        ]
    ]
):
    """
    Generates data of the format TTTTTSMTTT...SM
    where T is a token from the base vocabulary, S is a special induction token,
    and M is a memorised token that appears twice in the sequence.
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        device: str | torch.device,
        prefix_window: int,
        size: int = 100_000,
    ):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.prefix_window = prefix_window
        self.size = size
        self.induction_token = vocab_size + 1  # One additional token for the induction token
        self.device = device
        assert self.prefix_window < seq_len - 2, "S M … S M must fit."

    def __len__(self) -> int:
        return 2**31

    @torch.no_grad()  # pyright: ignore[reportUntypedFunctionDecorator]
    def generate_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        vocab_start = 1  # 0 is reserved for BOS
        vocab_end = self.vocab_size + vocab_start

        # For each sequence, we sample an `m` which serves as the target.
        memorised_token = torch.randint(vocab_start, vocab_end, (batch_size, 1), dtype=torch.long)

        # To begin with, our sequence is just `T*`
        tokens = torch.randint(
            vocab_start, vocab_end, (batch_size, self.seq_len - 2), dtype=torch.long
        )

        # We sample a random position in each sequence and insert the induction token
        # followed by the memory token -- our sequences are now: `T* S M T*`
        induction_token_location = torch.randint(
            1, self.prefix_window, (batch_size, 1), dtype=torch.long
        )
        tokens.scatter_(1, induction_token_location, self.induction_token)
        tokens.scatter_(1, induction_token_location + 1, memorised_token)

        # Finally, we add BOS tokens and an induction token at the final position.
        # Our sequences are now: `[BOS] T* S M T* S`
        tokens = torch.cat(
            (
                torch.zeros((batch_size, 1), dtype=torch.long),  # BOS token
                tokens,
                self.induction_token * torch.ones((batch_size, 1), dtype=torch.long),
            ),
            dim=1,
        )

        # Label is the memorised token that appears twice
        return tokens.to(self.device), memorised_token.to(self.device).squeeze(-1)


DataGenerationType = Literal[
    "exactly_one_active",
    "exactly_two_active",
    "exactly_three_active",
    "exactly_four_active",
    "exactly_five_active",
    "at_least_zero_active",
]


class SparseFeatureDataset(
    Dataset[
        tuple[
            Float[Tensor, "batch n_features"],
            Float[Tensor, "batch n_features"],
        ]
    ]
):
    def __init__(
        self,
        n_features: int,
        feature_probability: float,
        device: str,
        data_generation_type: DataGenerationType = "at_least_zero_active",
        value_range: tuple[float, float] = (0.0, 1.0),
        synced_inputs: list[list[int]] | None = None,
    ):
        self.n_features: int = n_features
        self.feature_probability: float = feature_probability
        self.device: str = device
        self.data_generation_type: DataGenerationType = data_generation_type
        self.value_range: tuple[float, float] = value_range
        self.synced_inputs: list[list[int]] | None = synced_inputs

    def __len__(self) -> int:
        return 2**31

    def sync_inputs(
        self, batch: Float[Tensor, "batch n_features"]
    ) -> Float[Tensor, "batch n_features"]:
        assert self.synced_inputs is not None
        all_indices = [item for sublist in self.synced_inputs for item in sublist]
        assert len(all_indices) == len(set(all_indices)), "Synced inputs must be non-overlapping"
        for indices in self.synced_inputs:
            mask = torch.zeros_like(batch, dtype=torch.bool)
            # First, get the samples for which there is a non-zero value for any of the indices
            non_zero_samples = (batch[..., indices] != 0.0).any(dim=-1)
            for idx in indices:
                mask[..., idx] = non_zero_samples
            # Now generate random values in value_range and apply them to the masked elements
            max_val, min_val = self.value_range
            random_values = torch.rand(batch.shape[0], self.n_features, device=self.device)
            random_values = random_values * (max_val - min_val) + min_val
            batch = torch.where(mask, random_values, batch)
        return batch

    def generate_batch(
        self, batch_size: int
    ) -> tuple[Float[Tensor, "batch n_features"], Float[Tensor, "batch n_features"]]:
        # TODO: This is a hack to keep backward compatibility. Probably best to have
        # data_generation_type: Literal["exactly_n_active", "at_least_zero_active"] and
        # data_generation_n: PositiveInt
        number_map = {
            "exactly_one_active": 1,
            "exactly_two_active": 2,
            "exactly_three_active": 3,
            "exactly_four_active": 4,
            "exactly_five_active": 5,
        }
        if self.data_generation_type in number_map:
            n = number_map[self.data_generation_type]
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
        """Generate a batch with exactly n features active per sample.

        Args:
            batch_size: Number of samples in the batch
            n: Number of features to activate per sample
        """
        if n > self.n_features:
            raise ValueError(
                f"Cannot activate {n} features when only {self.n_features} features exist"
            )

        batch = torch.zeros(batch_size, self.n_features, device=self.device)

        # Create indices for all features
        feature_indices = torch.arange(self.n_features, device=self.device)
        # Expand to batch size
        feature_indices = feature_indices.expand(batch_size, self.n_features)

        # For each instance in the batch, randomly permute the features
        perm = torch.rand_like(feature_indices.float()).argsort(dim=-1)
        permuted_features = feature_indices.gather(dim=-1, index=perm)

        # Take first n indices for each instance - guaranteed no duplicates
        active_features = permuted_features[..., :n]

        # Generate random values in value_range for the active features
        min_val, max_val = self.value_range
        random_values = torch.rand(batch_size, n, device=self.device)
        random_values = random_values * (max_val - min_val) + min_val

        # Place each active feature
        for i in range(n):
            batch.scatter_(
                dim=1, index=active_features[..., i : i + 1], src=random_values[..., i : i + 1]
            )

        return batch

    def _masked_batch_generator(self, batch_size: int) -> Float[Tensor, "batch_size n_features"]:
        """Generate a batch where each feature activates independently with probability
        `feature_probability`.
        """
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
        `feature_probability`.

        Ensures that there are no zero samples in the batch.

        Args:
            batch_size: Number of samples in the batch
            buffer_ratio: First generate `buffer_ratio * batch_size` samples and count the
                number of samples with all zeros. Then generate another `buffer_ratio *
                n_zeros` samples and fill in the zero samples. Continue until there are no zero
                samples.
        """
        buffer_size = int(batch_size * buffer_ratio)
        batch = torch.empty(0, device=self.device, dtype=torch.float32)
        n_samples_needed = batch_size
        while True:
            buffer = self._masked_batch_generator(buffer_size)
            # Get the indices of the non-zero samples in the buffer
            valid_indices = buffer.sum(dim=-1) != 0
            batch = torch.cat((batch, buffer[valid_indices][:n_samples_needed]))
            if len(batch) == batch_size:
                break
            else:
                # We don't have enough valid samples
                n_samples_needed = batch_size - len(batch)
                buffer_size = int(n_samples_needed * buffer_ratio)
        return batch
