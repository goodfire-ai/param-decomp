from typing import override

import torch
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import Dataset


class DeepLinearDataset(
    Dataset[tuple[Float[Tensor, "batch D"], Float[Tensor, "batch D"]]]
):
    """k-sparse dataset for the deep linear experiment.

    Each sample has exactly `k` randomly chosen active positions; each active position is
    drawn independently from Uniform[0, 1]. All other positions are zero.
    """

    def __init__(self, D: int, k: int, device: str):
        assert 0 < k <= D, f"Require 0 < k <= D, got k={k}, D={D}"
        self.D = D
        self.k = k
        self.device = device

    def __len__(self) -> int:
        return 2**31

    @override
    def __getitem__(self, _idx: int) -> tuple[Tensor, Tensor]:
        raise NotImplementedError("Use generate_batch instead")

    def generate_batch(
        self, batch_size: int
    ) -> tuple[Float[Tensor, "batch D"], Float[Tensor, "batch D"]]:
        batch = torch.zeros(batch_size, self.D, device=self.device)

        # Pick k unique indices per sample by argsorting random scores.
        scores = torch.rand(batch_size, self.D, device=self.device)
        active_indices = scores.argsort(dim=-1)[:, : self.k]

        values = torch.rand(batch_size, self.k, device=self.device)
        batch.scatter_(dim=1, index=active_indices, src=values)
        return batch, batch.clone().detach()
