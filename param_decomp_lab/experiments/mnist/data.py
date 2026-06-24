"""MNIST data for the memorization PD experiment.

Provides the raw-tensor MNIST loader, the deterministic memorized-set builder (label
corruption + optional subsampling), and an infinite batch iterator over the fixed
memorized set used by both pretraining and decomposition.
"""

from collections.abc import Iterator
from typing import override

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import IterableDataset
from torchvision import datasets

# Standard MNIST normalization constants.
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def load_raw_mnist(
    data_dir: str, normalize: bool = True, download: bool = True
) -> tuple[
    Float[Tensor, "n_train 784"],
    Int[Tensor, " n_train"],
    Float[Tensor, "n_test 784"],
    Int[Tensor, " n_test"],
]:
    """Load MNIST as flattened float tensors + integer labels (deterministic order)."""
    train = datasets.MNIST(root=data_dir, train=True, download=download)
    test = datasets.MNIST(root=data_dir, train=False, download=download)

    def to_x(data: Tensor) -> Tensor:
        x = data.float() / 255.0
        if normalize:
            x = (x - MNIST_MEAN) / MNIST_STD
        return x.reshape(x.shape[0], -1).contiguous()

    return (
        to_x(train.data),
        train.targets.long(),
        to_x(test.data),
        test.targets.long(),
    )


def build_memorized_set(
    train_labels: Int[Tensor, " n_train"],
    *,
    label_noise_p: float,
    label_noise_seed: int,
    n_train_examples: int | None,
    subsample_seed: int,
    n_classes: int = 10,
) -> tuple[Int[Tensor, " n"], Int[Tensor, " n"]]:
    """Return (indices into the full MNIST train set, possibly-corrupted labels).

    Subsampling (if `n_train_examples` is set) and label corruption are both deterministic
    given their seeds. A fraction `label_noise_p` of the selected examples have their label
    overwritten with a uniformly random class (which may coincide with the true class — at
    p=1 this yields ~90% wrong labels, learnable only by memorization).
    """
    n_full = train_labels.shape[0]

    # 1. Subsample a fixed index set (full set when n_train_examples is None).
    if n_train_examples is None or n_train_examples >= n_full:
        indices = torch.arange(n_full)
    else:
        gen = torch.Generator().manual_seed(subsample_seed)
        indices = torch.randperm(n_full, generator=gen)[:n_train_examples].sort().values

    labels = train_labels[indices].clone()

    # 2. Corrupt a deterministic fraction of the selected labels.
    if label_noise_p > 0.0:
        n = labels.shape[0]
        gen = torch.Generator().manual_seed(label_noise_seed)
        n_corrupt = int(round(label_noise_p * n))
        corrupt_pos = torch.randperm(n, generator=gen)[:n_corrupt]
        random_labels = torch.randint(0, n_classes, (n_corrupt,), generator=gen)
        labels[corrupt_pos] = random_labels

    return indices, labels


class MnistMemorizedDataset(IterableDataset[tuple[Tensor, Tensor]]):
    """Infinite (image, label) batch iterator over a fixed in-memory memorized set.

    Yields full pre-collated batches (use `DataLoader(ds, batch_size=None)`), matching the
    resid_mlp convention. When `shuffle` is True a fresh permutation is drawn each epoch;
    when False, batches are produced in a fixed order (used for eval so per-component
    density/L0 are measured over the whole memorized set deterministically).
    """

    def __init__(
        self,
        images: Float[Tensor, "n 784"],
        labels: Int[Tensor, " n"],
        batch_size: int,
        device: str,
        shuffle: bool = True,
        seed: int = 0,
    ):
        super().__init__()
        self.images = images.to(device)
        self.labels = labels.to(device)
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.seed = seed

    @override
    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        n = self.images.shape[0]
        gen = torch.Generator(device=self.device).manual_seed(self.seed)
        epoch = 0
        while True:
            order = (
                torch.randperm(n, generator=gen, device=self.device)
                if self.shuffle
                else torch.arange(n, device=self.device)
            )
            for start in range(0, n, self.batch_size):
                idx = order[start : start + self.batch_size]
                yield self.images[idx], self.labels[idx]
            epoch += 1
