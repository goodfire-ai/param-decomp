"""Shared dataloader utilities."""

from collections.abc import Generator

from datasets import IterableDataset
from torch.utils.data import DataLoader, DistributedSampler

from param_decomp.log import logger


def loop_dataloader[T](dl: DataLoader[T]) -> Generator[T]:
    """Loop over a dataloader, resetting the iterator when it is exhausted.

    Ensures that each epoch gets different data, even when using a distributed sampler.
    """
    epoch = 0
    dl_iter = iter(dl)
    while True:
        try:
            yield next(dl_iter)
        except StopIteration:
            logger.warning("Dataloader exhausted, resetting iterator.")
            epoch += 1
            if isinstance(dl.sampler, DistributedSampler):
                dl.sampler.set_epoch(epoch)
            if isinstance(dl.dataset, IterableDataset):
                dl.dataset.set_epoch(epoch)
            dl_iter = iter(dl)
            yield next(dl_iter)
