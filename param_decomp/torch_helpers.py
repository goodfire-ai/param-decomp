"""Small torch helpers used across the training loop and metrics."""

from collections.abc import Generator, Sequence
from typing import Protocol

import torch
import torch.nn as nn
from datasets import IterableDataset
from torch import Tensor
from torch.utils.data import DataLoader, DistributedSampler

from param_decomp.log import logger


def bf16_autocast(enabled: bool = True) -> torch.amp.autocast_mode.autocast:
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=enabled)


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


class _HasDevice(Protocol):
    device: torch.device


CanGetDevice = (
    nn.Module
    | _HasDevice
    | Tensor
    | dict[str, Tensor]
    | dict[str, _HasDevice]
    | Sequence[Tensor]
    | Sequence[_HasDevice]
)


def _get_obj_devices(d: CanGetDevice) -> set[torch.device]:
    if hasattr(d, "device"):
        assert isinstance(d.device, torch.device)  # pyright: ignore[reportAttributeAccessIssue]
        return {d.device}  # pyright: ignore[reportAttributeAccessIssue]
    elif isinstance(d, nn.Module):
        return {param.device for param in d.parameters()}
    elif isinstance(d, dict):
        return {obj.device for obj in d.values()}
    else:
        return {obj.device for obj in d}  # pyright: ignore[reportGeneralTypeIssues]


def get_obj_device(d: CanGetDevice) -> torch.device:
    """Get the device of an object. Asserts all parameters live on the same device."""
    devices = _get_obj_devices(d)
    assert len(devices) == 1, f"Object parameters are on multiple devices: {devices}"
    return devices.pop()
