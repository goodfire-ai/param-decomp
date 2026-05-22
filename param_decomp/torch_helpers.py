"""Small torch helpers used across the training loop and metrics."""

from collections.abc import Sequence
from typing import Any, Protocol

import torch
import torch.nn as nn
from torch import Tensor


def bf16_autocast(enabled: bool = True) -> torch.amp.autocast_mode.autocast:
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=enabled)


def runtime_cast[T](type_: type[T], obj: Any) -> T:
    """typecast with a runtime check"""
    if not isinstance(obj, type_):
        raise TypeError(f"Expected {type_}, got {type(obj)}")
    return obj


def combine_nonoverlapping_dicts(d1: dict[str, Any], d2: dict[str, Any]) -> None:
    """In-place merge of `d2` into `d1`, asserting no overlapping keys."""
    assert not set(d1.keys()) & set(d2.keys()), "The dictionaries must have no overlapping keys"
    d1.update(d2)


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
