"""Protocols for the callbacks `Trainer.run` invokes once per batch.

The lab ships concrete implementations in `param_decomp_lab.batch_and_loss_fns`.
"""

from typing import Any, Protocol

import torch
from jaxtyping import Float
from torch import Tensor, nn


class RunBatch(Protocol):
    """Callable that runs one batch through `model` and returns the output tensor."""

    def __call__(self, model: nn.Module, batch: Any) -> Tensor: ...


class ReconstructionLoss(Protocol):
    """Callable that compares `pred` against `target` and returns `(sum, n_elements)`.

    The first entry is the unreduced sum of per-element losses; the second is the count
    it summed over. Callers reduce `sum / n_elements` to a mean as needed.
    """

    def __call__(self, pred: Tensor, target: Tensor) -> tuple[Float[Tensor, ""], int]: ...


def move_batch_to_device(batch: Any, device: str | torch.device) -> Any:
    """Recursively move every `Tensor` in a (possibly nested) `batch` to `device`.

    Supports tensors, tuples, and dicts; passes other types through unchanged.
    """
    if isinstance(batch, Tensor):
        return batch.to(device)
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(x, device) for x in batch)
    if isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    return batch
