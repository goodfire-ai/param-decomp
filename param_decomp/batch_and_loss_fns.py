"""Protocols for the callbacks `optimize()` invokes once per batch.

The caller supplies concrete implementations — the lab ships a set in
`param_decomp_lab.batch_and_loss_fns` (`run_batch_passthrough`,
`run_batch_first_element`, `make_run_batch`, `recon_loss_mse`, `recon_loss_kl`),
and tests/experiments compose those into `optimize(run_batch=..., reconstruction_loss=...)`.
"""

from typing import Any, Protocol

import torch
from jaxtyping import Float
from torch import Tensor, nn


class RunBatch(Protocol):
    """Protocol for running a batch through a model and returning the output."""

    def __call__(self, model: nn.Module, batch: Any) -> Tensor: ...


class ReconstructionLoss(Protocol):
    """Protocol for computing reconstruction loss between predictions and targets."""

    def __call__(self, pred: Tensor, target: Tensor) -> tuple[Float[Tensor, ""], int]: ...


def move_batch_to_device(batch: Any, device: str | torch.device) -> Any:
    """Recursively move every Tensor in a (possibly nested) batch to `device`."""
    if isinstance(batch, Tensor):
        return batch.to(device)
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(x, device) for x in batch)
    if isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    return batch
