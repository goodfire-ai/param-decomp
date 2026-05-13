"""Batch handling and reconstruction loss functions for different model types.

These functions parameterize ComponentModel and training for different target model architectures.

Use ``make_run_batch(config.output_extract)`` when the experiment's output extraction is driven
by config (e.g. LM experiments). Import a concrete helper like ``run_batch_first_element`` or
``run_batch_passthrough`` when the experiment always runs batches the same way.
"""

import math
from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn

from param_decomp.utils.general_utils import runtime_cast


class RunBatch(Protocol):
    """Protocol for running a batch through a model and returning the output."""

    def __call__(self, model: nn.Module, batch: Any) -> Tensor: ...


class ReconstructionLoss(Protocol):
    """Protocol for computing reconstruction loss between predictions and targets."""

    def __call__(self, pred: Tensor, target: Tensor) -> tuple[Float[Tensor, ""], int]: ...


class ToDevice(Protocol):
    """Protocol for moving (and optionally pruning) a batch from the dataloader onto a device."""

    def __call__(self, batch: Any, device: str | torch.device) -> Any: ...


def move_batch_to_device(batch: Any, device: str | torch.device) -> Any:
    """Recursively move every Tensor in a (possibly nested) batch to `device`.

    Default `PDTarget.to_device`. Override on `PDTarget` for batches where only some fields
    belong on device (e.g. dict batches with large unused keys).
    """
    if isinstance(batch, Tensor):
        return batch.to(device)
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(x, device) for x in batch)
    if isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    return batch


@dataclass(frozen=True)
class PDTarget:
    """Target model bundle for PD.

    Bundles the model with everything needed to run a forward pass through it
    and compare its output to the component model's output. `reconstruction_loss`
    lives here (not separately) because it's coupled to `run_batch`'s output type:
    KL only makes sense for logits; MSE only makes sense for everything else.

    `to_device` is called once on each raw dataloader batch at the train/eval boundary,
    and its return value is what `run_batch` and every downstream loss/metric sees.
    Override the default `move_batch_to_device` if your dataloader yields a structure
    where only some fields belong on device (e.g. a dict with large unused keys).
    """

    model: nn.Module
    run_batch: RunBatch
    reconstruction_loss: ReconstructionLoss
    tied_weights: list[tuple[str, str]] | None = None
    to_device: ToDevice = move_batch_to_device


def run_batch_passthrough(model: nn.Module, batch: Any) -> Tensor:
    return runtime_cast(Tensor, model(batch))


def run_batch_first_element(model: nn.Module, batch: Any) -> Tensor:
    """Run model on the first element of a batch tuple (e.g. (input, labels) -> model(input))."""
    return runtime_cast(Tensor, model(batch[0]))


def make_run_batch(output_extract: int | str | None) -> RunBatch:
    """Creates a RunBatch function for a given configuration.

    NOTE: If you plan to override the RunBatch functionality, you can simply pass
    a custom RunBatch function into optimize and do not need to use this function at
    all.

    Args:
        output_extract: How to extract the tensor from model output.
            None: passthrough (model output is the tensor)
            int: index into model output tuple (e.g. 0 for first element)
            str: attribute name on model output (e.g. "logits")
    """
    match output_extract:
        case None:
            return run_batch_passthrough
        case int(idx):
            return lambda model, batch: model(batch)[idx]
        case str(attr):
            return lambda model, batch: getattr(model(batch), attr)


def recon_loss_mse(
    pred: Float[Tensor, "... d"],
    target: Float[Tensor, "... d"],
) -> tuple[Float[Tensor, ""], int]:
    """MSE reconstruction loss. Returns (sum_of_squared_errors, n_elements)."""
    assert pred.shape == target.shape
    squared_errors = (pred - target) ** 2
    return squared_errors.sum(), pred.numel()


def recon_loss_kl(
    pred: Float[Tensor, "... vocab"],
    target: Float[Tensor, "... vocab"],
) -> tuple[Float[Tensor, ""], int]:
    """KL divergence reconstruction loss for logits. Returns (sum_of_kl, n_positions)."""
    assert pred.shape == target.shape
    log_q = torch.log_softmax(pred, dim=-1)  # log Q
    p = torch.softmax(target, dim=-1)  # P
    kl_per_position = F.kl_div(log_q, p, reduction="none").sum(dim=-1)  # P · (log P − log Q)
    return kl_per_position.sum(), math.prod(pred.shape[:-1])
