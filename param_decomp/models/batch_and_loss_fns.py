"""Batch handling and reconstruction loss functions for different model types.

These functions parameterize ComponentModel and training for different target model architectures.

Use ``make_run_batch(config.output_extract)`` when the experiment's output extraction is driven
by config (e.g. LM experiments). Import a concrete helper like ``run_batch_first_element`` or
``run_batch_passthrough`` when the experiment always runs batches the same way.
"""

import math
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
