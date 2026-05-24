"""Caller-supplied conveniences for `optimize(run_batch=..., reconstruction_loss=...)`.

`optimize()` does not call any of these directly — it just invokes whatever the caller
hands it via the `RunBatch` / `ReconstructionLoss` protocols (in `param_decomp.batch_and_loss_fns`).
The helpers below are the implementations the in-repo experiments and tests use.
"""

from typing import Any

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn

from param_decomp.base_config import runtime_cast
from param_decomp.batch_and_loss_fns import RunBatch


def run_batch_passthrough(model: nn.Module, batch: Any) -> Tensor:
    """Run `model(batch)` and return its output unchanged."""
    return runtime_cast(Tensor, model(batch))


def run_batch_first_element(model: nn.Module, batch: Any) -> Tensor:
    """Run `model` on the first element of a batch tuple (e.g. ``(input, labels)``)."""
    return runtime_cast(Tensor, model(batch[0]))


def make_run_batch(output_extract: int | str | None) -> RunBatch:
    """Build a `RunBatch` that extracts a tensor from the model's raw output.

    Callers wanting more control can skip this and hand `optimize` a custom callable
    directly.

    Args:
        output_extract: How to pull a tensor out of `model(batch)`:
            ``None`` for passthrough, an ``int`` to index into a tuple output, or a
            ``str`` to read an attribute (e.g. ``"logits"``) off a structured output.

    Returns:
        A `RunBatch` callable applying the selected extraction.
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
    """Compute the elementwise MSE reconstruction loss.

    Args:
        pred: Component-model output.
        target: Target-model output of matching shape.

    Returns:
        Tuple of (sum of squared errors, number of scalar elements).
    """
    assert pred.shape == target.shape
    squared_errors = (pred - target) ** 2
    return squared_errors.sum(), pred.numel()


def calc_kl_divergence_lm(
    pred: Float[Tensor, "... vocab"],
    target: Float[Tensor, "... vocab"],
) -> Float[Tensor, ""]:
    """Compute the mean per-position KL divergence between two logits tensors.

    Args:
        pred: Predicted logits (treated as ``Q``).
        target: Target logits (treated as ``P``).

    Returns:
        Scalar KL divergence averaged over all positions.
    """
    sum_kl, n_positions = recon_loss_kl(pred=pred, target=target)
    return sum_kl / n_positions


def recon_loss_kl(
    pred: Float[Tensor, "... vocab"],
    target: Float[Tensor, "... vocab"],
) -> tuple[Float[Tensor, ""], int]:
    """Compute the KL reconstruction loss between two logits tensors.

    Args:
        pred: Predicted logits (``Q``).
        target: Target logits (``P``), matching shape.

    Returns:
        Tuple of (sum of per-position KL contributions, number of positions).
    """
    assert pred.shape == target.shape
    log_q = torch.log_softmax(pred, dim=-1)  # log Q
    p = torch.softmax(target, dim=-1)  # P
    n_positions = pred.numel() // pred.shape[-1]
    return F.kl_div(log_q, p, reduction="sum"), n_positions
