"""Caller-supplied conveniences for `optimize(run_batch=..., reconstruction_loss=...)`.

`optimize()` does not call any of these directly — it just invokes whatever the caller
hands it via the `RunBatch` / `ReconstructionLoss` protocols (in `param_decomp.models.batch_and_loss_fns`).
The helpers below are the implementations the in-repo experiments and tests use.
"""

from typing import Any

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn

from param_decomp.models.batch_and_loss_fns import RunBatch
from param_decomp.torch_helpers import runtime_cast


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


def calc_kl_divergence_lm(
    pred: Float[Tensor, "... vocab"],
    target: Float[Tensor, "... vocab"],
) -> Float[Tensor, ""]:
    """Calculate mean per-position KL divergence between two logits tensors."""
    sum_kl, n_positions = recon_loss_kl(pred=pred, target=target)
    return sum_kl / n_positions


def recon_loss_kl(
    pred: Float[Tensor, "... vocab"],
    target: Float[Tensor, "... vocab"],
) -> tuple[Float[Tensor, ""], int]:
    """KL divergence reconstruction loss for logits. Returns (sum_of_kl, n_positions)."""
    assert pred.shape == target.shape
    log_q = torch.log_softmax(pred, dim=-1)  # log Q
    p = torch.softmax(target, dim=-1)  # P
    n_positions = pred.numel() // pred.shape[-1]
    return F.kl_div(log_q, p, reduction="sum"), n_positions
