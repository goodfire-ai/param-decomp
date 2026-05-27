"""Lab-side `RunBatch` / `ReconstructionLoss` helpers.

Passed to `Trainer(run_batch=..., reconstruction_loss=...)`.
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
    """`RunBatch` extracting a tensor from `model(batch)`.

    `None` passes through; `int` indexes into a tuple; `str` reads an attribute (e.g.
    `"logits"`).
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
    """Elementwise MSE recon loss returning `(sum_squared_errors, n_elements)`."""
    assert pred.shape == target.shape
    squared_errors = (pred - target) ** 2
    return squared_errors.sum(), pred.numel()


def calc_kl_divergence_lm(
    pred: Float[Tensor, "... vocab"],
    target: Float[Tensor, "... vocab"],
) -> Float[Tensor, ""]:
    """Mean per-position KL between logits tensors. `pred = Q`, `target = P`."""
    sum_kl, n_positions = recon_loss_kl(pred=pred, target=target)
    return sum_kl / n_positions


def recon_loss_kl(
    pred: Float[Tensor, "... vocab"],
    target: Float[Tensor, "... vocab"],
) -> tuple[Float[Tensor, ""], int]:
    """KL recon loss returning `(sum_per_position_kl, n_positions)`. `pred = Q`, `target = P`."""
    assert pred.shape == target.shape
    log_q = torch.log_softmax(pred, dim=-1)  # log Q
    p = torch.softmax(target, dim=-1)  # P
    n_positions = pred.numel() // pred.shape[-1]
    return F.kl_div(log_q, p, reduction="sum"), n_positions
