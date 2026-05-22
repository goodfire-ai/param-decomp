"""Layerwise loss strategy: pairs the LM-head-bypass context with recon_loss.

Both pool A's layerwise loop and pool B's PPGD inner loop go through one of
two regimes:

  - **Fused**: bypass LM head; recon_loss is :func:`fused_linear_kl_div` against
    the saved lm_head weight. Forwards return pre-LM-head hidden state; the
    kernel applies LM head + KL in chunks (no vocab-scale tensor materialized).

  - **Unfused**: no bypass; the configured ``reconstruction_loss`` from PD
    config is used directly. Forwards return logits.

:class:`LayerwiseLossStrategy` encapsulates that pair so the runner never
branches on ``use_fused_kl`` — it just calls ``strategy.context`` and
``strategy.recon_loss``.
"""

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any

import torch.nn as nn
from torch import Tensor

from param_decomp.models.batch_and_loss_fns import ReconstructionLoss
from param_decomp.models.fused_linear_kl import fused_linear_kl_div


@contextmanager
def bypass_lm_head(target_model: nn.Module) -> Iterator[Any]:
    """Temporarily replace target_model.lm_head with nn.Identity for one step.

    With the bypass active, every forward through ``target_model`` returns the
    pre-LM-head hidden state ([b, seq, d_model]) instead of full-vocab logits
    ([b, seq, vocab]). The caller must apply the LM head another way — e.g.
    via :func:`fused_linear_kl_div` against the saved lm_head's weight.

    At Qwen vocab=152K, b=32+, s=1024+, the unfused path's vocab-scale tensors
    cost 5-40 GB per layerwise iter. Combined with the fused kernel, this
    drops peak memory to ``O(chunk_size · vocab)``.
    """
    saved = target_model.lm_head
    target_model.lm_head = nn.Identity()
    try:
        yield saved
    finally:
        target_model.lm_head = saved


def _make_fused_kl_recon_loss(lm_head_weight: Tensor) -> ReconstructionLoss:
    """Build a ``ReconstructionLoss`` that applies the fused linear+KL kernel.

    Both ``pred`` and ``target`` are expected to be pre-LM-head hidden states
    of shape ``[..., d_model]`` — the contract that holds when the caller is
    running under :func:`bypass_lm_head`. Returns ``(sum_kl, n_positions)``
    matching :func:`recon_loss_kl`'s contract.
    """

    def fn(pred: Tensor, target: Tensor) -> tuple[Tensor, int]:
        pred_flat = pred.reshape(-1, pred.shape[-1])
        target_flat = target.reshape(-1, target.shape[-1])
        return fused_linear_kl_div(pred_flat, target_flat, lm_head_weight)

    return fn


@dataclass(frozen=True)
class LayerwiseLossStrategy:
    """Pairs the LM-head-bypass context with the matching recon_loss callable.

    The shared contract:

    1. A context manager controlling model-forward output type for the step
       (swap lm_head for Identity → forwards return hidden state; or no-op →
       forwards return logits).
    2. A ``recon_loss(pred, target) -> (loss, n)`` callable whose pred/target
       shapes match what (1) made the forwards produce.
    """

    context: Callable[[nn.Module], AbstractContextManager[Any]]
    recon_loss: ReconstructionLoss

    @classmethod
    def fused(cls, lm_head_weight: Tensor) -> "LayerwiseLossStrategy":
        return cls(
            context=bypass_lm_head,
            recon_loss=_make_fused_kl_recon_loss(lm_head_weight),
        )

    @classmethod
    def unfused(cls, recon_loss: ReconstructionLoss) -> "LayerwiseLossStrategy":
        return cls(
            context=lambda _model: nullcontext(),
            recon_loss=recon_loss,
        )

    @classmethod
    def from_cfg(
        cls,
        target_model: nn.Module,
        use_fused_kl: bool,
        unfused_recon: ReconstructionLoss,
    ) -> "LayerwiseLossStrategy":
        """Resolve the strategy from a (target_model, use_fused_kl) pair."""
        if use_fused_kl:
            lm_head = target_model.lm_head
            assert isinstance(lm_head, nn.Linear), (
                f"expected target_model.lm_head to be nn.Linear; got {type(lm_head)}"
            )
            return cls.fused(lm_head.weight)
        return cls.unfused(unfused_recon)
