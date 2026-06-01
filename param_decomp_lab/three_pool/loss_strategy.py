"""Layerwise loss strategy: pairs the LM-head-bypass context with recon_loss.

Both the LW pool's layerwise loop and the PPGD pool's inner loop go through one
of two regimes:

  - **Fused**: bypass LM head; recon_loss is :func:`fused_linear_kl_div` against
    the model's lm_head weight. Forwards return pre-LM-head hidden state; the
    kernel applies LM head + KL in chunks (no vocab-scale tensor materialized).

  - **Unfused**: no bypass; the configured ``reconstruction_loss`` from PD
    config is used directly. Forwards return logits.

:class:`LayerwiseLossStrategy` encapsulates that pair so the runner never
branches on ``use_fused_kl`` — it just calls ``strategy.context()`` and
``strategy.recon_loss``.

The bypass is the vendored model's own :meth:`LMComponentModel.bypass_lm_head`
contextmanager (not a target-model monkeypatch): under it, every forward through
the component model returns the post-final-LN hidden state instead of logits.
"""

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass

from torch import Tensor

from param_decomp.batch_and_loss_fns import ReconstructionLoss
from param_decomp.fused_linear_kl import fused_linear_kl_div
from param_decomp_lab.experiments.lm.vendored.component_model import LMComponentModel


def _make_fused_kl_recon_loss(lm_head_weight: Tensor) -> ReconstructionLoss:
    """Build a ``ReconstructionLoss`` that applies the fused linear+KL kernel.

    Both ``pred`` and ``target`` are expected to be pre-LM-head hidden states
    of shape ``[..., d_model]`` — the contract that holds when the caller is
    running under ``bypass_lm_head``. Returns ``(sum_kl, n_positions)``
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

    1. A 0-arg context manager controlling model-forward output type for the step
       (bypass the lm_head → forwards return hidden state; or no-op → forwards
       return logits).
    2. A ``recon_loss(pred, target) -> (loss, n)`` callable whose pred/target
       shapes match what (1) made the forwards produce.
    """

    context: Callable[[], AbstractContextManager[object]]
    recon_loss: ReconstructionLoss

    @classmethod
    def fused(cls, component_model: LMComponentModel) -> "LayerwiseLossStrategy":
        # bypass_lm_head() yields lm_head.weight; the weight is stable across steps,
        # so capture it once for the fused-KL recon loss while toggling the bypass flag.
        with component_model.bypass_lm_head() as lm_head_weight:
            recon_loss = _make_fused_kl_recon_loss(lm_head_weight)
        return cls(context=component_model.bypass_lm_head, recon_loss=recon_loss)

    @classmethod
    def unfused(cls, recon_loss: ReconstructionLoss) -> "LayerwiseLossStrategy":
        @contextmanager
        def _noop() -> Iterator[None]:
            with nullcontext():
                yield

        return cls(context=_noop, recon_loss=recon_loss)

    @classmethod
    def from_cfg(
        cls,
        component_model: LMComponentModel,
        use_fused_kl: bool,
        unfused_recon: ReconstructionLoss,
    ) -> "LayerwiseLossStrategy":
        """Resolve the strategy from a (component_model, use_fused_kl) pair."""
        if use_fused_kl:
            return cls.fused(component_model)
        return cls.unfused(unfused_recon)
