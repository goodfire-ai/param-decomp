"""Memory-efficient fused linear + KL-divergence loss.

Computes ``KL(softmax(target_logits) || softmax(pred_logits))`` where
``pred_logits = pred_hidden @ lm_head_weight.T`` and ``target_logits =
target_hidden @ lm_head_weight.T``, **without ever materializing the full
[N, vocab] logits tensor.**

This is the same algorithmic substitute for the unfused pattern
::
    pred_logits = pred_hidden @ lm_head_weight.T  # [N, vocab]
    target_logits = target_hidden @ lm_head_weight.T  # [N, vocab]
    loss = recon_loss_kl(pred_logits, target_logits)

but processes the (N=batch*seq) dimension in chunks so peak memory is
``O(chunk_size * vocab)`` instead of ``O(N * vocab)``. For Qwen vocab=152K and
b=64 s=2048 (N=131K), the unfused path materializes ~40-80 GB of vocab-scale
tensors; this kernel keeps it under ~50 MB at chunk_size=128.

Design notes:
  - Only ``pred_hidden`` requires grad. ``target_hidden`` is the cached target
    pre-LM-head activation (frozen target). ``lm_head_weight`` is the target's
    frozen LM head — no grad either.
  - Backward grad-on-pred-hidden is computed during *forward* and saved for
    backward (cheap; just one [N, d_model] tensor). This is the same shape
    pattern as the standard autograd path but with chunked vocab-tensor
    materialization.
  - Inspired by Liger Kernel's ``LigerFusedLinearJSD`` chunked-loss pattern
    (Liger has no fused-linear-KL variant as of 2026-01).
"""

from typing import Any, cast, override

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor


class FusedLinearKLDiv(torch.autograd.Function):
    """Fused linear + KL divergence loss with vocab-dimension chunking.

    Forward signature mirrors the unfused pair (matmul + ``recon_loss_kl``).
    Backward provides grad w.r.t. ``pred_hidden`` only (the other inputs are
    frozen — target's LM head and the cached target hidden states).
    """

    @staticmethod
    @override
    def forward(
        ctx: Any,
        pred_hidden: Float[Tensor, "n d_model"],
        target_hidden: Float[Tensor, "n d_model"],
        lm_head_weight: Float[Tensor, "vocab d_model"],
        chunk_size: int = 1024,
    ) -> tuple[Float[Tensor, ""], int]:
        assert pred_hidden.shape == target_hidden.shape, (
            f"pred_hidden {pred_hidden.shape} vs target_hidden {target_hidden.shape}"
        )
        assert pred_hidden.shape[-1] == lm_head_weight.shape[-1], (
            f"d_model mismatch: pred_hidden {pred_hidden.shape[-1]} vs lm_head_weight {lm_head_weight.shape[-1]}"
        )
        n, _ = pred_hidden.shape
        device = pred_hidden.device
        # Accumulator dtype: fp32 for numeric stability when summing many small
        # KL contributions. The chunk compute can stay in pred_hidden's dtype
        # (typically bf16 under autocast).
        total_loss = torch.zeros((), device=device, dtype=torch.float32)
        grad_pred_hidden = torch.zeros_like(pred_hidden)

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            ph = pred_hidden[start:end]
            th = target_hidden[start:end]

            # Forward: chunk-local logits + KL, no grad inside the chunk (we
            # compute the grad-on-pred-hidden contribution analytically below).
            with torch.no_grad():
                pred_logits = ph @ lm_head_weight.t()  # [chunk, vocab]
                target_logits = th @ lm_head_weight.t()  # [chunk, vocab]

                log_q = F.log_softmax(pred_logits, dim=-1)
                p = F.softmax(target_logits, dim=-1)
                # F.kl_div(log_q, p, reduction='none') = p * (log p - log q)
                # Summing over vocab gives per-position KL.
                kl_per_pos = F.kl_div(log_q, p, reduction="none").sum(dim=-1)
                total_loss = total_loss + kl_per_pos.sum().to(torch.float32)

                # Analytical grad of (per-position-KL summed over chunk) w.r.t.
                # pred_logits is (softmax(pred_logits) - softmax(target_logits)).
                # Chain through the linear: ∂L/∂pred_hidden = (q - p) @ W.
                q = log_q.exp()
                grad_pred_logits = q - p  # [chunk, vocab], same dtype as ph
                grad_pred_hidden[start:end] = grad_pred_logits @ lm_head_weight

        ctx.save_for_backward(grad_pred_hidden)
        # Return loss + n (mirrors recon_loss_kl's return type).
        # Cast back to pred_hidden dtype for downstream sum (matches the
        # unfused recon_loss_kl behaviour).
        return total_loss.to(pred_hidden.dtype), n

    @staticmethod
    @override
    def backward(
        ctx: Any,
        *grad_outputs: Any,
    ) -> tuple[Tensor | None, None, None, None]:
        grad_loss = grad_outputs[0]
        (grad_pred_hidden,) = ctx.saved_tensors
        # grad_pred_hidden was computed for loss-coefficient 1.0; scale by the
        # incoming gradient on the loss output.
        return grad_loss * grad_pred_hidden, None, None, None


def fused_linear_kl_div(
    pred_hidden: Float[Tensor, "n d_model"],
    target_hidden: Float[Tensor, "n d_model"],
    lm_head_weight: Float[Tensor, "vocab d_model"],
    chunk_size: int = 1024,
) -> tuple[Float[Tensor, ""], int]:
    """Functional wrapper around :class:`FusedLinearKLDiv`.

    Returns (sum_of_kl_over_positions, n_positions) — matching the shape of
    ``recon_loss_kl`` so the call site is a drop-in replacement once the
    upstream code is refactored to expose pred_hidden + target_hidden +
    lm_head_weight (instead of pre-materialized logits).
    """
    return cast(
        "tuple[Float[Tensor, '']  , int]",
        FusedLinearKLDiv.apply(pred_hidden, target_hidden, lm_head_weight, chunk_size),
    )
