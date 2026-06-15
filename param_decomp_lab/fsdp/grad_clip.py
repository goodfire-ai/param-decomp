"""FSDP2/DTensor-aware global grad-norm clip over a flat param list.

torch 2.11's `torch.nn.utils.clip_grad_norm_` already handles DTensor `.grad`s
correctly: `aten.linalg_vector_norm` has a registered DTensor sharding strategy
(`NormReduction`), so the norm it computes is the GLOBAL p-norm reduced across the
FSDP mesh, and the in-place scale touches each local shard. There is no per-shard
under-counting to correct, so we do NOT need the 3-pool disjoint-subset reduction
machinery (`param_decomp.grad_clip.cross_pool_clip_grad_norm`) here.

This wrapper exists only to keep the FSDP trainer off the per-step CPU sync that
reading the returned norm would force. `clip_grad_norm_` returns the total norm as
a (DTensor-wrapped) scalar; materializing it for logging triggers a collective +
device→host sync that blocks the step on the full backward. We discard it by
default and clip with no readback.
"""

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_


def clip_grad_norm_no_sync(params: list[nn.Parameter], max_norm: float) -> None:
    """Clip the global L2 grad-norm of `params` in place without any host sync.

    `params` may hold DTensor `.grad`s under FSDP2; the global norm is reduced
    across the mesh by `clip_grad_norm_`. The returned norm is intentionally
    dropped — reading it would force a per-step device→host sync.
    """
    assert params, "clip_grad_norm_no_sync called with empty params"
    clip_grad_norm_(params, max_norm)


def clip_grad_norm_with_norm(params: list[nn.Parameter], max_norm: float) -> torch.Tensor:
    """Clip in place and return the pre-clip global L2 grad-norm.

    Use only when the caller actually logs the norm and accepts the per-step
    device→host sync that materializing it costs. The returned tensor may be a
    DTensor under FSDP2; call `.full_tensor()` / `.item()` to read it.
    """
    assert params, "clip_grad_norm_with_norm called with empty params"
    return clip_grad_norm_(params, max_norm)
