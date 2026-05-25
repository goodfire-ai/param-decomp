"""Cross-pool gradient norm clipping.

Single-pool training uses ``torch.nn.utils.clip_grad_norm_`` directly: it
computes the L2 norm of all the rank's grads and scales them in place if the
norm exceeds the threshold. That's correct when every rank owns the same set
of parameters and the DDP all-reduce has already produced identical grads on
every rank.

In multi-pool training (2-pool, 3-pool), ranks own DISJOINT parameter subsets
(different sites' V/U + CI fn entries across LW blocks). The "global" L2 norm
that 1-pool clips on is ``sqrt(sum over all ranks of sum_sq_local)``, not a
single rank's local norm. This module provides that reduction.

Equivalence note: when every rank has the same parameters (DDP within a
block) ``sum_sq_local`` is identical on every replica, so a naive cross-rank
SUM would double-count by ``n_replicas``. Dividing by ``n_replicas`` after
the SUM recovers the global value.
"""

import torch
import torch.distributed as dist
from torch import Tensor, nn


def cross_pool_clip_grad_norm(
    params: list[nn.Parameter],
    max_norm: float,
    *,
    group: dist.ProcessGroup,
    n_replicas: int,
) -> Tensor:
    """L2-norm grad clip across a pool, matching ``clip_grad_norm_``'s semantics.

    Each rank's ``params`` are this rank's owned subset (which may be disjoint
    across blocks within the pool). ``n_replicas`` is the number of DDP partners
    that hold identical copies of the parameters on this rank — for an LW block
    of size ``n_per_block``, this is ``n_per_block``; for a fully-replicated CI
    pool, it's ``n_ci``.

    Returns the pre-clip global L2 norm as a 0-dim tensor on the param device,
    matching ``torch.nn.utils.clip_grad_norm_``'s return contract.
    """
    assert params, "cross_pool_clip_grad_norm called with empty params"
    device = params[0].device
    sq_local = torch.zeros((), device=device, dtype=torch.float32)
    for p in params:
        if p.grad is not None:
            sq_local = sq_local + (p.grad.detach().to(torch.float32) ** 2).sum()
    dist.all_reduce(sq_local, op=dist.ReduceOp.SUM, group=group)
    sq_global = sq_local / n_replicas
    total_norm = sq_global.sqrt()
    # Match torch.nn.utils.clip_grad_norm_'s convention: scale only if norm
    # exceeds max_norm; otherwise leave grads untouched.
    clip_coef = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
    if clip_coef.item() < 1.0:
        for p in params:
            if p.grad is not None:
                p.grad.mul_(clip_coef)
    return total_norm
