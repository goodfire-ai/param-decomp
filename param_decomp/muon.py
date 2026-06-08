"""Muon optimizer (MomentUm Orthogonalized by Newton-Schulz).

Muon takes a momentum-SGD update and orthogonalizes it via a quintic Newton-Schulz
iteration before applying it, which empirically conditions the updates for hidden
weight matrices better than AdamW. Reference: https://github.com/KellerJordan/Muon.

Orthogonalization only makes sense for matrices, so parameters with fewer than 2
dimensions (e.g. a per-component bias vector) fall back to a decoupled-weight-decay Adam
update inside the same optimizer — the caller manages one optimizer object regardless of
how its parameters are shaped.

Under DDP the gradients are all-reduced before `step`, so each rank runs the identical
(deterministic) Newton-Schulz on identical inputs and the parameters stay in sync —
no Muon-specific distributed logic is needed.
"""

from collections.abc import Callable, Iterable
from typing import Any, override

import torch
from jaxtyping import Float
from torch import Tensor
from torch.optim.optimizer import Optimizer


def _orthogonalize_via_newton_schulz(
    grad: Float[Tensor, "rows cols"], steps: int
) -> Float[Tensor, "rows cols"]:
    """Approximate the orthogonal factor `U @ V.T` of `grad`'s SVD via a quintic iteration.

    The quintic coefficients are tuned (per Keller Jordan) to push the singular values
    toward 1 from below, so the iteration converges without the singular values ever
    overshooting. Runs in bf16; the caller casts the result back to the parameter dtype.
    """
    assert grad.ndim >= 2, f"Newton-Schulz needs a matrix, got shape {tuple(grad.shape)}"
    a, b, c = 3.4445, -4.7750, 2.0315
    x = grad.bfloat16()
    transposed = x.size(-2) > x.size(-1)
    if transposed:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        gram = x @ x.mT
        update = b * gram + c * (gram @ gram)
        x = a * x + update @ x
    if transposed:
        x = x.mT
    return x


class Muon(Optimizer):
    """Momentum-orthogonalized optimizer for matrix parameters, Adam fallback otherwise.

    Parameters with `ndim >= 2` get the Muon update; lower-dim parameters get a decoupled
    Adam update keyed off `adam_betas` / `adam_eps`. All parameters share the group's `lr`
    and `weight_decay` (the training loop rewrites `lr` per step from the schedule).
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        *,
        lr: float,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        nesterov: bool = True,
        ns_steps: int = 5,
        adam_betas: tuple[float, float] = (0.9, 0.95),
        adam_eps: float = 1e-8,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adam_betas=adam_betas,
            adam_eps=adam_eps,
        )
        super().__init__(params, defaults)

    @override
    def step(self, closure: Callable[[], float] | None = None) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]  # torch stub overloads `step` as returning float
        assert closure is None, "Muon does not support a closure"
        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if p.ndim >= 2:
                        self._muon_update(p, p.grad, state, group)
                    else:
                        self._adam_update(p, p.grad, state, group)

    def _muon_update(
        self, p: Tensor, grad: Tensor, state: dict[str, Any], group: dict[str, Any]
    ) -> None:
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(grad)
        buf = state["momentum_buffer"]
        buf.lerp_(grad, 1 - group["momentum"])
        # Nesterov: look ahead by blending the fresh grad with the momentum buffer.
        direction = grad.lerp(buf, group["momentum"]) if group["nesterov"] else buf
        update = _orthogonalize_via_newton_schulz(direction, group["ns_steps"])
        # Keep the update's RMS scale-invariant to the matrix's aspect ratio.
        scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
        if group["weight_decay"] != 0:
            p.mul_(1 - group["lr"] * group["weight_decay"])
        p.add_(update.to(p.dtype), alpha=-group["lr"] * scale)

    def _adam_update(
        self, p: Tensor, grad: Tensor, state: dict[str, Any], group: dict[str, Any]
    ) -> None:
        beta1, beta2 = group["adam_betas"]
        if "step" not in state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(grad)
            state["exp_avg_sq"] = torch.zeros_like(grad)
        state["step"] += 1
        step = state["step"]
        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step
        if group["weight_decay"] != 0:
            p.mul_(1 - group["lr"] * group["weight_decay"])
        denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(group["adam_eps"])
        p.addcdiv_(exp_avg, denom, value=-group["lr"] / bias_correction1)
