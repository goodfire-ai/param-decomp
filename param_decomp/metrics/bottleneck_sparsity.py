"""`L_p` sparsity penalty on the CI fn's bottleneck codes.

Only meaningful when the CI fn has a sparse bottleneck (`CiBottleneckConfig`); `update`
asserts that codes are present. The hard JumpReLU gate produces exact zeros, so alongside
the loss this metric reports the exact code L0 and the fraction of code dims that never
fired during the eval pass.
"""

from typing import Literal, override

import torch
from jaxtyping import Float
from pydantic import NonNegativeFloat, PositiveFloat
from torch import Tensor
from torch.distributed import ReduceOp

from param_decomp.base_config import Probability
from param_decomp.distributed import all_reduce
from param_decomp.metrics.base import LossMetricConfig, Metric, MetricResult
from param_decomp.metrics.context import MetricContext


class BottleneckSparsityLossConfig(LossMetricConfig):
    """Config for the `L_p` penalty on bottleneck codes.

    `pnorm < 1` is the canonical sparsity-inducing exponent; `eps` keeps the gradient
    bounded at exactly-zero codes.

    `warmup_end_frac > 0` linearly ramps the penalty from 0 at the start of training to
    full strength at that fraction of training. The gate's STE passes no gradient to
    gated-off elements, so dims crushed before the code is useful cannot revive; warmup
    lets the code become load-bearing before sparsity pressure is applied.
    """

    type: Literal["BottleneckSparsityLoss"] = "BottleneckSparsityLoss"
    pnorm: PositiveFloat = 0.9
    eps: NonNegativeFloat = 1e-6
    warmup_end_frac: Probability = 0.0


def _bottleneck_sparsity_loss(
    codes: Float[Tensor, "... D"], pnorm: float, eps: float
) -> Float[Tensor, ""]:
    """Mean over batch/seq of `sum_d (|z_d| + eps)^p`."""
    return ((codes.abs() + eps) ** pnorm).sum(dim=-1).mean()


class BottleneckSparsityLoss(Metric[BottleneckSparsityLossConfig]):
    """`L_p` penalty (p < 1) driving sparsity of the CI bottleneck code."""

    log_namespace = "loss"
    short_name = "BneckSparse"

    @override
    def reset(self) -> None:
        self.loss_sum = torch.zeros((), device=self.device)
        self.l0_sum = torch.zeros((), device=self.device)
        self.dim_fired_counts: Tensor | None = None
        self.n_examples = torch.zeros((), device=self.device, dtype=torch.long)

    @override
    def update(self, ctx: MetricContext) -> Tensor:
        """Returns the warmup-scaled live loss; accumulators track the unscaled penalty."""
        codes = ctx.ci.bottleneck_codes
        assert codes is not None, (
            "BottleneckSparsityLoss requires a CI fn with a bottleneck "
            "(set ci_config.simple_transformer_ci_cfg.bottleneck)"
        )
        loss = _bottleneck_sparsity_loss(codes, pnorm=self.cfg.pnorm, eps=self.cfg.eps)

        n = codes.shape[:-1].numel()
        fired = codes.detach() != 0
        if self.dim_fired_counts is None:
            self.dim_fired_counts = torch.zeros(
                codes.shape[-1], device=self.device, dtype=torch.long
            )
        self.dim_fired_counts += fired.reshape(-1, codes.shape[-1]).sum(dim=0)
        self.l0_sum += fired.sum(dim=-1).float().mean() * n
        self.loss_sum += loss.detach() * n
        self.n_examples += n

        if self.cfg.warmup_end_frac > 0:
            return loss * min(1.0, ctx.current_frac_of_training / self.cfg.warmup_end_frac)
        return loss

    @override
    def compute(self) -> MetricResult:
        assert self.dim_fired_counts is not None, "compute() called before any update()"
        loss_sum = all_reduce(self.loss_sum, op=ReduceOp.SUM)
        l0_sum = all_reduce(self.l0_sum, op=ReduceOp.SUM)
        dim_fired_counts = all_reduce(self.dim_fired_counts, op=ReduceOp.SUM)
        n_examples = all_reduce(self.n_examples, op=ReduceOp.SUM)

        name = type(self).__name__
        return {
            name: loss_sum / n_examples,
            f"{name}_code_l0": l0_sum / n_examples,
            f"{name}_frac_dims_never_fired": (dim_fired_counts == 0).float().mean(),
        }
