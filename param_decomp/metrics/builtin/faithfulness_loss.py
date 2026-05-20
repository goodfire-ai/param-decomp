from typing import override

import torch
from jaxtyping import Float
from torch import Tensor
from torch.distributed import ReduceOp

from param_decomp.metrics.base import LossMetricConfig, Metric, MetricResult
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.registry import register_metric
from param_decomp.models.component_model import ComponentModel
from param_decomp.utils.distributed_utils import all_reduce


class FaithfulnessLossConfig(LossMetricConfig):
    pass


def faithfulness_loss(
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]],
) -> Float[Tensor, ""]:
    """Pure compute helper preserved for direct callers (tests, notebooks)."""
    assert weight_deltas, "Empty weight deltas"
    device = next(iter(weight_deltas.values())).device
    sum_loss = torch.zeros((), device=device)
    total_params = 0
    for delta in weight_deltas.values():
        sum_loss = sum_loss + (delta**2).sum()
        total_params += delta.numel()
    return sum_loss / total_params


@register_metric
class FaithfulnessLoss(Metric[FaithfulnessLossConfig]):
    """MSE between the target weights and the sum of the components."""

    section = "loss"
    config_type = FaithfulnessLossConfig
    short_name = "Faith"

    def __init__(self, cfg: FaithfulnessLossConfig, *, model: ComponentModel, device: str) -> None:
        self.cfg = cfg
        self.device = device
        self.reset()

    @override
    def reset(self) -> None:
        self.sum_loss = torch.zeros((), device=self.device)
        self.n_batches = torch.zeros((), device=self.device, dtype=torch.long)

    @override
    def update(self, ctx: MetricContext) -> Tensor:
        loss = faithfulness_loss(ctx.weight_deltas)
        self.sum_loss += loss.detach()
        self.n_batches += 1
        return loss

    @override
    def compute(self) -> MetricResult:
        sum_loss = all_reduce(self.sum_loss, op=ReduceOp.SUM)
        n_batches = all_reduce(self.n_batches, op=ReduceOp.SUM)
        return sum_loss / n_batches
