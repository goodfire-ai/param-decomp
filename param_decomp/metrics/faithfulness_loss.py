from typing import Any, ClassVar, override

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import ReduceOp

from param_decomp.metrics.base import Metric
from param_decomp.models.component_model import ComponentModel
from param_decomp.utils.distributed_utils import all_reduce


def _faithfulness_loss_compute(
    sum_loss: Float[Tensor, ""], total_params: Int[Tensor, ""] | int
) -> Float[Tensor, ""]:
    return sum_loss / total_params


def faithfulness_loss(model: ComponentModel) -> Float[Tensor, ""]:
    """MSE of `W_target - V@U` over all sites. Computed inside each site's forward
    so FSDP2 gather/reduce hooks fire."""
    sum_sq, numel = model.calc_faithfulness_terms()
    return _faithfulness_loss_compute(sum_sq, numel)


class FaithfulnessLoss(Metric):
    """MSE between the target weights and the sum of the components."""

    metric_section: ClassVar[str] = "loss"

    def __init__(self, model: ComponentModel, device: str) -> None:
        self.model = model
        self.sum_loss = torch.tensor(0.0, device=device)
        self.total_params = torch.tensor(0, device=device)

    @override
    def update(self, **_: Any) -> None:
        sum_sq, numel = self.model.calc_faithfulness_terms()
        self.sum_loss = self.sum_loss + sum_sq
        self.total_params = self.total_params + numel

    @override
    def compute(self) -> Float[Tensor, ""]:
        sum_loss = all_reduce(self.sum_loss, op=ReduceOp.SUM)
        total_params = all_reduce(self.total_params, op=ReduceOp.SUM)
        return _faithfulness_loss_compute(sum_loss, total_params)
