from typing import Any, ClassVar, override

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import ReduceOp

from param_decomp.metrics.base import Metric
from param_decomp.models.component_model import ComponentModel
from param_decomp.utils.distributed_utils import all_reduce
from param_decomp.utils.general_utils import get_obj_device


def _faithfulness_loss_update(
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]],
) -> tuple[Float[Tensor, ""], int]:
    assert weight_deltas, "Empty weight deltas"
    device = get_obj_device(weight_deltas)
    sum_loss = torch.tensor(0.0, device=device)
    total_params = 0
    for delta in weight_deltas.values():
        sum_loss += (delta**2).sum()
        total_params += delta.numel()
    return sum_loss, total_params


def _faithfulness_loss_compute(
    sum_loss: Float[Tensor, ""], total_params: Int[Tensor, ""] | int
) -> Float[Tensor, ""]:
    return sum_loss / total_params


def faithfulness_loss(weight_deltas: dict[str, Float[Tensor, "d_out d_in"]]) -> Float[Tensor, ""]:
    sum_loss, total_params = _faithfulness_loss_update(weight_deltas)
    return _faithfulness_loss_compute(sum_loss, total_params)


class FaithfulnessLoss(Metric):
    """MSE between the target weights and the sum of the components."""

    metric_section: ClassVar[str] = "loss"

    def __init__(self, model: ComponentModel, device: str) -> None:
        self.model = model
        self.sum_loss = torch.tensor(0.0, device=device)
        self.total_params = torch.tensor(0, device=device)

    @override
    def update(self, *, weight_deltas: dict[str, Float[Tensor, "d_out d_in"]], **_: Any) -> None:
        sum_loss, total_params = _faithfulness_loss_update(weight_deltas)
        self.sum_loss += sum_loss
        self.total_params += total_params

    @override
    def compute(self) -> Float[Tensor, ""]:
        sum_loss = all_reduce(self.sum_loss, op=ReduceOp.SUM)
        total_params = all_reduce(self.total_params, op=ReduceOp.SUM)
        return _faithfulness_loss_compute(sum_loss, total_params)
