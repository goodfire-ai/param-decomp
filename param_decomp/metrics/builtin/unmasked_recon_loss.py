from typing import Any, override

import torch
from jaxtyping import Float
from torch import Tensor
from torch.distributed import ReduceOp

from param_decomp.metrics.base import LossMetricConfig, Metric, MetricResult
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.registry import register_metric
from param_decomp.models.batch_and_loss_fns import ReconstructionLoss
from param_decomp.models.component_model import ComponentModel
from param_decomp.models.components import make_mask_infos
from param_decomp.utils.distributed_utils import all_reduce
from param_decomp.utils.general_utils import get_obj_device


class UnmaskedReconLossConfig(LossMetricConfig):
    pass


def _unmasked_recon_loss_update(
    model: ComponentModel,
    batch: Any,
    target_out: Tensor,
    reconstruction_loss: ReconstructionLoss,
) -> tuple[Float[Tensor, ""], int]:
    device = get_obj_device(model)
    all_ones_mask_infos = make_mask_infos(
        {
            module_path: torch.ones(model.module_to_c[module_path], device=device)
            for module_path in model.target_module_paths
        }
    )
    out = model(batch, mask_infos=all_ones_mask_infos)
    return reconstruction_loss(out, target_out)


@register_metric
class UnmaskedReconLoss(Metric[UnmaskedReconLossConfig]):
    """Recon loss using the unmasked components and without the delta component."""

    section = "loss"
    config_type = UnmaskedReconLossConfig
    short_name = "UnmaskedRecon"

    def __init__(self, cfg: UnmaskedReconLossConfig, *, model: ComponentModel, device: str) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device
        self.reset()

    @override
    def reset(self) -> None:
        self.sum_loss = torch.zeros((), device=self.device)
        self.n_examples = torch.zeros((), device=self.device, dtype=torch.long)

    @override
    def update(self, ctx: MetricContext) -> Tensor:
        sum_loss, n = _unmasked_recon_loss_update(
            model=self.model,
            batch=ctx.batch,
            target_out=ctx.target_out,
            reconstruction_loss=ctx.reconstruction_loss,
        )
        self.sum_loss += sum_loss.detach()
        self.n_examples += n
        return sum_loss / n

    @override
    def compute(self) -> MetricResult:
        sum_loss = all_reduce(self.sum_loss, op=ReduceOp.SUM)
        n_examples = all_reduce(self.n_examples, op=ReduceOp.SUM)
        return sum_loss / n_examples
