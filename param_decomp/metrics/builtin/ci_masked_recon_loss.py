from typing import Any

import torch
from jaxtyping import Float
from torch import Tensor
from torch.distributed import ReduceOp

from param_decomp.metrics.base import LossMetricConfig
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.registry import register_metric
from param_decomp.models.batch_and_loss_fns import ReconstructionLoss
from param_decomp.models.component_model import ComponentModel
from param_decomp.models.components import make_mask_infos
from param_decomp.utils.distributed_utils import all_reduce


class CIMaskedReconLossConfig(LossMetricConfig):
    pass


def _ci_masked_recon_loss_update(
    model: ComponentModel,
    batch: Any,
    target_out: Tensor,
    ci: dict[str, Float[Tensor, "... C"]],
    reconstruction_loss: ReconstructionLoss,
) -> tuple[Float[Tensor, ""], int]:
    mask_infos = make_mask_infos(ci, weight_deltas_and_masks=None)
    out = model(batch, mask_infos=mask_infos)
    return reconstruction_loss(out, target_out)


def ci_masked_recon_loss(
    model: ComponentModel,
    batch: Any,
    target_out: Tensor,
    ci: dict[str, Float[Tensor, "... C"]],
    reconstruction_loss: ReconstructionLoss,
) -> Float[Tensor, ""]:
    """Pure compute helper preserved for direct callers (tests, notebooks)."""
    sum_loss, n = _ci_masked_recon_loss_update(model, batch, target_out, ci, reconstruction_loss)
    return sum_loss / n


@register_metric
class CIMaskedReconLoss:
    """Recon loss when masking with CI values directly on all component layers."""

    section = "loss"
    config_type = CIMaskedReconLossConfig
    short_name = "CIMaskRecon"

    def __init__(self, cfg: CIMaskedReconLossConfig, *, model: ComponentModel, device: str) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.sum_loss = torch.zeros((), device=self.device)
        self.n_examples = torch.zeros((), device=self.device, dtype=torch.long)

    def update(self, ctx: MetricContext) -> Tensor:
        sum_loss, n = _ci_masked_recon_loss_update(
            model=self.model,
            batch=ctx.batch,
            target_out=ctx.target_out,
            ci=ctx.ci.lower_leaky,
            reconstruction_loss=ctx.reconstruction_loss,
        )
        self.sum_loss += sum_loss.detach()
        self.n_examples += n
        return sum_loss / n

    def compute(self) -> Float[Tensor, ""]:
        sum_loss = all_reduce(self.sum_loss, op=ReduceOp.SUM)
        n_examples = all_reduce(self.n_examples, op=ReduceOp.SUM)
        return sum_loss / n_examples
