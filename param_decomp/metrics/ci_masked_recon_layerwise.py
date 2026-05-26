from typing import Any, Literal, override

import torch
from jaxtyping import Float
from torch import Tensor
from torch.distributed import ReduceOp

from param_decomp.batch_and_loss_fns import ReconstructionLoss
from param_decomp.component_model import ComponentModel
from param_decomp.distributed import all_reduce
from param_decomp.masks import make_mask_infos
from param_decomp.metrics.base import LossMetricConfig, Metric, MetricResult
from param_decomp.metrics.context import MetricContext
from param_decomp.torch_helpers import get_obj_device


class CIMaskedReconLayerwiseLossConfig(LossMetricConfig):
    type: Literal["CIMaskedReconLayerwiseLoss"] = "CIMaskedReconLayerwiseLoss"


def _ci_masked_recon_layerwise_loss_update(
    model: ComponentModel,
    batch: Any,
    target_out: Tensor,
    ci: dict[str, Float[Tensor, "... C"]],
    reconstruction_loss: ReconstructionLoss,
) -> tuple[Float[Tensor, ""], int]:
    sum_loss = torch.zeros((), device=get_obj_device(model))
    n_examples = 0
    mask_infos = make_mask_infos(ci, weight_deltas_and_masks=None)
    for module_name, mask_info in mask_infos.items():
        out = model(batch, mask_infos={module_name: mask_info})
        loss, batch_n = reconstruction_loss(out, target_out)
        sum_loss = sum_loss + loss
        n_examples += batch_n
    return sum_loss, n_examples


def ci_masked_recon_layerwise_loss(
    model: ComponentModel,
    batch: Any,
    target_out: Tensor,
    ci: dict[str, Float[Tensor, "... C"]],
    reconstruction_loss: ReconstructionLoss,
) -> Float[Tensor, ""]:
    """Compute layerwise CI-masked recon loss directly (helper for tests/notebooks)."""
    sum_loss, n = _ci_masked_recon_layerwise_loss_update(
        model, batch, target_out, ci, reconstruction_loss
    )
    return sum_loss / n


class CIMaskedReconLayerwiseLoss(Metric[CIMaskedReconLayerwiseLossConfig]):
    """Recon loss masking one layer at a time with `ci.lower_leaky`.

    Sums the per-layer recon losses.
    """

    log_namespace = "loss"
    short_name = "CIMaskReconLayer"

    @override
    def reset(self) -> None:
        self.sum_loss = torch.zeros((), device=self.device)
        self.n_examples = torch.zeros((), device=self.device, dtype=torch.long)

    @override
    def update(self, ctx: MetricContext) -> Tensor:
        sum_loss, n = _ci_masked_recon_layerwise_loss_update(
            model=self.model,
            batch=ctx.batch,
            target_out=ctx.target_out,
            ci=ctx.ci.lower_leaky,
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
