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


class CIMaskedReconLossConfig(LossMetricConfig):
    type: Literal["CIMaskedReconLoss"] = "CIMaskedReconLoss"


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
    """Compute CI-masked recon loss directly (helper preserved for tests/notebooks)."""
    sum_loss, n = _ci_masked_recon_loss_update(model, batch, target_out, ci, reconstruction_loss)
    return sum_loss / n


class CIMaskedReconLoss(Metric[CIMaskedReconLossConfig]):
    """Recon loss: forward with `mask = ci.lower_leaky` on every target layer.

    Scores reconstruction against the target output.
    """

    log_namespace = "loss"
    short_name = "CIMaskRecon"

    @override
    def reset(self) -> None:
        self.sum_loss = torch.zeros((), device=self.device)
        self.n_examples = torch.zeros((), device=self.device, dtype=torch.long)

    @override
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

    @override
    def compute(self) -> MetricResult:
        sum_loss = all_reduce(self.sum_loss, op=ReduceOp.SUM)
        n_examples = all_reduce(self.n_examples, op=ReduceOp.SUM)
        return sum_loss / n_examples
