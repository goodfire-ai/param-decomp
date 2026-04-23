from typing import Any, ClassVar, override

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import ReduceOp

from spd.metrics.base import Metric
from spd.models.batch_and_loss_fns import ReconstructionLoss
from spd.models.component_model import CIOutputs, ComponentModel
from spd.models.components import make_mask_infos
from spd.utils.distributed_utils import all_reduce


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


def _ci_masked_recon_loss_compute(
    sum_loss: Float[Tensor, ""], n_examples: Int[Tensor, ""] | int
) -> Float[Tensor, ""]:
    return sum_loss / n_examples


def ci_masked_recon_loss(
    model: ComponentModel,
    batch: Any,
    target_out: Tensor,
    ci: dict[str, Float[Tensor, "... C"]],
    reconstruction_loss: ReconstructionLoss,
) -> Float[Tensor, ""]:
    sum_loss, n_examples = _ci_masked_recon_loss_update(
        model=model,
        batch=batch,
        target_out=target_out,
        ci=ci,
        reconstruction_loss=reconstruction_loss,
    )
    return _ci_masked_recon_loss_compute(sum_loss, n_examples)


class CIMaskedReconLoss(Metric):
    """Recon loss when masking with CI values directly on all component layers."""

    metric_section: ClassVar[str] = "loss"

    def __init__(
        self,
        model: ComponentModel,
        device: str,
        reconstruction_loss: ReconstructionLoss,
    ) -> None:
        self.model = model
        self.reconstruction_loss = reconstruction_loss
        self.sum_loss = torch.tensor(0.0, device=device)
        self.n_examples = torch.tensor(0, device=device)

    @override
    def update(
        self,
        *,
        batch: Any,
        target_out: Tensor,
        ci: CIOutputs,
        **_: Any,
    ) -> None:
        sum_loss, n_examples = _ci_masked_recon_loss_update(
            model=self.model,
            batch=batch,
            target_out=target_out,
            ci=ci.lower_leaky,
            reconstruction_loss=self.reconstruction_loss,
        )
        self.sum_loss += sum_loss
        self.n_examples += n_examples

    @override
    def compute(self) -> Float[Tensor, ""]:
        sum_loss = all_reduce(self.sum_loss, op=ReduceOp.SUM)
        n_examples = all_reduce(self.n_examples, op=ReduceOp.SUM)
        return _ci_masked_recon_loss_compute(sum_loss, n_examples)
