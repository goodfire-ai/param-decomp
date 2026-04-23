from typing import Any, ClassVar, override

import torch
from jaxtyping import Float
from torch import Tensor
from torch.distributed import ReduceOp

from spd.configs import PGDConfig
from spd.metrics.base import Metric
from spd.metrics.pgd_utils import pgd_masked_recon_loss_update
from spd.models.batch_and_loss_fns import ReconstructionLoss
from spd.models.component_model import CIOutputs, ComponentModel
from spd.routing import AllLayersRouter
from spd.utils.distributed_utils import all_reduce


def pgd_recon_loss(
    *,
    model: ComponentModel,
    batch: Any,
    target_out: Tensor,
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    pgd_config: PGDConfig,
    reconstruction_loss: ReconstructionLoss,
) -> Float[Tensor, ""]:
    sum_loss, n_examples = pgd_masked_recon_loss_update(
        model=model,
        batch=batch,
        ci=ci,
        weight_deltas=weight_deltas,
        target_out=target_out,
        router=AllLayersRouter(),
        pgd_config=pgd_config,
        reconstruction_loss=reconstruction_loss,
    )
    return sum_loss / n_examples


class PGDReconLoss(Metric):
    """Recon loss when masking with adversarially-optimized values and routing to all component
    layers."""

    metric_section: ClassVar[str] = "loss"

    def __init__(
        self,
        model: ComponentModel,
        device: str,
        pgd_config: PGDConfig,
        use_delta_component: bool,
        reconstruction_loss: ReconstructionLoss,
    ) -> None:
        self.model = model
        self.pgd_config: PGDConfig = pgd_config
        self.use_delta_component: bool = use_delta_component
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
        weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
        **_: Any,
    ) -> None:
        sum_loss, n_examples = pgd_masked_recon_loss_update(
            model=self.model,
            batch=batch,
            ci=ci.lower_leaky,
            weight_deltas=weight_deltas if self.use_delta_component else None,
            target_out=target_out,
            router=AllLayersRouter(),
            pgd_config=self.pgd_config,
            reconstruction_loss=self.reconstruction_loss,
        )
        self.sum_loss += sum_loss
        self.n_examples += n_examples

    @override
    def compute(self) -> Float[Tensor, ""]:
        sum_loss = all_reduce(self.sum_loss, op=ReduceOp.SUM)
        n_examples = all_reduce(self.n_examples, op=ReduceOp.SUM)
        return sum_loss / n_examples
