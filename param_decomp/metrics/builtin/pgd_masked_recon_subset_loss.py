from typing import Annotated

import torch
from jaxtyping import Float
from pydantic import Field
from torch import Tensor
from torch.distributed import ReduceOp

from param_decomp.configs import SubsetRoutingType, UniformKSubsetRoutingConfig
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.pgd_utils import PGDConfig, pgd_masked_recon_loss_update
from param_decomp.metrics.registry import register_metric
from param_decomp.models.component_model import ComponentModel
from param_decomp.routing import get_subset_router
from param_decomp.utils.distributed_utils import all_reduce


class PGDReconSubsetLossConfig(PGDConfig):
    routing: Annotated[
        SubsetRoutingType, Field(discriminator="type", default=UniformKSubsetRoutingConfig())
    ]


@register_metric
class PGDReconSubsetLoss:
    """Recon loss when masking with adversarially-optimized values and routing to subsets of
    component layers."""

    section = "loss"
    config_type = PGDReconSubsetLossConfig
    short_name = "PGDReconSub"

    def __init__(
        self, cfg: PGDReconSubsetLossConfig, *, model: ComponentModel, device: str
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device
        self.router = get_subset_router(cfg.routing, device)
        self.reset()

    def reset(self) -> None:
        self.sum_loss = torch.zeros((), device=self.device)
        self.n_examples = torch.zeros((), device=self.device, dtype=torch.long)

    def update(self, ctx: MetricContext) -> Tensor:
        wd = ctx.weight_deltas if ctx.config.use_delta_component else None
        sum_loss, n = pgd_masked_recon_loss_update(
            model=self.model,
            batch=ctx.batch,
            ci=ctx.ci.lower_leaky,
            weight_deltas=wd,
            target_out=ctx.target_out,
            router=self.router,
            pgd_config=self.cfg,
            reconstruction_loss=ctx.reconstruction_loss,
        )
        self.sum_loss += sum_loss.detach()
        self.n_examples += n
        return sum_loss / n

    def compute(self) -> Float[Tensor, ""]:
        sum_loss = all_reduce(self.sum_loss, op=ReduceOp.SUM)
        n_examples = all_reduce(self.n_examples, op=ReduceOp.SUM)
        return sum_loss / n_examples
