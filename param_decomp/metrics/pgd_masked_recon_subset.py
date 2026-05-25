from typing import Annotated, Literal, override

import torch
from pydantic import Field
from torch import Tensor
from torch.distributed import ReduceOp

from param_decomp.component_model import ComponentModel
from param_decomp.distributed import all_reduce
from param_decomp.masks import SubsetRoutingType, UniformKSubsetRoutingConfig, get_subset_router
from param_decomp.metrics.base import Metric, MetricResult
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.pgd_utils import PGDConfig, pgd_masked_recon_loss_update


class PGDReconSubsetLossConfig(PGDConfig):
    """Config for `PGDReconSubsetLoss`.

    Attributes:
        type: Discriminator literal `"PGDReconSubsetLoss"`.
        routing: Subset-routing strategy that selects which layers receive masks on
            each forward.
    """

    type: Literal["PGDReconSubsetLoss"] = "PGDReconSubsetLoss"
    routing: Annotated[
        SubsetRoutingType, Field(discriminator="type", default=UniformKSubsetRoutingConfig())
    ]


class PGDReconSubsetLoss(Metric[PGDReconSubsetLossConfig]):
    """Recon loss with adversarially-optimized masks routed to subsets of component layers.

    Runs `cfg.n_steps` of per-step PGD on fresh adversarial sources each batch, with
    masks applied only on a routed subset of layers (per `cfg.routing`).
    """

    log_namespace = "loss"
    short_name = "PGDReconSub"

    @override
    def bind(self, *, model: ComponentModel, device: str) -> None:
        super().bind(model=model, device=device)
        self.router = get_subset_router(self.cfg.routing, device)

    @override
    def reset(self) -> None:
        self.sum_loss = torch.zeros((), device=self.device)
        self.n_examples = torch.zeros((), device=self.device, dtype=torch.long)

    @override
    def update(self, ctx: MetricContext) -> Tensor:
        wd = ctx.weight_deltas if ctx.use_delta_component else None
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

    @override
    def compute(self) -> MetricResult:
        sum_loss = all_reduce(self.sum_loss, op=ReduceOp.SUM)
        n_examples = all_reduce(self.n_examples, op=ReduceOp.SUM)
        return sum_loss / n_examples
