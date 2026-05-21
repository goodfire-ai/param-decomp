from typing import Any, override

import torch
from jaxtyping import Float
from torch import Tensor
from torch.distributed import ReduceOp

from param_decomp.metrics.base import Metric, MetricResult
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.pgd_utils import PGDConfig, pgd_masked_recon_loss_update
from param_decomp.models.batch_and_loss_fns import ReconstructionLoss
from param_decomp.models.component_model import ComponentModel
from param_decomp.routing import AllLayersRouter
from param_decomp.utils.distributed_utils import all_reduce


class PGDReconLossConfig(PGDConfig):
    pass


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
    """Pure compute helper preserved for direct callers (tests, notebooks)."""
    sum_loss, n = pgd_masked_recon_loss_update(
        model=model,
        batch=batch,
        ci=ci,
        weight_deltas=weight_deltas,
        target_out=target_out,
        router=AllLayersRouter(),
        pgd_config=pgd_config,
        reconstruction_loss=reconstruction_loss,
    )
    return sum_loss / n


class PGDReconLoss(Metric[PGDReconLossConfig]):
    """Recon loss when masking with adversarially-optimized values and routing to all component
    layers."""

    section = "loss"
    config_type = PGDReconLossConfig
    short_name = "PGDRecon"

    @override
    def reset(self) -> None:
        self.sum_loss = torch.zeros((), device=self.device)
        self.n_examples = torch.zeros((), device=self.device, dtype=torch.long)

    @override
    def update(self, ctx: MetricContext) -> Tensor:
        wd = ctx.weight_deltas if ctx.config.use_delta_component else None
        sum_loss, n = pgd_masked_recon_loss_update(
            model=self.model,
            batch=ctx.batch,
            ci=ctx.ci.lower_leaky,
            weight_deltas=wd,
            target_out=ctx.target_out,
            router=AllLayersRouter(),
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
