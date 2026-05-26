from typing import Any, Literal, override

import torch
from jaxtyping import Float
from torch import Tensor
from torch.distributed import ReduceOp

from param_decomp.batch_and_loss_fns import ReconstructionLoss
from param_decomp.component_model import ComponentModel
from param_decomp.distributed import all_reduce
from param_decomp.masks import AllLayersRouter, SamplingType, calc_stochastic_component_mask_info
from param_decomp.metrics.base import LossMetricConfig, Metric, MetricResult
from param_decomp.metrics.context import MetricContext
from param_decomp.torch_helpers import get_obj_device


class StochasticReconLossConfig(LossMetricConfig):
    type: Literal["StochasticReconLoss"] = "StochasticReconLoss"


def _stochastic_recon_loss_update(
    model: ComponentModel,
    sampling: SamplingType,
    n_mask_samples: int,
    batch: Any,
    target_out: Tensor,
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    reconstruction_loss: ReconstructionLoss,
) -> tuple[Float[Tensor, ""], int]:
    assert ci, "Empty ci"
    sum_loss = torch.zeros((), device=get_obj_device(ci))
    n_examples = 0
    for _ in range(n_mask_samples):
        stoch_mask_infos = calc_stochastic_component_mask_info(
            causal_importances=ci,
            component_mask_sampling=sampling,
            weight_deltas=weight_deltas,
            router=AllLayersRouter(),
        )
        out = model(batch, mask_infos=stoch_mask_infos)
        loss, batch_n = reconstruction_loss(out, target_out)
        sum_loss = sum_loss + loss
        n_examples += batch_n
    return sum_loss, n_examples


def stochastic_recon_loss(
    model: ComponentModel,
    sampling: SamplingType,
    n_mask_samples: int,
    batch: Any,
    target_out: Tensor,
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    reconstruction_loss: ReconstructionLoss,
) -> Float[Tensor, ""]:
    """Compute stochastic recon loss directly (helper for tests/notebooks)."""
    sum_loss, n = _stochastic_recon_loss_update(
        model=model,
        sampling=sampling,
        n_mask_samples=n_mask_samples,
        batch=batch,
        target_out=target_out,
        ci=ci,
        weight_deltas=weight_deltas,
        reconstruction_loss=reconstruction_loss,
    )
    return sum_loss / n


class StochasticReconLoss(Metric[StochasticReconLossConfig]):
    """Stochastic recon loss summed across mask samples.

    For each of `ctx.n_mask_samples` draws, samples a stochastic component mask
    (parameterised by CI values) on every layer and accumulates recon loss.
    """

    log_namespace = "loss"
    short_name = "StochRecon"

    @override
    def reset(self) -> None:
        self.sum_loss = torch.zeros((), device=self.device)
        self.n_examples = torch.zeros((), device=self.device, dtype=torch.long)

    @override
    def update(self, ctx: MetricContext) -> Tensor:
        wd = ctx.weight_deltas if ctx.use_delta_component else None
        sum_loss, n = _stochastic_recon_loss_update(
            model=self.model,
            sampling=ctx.sampling,
            n_mask_samples=ctx.n_mask_samples,
            batch=ctx.batch,
            target_out=ctx.target_out,
            ci=ctx.ci.lower_leaky,
            weight_deltas=wd,
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
