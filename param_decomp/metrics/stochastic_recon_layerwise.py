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


class StochasticReconLayerwiseLossConfig(LossMetricConfig):
    type: Literal["StochasticReconLayerwiseLoss"] = "StochasticReconLayerwiseLoss"


def _stochastic_recon_layerwise_loss_update(
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
    stochastic_mask_infos_list = [
        calc_stochastic_component_mask_info(
            causal_importances=ci,
            component_mask_sampling=sampling,
            weight_deltas=weight_deltas,
            router=AllLayersRouter(),
        )
        for _ in range(n_mask_samples)
    ]
    for stoch_mask_infos in stochastic_mask_infos_list:
        for module_name, mask_info in stoch_mask_infos.items():
            out = model(batch, mask_infos={module_name: mask_info})
            loss, batch_n = reconstruction_loss(out, target_out)
            sum_loss = sum_loss + loss
            n_examples += batch_n
    return sum_loss, n_examples


def stochastic_recon_layerwise_loss(
    model: ComponentModel,
    sampling: SamplingType,
    n_mask_samples: int,
    batch: Any,
    target_out: Tensor,
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    reconstruction_loss: ReconstructionLoss,
) -> Float[Tensor, ""]:
    """Compute layerwise stochastic recon loss directly (helper for tests/notebooks)."""
    sum_loss, n = _stochastic_recon_layerwise_loss_update(
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


class StochasticReconLayerwiseLoss(Metric[StochasticReconLayerwiseLossConfig]):
    """Stochastic recon loss applied one layer at a time.

    Samples per-layer masks per draw but applies only one layer's mask per forward;
    sums the per-layer per-sample recon losses.
    """

    log_namespace = "loss"
    short_name = "StochReconLayer"

    @override
    def reset(self) -> None:
        self.sum_loss = torch.zeros((), device=self.device)
        self.n_examples = torch.zeros((), device=self.device, dtype=torch.long)

    @override
    def update(self, ctx: MetricContext) -> Tensor:
        wd = ctx.weight_deltas if ctx.use_delta_component else None
        sum_loss, n = _stochastic_recon_layerwise_loss_update(
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
