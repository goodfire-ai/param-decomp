from typing import Annotated, Any

import torch
from jaxtyping import Float
from pydantic import Field
from torch import Tensor
from torch.distributed import ReduceOp

from param_decomp.configs import SamplingType, SubsetRoutingType, UniformKSubsetRoutingConfig
from param_decomp.metrics.base import LossMetricConfig
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.registry import register_metric
from param_decomp.models.batch_and_loss_fns import ReconstructionLoss
from param_decomp.models.component_model import ComponentModel
from param_decomp.routing import Router, get_subset_router
from param_decomp.utils.component_utils import calc_stochastic_component_mask_info
from param_decomp.utils.distributed_utils import all_reduce
from param_decomp.utils.general_utils import get_obj_device


class StochasticReconSubsetLossConfig(LossMetricConfig):
    routing: Annotated[
        SubsetRoutingType, Field(discriminator="type", default=UniformKSubsetRoutingConfig())
    ]


def _stochastic_recon_subset_loss_update(
    model: ComponentModel,
    sampling: SamplingType,
    n_mask_samples: int,
    batch: Any,
    target_out: Tensor,
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    router: Router,
    reconstruction_loss: ReconstructionLoss,
) -> tuple[Float[Tensor, ""], int]:
    assert ci, "Empty ci"
    sum_loss = torch.zeros((), device=get_obj_device(ci))
    n_examples = 0
    stoch_mask_infos_list = [
        calc_stochastic_component_mask_info(
            causal_importances=ci,
            component_mask_sampling=sampling,
            weight_deltas=weight_deltas,
            router=router,
        )
        for _ in range(n_mask_samples)
    ]
    for stoch_mask_infos in stoch_mask_infos_list:
        out = model(batch, mask_infos=stoch_mask_infos)
        loss, batch_n = reconstruction_loss(out, target_out)
        sum_loss = sum_loss + loss
        n_examples += batch_n
    return sum_loss, n_examples


def stochastic_recon_subset_loss(
    model: ComponentModel,
    sampling: SamplingType,
    n_mask_samples: int,
    batch: Any,
    target_out: Tensor,
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    routing: SubsetRoutingType,
    reconstruction_loss: ReconstructionLoss,
) -> Float[Tensor, ""]:
    """Pure compute helper preserved for direct callers (tests, notebooks)."""
    sum_loss, n = _stochastic_recon_subset_loss_update(
        model=model,
        sampling=sampling,
        n_mask_samples=n_mask_samples,
        batch=batch,
        target_out=target_out,
        ci=ci,
        weight_deltas=weight_deltas,
        router=get_subset_router(routing, device=get_obj_device(model)),
        reconstruction_loss=reconstruction_loss,
    )
    return sum_loss / n


@register_metric
class StochasticReconSubsetLoss:
    """Recon loss when sampling with stochastic masks and routing to subsets of component layers."""

    name = "stochastic_recon_subset"
    section = "loss"
    config_type = StochasticReconSubsetLossConfig
    short_name = "StochReconSub"

    def __init__(
        self, cfg: StochasticReconSubsetLossConfig, *, model: ComponentModel, device: str
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
        sum_loss, n = _stochastic_recon_subset_loss_update(
            model=self.model,
            sampling=ctx.config.sampling,
            n_mask_samples=ctx.config.n_mask_samples,
            batch=ctx.batch,
            target_out=ctx.target_out,
            ci=ctx.ci.lower_leaky,
            weight_deltas=wd,
            router=self.router,
            reconstruction_loss=ctx.reconstruction_loss,
        )
        self.sum_loss += sum_loss.detach()
        self.n_examples += n
        return sum_loss / n

    def compute(self) -> Float[Tensor, ""]:
        sum_loss = all_reduce(self.sum_loss, op=ReduceOp.SUM)
        n_examples = all_reduce(self.n_examples, op=ReduceOp.SUM)
        return sum_loss / n_examples
