from typing import Annotated, Any, Literal, override

import torch
from jaxtyping import Float
from pydantic import Field
from torch import Tensor
from torch.distributed import ReduceOp

from param_decomp.batch_and_loss_fns import ReconstructionLoss
from param_decomp.component_model import ComponentModel
from param_decomp.distributed import all_reduce
from param_decomp.masks import (
    Router,
    SubsetRoutingType,
    UniformKSubsetRoutingConfig,
    get_subset_router,
    make_mask_infos,
)
from param_decomp.metrics.base import LossMetricConfig, Metric, MetricResult
from param_decomp.metrics.context import MetricContext


class CIMaskedReconSubsetLossConfig(LossMetricConfig):
    type: Literal["CIMaskedReconSubsetLoss"] = "CIMaskedReconSubsetLoss"
    routing: Annotated[
        SubsetRoutingType, Field(discriminator="type", default=UniformKSubsetRoutingConfig())
    ]


def _ci_masked_recon_subset_loss_update(
    model: ComponentModel,
    batch: Any,
    target_out: Tensor,
    ci: dict[str, Float[Tensor, "... C"]],
    router: Router,
    reconstruction_loss: ReconstructionLoss,
) -> tuple[Float[Tensor, ""], int]:
    subset_routing_masks = router.get_masks(
        module_names=model.target_module_paths,
        mask_shape=next(iter(ci.values())).shape[:-1],
    )
    mask_infos = make_mask_infos(
        component_masks=ci,
        routing_masks=subset_routing_masks,
        weight_deltas_and_masks=None,
    )
    out = model(batch, mask_infos=mask_infos)
    return reconstruction_loss(out, target_out)


def ci_masked_recon_subset_loss(
    model: ComponentModel,
    batch: Any,
    target_out: Tensor,
    ci: dict[str, Float[Tensor, "... C"]],
    routing: SubsetRoutingType,
    reconstruction_loss: ReconstructionLoss,
) -> Float[Tensor, ""]:
    """Compute CI-masked subset recon loss directly (helper for tests/notebooks)."""
    from param_decomp.torch_helpers import get_obj_device

    sum_loss, n = _ci_masked_recon_subset_loss_update(
        model=model,
        batch=batch,
        target_out=target_out,
        ci=ci,
        router=get_subset_router(routing, device=get_obj_device(model)),
        reconstruction_loss=reconstruction_loss,
    )
    return sum_loss / n


class CIMaskedReconSubsetLoss(Metric[CIMaskedReconSubsetLossConfig]):
    """Recon loss applying the CI mask only on a routed subset of layers.

    Subset chosen per `cfg.routing`; the remaining layers run with the original target
    weights.
    """

    log_namespace = "loss"
    short_name = "CIMaskReconSub"

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
        sum_loss, n = _ci_masked_recon_subset_loss_update(
            model=self.model,
            batch=ctx.batch,
            target_out=ctx.target_out,
            ci=ctx.ci.lower_leaky,
            router=self.router,
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
