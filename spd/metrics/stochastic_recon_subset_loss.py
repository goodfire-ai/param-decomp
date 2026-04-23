from typing import Any, ClassVar, override

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import ReduceOp

from spd.configs import SamplingType, SubsetRoutingType
from spd.metrics.base import Metric
from spd.models.batch_and_loss_fns import ReconstructionLoss
from spd.models.component_model import CIOutputs, ComponentModel
from spd.routing import Router, get_subset_router
from spd.utils.component_utils import calc_stochastic_component_mask_info
from spd.utils.distributed_utils import all_reduce
from spd.utils.general_utils import get_obj_device


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
    device = get_obj_device(ci)
    sum_loss = torch.tensor(0.0, device=device)
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
        loss, batch_n_examples = reconstruction_loss(out, target_out)
        sum_loss += loss
        n_examples += batch_n_examples
    return sum_loss, n_examples


def _stochastic_recon_subset_loss_compute(
    sum_loss: Float[Tensor, ""], n_examples: Int[Tensor, ""] | int
) -> Float[Tensor, ""]:
    return sum_loss / n_examples


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
    sum_loss, n_examples = _stochastic_recon_subset_loss_update(
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
    return _stochastic_recon_subset_loss_compute(sum_loss, n_examples)


class StochasticReconSubsetLoss(Metric):
    """Recon loss when sampling with stochastic masks and routing to subsets of component layers."""

    metric_section: ClassVar[str] = "loss"

    def __init__(
        self,
        model: ComponentModel,
        device: str,
        sampling: SamplingType,
        use_delta_component: bool,
        n_mask_samples: int,
        routing: SubsetRoutingType,
        reconstruction_loss: ReconstructionLoss,
    ) -> None:
        self.model = model
        self.sampling: SamplingType = sampling
        self.use_delta_component: bool = use_delta_component
        self.n_mask_samples: int = n_mask_samples
        self.router = get_subset_router(routing, device)
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
        weight_deltas: dict[str, Float[Tensor, "d_out d_in"]],
        **_: Any,
    ) -> None:
        sum_loss, n_examples = _stochastic_recon_subset_loss_update(
            model=self.model,
            sampling=self.sampling,
            n_mask_samples=self.n_mask_samples,
            batch=batch,
            target_out=target_out,
            ci=ci.lower_leaky,
            weight_deltas=weight_deltas if self.use_delta_component else None,
            router=self.router,
            reconstruction_loss=self.reconstruction_loss,
        )
        self.sum_loss += sum_loss
        self.n_examples += n_examples

    @override
    def compute(self) -> Float[Tensor, ""]:
        sum_loss = all_reduce(self.sum_loss, op=ReduceOp.SUM)
        n_examples = all_reduce(self.n_examples, op=ReduceOp.SUM)
        return _stochastic_recon_subset_loss_compute(sum_loss, n_examples)
