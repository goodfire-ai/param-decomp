from typing import Any, ClassVar, override

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import ReduceOp

from spd.configs import SamplingType
from spd.metrics.base import Metric
from spd.models.component_model import CIOutputs, ComponentModel
from spd.models.components import ComponentsMaskInfo, make_mask_infos
from spd.routing import AllLayersRouter
from spd.utils.component_utils import calc_stochastic_component_mask_info
from spd.utils.distributed_utils import all_reduce

PerModuleMSE = dict[str, tuple[Float[Tensor, ""], int]]


def calc_hidden_acts_mse(
    model: ComponentModel,
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    mask_infos: dict[str, ComponentsMaskInfo],
    target_acts: dict[str, Float[Tensor, "..."]],
) -> tuple[PerModuleMSE, Float[Tensor, "..."]]:
    """Forward with mask_infos and compute per-module MSE against target output activations.

    Returns the per-module MSE dict and the component model's output tensor.
    """
    result = model(batch, mask_infos=mask_infos, cache_type="output")
    per_module: PerModuleMSE = {}
    for layer_name, target in target_acts.items():
        assert layer_name in result.cache, f"{layer_name} not in comp_cache"
        mse = F.mse_loss(result.cache[layer_name], target, reduction="sum")
        per_module[layer_name] = (mse, target.numel())
    return per_module, result.output


def _sum_per_module_mse(per_module: PerModuleMSE) -> tuple[Float[Tensor, ""], int]:
    device = next(iter(per_module.values()))[0].device
    total_mse = torch.tensor(0.0, device=device)
    total_n = 0
    for mse, n in per_module.values():
        total_mse = total_mse + mse
        total_n += n
    return total_mse, total_n


def _accumulate_per_module(accum: PerModuleMSE, per_module: PerModuleMSE) -> None:
    for key, (mse, n) in per_module.items():
        if key in accum:
            prev_mse, prev_n = accum[key]
            accum[key] = (prev_mse + mse, prev_n + n)
        else:
            accum[key] = (mse, n)


def _stochastic_hidden_acts_recon_loss_update(
    model: ComponentModel,
    sampling: SamplingType,
    n_mask_samples: int,
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
) -> PerModuleMSE:
    assert ci, "Empty ci"

    target_acts = model(batch, cache_type="output").cache

    accum: PerModuleMSE = {}
    stoch_mask_infos_list = [
        calc_stochastic_component_mask_info(
            causal_importances=ci,
            component_mask_sampling=sampling,
            weight_deltas=weight_deltas,
            router=AllLayersRouter(),
        )
        for _ in range(n_mask_samples)
    ]
    for stoch_mask_infos in stoch_mask_infos_list:
        per_module, _ = calc_hidden_acts_mse(
            model=model,
            batch=batch,
            mask_infos=stoch_mask_infos,
            target_acts=target_acts,
        )
        _accumulate_per_module(accum, per_module)

    return accum


def _hidden_acts_recon_loss_compute(
    sum_mse: Float[Tensor, ""], n_examples: Int[Tensor, ""] | int
) -> Float[Tensor, ""]:
    return sum_mse / n_examples


def compute_per_module_metrics(
    class_name: str,
    per_module_sum_mse: dict[str, Tensor],
    per_module_n_examples: dict[str, Tensor],
) -> dict[str, Float[Tensor, ""]]:
    assert per_module_sum_mse, "No data accumulated"
    keys = list(per_module_sum_mse.keys())
    stacked_mse = torch.stack([per_module_sum_mse[k] for k in keys])
    stacked_n = torch.stack([per_module_n_examples[k].float() for k in keys])
    stacked_mse = all_reduce(stacked_mse, op=ReduceOp.SUM)
    stacked_n = all_reduce(stacked_n, op=ReduceOp.SUM)

    out: dict[str, Float[Tensor, ""]] = {}
    for i, key in enumerate(keys):
        out[f"{class_name}/{key}"] = stacked_mse[i] / stacked_n[i]
    out[class_name] = stacked_mse.sum() / stacked_n.sum()
    return out


def stochastic_hidden_acts_recon_loss(
    model: ComponentModel,
    sampling: SamplingType,
    n_mask_samples: int,
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
) -> Float[Tensor, ""]:
    per_module = _stochastic_hidden_acts_recon_loss_update(
        model=model,
        sampling=sampling,
        n_mask_samples=n_mask_samples,
        batch=batch,
        ci=ci,
        weight_deltas=weight_deltas,
    )
    sum_mse, n_examples = _sum_per_module_mse(per_module)
    return _hidden_acts_recon_loss_compute(sum_mse, n_examples)


class StochasticHiddenActsReconLoss(Metric):
    """Reconstruction loss between target and stochastic hidden activations when sampling with stochastic masks."""

    slow: ClassVar[bool] = True
    metric_section: ClassVar[str] = "loss"

    def __init__(
        self,
        model: ComponentModel,
        device: str,
        sampling: SamplingType,
        use_delta_component: bool,
        n_mask_samples: int,
    ) -> None:
        self.model = model
        self.sampling: SamplingType = sampling
        self.use_delta_component: bool = use_delta_component
        self.n_mask_samples: int = n_mask_samples
        self.device = device
        self.per_module_sum_mse: dict[str, Tensor] = {}
        self.per_module_n_examples: dict[str, Tensor] = {}

    @override
    def update(
        self,
        *,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        ci: CIOutputs,
        weight_deltas: dict[str, Float[Tensor, "d_out d_in"]],
        **_: Any,
    ) -> None:
        per_module = _stochastic_hidden_acts_recon_loss_update(
            model=self.model,
            sampling=self.sampling,
            n_mask_samples=self.n_mask_samples,
            batch=batch,
            ci=ci.lower_leaky,
            weight_deltas=weight_deltas if self.use_delta_component else None,
        )
        for key, (mse, n) in per_module.items():
            if key not in self.per_module_sum_mse:
                self.per_module_sum_mse[key] = torch.tensor(0.0, device=self.device)
                self.per_module_n_examples[key] = torch.tensor(0, device=self.device)
            self.per_module_sum_mse[key] += mse.detach()
            self.per_module_n_examples[key] += n

    @override
    def compute(self) -> dict[str, Float[Tensor, ""]]:
        return compute_per_module_metrics(
            class_name=type(self).__name__,
            per_module_sum_mse=self.per_module_sum_mse,
            per_module_n_examples=self.per_module_n_examples,
        )


class CIHiddenActsReconLoss(Metric):
    """Reconstruction loss between target and component hidden activations when masking with CI values."""

    slow: ClassVar[bool] = True
    metric_section: ClassVar[str] = "loss"

    def __init__(self, model: ComponentModel, device: str) -> None:
        self.model = model
        self.device = device
        self.per_module_sum_mse: dict[str, Tensor] = {}
        self.per_module_n_examples: dict[str, Tensor] = {}

    @override
    def update(
        self,
        *,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        ci: CIOutputs,
        **_: Any,
    ) -> None:
        target_acts = self.model(batch, cache_type="output").cache
        mask_infos = make_mask_infos(ci.lower_leaky, weight_deltas_and_masks=None)
        per_module, _output = calc_hidden_acts_mse(
            model=self.model,
            batch=batch,
            mask_infos=mask_infos,
            target_acts=target_acts,
        )
        for key, (mse, n) in per_module.items():
            if key not in self.per_module_sum_mse:
                self.per_module_sum_mse[key] = torch.tensor(0.0, device=self.device)
                self.per_module_n_examples[key] = torch.tensor(0, device=self.device)
            self.per_module_sum_mse[key] += mse.detach()
            self.per_module_n_examples[key] += n

    @override
    def compute(self) -> dict[str, Float[Tensor, ""]]:
        return compute_per_module_metrics(
            class_name=type(self).__name__,
            per_module_sum_mse=self.per_module_sum_mse,
            per_module_n_examples=self.per_module_n_examples,
        )
