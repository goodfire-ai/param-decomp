from typing import Literal, override

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import ReduceOp

from param_decomp.component_model import ComponentModel
from param_decomp.distributed import all_reduce
from param_decomp.masks import (
    AllLayersRouter,
    ComponentsMaskInfo,
    SamplingType,
    calc_stochastic_component_mask_info,
)
from param_decomp.metrics.base import LossMetricConfig, Metric, MetricResult
from param_decomp.metrics.context import MetricContext

PerModuleMSE = dict[str, tuple[Float[Tensor, ""], int]]


class StochasticHiddenActsReconLossConfig(LossMetricConfig):
    type: Literal["StochasticHiddenActsReconLoss"] = "StochasticHiddenActsReconLoss"


def calc_hidden_acts_mse(
    model: ComponentModel,
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    mask_infos: dict[str, ComponentsMaskInfo],
    target_acts: dict[str, Float[Tensor, "..."]],
) -> tuple[PerModuleMSE, Float[Tensor, "..."]]:
    """Forward with `mask_infos` and compute per-module MSE against `target_acts`.

    Returns `({module_path: (summed_mse, n_elements)}, model_output)`.
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
    total_mse = torch.zeros((), device=device)
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


def _stochastic_hidden_acts_update(
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
    for _ in range(n_mask_samples):
        stoch_mask_infos = calc_stochastic_component_mask_info(
            causal_importances=ci,
            component_mask_sampling=sampling,
            weight_deltas=weight_deltas,
            router=AllLayersRouter(),
        )
        per_module, _ = calc_hidden_acts_mse(
            model=model, batch=batch, mask_infos=stoch_mask_infos, target_acts=target_acts
        )
        _accumulate_per_module(accum, per_module)
    return accum


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


class _HiddenActsAccumulator:
    """Shared accumulator state for hidden_acts metrics."""

    def __init__(self, device: str) -> None:
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.per_module_sum_mse: dict[str, Tensor] = {}
        self.per_module_n_examples: dict[str, Tensor] = {}

    def accumulate(self, per_module: PerModuleMSE) -> tuple[Float[Tensor, ""], int]:
        for key, (mse, n) in per_module.items():
            if key not in self.per_module_sum_mse:
                self.per_module_sum_mse[key] = torch.zeros((), device=self.device)
                self.per_module_n_examples[key] = torch.zeros(
                    (), device=self.device, dtype=torch.long
                )
            self.per_module_sum_mse[key] += mse.detach()
            self.per_module_n_examples[key] += n
        return _sum_per_module_mse(per_module)


class StochasticHiddenActsReconLoss(Metric[StochasticHiddenActsReconLossConfig]):
    """Per-module MSE between masked-model and target-model output activations.

    Summed across `ctx.n_mask_samples` stochastic mask draws. `compute()` returns one
    entry per module plus a combined total.
    """

    log_namespace = "loss"
    slow = True
    short_name = "StochHiddenActRecon"

    @override
    def reset(self) -> None:
        self._accum = _HiddenActsAccumulator(self.device)

    @override
    def update(self, ctx: MetricContext) -> Tensor:
        wd = ctx.weight_deltas if ctx.use_delta_component else None
        per_module = _stochastic_hidden_acts_update(
            model=self.model,
            sampling=ctx.sampling,
            n_mask_samples=ctx.n_mask_samples,
            batch=ctx.batch,
            ci=ctx.ci.lower_leaky,
            weight_deltas=wd,
        )
        sum_loss, n = self._accum.accumulate(per_module)
        return sum_loss / n

    @override
    def compute(self) -> MetricResult:
        return compute_per_module_metrics(
            class_name=type(self).__name__,
            per_module_sum_mse=self._accum.per_module_sum_mse,
            per_module_n_examples=self._accum.per_module_n_examples,
        )


def stochastic_hidden_acts_recon_loss(
    model: ComponentModel,
    sampling: SamplingType,
    n_mask_samples: int,
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
) -> Float[Tensor, ""]:
    per_module = _stochastic_hidden_acts_update(
        model=model,
        sampling=sampling,
        n_mask_samples=n_mask_samples,
        batch=batch,
        ci=ci,
        weight_deltas=weight_deltas,
    )
    sum_mse, n = _sum_per_module_mse(per_module)
    return sum_mse / n
