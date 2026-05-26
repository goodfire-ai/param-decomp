from typing import Literal, override

from param_decomp.base_config import BaseConfig
from param_decomp.masks import make_mask_infos
from param_decomp.metrics.base import Metric, MetricResult
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.stochastic_hidden_acts_recon import (
    _HiddenActsAccumulator,
    calc_hidden_acts_mse,
    compute_per_module_metrics,
)


class CIHiddenActsReconLossConfig(BaseConfig):
    type: Literal["CIHiddenActsReconLoss"] = "CIHiddenActsReconLoss"


class CIHiddenActsReconLoss(Metric[CIHiddenActsReconLossConfig]):
    """Per-module MSE between target and CI-masked component hidden activations."""

    log_namespace = "loss"
    slow = True
    short_name = "CIHiddenActRecon"

    @override
    def reset(self) -> None:
        self._accum = _HiddenActsAccumulator(self.device)

    @override
    def update(self, ctx: MetricContext) -> None:
        target_acts = self.model(ctx.batch, cache_type="output").cache
        mask_infos = make_mask_infos(ctx.ci.lower_leaky, weight_deltas_and_masks=None)
        per_module, _ = calc_hidden_acts_mse(
            model=self.model, batch=ctx.batch, mask_infos=mask_infos, target_acts=target_acts
        )
        self._accum.accumulate(per_module)
        return None

    @override
    def compute(self) -> MetricResult:
        return compute_per_module_metrics(
            class_name=type(self).__name__,
            per_module_sum_mse=self._accum.per_module_sum_mse,
            per_module_n_examples=self._accum.per_module_n_examples,
        )
