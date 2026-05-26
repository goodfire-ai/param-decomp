from typing import Literal, override

import torch
from torch import Tensor
from torch.distributed import ReduceOp

from param_decomp.base_config import BaseConfig
from param_decomp.distributed import all_reduce
from param_decomp.metrics.base import Metric, MetricResult
from param_decomp.metrics.context import MetricContext
from param_decomp_lab.eval_metrics.plotting import plot_mean_component_cis_both_scales


class CIMeanPerComponentConfig(BaseConfig):
    type: Literal["CIMeanPerComponent"] = "CIMeanPerComponent"


class CIMeanPerComponent(Metric[CIMeanPerComponentConfig]):
    """Per-layer plot of mean CI per component, sorted descending (linear + log y)."""

    log_namespace = "figures"
    slow = True
    short_name = "CIMeanPerComp"

    @override
    def reset(self) -> None:
        self.component_ci_sums: dict[str, Tensor] = {
            module_name: torch.zeros(self.model.module_to_c[module_name], device=self.device)
            for module_name in self.model.components
        }
        self.examples_seen: dict[str, Tensor] = {
            module_name: torch.zeros((), device=self.device, dtype=torch.long)
            for module_name in self.model.components
        }

    @override
    def update(self, ctx: MetricContext) -> None:
        for module_name, ci_vals in ctx.ci.lower_leaky.items():
            n_leading_dims = ci_vals.ndim - 1
            n_examples = ci_vals.shape[:n_leading_dims].numel()
            self.examples_seen[module_name] += n_examples
            leading_dim_idxs = tuple(range(n_leading_dims))
            self.component_ci_sums[module_name] += ci_vals.detach().sum(dim=leading_dim_idxs)
        return None

    @override
    def compute(self) -> MetricResult:
        mean_component_cis = {}
        for module_name in self.model.components:
            summed_ci = all_reduce(self.component_ci_sums[module_name], op=ReduceOp.SUM)
            examples_reduced = all_reduce(self.examples_seen[module_name], op=ReduceOp.SUM)
            mean_component_cis[module_name] = summed_ci / examples_reduced
        img_linear, img_log = plot_mean_component_cis_both_scales(mean_component_cis)
        return {
            "ci_mean_per_component": img_linear,
            "ci_mean_per_component_log": img_log,
        }
