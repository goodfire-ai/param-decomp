from typing import override

import torch
from einops import reduce
from torch import Tensor
from torch.distributed import ReduceOp

from param_decomp.metrics.base import Metric, MetricConfig, MetricResult
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.registry import register_metric
from param_decomp.plotting import plot_component_activation_density
from param_decomp.utils.distributed_utils import all_reduce


class ComponentActivationDensityConfig(MetricConfig):
    ci_alive_threshold: float = 0.0


@register_metric
class ComponentActivationDensity(Metric[ComponentActivationDensityConfig]):
    """Activation density for each component."""

    section = "figures"
    config_type = ComponentActivationDensityConfig
    slow = True
    short_name = "CompActDens"

    @override
    def reset(self) -> None:
        self.n_examples: Tensor = torch.zeros((), device=self.device, dtype=torch.long)
        self.component_activation_counts: dict[str, Tensor] = {
            module_name: torch.zeros(self.model.module_to_c[module_name], device=self.device)
            for module_name in self.model.components
        }

    @override
    def update(self, ctx: MetricContext) -> None:
        n_examples_this_batch = next(iter(ctx.ci.lower_leaky.values())).shape[:-1].numel()
        self.n_examples += n_examples_this_batch
        for module_name, ci_vals in ctx.ci.lower_leaky.items():
            active_components = ci_vals > self.cfg.ci_alive_threshold
            n_activations_per_component = reduce(active_components, "... C -> C", "sum")
            self.component_activation_counts[module_name] += n_activations_per_component
        return None

    @override
    def compute(self) -> MetricResult:
        activation_densities = {}
        n_examples_reduced = all_reduce(self.n_examples, op=ReduceOp.SUM)
        for module_name in self.model.components:
            counts_reduced = all_reduce(
                self.component_activation_counts[module_name], op=ReduceOp.SUM
            )
            activation_densities[module_name] = counts_reduced / n_examples_reduced
        fig = plot_component_activation_density(activation_densities)
        return {"component_activation_density": fig}
