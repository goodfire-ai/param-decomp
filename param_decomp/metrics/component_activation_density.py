from typing import Any, ClassVar, override

import torch
from einops import reduce
from jaxtyping import Int
from PIL import Image
from torch import Tensor
from torch.distributed import ReduceOp

from param_decomp.metrics.base import Metric
from param_decomp.models.component_model import CIOutputs, ComponentModel
from param_decomp.plotting import plot_component_activation_density
from param_decomp.utils.distributed_utils import all_reduce


class ComponentActivationDensity(Metric):
    """Activation density for each component."""

    slow: ClassVar[bool] = True
    metric_section: ClassVar[str] = "figures"

    def __init__(self, model: ComponentModel, device: str, ci_alive_threshold: float) -> None:
        self.model = model
        self.ci_alive_threshold = ci_alive_threshold

        self.n_examples: Int[Tensor, ""] = torch.tensor(0.0, device=device)
        self.component_activation_counts: dict[str, Tensor] = {
            module_name: torch.zeros(model.module_to_c[module_name], device=device)
            for module_name in model.components
        }

    @override
    def update(self, *, ci: CIOutputs, **_: Any) -> None:
        n_examples_this_batch = next(iter(ci.lower_leaky.values())).shape[:-1].numel()
        self.n_examples += n_examples_this_batch

        for module_name, ci_vals in ci.lower_leaky.items():
            active_components = ci_vals > self.ci_alive_threshold
            n_activations_per_component = reduce(active_components, "... C -> C", "sum")
            self.component_activation_counts[module_name] += n_activations_per_component

    @override
    def compute(self) -> dict[str, Image.Image]:
        activation_densities = {}
        n_examples_reduced = all_reduce(self.n_examples, op=ReduceOp.SUM)
        for module_name in self.model.components:
            counts_reduced = all_reduce(
                self.component_activation_counts[module_name], op=ReduceOp.SUM
            )
            activation_densities[module_name] = counts_reduced / n_examples_reduced

        fig = plot_component_activation_density(activation_densities)
        return {"component_activation_density": fig}
