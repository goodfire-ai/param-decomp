from collections import defaultdict
from typing import Any, ClassVar, override

import torch
from jaxtyping import Float
from PIL import Image
from torch import Tensor

from param_decomp.metrics.base import Metric
from param_decomp.models.component_model import CIOutputs, ComponentModel
from param_decomp.plotting import plot_ci_values_histograms
from param_decomp.utils.distributed_utils import gather_all_tensors


class CIHistograms(Metric):
    slow: ClassVar[bool] = True
    metric_section: ClassVar[str] = "figures"

    def __init__(
        self,
        model: ComponentModel,
        n_batches_accum: int | None = None,
    ):
        self.n_batches_accum = n_batches_accum
        self.batches_seen = 0

        self.lower_leaky_causal_importances = defaultdict[str, list[Float[Tensor, "... C"]]](list)
        self.pre_sigmoid_causal_importances = defaultdict[str, list[Float[Tensor, "... C"]]](list)

    @override
    def update(self, *, ci: CIOutputs, **_: Any) -> None:
        if self.n_batches_accum is not None and self.batches_seen >= self.n_batches_accum:
            return

        self.batches_seen += 1

        for k, v in ci.lower_leaky.items():
            self.lower_leaky_causal_importances[k].append(v)

        for k, v in ci.pre_sigmoid.items():
            self.pre_sigmoid_causal_importances[k].append(v)

    @override
    def compute(self) -> dict[str, Image.Image]:
        if self.batches_seen == 0:
            raise RuntimeError("No batches seen yet")

        lower_leaky_cis: dict[str, Float[Tensor, "... C"]] = {}
        for module_name, ci_list in self.lower_leaky_causal_importances.items():
            lower_leaky_cis[module_name] = torch.cat(
                gather_all_tensors(torch.cat(ci_list, dim=0)), dim=0
            )

        pre_sigmoid_cis: dict[str, Float[Tensor, "... C"]] = {}
        for module_name, ci_list in self.pre_sigmoid_causal_importances.items():
            pre_sigmoid_cis[module_name] = torch.cat(
                gather_all_tensors(torch.cat(ci_list, dim=0)), dim=0
            )

        lower_leaky_fig = plot_ci_values_histograms(causal_importances=lower_leaky_cis)
        pre_sigmoid_fig = plot_ci_values_histograms(causal_importances=pre_sigmoid_cis)

        return {
            "causal_importance_values": lower_leaky_fig,
            "causal_importance_values_pre_sigmoid": pre_sigmoid_fig,
        }
