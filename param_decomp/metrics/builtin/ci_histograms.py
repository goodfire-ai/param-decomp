from collections import defaultdict

import torch
from jaxtyping import Float
from PIL import Image
from torch import Tensor

from param_decomp.metrics.base import MetricConfig
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.registry import register_metric
from param_decomp.models.component_model import ComponentModel
from param_decomp.plotting import plot_ci_values_histograms
from param_decomp.utils.distributed_utils import gather_all_tensors


class CIHistogramsConfig(MetricConfig):
    n_batches_accum: int | None


@register_metric
class CIHistograms:
    section = "figures"
    config_type = CIHistogramsConfig
    slow = True
    short_name = "CIHist"

    def __init__(self, cfg: CIHistogramsConfig, *, model: ComponentModel, device: str) -> None:
        self.cfg = cfg
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.batches_seen = 0
        self.lower_leaky_causal_importances = defaultdict[str, list[Float[Tensor, "... C"]]](list)
        self.pre_sigmoid_causal_importances = defaultdict[str, list[Float[Tensor, "... C"]]](list)

    def update(self, ctx: MetricContext) -> None:
        if self.cfg.n_batches_accum is not None and self.batches_seen >= self.cfg.n_batches_accum:
            return None
        self.batches_seen += 1
        for k, v in ctx.ci.lower_leaky.items():
            self.lower_leaky_causal_importances[k].append(v.detach())
        for k, v in ctx.ci.pre_sigmoid.items():
            self.pre_sigmoid_causal_importances[k].append(v.detach())
        return None

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
