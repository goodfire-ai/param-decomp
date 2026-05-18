import re
from collections import defaultdict

import torch
import wandb.plot
from torch.distributed import ReduceOp

from param_decomp.metrics.base import MetricConfig
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.registry import register_metric
from param_decomp.models.component_model import ComponentModel
from param_decomp.utils.component_utils import calc_ci_l_zero
from param_decomp.utils.distributed_utils import all_reduce


class CI_L0Config(MetricConfig):
    groups: dict[str, list[str]] | None


@register_metric
class CI_L0:
    """L0 metric for CI values."""

    section = "l0"
    config_type = CI_L0Config
    short_name = "CI_L0"

    def __init__(self, cfg: CI_L0Config, *, model: ComponentModel, device: str) -> None:
        self.cfg = cfg
        self.device = device
        self.model = model
        self.reset()

    def reset(self) -> None:
        self.l0_values: defaultdict[str, list[float]] = defaultdict(list)
        self.threshold: float | None = None

    def update(self, ctx: MetricContext) -> None:
        threshold = ctx.config.ci_alive_threshold
        self.threshold = threshold
        group_sums: dict[str, float] = defaultdict(float) if self.cfg.groups else {}
        for layer_name, layer_ci in ctx.ci.lower_leaky.items():
            l0_val = calc_ci_l_zero(layer_ci, threshold)
            self.l0_values[layer_name].append(l0_val)
            if self.cfg.groups:
                for group_name, patterns in self.cfg.groups.items():
                    for pattern in patterns:
                        if re.match(pattern.replace("*", ".*"), layer_name):
                            group_sums[group_name] += l0_val
                            break
        for group_name, group_sum in group_sums.items():
            self.l0_values[group_name].append(group_sum)
        return None

    def compute(self) -> dict[str, float | wandb.plot.CustomChart]:
        assert self.threshold is not None, "compute called before any update"
        out: dict[str, float | wandb.plot.CustomChart] = {}
        table_data = []
        for key, l0s in self.l0_values.items():
            global_sum = all_reduce(torch.tensor(l0s, device=self.device).sum(), op=ReduceOp.SUM)
            global_count = all_reduce(torch.tensor(len(l0s), device=self.device), op=ReduceOp.SUM)
            avg_l0 = (global_sum / global_count).item()
            out[f"{self.threshold}_{key}"] = avg_l0
            table_data.append((key, avg_l0))
        out["bar_chart"] = wandb.plot.bar(
            table=wandb.Table(columns=["layer", "l0"], data=table_data),
            label="layer",
            value="l0",
            title=f"L0_{self.threshold}",
        )
        return out
