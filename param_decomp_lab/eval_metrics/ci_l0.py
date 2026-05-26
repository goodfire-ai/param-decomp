import re
from collections import defaultdict
from typing import Literal, override

import torch
import wandb.plot
from jaxtyping import Float
from torch import Tensor
from torch.distributed import ReduceOp

from param_decomp.base_config import BaseConfig
from param_decomp.distributed import all_reduce
from param_decomp.metrics.base import Metric, MetricResult
from param_decomp.metrics.context import MetricContext


def calc_ci_l_zero(ci: Float[Tensor, "... C"], threshold: float) -> float:
    """Mean number of CI entries above `threshold` per example."""
    return (ci > threshold).float().sum(-1).mean().item()


class CI_L0Config(BaseConfig):
    """`groups` maps `{group_name: [fnmatch-style layer pattern, ...]}`.

    Matching layers' L0s are summed into the group and logged under the group's name.
    """

    type: Literal["CI_L0"] = "CI_L0"
    groups: dict[str, list[str]] | None
    ci_alive_threshold: float = 0.0


class CI_L0(Metric[CI_L0Config]):
    """Mean L0 of CI values per layer, with optional grouped aggregates."""

    log_namespace = "l0"
    short_name = "CI_L0"

    @override
    def reset(self) -> None:
        self.l0_values: defaultdict[str, list[float]] = defaultdict(list)

    @override
    def update(self, ctx: MetricContext) -> None:
        group_sums: dict[str, float] = defaultdict(float) if self.cfg.groups else {}
        for layer_name, layer_ci in ctx.ci.lower_leaky.items():
            l0_val = calc_ci_l_zero(layer_ci, self.cfg.ci_alive_threshold)
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

    @override
    def compute(self) -> MetricResult:
        threshold = self.cfg.ci_alive_threshold
        out: dict[str, float | wandb.plot.CustomChart] = {}
        table_data = []
        for key, l0s in self.l0_values.items():
            global_sum = all_reduce(torch.tensor(l0s, device=self.device).sum(), op=ReduceOp.SUM)
            global_count = all_reduce(torch.tensor(len(l0s), device=self.device), op=ReduceOp.SUM)
            avg_l0 = (global_sum / global_count).item()
            out[f"{threshold}_{key}"] = avg_l0
            table_data.append((key, avg_l0))
        out["bar_chart"] = wandb.plot.bar(
            table=wandb.Table(columns=["layer", "l0"], data=table_data),
            label="layer",
            value="l0",
            title=f"L0_{threshold}",
        )
        return out
