import re
from collections import defaultdict
from typing import Any, ClassVar, override

import torch
import wandb
from torch.distributed import ReduceOp

from param_decomp.metrics.base import Metric
from param_decomp.models.component_model import CIOutputs, ComponentModel
from param_decomp.utils.component_utils import calc_ci_l_zero
from param_decomp.utils.distributed_utils import all_reduce


class CI_L0(Metric):
    """L0 metric for CI values.

    NOTE: Assumes all batches and sequences are the same size.
    """

    metric_section: ClassVar[str] = "l0"

    def __init__(
        self,
        model: ComponentModel,
        device: str,
        ci_alive_threshold: float,
        groups: dict[str, list[str]] | None = None,
    ) -> None:
        self.l0_threshold = ci_alive_threshold
        self.groups = groups
        self.device = device

        all_keys = model.target_module_paths.copy()
        if groups:
            all_keys += list(groups.keys())

        self.l0_values = defaultdict[str, list[float]](list)

    @override
    def update(self, *, ci: CIOutputs, **_: Any) -> None:
        group_sums = defaultdict(float) if self.groups else {}
        for layer_name, layer_ci in ci.lower_leaky.items():
            l0_val = calc_ci_l_zero(layer_ci, self.l0_threshold)
            self.l0_values[layer_name].append(l0_val)

            if self.groups:
                for group_name, patterns in self.groups.items():
                    for pattern in patterns:
                        if re.match(pattern.replace("*", ".*"), layer_name):
                            group_sums[group_name] += l0_val
                            break
        for group_name, group_sum in group_sums.items():
            self.l0_values[group_name].append(group_sum)

    @override
    def compute(self) -> dict[str, float | wandb.plot.CustomChart]:
        out = {}
        table_data = []

        for key, l0s in self.l0_values.items():
            global_sum = all_reduce(torch.tensor(l0s, device=self.device).sum(), op=ReduceOp.SUM)
            global_count = all_reduce(torch.tensor(len(l0s), device=self.device), op=ReduceOp.SUM)
            avg_l0 = (global_sum / global_count).item()
            out[f"{self.l0_threshold}_{key}"] = avg_l0
            table_data.append((key, avg_l0))
        bar_chart = wandb.plot.bar(
            table=wandb.Table(columns=["layer", "l0"], data=table_data),
            label="layer",
            value="l0",
            title=f"L0_{self.l0_threshold}",
        )
        out["bar_chart"] = bar_chart
        return out
