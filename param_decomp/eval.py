"""Evaluation output helpers."""

from typing import Any

from PIL import Image
from torch import Tensor
from torch.types import Number
from wandb.plot.custom_chart import CustomChart

from param_decomp.metrics.base import Metric
from param_decomp.torch_helpers import combine_nonoverlapping_dicts

MetricOutType = dict[str, str | Number | Image.Image | CustomChart]


def _clean_metric_output(
    section: str,
    metric_name: str,
    computed_raw: Any,
) -> MetricOutType:
    """Normalize metric.compute() output into a {key: scalar|image|chart} map.

    Accepts either a scalar tensor or a dict of strings to scalars/images/tensors.
    """
    computed: MetricOutType = {}
    match computed_raw:
        case Tensor():
            assert computed_raw.numel() == 1, (
                f"Only scalar tensors supported, got shape {computed_raw.shape}"
            )
            computed[f"{section}/{metric_name}"] = computed_raw.item()
        case dict():
            for k, v in computed_raw.items():
                assert isinstance(k, str), f"Only string keys supported, got {type(k)}"
                assert isinstance(v, str | Number | Image.Image | CustomChart | Tensor), (
                    f"{type(v)} not supported"
                )
                if isinstance(v, Tensor):
                    v = v.item()
                computed[f"{section}/{k}"] = v
        case _:
            raise ValueError(f"Unsupported type: {type(computed_raw)}")
    return computed


def collect_metric_outputs(active: list[Metric[Any]]) -> MetricOutType:
    """Compute and flatten each metric's output into a single key→value map."""
    outputs: MetricOutType = {}
    for m in active:
        cleaned = _clean_metric_output(
            section=m.section,
            metric_name=type(m).__name__,
            computed_raw=m.compute(),
        )
        combine_nonoverlapping_dicts(outputs, cleaned)
    return outputs
