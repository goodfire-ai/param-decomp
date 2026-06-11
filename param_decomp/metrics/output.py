"""Normalise each `Metric.compute()` result into a flat key→value map.

Non-tensor map values are passed through for the caller's sink to interpret.
"""

from typing import Any

from torch import Tensor

from param_decomp.metrics.base import Metric

MetricOutType = dict[str, Any]


def _clean_metric_output(
    log_namespace: str,
    metric_name: str,
    computed_raw: Any,
) -> MetricOutType:
    """Normalize one `compute()` return.

    Accepts a scalar tensor (emitted as `{log_namespace}/{metric_name}`) or a dict
    (keys prefixed by `log_namespace`). Non-tensor dict values pass through to the
    concrete sink so core stays logging-backend agnostic.
    """
    computed: MetricOutType = {}
    match computed_raw:
        case Tensor():
            assert computed_raw.numel() == 1, (
                f"Only scalar tensors supported, got shape {computed_raw.shape}"
            )
            computed[f"{log_namespace}/{metric_name}"] = computed_raw.item()
        case dict():
            for k, v in computed_raw.items():
                assert isinstance(k, str), f"Only string keys supported, got {type(k)}"
                if isinstance(v, Tensor):
                    assert v.numel() == 1, f"Only scalar tensors supported, got shape {v.shape}"
                    v = v.item()
                computed[f"{log_namespace}/{k}"] = v
        case _:
            raise ValueError(f"Unsupported type: {type(computed_raw)}")
    return computed


def collect_metric_outputs(active: list[Metric[Any]]) -> MetricOutType:
    """Compute and flatten each metric's output into a single key→value map."""
    outputs: MetricOutType = {}
    for m in active:
        cleaned = _clean_metric_output(
            log_namespace=m.log_namespace,
            metric_name=m.instance_key,
            computed_raw=m.compute(),
        )
        assert not set(outputs) & set(cleaned)
        outputs.update(cleaned)
    return outputs
