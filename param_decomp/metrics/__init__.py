"""Metrics package.

Loss metrics ship in this package and form a pydantic discriminated union over their
`type` literals. `PDConfig.loss_metrics` validates each entry into the right
`*LossMetricConfig` subclass directly — there is no class-name registry for validation.
The runtime `type` → `Metric` class dispatch (used by `optimize()` for instantiation)
lives in `param_decomp.metrics.loss_metrics.LOSS_METRIC_CLASSES`.

Eval metrics are caller-supplied: experiments instantiate `Metric` objects directly and pass
them to `optimize(eval_metrics=...)`.
"""

from param_decomp.metrics.base import LossMetricConfig, Metric
from param_decomp.metrics.context import MetricContext

__all__ = [
    "LossMetricConfig",
    "Metric",
    "MetricContext",
]
