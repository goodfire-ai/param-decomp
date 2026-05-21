"""Metrics package.

Loss metrics ship in this package and are referenced from `PDConfig.loss_metrics` by class
name. The mapping from class name to class is `param_decomp.metrics.loss_metrics.LOSS_METRICS`
— there is no auto-registration; new loss metrics are added by appending them there.

Eval metrics are caller-supplied: experiments instantiate `Metric` objects directly and pass
them to `optimize(eval_metrics=...)`.

`LOSS_METRICS` lives in its own submodule to keep this package's top-level imports cheap and
avoid circular imports through `param_decomp.configs`.
"""

from param_decomp.metrics.base import LossMetricConfig, Metric
from param_decomp.metrics.context import MetricContext

__all__ = [
    "LossMetricConfig",
    "Metric",
    "MetricContext",
]
