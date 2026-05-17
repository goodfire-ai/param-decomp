"""Metrics package.

Each metric module under `param_decomp/metrics/` defines its pydantic config + a
`@register_metric`-decorated Metric class. `discover_metrics()` (called from `configs.py` after
its own class definitions are complete) walks this package and imports every metric module,
firing all the decorators so `METRIC_REGISTRY` is populated before any `PDConfig` validates.

This package's `__init__.py` deliberately does NOT auto-discover at import time, because
`configs.py` imports `metrics.base` early (which triggers this `__init__.py`), and the metric
modules in turn import `configs.py`. If we ran discovery here we would re-enter `configs.py` mid-
load. Instead, discovery is invoked explicitly from the bottom of `configs.py`.

Importers who want pure-function helpers (e.g. `faithfulness_loss`) should import them from the
specific submodule, e.g. `from param_decomp.metrics.faithfulness_loss import faithfulness_loss`.
"""

import importlib
import pkgutil

from param_decomp.metrics.base import LossMetricConfig, Metric, MetricConfig
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.registry import METRIC_REGISTRY, register_metric

_INFRASTRUCTURE = {"base", "context", "registry", "pgd_utils"}
_discovered = False


def discover_metrics() -> None:
    """Import every metric module so its `@register_metric` decorator fires.

    Called from the bottom of `configs.py` after class definitions complete. Subsequent calls are
    no-ops.
    """
    global _discovered
    if _discovered:
        return
    _discovered = True
    for info in pkgutil.iter_modules(__path__):
        if info.name in _INFRASTRUCTURE or info.name.startswith("_"):
            continue
        importlib.import_module(f"{__name__}.{info.name}")


__all__ = [
    "METRIC_REGISTRY",
    "LossMetricConfig",
    "Metric",
    "MetricConfig",
    "MetricContext",
    "discover_metrics",
    "register_metric",
]
