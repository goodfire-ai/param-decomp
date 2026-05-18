"""Metrics package.

Built-in metric modules live under `param_decomp/metrics/builtin/`. Each defines its pydantic
config + a `@register_metric`-decorated Metric class. `discover_metrics()` walks the `builtin/`
subpackage and imports every module, firing all the decorators so `METRIC_REGISTRY` is
populated before `PDConfig.loss_metrics` / `PDConfig.eval_metrics` are parsed.

This package's `__init__.py` deliberately does NOT auto-discover at import time, because
`configs.py` imports `metrics.base` early (which triggers this `__init__.py`), and the metric
modules in turn import `configs.py`. If we ran discovery here we would re-enter `configs.py`
mid-load. Instead, `PDConfig` invokes discovery during validation, after `configs.py` has
finished defining its classes.

Importers who want pure-function helpers (e.g. `faithfulness_loss`) should import them from the
specific submodule, e.g.
`from param_decomp.metrics.builtin.faithfulness_loss import faithfulness_loss`.
"""

import importlib
import pkgutil

from param_decomp.metrics.base import LossMetricConfig, Metric, MetricConfig
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.registry import (
    METRIC_REGISTRY,
    import_metric_module,
    register_metric,
)

from . import builtin

_discovered = False


def discover_metrics() -> None:
    """Import every built-in metric module so its `@register_metric` decorator fires.

    Called by `PDConfig` validation before metric config fields are parsed. Subsequent calls are
    no-ops.
    """
    global _discovered
    if _discovered:
        return
    _discovered = True
    for info in pkgutil.iter_modules(builtin.__path__):
        if info.name.startswith("_"):
            continue
        importlib.import_module(f"{builtin.__name__}.{info.name}")


__all__ = [
    "METRIC_REGISTRY",
    "LossMetricConfig",
    "Metric",
    "MetricConfig",
    "MetricContext",
    "discover_metrics",
    "import_metric_module",
    "register_metric",
]
