"""Auto-registration for metrics.

Metric classes decorated with `@register_metric` are inserted into `METRIC_REGISTRY` keyed by
their class name. `PDConfig` validation calls `discover_metrics()` before parsing
`loss_metrics` / `eval_metrics`, so built-in decorators fire before registry lookup. External
users can register their own metrics by listing additional modules in `PDConfig.metric_modules`;
the validator imports each entry after built-in discovery.
"""

from param_decomp.metrics.base import Metric

METRIC_REGISTRY: dict[str, type[Metric]] = {}


def register_metric[T: type](cls: T) -> T:
    """Insert `cls` into METRIC_REGISTRY by its class name. Returns `cls` unchanged.

    Typed permissively so structural Protocol satisfaction of `Metric` is not statically checked
    at decoration time; checks happen when registry users iterate values typed as `type[Metric]`.
    """
    name = cls.__name__
    assert name not in METRIC_REGISTRY, (
        f"duplicate metric name {name!r}: {METRIC_REGISTRY[name]} vs {cls}"
    )
    METRIC_REGISTRY[name] = cls  # type: ignore[assignment]
    return cls
