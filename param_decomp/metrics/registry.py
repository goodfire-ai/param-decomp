"""Auto-registration for metrics.

Metric classes decorated with `@register_metric` are inserted into `METRIC_REGISTRY` keyed by
their `name` ClassVar. The `param_decomp.metrics` package walks its own modules at import time
so all decorators fire once.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from param_decomp.metrics.base import Metric

METRIC_REGISTRY: "dict[str, type[Metric]]" = {}


def register_metric[T: type](cls: T) -> T:
    """Insert `cls` into METRIC_REGISTRY by its `name` ClassVar. Returns `cls` unchanged.

    Typed permissively (any class with a `name: str` ClassVar) so structural Protocol satisfaction
    of `Metric` is not statically checked at decoration time — checks happen when registry users
    iterate values typed as `type[Metric]`.
    """
    name = cls.name  # type: ignore[attr-defined]
    assert name not in METRIC_REGISTRY, (
        f"duplicate metric name {name!r}: {METRIC_REGISTRY[name]} vs {cls}"
    )
    METRIC_REGISTRY[name] = cls  # type: ignore[assignment]
    return cls
