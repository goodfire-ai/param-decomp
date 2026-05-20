"""Built-in metrics that ship with PD.

Each module here defines a pydantic config + `@register_metric`-decorated Metric class.
`param_decomp.metrics.discover_metrics()` walks this package and imports every module to
populate `METRIC_REGISTRY`.
"""
