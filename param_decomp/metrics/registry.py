"""Auto-registration for metrics.

Metric classes decorated with `@register_metric` are inserted into `METRIC_REGISTRY` keyed by
their `name` ClassVar. The `param_decomp.metrics` package walks its own modules at import time
so all decorators fire once. External users can register their own metrics by listing additional
modules in `PDConfig.metric_modules`; the validator calls `import_metric_module` on each entry
before `loss_metrics`/`eval_metrics` are parsed.
"""

import importlib
import importlib.util
import sys
from pathlib import Path
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


def _looks_like_path(spec: str) -> bool:
    return spec.endswith(".py") or "/" in spec or "\\" in spec


def import_metric_module(spec: str) -> None:
    """Import an external metric module so its `@register_metric` decorators fire.

    `spec` is either a dotted module name (`my_pkg.my_metrics`) or an absolute path to a `.py`
    file (`/home/me/mymetrics.py`). File-path entries must be absolute — relative paths are
    ambiguous against CWD and would silently break under SLURM re-execution. Both branches are
    idempotent: re-importing the same spec is a no-op.
    """
    if _looks_like_path(spec):
        path = Path(spec)
        assert path.is_absolute(), (
            f"metric_modules file paths must be absolute, got {spec!r}. "
            "Use ${PWD}/... in YAML if you need a CWD-relative spec."
        )
        assert path.exists(), f"metric_modules path does not exist: {path}"
        module_name = f"_param_decomp_metric_module_{path.resolve()}"
        if module_name in sys.modules:
            return
        loader_spec = importlib.util.spec_from_file_location(module_name, path)
        assert loader_spec is not None and loader_spec.loader is not None, (
            f"failed to build import spec for {path}"
        )
        module = importlib.util.module_from_spec(loader_spec)
        sys.modules[module_name] = module
        loader_spec.loader.exec_module(module)
    else:
        importlib.import_module(spec)
