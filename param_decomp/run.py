"""The `RunConfig` object: serializable spec for a PD run.

Holds the driver import path plus the three determinism-tier configs (``pd``,
``logging``, ``runtime``). Driver-specific subclasses (``LMRunConfig``, ``TMSRunConfig``,
``ResidMLPRunConfig``) add ``target`` / ``data`` and are pointed at by each driver's
``config_type``.

Written to ``run_config.yaml`` beside the checkpoint, passed to the worker,
and re-read on reload. One type, one shape, everywhere.

``RunConfig.from_dict(...)`` dispatches to the right subclass: it reads
``driver_path``, loads the driver, and routes ``model_validate`` to
``driver.config_type``. Callers use ``RunConfig.from_dict(data)`` /
``RunConfig.from_file(path)`` and get back the appropriate subtype.
"""

from pathlib import Path
from typing import Any, Self, override

import yaml
from pydantic import Field, model_validator

from param_decomp.base_config import BaseConfig
from param_decomp.configs import LoggingConfig, PDConfig, RuntimeConfig
from param_decomp.driver_path import load_driver
from param_decomp.utils.run_utils import generate_run_id

RUN_CONFIG_FILENAME = "run_config.yaml"


class RunConfig(BaseConfig):
    """Top-level run config.

    ``run_id`` identifies the output directory and W&B run. Fresh ``RunConfig``
    objects generate one automatically; YAML / dict inputs that already
    contain a value preserve it.

    ``driver_path`` is the ``module:attr`` import path of the experiment driver
    used to build the target model and dataloaders. ``None`` for notebook
    callers of ``run_pd`` who build their own ``PDTarget``.
    """

    name: str | None = None
    run_id: str = Field(default_factory=lambda: generate_run_id("param_decomp"))
    driver_path: str | None
    pd: PDConfig
    logging: LoggingConfig
    runtime: RuntimeConfig
    view_meta: dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form labels for downstream grouping/coloring/reports (e.g. "
        "`{'lr_ratio': 0.1, 'size': 'medium'}`). Populated by sweep generators; surfaced "
        "to W&B under a `view_meta/` prefix.",
    )

    @model_validator(mode="after")
    def validate_metric_overlap(self) -> Self:
        overlap = sorted(set(self.pd.loss_metrics) & set(self.logging.eval_metrics))
        assert not overlap, (
            f"The same metric was set under both pd.loss_metrics and logging.eval_metrics: "
            f"{overlap}. Loss metrics are automatically evaluated; remove the "
            "logging.eval_metrics entry, or move it out of pd.loss_metrics if you want eval-only."
        )
        return self

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunConfig":
        """Parse a dict (e.g. from YAML) into the right `RunConfig` subclass.

        Dispatch is driven entirely by ``data["driver_path"]``: when set, the
        driver's ``config_type`` is used; when ``None``, the bare ``RunConfig``
        is used. The caller class (``cls``) is intentionally not consulted —
        every caller of this method uses ``RunConfig.from_dict(...)`` and the
        right subtype falls out of the driver. Callers that need the concrete
        subtype narrow with ``isinstance(run_cfg, driver.config_type)``.
        """
        driver_path = data.get("driver_path")
        if driver_path is None:
            return RunConfig.model_validate(data)
        return load_driver(driver_path).config_type.model_validate(data)

    @classmethod
    @override
    def from_file(cls, path: Path | str) -> "RunConfig":
        path = Path(path)
        assert path.exists(), f"{RUN_CONFIG_FILENAME} not found at {path}"
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.model_dump(mode="json"), f, default_flow_style=False, sort_keys=False)
