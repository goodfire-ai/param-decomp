"""The `Run` object: one type for "what a PD run is".

Holds the driver import path plus three configs (``pd``, ``logging``, ``runtime``)
that split algorithm / substrate / observation. Driver-specific subclasses
(``LMRun``, ``TMSRun``, ``ResidMLPRun``) add ``target`` / ``data`` and are pointed
at by each driver's ``config_type``.

Written to ``run_metadata.yaml`` beside the checkpoint, passed to the worker,
and re-read on reload. One type, one shape, everywhere.

A ``mode="wrap"`` model validator on the base ``Run`` dispatches to the right
subclass when validating a dict: it reads ``driver_path``, loads the driver,
and re-routes ``model_validate`` to ``driver.config_type``. Callers therefore
use ``Run.model_validate(data)`` / ``Run.from_file(path)`` (inherited from
``BaseConfig``) and get back the appropriate subtype.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, ValidatorFunctionWrapHandler, model_validator

from param_decomp.base_config import BaseConfig
from param_decomp.configs import LoggingConfig, PDConfig, RuntimeConfig
from param_decomp.driver_path import load_driver
from param_decomp.utils.run_utils import generate_run_id

RUN_METADATA_FILENAME = "run_metadata.yaml"

_BASE_RUN_FIELDS = frozenset({"run_id", "driver_path", "pd", "logging", "runtime"})


class Run(BaseConfig):
    """Top-level run config.

    ``run_id`` identifies the output directory and W&B run. Fresh ``Run``
    objects generate one automatically; YAML / dict inputs that already
    contain a value preserve it.

    ``driver_path`` is the ``module:attr`` import path of the experiment driver
    used to build the target model and dataloaders. ``None`` for notebook
    callers of ``run_pd`` who build their own ``PDTarget``.
    """

    run_id: str = Field(default_factory=lambda: generate_run_id("param_decomp"))
    driver_path: str | None
    pd: PDConfig
    logging: LoggingConfig
    runtime: RuntimeConfig

    @model_validator(mode="wrap")
    @classmethod
    def _dispatch_to_subclass(cls, data: Any, handler: ValidatorFunctionWrapHandler) -> "Run":
        """Route base-``Run`` validation to the driver's ``config_type`` subclass.

        Only triggers when the caller asked for the base class (``cls is Run``)
        and is passing a dict — direct ``LMRun.model_validate(...)`` calls and
        validation of an already-constructed model fall through unchanged.
        """
        if cls is Run and isinstance(data, dict):
            driver_path = data.get("driver_path")
            if driver_path is not None:
                subclass = load_driver(driver_path).config_type
                if subclass is not Run:
                    return subclass.model_validate(data)
            else:
                extras = set(data) - _BASE_RUN_FIELDS
                if extras:
                    raise ValueError(
                        f"Config has extra fields {sorted(extras)} but no driver_path. "
                        "Set `driver_path: module:Driver` so the right Run subclass can "
                        "be selected, or remove the extra fields for a notebook-style run."
                    )
        return handler(data)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.model_dump(mode="json"), f, default_flow_style=False, sort_keys=False)
