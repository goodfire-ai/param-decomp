"""The `Run` object: one type for "what a PD run is".

Holds the driver import path plus the three determinism-tier configs (``pd``,
``logging``, ``runtime``). Driver-specific subclasses (``LMRun``, ``TMSRun``,
``ResidMLPRun``) add ``target`` / ``data`` and are pointed at by each driver's
``config_type``.

Written to ``run_metadata.yaml`` beside the checkpoint, passed to the worker,
and re-read on reload. One type, one shape, everywhere.
"""

import importlib
from pathlib import Path
from typing import Any, override

import yaml

from param_decomp.base_config import BaseConfig
from param_decomp.configs import LoggingConfig, PDConfig, RuntimeConfig

RUN_METADATA_FILENAME = "run_metadata.yaml"


class Run(BaseConfig):
    """Top-level run config.

    The three nested configs form a determinism ladder:

    1. Same ``pd`` + same ``runtime`` → bit-identical trained weights.
    2. Same ``pd``, different ``runtime`` → same algorithm, weights differ only
       via numerical effects (precision, device).
    3. Same ``pd`` + same ``runtime``, different ``logging`` → bit-identical
       weights; only what was observed differs. ``logging`` carries
       ``wandb_project`` / ``wandb_run_name`` / ``view_meta``.

    ``driver_path`` is the ``module:attr`` import path of the experiment driver
    used to build the target model and dataloaders. ``None`` for notebook
    callers of ``run_pd`` who build their own ``PDTarget``.
    """

    driver_path: str | None
    pd: PDConfig
    logging: LoggingConfig
    runtime: RuntimeConfig

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Run":
        """Parse a dict (e.g. from YAML) into the right `Run` subclass.

        Looks up ``driver_path`` → driver → ``config_type`` and validates the
        dict against that subclass. When ``driver_path`` is ``None``, validates
        as a bare ``Run``. Callers that need the concrete subtype narrow with
        ``isinstance(run, driver.config_type)``.
        """
        driver_path = data.get("driver_path")
        if driver_path is None:
            return cls.model_validate(data)
        return _load_config_type(driver_path).model_validate(data)

    @classmethod
    @override
    def from_file(cls, path: Path | str) -> "Run":
        path = Path(path)
        assert path.exists(), f"{RUN_METADATA_FILENAME} not found at {path}"
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.model_dump(mode="json"), f, default_flow_style=False, sort_keys=False)


def _load_config_type(driver_path: str) -> type[Run]:
    """Resolve a ``module:attr`` driver path to its ``config_type`` (a ``Run`` subclass).

    Inlined here (rather than reusing ``experiments.driver.load_driver``) to avoid a
    static import cycle between this module and ``experiments.driver``.
    """
    module_path, sep, attr = driver_path.partition(":")
    if sep == "":
        raise ValueError(f"Driver path must be of the form 'module:attr', got {driver_path!r}")
    driver = getattr(importlib.import_module(module_path), attr)
    if isinstance(driver, type):
        driver = driver()
    return driver.config_type
