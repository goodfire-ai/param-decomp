"""The `RunConfig` object: serializable spec for a PD run.

Holds the driver import path plus the three determinism-tier configs (``pd``,
``logging``, ``runtime``). Driver-specific subclasses (``LMRunConfig``, ``TMSRunConfig``,
``ResidMLPRunConfig``) add ``target`` / ``data`` and are pointed at by each driver's
``config_type``.

Written to ``run_config.yaml`` beside the checkpoint, passed to the worker,
and re-read on reload. One type, one shape, everywhere.

For loading from YAML/dict, use ``resolve_run(data)`` from ``param_decomp.compose``
which returns both the config and driver in one call.

Notebook users who build their own target model and dataloaders should call
``optimize()`` directly and skip RunConfig entirely.
"""

from pathlib import Path
from typing import Any, Self, override

import yaml
from pydantic import Field, model_validator

from param_decomp.base_config import BaseConfig
from param_decomp.configs import LoggingConfig, PDConfig, RuntimeConfig
from param_decomp.utils.run_utils import generate_run_id

RUN_CONFIG_FILENAME = "run_config.yaml"


class RunConfig(BaseConfig):
    """Top-level run config.

    ``run_id`` identifies the output directory and W&B run. Fresh ``RunConfig``
    objects generate one automatically; YAML / dict inputs that already
    contain a value preserve it.

    ``driver_path`` is the ``module:attr`` import path of the experiment driver
    used to build the target model and dataloaders.
    """

    name: str | None = None
    run_id: str = Field(default_factory=lambda: generate_run_id("param_decomp"))
    driver_path: str
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
    @override
    def from_file(cls, path: Path | str) -> "RunConfig":
        """Load from YAML.

        Returns the base ``RunConfig`` type. For driver-specific subclasses
        (and to get the driver), use ``resolve_run()`` from ``param_decomp.compose``
        which returns both the config and driver.
        """
        path = Path(path)
        assert path.exists(), f"{RUN_CONFIG_FILENAME} not found at {path}"
        with open(path) as f:
            return cls.model_validate(yaml.safe_load(f))

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.model_dump(mode="json"), f, default_flow_style=False, sort_keys=False)
