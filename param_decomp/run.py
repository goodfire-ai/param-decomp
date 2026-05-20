"""The `RunConfig` object: serializable spec for a driver-mediated PD run.

Pure data. Holds the driver import path, the three determinism-tier configs
(``pd``, ``logging``, ``runtime``), and two driver-private payloads
(``target``, ``data``) as raw dicts. Core never inspects ``target`` / ``data`` —
the driver named by ``driver_path`` validates them via its own pydantic models.

Written to ``run_config.yaml`` beside the checkpoint, passed to the worker,
and re-read on reload. One type, one shape, everywhere.

**Scope**: ``RunConfig`` only exists for driver-mediated runs. Notebook /
script callers do **not** construct a ``RunConfig`` — they call ``optimize``
directly with a ``PDTarget`` + dataloaders + ``PDConfig`` / ``LoggingConfig``
/ ``RuntimeConfig``. See ``param_decomp/run_pd.py``.
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
    """Top-level driver-mediated run config.

    ``run_id`` identifies the output directory and W&B run. Fresh ``RunConfig``
    objects generate one automatically; YAML / dict inputs that already
    contain a value preserve it.

    ``driver_path`` is the ``module:attr`` import path of the experiment driver
    used to build the target model and dataloaders. **Required** — there is no
    "notebook" flavour of ``RunConfig``; notebook callers skip this class
    entirely and call ``optimize`` directly.

    ``target`` and ``data`` are driver-private payloads. Core stores them as
    raw dicts; the driver named by ``driver_path`` validates them with its own
    pydantic models (e.g. ``LMTargetConfig.model_validate(run_cfg.target)``).
    """

    name: str | None = None
    run_id: str = Field(default_factory=lambda: generate_run_id("param_decomp"))
    driver_path: str
    pd: PDConfig
    logging: LoggingConfig
    runtime: RuntimeConfig
    target: dict[str, Any]
    data: dict[str, Any]
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
        path = Path(path)
        assert path.exists(), f"{RUN_CONFIG_FILENAME} not found at {path}"
        with open(path) as f:
            return cls.model_validate(yaml.safe_load(f))

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.model_dump(mode="json"), f, default_flow_style=False, sort_keys=False)
