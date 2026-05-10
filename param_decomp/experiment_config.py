"""Public helpers for saved PD experiment manifests.

This module intentionally does not import concrete experiments. Saved runs are open-world:
the manifest records the driver import path needed to parse the spec and rebuild runtime
objects when that is possible.
"""

from pathlib import Path
from typing import Any

import yaml

from param_decomp.experiments.driver import (
    EXPERIMENT_CONFIG_FILENAME,
    ExperimentManifest,
    ExperimentSpec,
    parse_driver_spec,
)

ExperimentConfig = ExperimentManifest


def parse_experiment_config(data: dict[str, Any]) -> ExperimentManifest:
    return ExperimentManifest.model_validate(data)


def load_experiment_config(run_dir: Path) -> ExperimentManifest:
    path = run_dir / EXPERIMENT_CONFIG_FILENAME
    assert path.exists(), f"{EXPERIMENT_CONFIG_FILENAME} not found at {path}"
    with open(path) as f:
        return parse_experiment_config(yaml.safe_load(f))


__all__ = [
    "EXPERIMENT_CONFIG_FILENAME",
    "ExperimentConfig",
    "ExperimentManifest",
    "ExperimentSpec",
    "load_experiment_config",
    "parse_driver_spec",
    "parse_experiment_config",
]
