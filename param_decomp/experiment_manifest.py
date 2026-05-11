"""Public helpers for saved PD experiment manifests.

This module intentionally does not import concrete experiments. Saved runs are open-world:
the manifest records the driver import path needed to parse the experiment config and rebuild
runtime objects when that is possible.
"""

from pathlib import Path
from typing import Any

import yaml

from param_decomp.experiments.constants import EXPERIMENT_MANIFEST_FILENAME
from param_decomp.experiments.driver import (
    ExperimentConfig,
    ExperimentManifest,
    parse_manifest_experiment_config,
)


def parse_experiment_manifest(data: dict[str, Any]) -> ExperimentManifest:
    return ExperimentManifest.model_validate(data)


def load_experiment_manifest(run_dir: Path) -> ExperimentManifest:
    path = run_dir / EXPERIMENT_MANIFEST_FILENAME
    assert path.exists(), f"{EXPERIMENT_MANIFEST_FILENAME} not found at {path}"
    with open(path) as f:
        return parse_experiment_manifest(yaml.safe_load(f))


__all__ = [
    "EXPERIMENT_MANIFEST_FILENAME",
    "ExperimentConfig",
    "ExperimentManifest",
    "load_experiment_manifest",
    "parse_experiment_manifest",
    "parse_manifest_experiment_config",
]
