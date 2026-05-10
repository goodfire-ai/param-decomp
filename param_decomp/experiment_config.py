"""Discriminated union over per-experiment configs persisted alongside a PD run.

`experiment_config.yaml` lives in each saved decomposition directory and round-trips
through this union. Adding a new registered experiment requires:

1. Subclass `BaseExperimentConfig` (in `param_decomp/experiments/_base.py`) and
   implement its three methods (`load_target`, `build_dataloaders`, `display_name`).
2. Add the new variant to `ExperimentConfig` here.

The `BaseExperimentConfig` ABC is enforced at class-definition time, and
basedpyright still narrows `ExperimentConfig` exhaustively wherever it is matched.
"""

from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import Field, TypeAdapter

from param_decomp.experiments._base import BaseExperimentConfig as BaseExperimentConfig
from param_decomp.experiments._base import LoadedTarget as LoadedTarget
from param_decomp.experiments.ih.experiment import IHExperimentConfig
from param_decomp.experiments.lm.experiment import LMExperimentConfig
from param_decomp.experiments.resid_mlp.experiment import ResidMLPExperimentConfig
from param_decomp.experiments.tms.experiment import TMSExperimentConfig

ExperimentConfig = Annotated[
    LMExperimentConfig | TMSExperimentConfig | ResidMLPExperimentConfig | IHExperimentConfig,
    Field(discriminator="kind"),
]

_EXPERIMENT_CONFIG_ADAPTER: TypeAdapter[ExperimentConfig] = TypeAdapter(ExperimentConfig)

EXPERIMENT_CONFIG_FILENAME = "experiment_config.yaml"


def parse_experiment_config(data: dict[str, Any]) -> ExperimentConfig:
    return _EXPERIMENT_CONFIG_ADAPTER.validate_python(data)


def load_experiment_config(run_dir: Path) -> ExperimentConfig:
    path = run_dir / EXPERIMENT_CONFIG_FILENAME
    assert path.exists(), f"experiment_config.yaml not found at {path}"
    with open(path) as f:
        return parse_experiment_config(yaml.safe_load(f))
