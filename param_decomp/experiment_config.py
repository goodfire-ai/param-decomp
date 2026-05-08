"""Discriminated union over per-experiment configs persisted alongside a PD run.

`experiment_config.yaml` lives in each saved decomposition directory and round-trips
through this union. Adding a new registered experiment requires a single edit here:
extending `ExperimentConfig` and `display_name` with the new variant. basedpyright
then exhaustively flags every dispatch site that needs a new `case`.
"""

from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import Field, TypeAdapter

from param_decomp.experiments.ih.configs import IHExperimentConfig
from param_decomp.experiments.lm.configs import LMExperimentConfig
from param_decomp.experiments.resid_mlp.configs import ResidMLPExperimentConfig
from param_decomp.experiments.tms.configs import TMSExperimentConfig

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


def display_name(exp: ExperimentConfig) -> str:
    match exp:
        case LMExperimentConfig(target=t, data=d):
            return f"LM: {t.model_class.rsplit('.', 1)[-1]} on {d.dataset_name}"
        case TMSExperimentConfig(target=t):
            return f"TMS: {t.run_path}"
        case ResidMLPExperimentConfig(target=t):
            return f"ResidMLP: {t.run_path}"
        case IHExperimentConfig(target=t):
            return f"IH: {t.run_path}"
