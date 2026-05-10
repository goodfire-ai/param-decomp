"""Public PD API.

Core entrypoints:
    - `run_pd`: train a parameter decomposition.
    - `load_pd`: load a saved PD run as a `ComponentModel`.

Core types:
    - `PDConfig`: training/algorithm config.
    - `PDTarget`: target model + run_batch + reconstruction_loss.
    - `PreparedExperiment`: target + dataloaders + manifest produced by an experiment driver.
    - `PDRunInfo`: handle to a saved run (config, checkpoint path, experiment config).
"""

from param_decomp.configs import PDConfig
from param_decomp.experiments.driver import (
    ExperimentDriver,
    ExperimentManifest,
    PreparedExperiment,
    RunArtifact,
)
from param_decomp.load import load_pd
from param_decomp.models.batch_and_loss_fns import PDTarget
from param_decomp.models.component_model import PDRunInfo
from param_decomp.run_param_decomp import run_pd

__all__ = [
    "PDConfig",
    "ExperimentDriver",
    "ExperimentManifest",
    "PDRunInfo",
    "PDTarget",
    "PreparedExperiment",
    "RunArtifact",
    "load_pd",
    "run_pd",
]
