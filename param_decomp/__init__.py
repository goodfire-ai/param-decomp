"""Public PD API.

Core entrypoints:
    - `run_pd`: train a parameter decomposition.
    - `load_pd`: load a saved PD run as a `ComponentModel`.

Core types:
    - `PDConfig`: training/algorithm config.
    - `PDTarget`: target model + run_batch + reconstruction_loss.
    - `ExperimentConfig`: user-authored experiment recipe parsed by a driver.
    - `ExperimentDriver`: Protocol for the open-world experiment extension point.
    - `PDRun`: handle to a saved run (manifest, checkpoint, parsed config).
"""

from param_decomp.configs import PDConfig
from param_decomp.experiments.driver import ExperimentConfig, ExperimentDriver
from param_decomp.load import load_pd
from param_decomp.models.batch_and_loss_fns import PDTarget
from param_decomp.pd_run import PDRun
from param_decomp.run_param_decomp import run_pd

__all__ = [
    "ExperimentConfig",
    "ExperimentDriver",
    "PDConfig",
    "PDRun",
    "PDTarget",
    "load_pd",
    "run_pd",
]
