"""Public PD API.

Core entrypoints:
    - `run_pd`: train a parameter decomposition.
    - `load_pd`: load a saved PD run as a `ComponentModel`.

Core types:
    - `PDConfig`: training/algorithm config.
    - `PDTarget`: target model + run_batch + reconstruction_loss.
    - `ExperimentConfig`: user-authored experiment recipe parsed by a driver.
    - `ExperimentDriver`: Protocol for the open-world experiment extension point.
    - `PDRun`: handle to a saved run (metadata, checkpoint, parsed config).
    - `RunMetadata`: typed envelope for the on-disk ``run_metadata.yaml``.
"""

from param_decomp.configs import PDConfig
from param_decomp.experiments.driver import ExperimentConfig, ExperimentDriver
from param_decomp.load import load_pd
from param_decomp.models.batch_and_loss_fns import PDTarget
from param_decomp.run_metadata import RunMetadata
from param_decomp.run_pd import run_pd
from param_decomp.saved_run import PDRun

__all__ = [
    "ExperimentConfig",
    "ExperimentDriver",
    "PDConfig",
    "PDRun",
    "PDTarget",
    "RunMetadata",
    "load_pd",
    "run_pd",
]
