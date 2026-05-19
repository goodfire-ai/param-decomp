"""Public PD API.

Core entrypoints:
    - `run_pd`: train a parameter decomposition.
    - `load_component_model`: load a saved PD run as a `ComponentModel`.

Core types:
    - `PDConfig`: training/algorithm config.
    - `PDTarget`: target model + run_batch + reconstruction_loss.
    - `ExperimentConfig`: user-authored experiment recipe parsed by a driver.
    - `ExperimentDriver`: Protocol for the open-world experiment extension point.
    - `PDRun`: handle to a saved run (spec, checkpoint, parsed config).
    - `RunSpec`: typed envelope for the on-disk ``run_metadata.yaml``.
"""

from param_decomp.configs import PDConfig
from param_decomp.experiments.driver import ExperimentConfig, ExperimentDriver
from param_decomp.models.batch_and_loss_fns import PDTarget
from param_decomp.run_pd import run_pd
from param_decomp.run_spec import RunSpec
from param_decomp.saved_run import PDRun, load_component_model

__all__ = [
    "ExperimentConfig",
    "ExperimentDriver",
    "PDConfig",
    "PDRun",
    "PDTarget",
    "RunSpec",
    "load_component_model",
    "run_pd",
]
