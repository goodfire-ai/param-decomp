"""Public PD API.

Core entrypoints:
    - `run_pd`: train a parameter decomposition.
    - `load_component_model`: load a saved PD run as a `ComponentModel`.

Core types:
    - `PDConfig`: training/algorithm config.
    - `PDTarget`: target model + run_batch + reconstruction_loss.
    - `Run`: one type for "what a PD run is" — driver_path + pd/logging/runtime.
      Driver-specific subclasses (LMRun, TMSRun, ResidMLPRun) add target/data.
    - `ExperimentDriver`: Protocol for the open-world experiment extension point.
    - `PDRun`: handle to a saved run (Run config, checkpoint).
"""

from param_decomp.configs import PDConfig
from param_decomp.experiments.driver import ExperimentDriver
from param_decomp.models.batch_and_loss_fns import PDTarget
from param_decomp.run import Run
from param_decomp.run_pd import run_pd
from param_decomp.saved_run import PDRun, load_component_model

__all__ = [
    "ExperimentDriver",
    "PDConfig",
    "PDRun",
    "PDTarget",
    "Run",
    "load_component_model",
    "run_pd",
]
