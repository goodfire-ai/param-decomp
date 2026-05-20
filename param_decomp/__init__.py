"""Public PD API.

Core entrypoints:
    - `run_pd`: train a parameter decomposition from a `RunConfig`. Uses the driver
      to materialize the target model and dataloaders.
    - `optimize`: pure training loop. Notebook users who build their own target model
      and dataloaders should call this directly (skipping `RunConfig` entirely).
    - `load_component_model`: load a saved PD run as a `ComponentModel`.

Composition root:
    - `resolve_run`: parse a dict (from YAML) into (RunConfig, ExperimentDriver).

Core types:
    - `PDConfig`: training/algorithm config.
    - `PDTarget`: target model + run_batch + reconstruction_loss.
    - `RunConfig`: serializable spec for a PD run — driver_path + pd/logging/runtime.
      Driver-specific subclasses (LMRunConfig, TMSRunConfig, ResidMLPRunConfig) add target/data.
    - `ExperimentDriver`: Protocol for the open-world experiment extension point.
    - `PDRun`: handle to a saved run (RunConfig, checkpoint, driver).
"""

from param_decomp.compose import resolve_run
from param_decomp.configs import PDConfig
from param_decomp.experiments.driver import ExperimentDriver
from param_decomp.models.batch_and_loss_fns import PDTarget
from param_decomp.run import RunConfig
from param_decomp.run_pd import optimize, run_pd
from param_decomp.saved_run import PDRun, load_component_model

__all__ = [
    "ExperimentDriver",
    "PDConfig",
    "PDRun",
    "PDTarget",
    "RunConfig",
    "load_component_model",
    "optimize",
    "resolve_run",
    "run_pd",
]
