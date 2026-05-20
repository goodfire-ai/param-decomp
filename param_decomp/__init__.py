"""Public PD API.

Two ways to train:
    - `optimize(target, train_loader, eval_loader, *, pd_config, logging_config,
      runtime_config, device, out_dir=None)` — notebook / script entry point.
      Pure trainer. Caller provides everything explicitly. Wandb is opportunistic
      (init it yourself beforehand if you want).
    - `run_pd(run_cfg, *, device, ...)` — driver-mediated entry point. Reads a
      `RunConfig` (with a `driver_path`), materializes target+loaders from the
      driver, writes the spec to disk, inits wandb, then calls `optimize`. Used
      by `pd-run` / `_worker.py`.

Reload:
    - `load_component_model(path)` — driver-mediated reload.
    - `PDRun.from_path(path)` — handle to a saved run.

Core types:
    - `PDConfig`: training/algorithm config.
    - `PDTarget`: target model + run_batch + reconstruction_loss.
    - `RunConfig`: serializable spec for a driver-mediated run.
      Driver-specific subclasses (LMRunConfig, TMSRunConfig, ResidMLPRunConfig) add target/data.
    - `ExperimentDriver`: Protocol for the open-world experiment extension point.
    - `PDRun`: handle to a saved run.

Composition root:
    - `materialize_run(run_cfg, *, device, dist_state=None) -> (target, train_loader, eval_loader)`
      — driver-mediated callers turn a `RunConfig` into the tuple `optimize` needs.
    - `PDRun` — the run's domain object: both write-side (during training:
      `.log()`, `.console()`, `.checkpoint()`, `.finish()`) and read-side
      (on reload: `.load_model()`, `.load_target()`, `.build_*_loader()`).
      Construct via `PDRun.for_run(run_cfg, ...)` (driver-mediated training),
      `PDRun.local(out_dir)` / `PDRun.with_wandb(out_dir, project=..., ...)`
      (notebook), `PDRun.silent()` (no persistence), or `PDRun.from_path(path)`
      (reload).
"""

from param_decomp.configs import PDConfig
from param_decomp.experiments.driver import ExperimentDriver
from param_decomp.models.batch_and_loss_fns import PDTarget
from param_decomp.pd_run import PDRun, load_component_model
from param_decomp.run import RunConfig
from param_decomp.run_pd import materialize_run, optimize, run_pd

__all__ = [
    "ExperimentDriver",
    "PDConfig",
    "PDRun",
    "PDTarget",
    "RunConfig",
    "load_component_model",
    "materialize_run",
    "optimize",
    "run_pd",
]
