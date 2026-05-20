"""Public PD API.

Three-phase lifecycle:

    RunConfig   →   RunSink    →   SavedRun
    (recipe)        (writer        (reader
                     during          after)
                     training)

Each phase is its own type — separate concerns, separate lifetimes.

Two ways to train:
    - `optimize(target, train_loader, eval_loader, *, pd_config, logging_config,
      runtime_config, device, sink)` — notebook / script entry point. Pure
      trainer. Caller provides target/loaders/configs explicitly, plus a
      `RunSink` for outputs.
    - `run_pd(run_cfg, *, device, ...)` — driver-mediated entry point. Reads a
      `RunConfig`, materializes target+loaders from the driver, builds a
      `RunSink.for_run`, then calls `optimize`. Used by `pd-run` / `_worker.py`.

Reload:
    - `load_component_model(path)` — driver-mediated reload.
    - `SavedRun.from_path(path)` — handle to a saved run for full reload
      (model + dataloaders + target via the driver).

Core types:
    - `PDConfig`: training/algorithm config.
    - `PDTarget`: target model + run_batch + reconstruction_loss.
    - `RunConfig`: serializable spec for a driver-mediated run. Single class —
      `target` and `data` are stored as raw `dict[str, Any]` payloads and
      interpreted by the driver (no per-driver subclass).
    - `ExperimentDriver`: Protocol for the open-world experiment extension
      point. Drivers validate their typed `target` / `data` via
      `validate_config` and re-parse inside each `build_*` method.
    - `RunSink`: output channels (local files + opportunistic wandb +
      checkpoints) for a training run. Constructors:
      `RunSink.for_run(run_cfg, ...)` (driver-mediated),
      `RunSink.local(out_dir)` / `RunSink.with_wandb(out_dir, project=..., ...)`
      (notebook), `RunSink.silent()` (no persistence).
    - `SavedRun`: handle to a completed run on disk / W&B.
      `SavedRun.from_path(path)` resolves spec + checkpoint + driver.

Composition root:
    - `materialize_run(run_cfg, *, device, dist_state=None, driver=None) ->
      (target, train_loader, eval_loader)` — driver-mediated callers turn a
      `RunConfig` into the tuple `optimize` needs.
"""

from param_decomp.configs import PDConfig
from param_decomp.experiments.driver import ExperimentDriver
from param_decomp.models.batch_and_loss_fns import PDTarget
from param_decomp.run import RunConfig
from param_decomp.run_pd import materialize_run, optimize, run_pd
from param_decomp.run_sink import RunSink
from param_decomp.saved_run import SavedRun, load_component_model

__all__ = [
    "ExperimentDriver",
    "PDConfig",
    "PDTarget",
    "RunConfig",
    "RunSink",
    "SavedRun",
    "load_component_model",
    "materialize_run",
    "optimize",
    "run_pd",
]
