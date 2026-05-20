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
    - `run_pd(run_cfg, *, device, ...)` — recipe-mediated entry point. Reads a
      `RunConfig`, materializes target+loaders from the recipe, builds a
      `RunSink.for_run`, then calls `optimize`. Used by `pd-run` / `_worker.py`.

Reload:
    - `load_component_model(path)` — recipe-mediated reload.
    - `SavedRun.from_path(path)` — handle to a saved run for full reload
      (model + dataloaders + target via the recipe).

Core types:
    - `PDConfig`: training/algorithm config.
    - `PDTarget`: target model + run_batch + reconstruction_loss.
    - `RunConfig`: serializable spec for a recipe-mediated run.
    - `RunRecipe`: Protocol for the open-world reload/materialization extension point.
    - `RunSink`: output channels (local files + opportunistic wandb +
      checkpoints) for a training run. Constructors:
      `RunSink.for_run(run_cfg, ...)` (recipe-mediated),
      `RunSink.local(out_dir)` / `RunSink.with_wandb(out_dir, project=..., ...)`
      (notebook), `RunSink.silent()` (no persistence).
    - `SavedRun`: handle to a completed run on disk / W&B.
      `SavedRun.from_path(path)` resolves spec + checkpoint + recipe.

Composition root:
    - `materialize_run(run_cfg, *, device, dist_state=None, recipe=None) ->
      (target, train_loader, eval_loader)` — recipe-mediated callers turn a
      `RunConfig` into the tuple `optimize` needs.
"""

from param_decomp.configs import PDConfig
from param_decomp.models.batch_and_loss_fns import PDTarget
from param_decomp.recipes import RunRecipe
from param_decomp.run import RecipeRef, RunConfig
from param_decomp.run_pd import materialize_run, optimize, run_pd
from param_decomp.run_sink import RunSink
from param_decomp.saved_run import SavedRun, load_component_model

__all__ = [
    "PDConfig",
    "PDTarget",
    "RecipeRef",
    "RunConfig",
    "RunRecipe",
    "RunSink",
    "SavedRun",
    "load_component_model",
    "materialize_run",
    "optimize",
    "run_pd",
]
