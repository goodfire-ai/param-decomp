"""Public PD API.

One entry point:

    optimize(target_model, train_loader, eval_loader, *, run_batch, reconstruction_loss,
             pd_config, runtime_config, sink, eval_metrics, device)

Caller builds:
    - `target_model`: the `nn.Module` whose weights you want to decompose.
    - `train_loader` / `eval_loader`: dataloaders. `optimize` re-iters them with
      `loop_dataloader`, so finite loaders restart.
    - `run_batch`: `(model, batch) -> Tensor` — how a forward pass is invoked.
      Pre-built helpers: `run_batch_passthrough`, `run_batch_first_element`,
      `make_run_batch(output_extract)` in `param_decomp.models.batch_and_loss_fns`.
    - `reconstruction_loss`: `(pred, target) -> (loss, n_examples)`. Helpers:
      `recon_loss_mse`, `recon_loss_kl` in the same module.
    - `pd_config`: `PDConfig` — the PD algorithm spec (CI fn, loss-metric mix,
      module patterns, optimizers, schedules, seed, tied weights).
    - `runtime_config`: `RuntimeConfig` — substrate (device, autocast, dp).
    - `sink`: `RunSink` — output channels + cadence. Use
      `RunSink.local(out_dir, train_log_freq=..., eval_freq=..., ...)`,
      `RunSink.with_wandb(out_dir, project=..., train_log_freq=..., ...)`, or
      `RunSink.silent(...)` for tests / quick interactive runs.
    - `eval_metrics`: list of pre-instantiated eval `Metric` objects. `optimize`
      binds them to the built `ComponentModel` internally.
"""

from param_decomp.configs import PDConfig, RuntimeConfig
from param_decomp.metrics.base import LossMetricConfig, Metric, MetricConfig
from param_decomp.models.batch_and_loss_fns import ReconstructionLoss, RunBatch
from param_decomp.optimize import optimize
from param_decomp.run_sink import RunSink

__all__ = [
    "LossMetricConfig",
    "Metric",
    "MetricConfig",
    "PDConfig",
    "ReconstructionLoss",
    "RunBatch",
    "RunSink",
    "RuntimeConfig",
    "optimize",
]
