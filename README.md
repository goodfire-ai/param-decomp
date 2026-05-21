# Parameter Decomposition

Training tools for parameter decomposition on neural networks. For a compact implementation of
the core method, see [`nano_param_decomp/`](nano_param_decomp/).

## References

- **VPD paper (April 2026):** https://www.goodfire.ai/research/interpreting-lm-parameters.
  Canonical run: `goodfire/spd/runs/s-55ea3f9b`.
- **SPD paper (June 2025):** https://arxiv.org/abs/2506.20790. Paper branch:
  [`spd-paper`](https://github.com/goodfire-ai/param-decomp/tree/spd-paper).

## Install

```bash
make install-dev  # package, dev dependencies, pre-commit hooks
make install      # package only
```

## Run Experiments

Each in-repo experiment is a self-contained script that reads a YAML and calls `optimize()`:

```bash
pd-tms       param_decomp/experiments/tms/tms_5-2_config.yaml
pd-resid-mlp param_decomp/experiments/resid_mlp/resid_mlp1_config.yaml
pd-lm        param_decomp/experiments/lm/ss_llama_simple_mlp-2L.yaml
```

For a brand-new experiment, write your own `run.py` that builds the target model, the
train/eval dataloaders, the eval `Metric` list, the `PDConfig` and `RuntimeConfig`, and a
`RunSink`, then calls `optimize(...)`:

```python
from param_decomp import PDConfig, RunSink, RuntimeConfig, optimize
from param_decomp.models.batch_and_loss_fns import recon_loss_mse, run_batch_first_element

optimize(
    target_model=my_target_module,
    train_loader=train_loader,
    eval_loader=eval_loader,
    run_batch=run_batch_first_element,
    reconstruction_loss=recon_loss_mse,
    pd_config=PDConfig(...),
    runtime_config=RuntimeConfig(),
    sink=RunSink.local(out_dir, train_log_freq=100, eval_freq=1000, slow_eval_freq=5000,
                       n_eval_steps=10),
    eval_metrics=[...],   # list of pre-instantiated Metric objects
    device=device,
)
```

The three in-repo `run.py` files
([tms](param_decomp/experiments/tms/run.py),
 [resid_mlp](param_decomp/experiments/resid_mlp/run.py),
 [lm](param_decomp/experiments/lm/run.py)) are reference examples.

## Metrics

Configure training losses in `pd.loss_metrics`; keys are registered metric class names. Loss
metrics must set `coeff`; they are evaluated automatically.

Eval-only metrics are constructed by your `run.py` and passed to `optimize(eval_metrics=...)`.
The `experiments.utils.build_eval_metrics(...)` helper converts a YAML
`logging.eval_metrics` dict-of-config into a `list[Metric]`.

You can register your own metrics by listing importable dotted modules in `pd.metric_modules`.
`PDConfig` imports those modules before resolving metric names, so any classes decorated with
`@register_metric` are available from YAML:

```yaml
pd:
  metric_modules:
    - my_project.pd_metrics
  loss_metrics:
    MyCustomLoss:
      coeff: 0.1
      scale: 3.0
logging:
  eval_metrics:
    MyCustomEvalMetric: {}
```

Custom metric modules define a Pydantic config plus a metric class satisfying `__init__(cfg)`,
`bind(*, model, device)`, `reset()`, `update(ctx)`, and `compute()`. Use `LossMetricConfig` for
trainable losses and `MetricConfig` for eval-only metrics; see
[`param_decomp/metrics/base.py`](param_decomp/metrics/base.py).

## Development

```bash
make check     # ruff format/lint + basedpyright
make type      # basedpyright only
make format    # ruff lint + format
make test      # tests not marked slow
make test-all  # all tests
```
