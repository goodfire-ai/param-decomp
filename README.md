# Parameter Decomposition

Training tools for parameter decomposition on neural networks. For a compact implementation of
the core method, see [`nano_param_decomp/`](nano_param_decomp/).

## References

- **VPD paper (April 2026):** https://www.goodfire.ai/research/interpreting-lm-parameters. [VPD Code Release](https://github.com/goodfire-ai/param-decomp/releases/tag/vpd-paper)
  Canonical 4L-pile run: `goodfire/spd/runs/s-55ea3f9b`.
- **SPD paper (June 2025):** https://arxiv.org/abs/2506.20790. [SPD Code Release](https://github.com/goodfire-ai/param-decomp/releases/tag/v1).

## Install

This repo contains two Python distributions:

- `param-decomp`: the core library, importing as `param_decomp`
- `param-decomp-lab`: in-repo experiments, app, postprocessing, and CLI tooling, importing as
  `param_decomp_lab`

```bash
make install-dev  # workspace dev install: core + lab + dev dependencies + pre-commit hooks
make install      # core package only
make install-lab  # core + lab packages, without dev dependencies
```

## Run Experiments

The `pd-*` commands are installed by `param-decomp-lab`. Each in-repo experiment is a
self-contained script that reads a YAML and calls `optimize()`:

```bash
pd-tms       param_decomp_lab/experiments/tms/tms_5-2_config.yaml
pd-resid-mlp param_decomp_lab/experiments/resid_mlp/resid_mlp1_config.yaml
pd-lm        param_decomp_lab/experiments/lm/pile_llama_simple_mlp-4L.yaml
```

For a brand-new experiment, write your own `run.py` that builds the target model, the
train/eval dataloaders, the eval `Metric` list, the `PDConfig` and `RuntimeConfig`, a
`Cadence` (when to emit), and a `RunSink` (where output goes), then calls `optimize(...)`:

```python
from param_decomp.configs import Cadence, PDConfig, RuntimeConfig
from param_decomp.optimize import EvalLoop, optimize
from param_decomp_lab.batch_and_loss_fns import recon_loss_mse, run_batch_first_element
from param_decomp_lab.run_sink import RunSink

optimize(
    target_model=my_target_module,
    train_loader=train_loader,
    run_batch=run_batch_first_element,
    reconstruction_loss=recon_loss_mse,
    pd_config=PDConfig(...),
    runtime_config=RuntimeConfig(device=device),
    cadence=Cadence(
        train_log_every=100,
        save_every=5000,
    ),
    sink=RunSink.local(out_dir),
    eval_loop=EvalLoop(
        loader=eval_loader,
        metrics=[...],  # list of pre-instantiated Metric objects
        n_steps=10,
        every=1000,
        slow_every=5000,
    ),
)
```

The three in-repo `run.py` files
([tms](param_decomp_lab/experiments/tms/run.py),
 [resid_mlp](param_decomp_lab/experiments/resid_mlp/run.py),
 [lm](param_decomp_lab/experiments/lm/run.py)) are reference examples.

## Metrics

Configure training losses in `pd.loss_metrics` as a list of `{type: "<ClassName>", ...}`
entries. The `type` literal dispatches to a `Metric` subclass via
`param_decomp.metrics.dispatch.LOSS_METRIC_CLASSES`. Loss metrics must set `coeff`; they
are evaluated automatically alongside dedicated eval metrics. New loss metrics are added
by defining the class in `param_decomp/metrics/`, appending the config to
`AnyLossMetricConfig` in `configs.py`, and appending the class to `LOSS_METRIC_CLASSES`.

Eval metrics are caller-supplied: instantiate `Metric` objects in your `run.py` and pass
them via `EvalLoop(metrics=...)`. The in-repo experiments validate the YAML
`eval.metrics` list via the `AnyEvalMetricConfig` discriminated union on `EvalConfig`,
then dispatch each entry through `EVAL_METRIC_CLASSES` (both in
`param_decomp_lab.eval_metrics`):

```python
eval_metrics = [EVAL_METRIC_CLASSES[m.type](m) for m in cfg.eval.metrics]
```

## Packaging

The root `pyproject.toml` builds only the core `param-decomp` distribution. Lab scripts
and experiment tooling live in `param_decomp_lab/pyproject.toml` as the separate
`param-decomp-lab` distribution. Local development uses the uv workspace, so absolute
imports for both packages work after `make install-dev`.

Metric classes define a Pydantic config plus a class satisfying `__init__(cfg)`,
`bind(*, model, device)`, `reset()`, `update(ctx)`, and `compute()`. Use `LossMetricConfig`
for trainable losses and subclass `BaseConfig` directly for eval-only metrics; see
[`param_decomp/metrics/base.py`](param_decomp/metrics/base.py).

## Development

```bash
make check     # ruff format/lint + basedpyright
make type      # basedpyright only
make format    # ruff lint + format
make test      # tests not marked slow
make test-all  # all tests
```
