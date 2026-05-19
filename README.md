# Parameter Decomposition

Training, post-processing, and visualization tools for parameter decomposition on neural networks.
For a compact implementation of the core method, see [`nano_param_decomp/`](nano_param_decomp/).

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

```bash
pd-run <name>          # submit a SLURM job
pd-run <name> --local  # run in this process
```

Useful built-ins:

Run an experiment with `pd-run <name>`. This submits a SLURM job by default (with a git snapshot
for reproducibility); add `--local` to run in this process instead. Compute substrate (device,
data parallelism, autocast) lives in the YAML's `runtime:` block — there are no CLI overrides.
Sweeps run as `pd-run --sweep_generator_path /abs/path/file.py:func --n_agents N`. The two main
language-model decompositions:

- **`pile_llama_simple_mlp-4L`** — 4-layer Llama (MLP-only) on the Pile; the VPD paper run
  [`goodfire/spd/runs/s-55ea3f9b`](https://wandb.ai/goodfire/spd/runs/s-55ea3f9b)
  ([config](param_decomp/experiments/lm/pile_llama_simple_mlp-4L.yaml)).
- `ss_llama_simple_mlp-2L`: smaller SimpleStories LM decomposition
  ([config](param_decomp/experiments/lm/ss_llama_simple_mlp-2L.yaml)).

Other YAML configs under [`param_decomp/experiments/`](param_decomp/experiments) are
auto-discovered. The LM experiment supports HuggingFace-loadable models with `nn.Linear`,
`nn.Embedding`, or `transformers.modeling_utils.Conv1D` target modules.

For custom experiments, either call `run_pd(...)` directly or provide a YAML-driven
`ExperimentDriver`:

- **Call `run_pd` directly** — build a `PDTarget` (model + `run_batch` + reconstruction loss;
  helpers in [`batch_and_loss_fns.py`](param_decomp/models/batch_and_loss_fns.py)) and call
  `run_pd(config, logging_config, runtime_config, target, train_loader, eval_loader, device)`.
  Reload with `load_component_model(path, target=...)`. Best for notebooks/scripts.
- **Package it as a YAML-driven experiment** — define your experiment as a Pydantic `Run`
  subclass (adding `target` / `data` fields) plus an `ExperimentDriver` class (a small adapter
  exposing `build_target` and `build_dataloaders`; see
  [`driver.py`](param_decomp/experiments/driver.py) for the interface and
  [`tms/experiment.py`](param_decomp/experiments/tms/experiment.py) for the smallest example),
  put `driver_path: my_pkg.my_exp:MyDriver` at the top of your YAML, then run
  `pd-run --config_path my_config.yaml`. This is what built-in experiments do, and is needed for
  sweeps and for self-reloading runs via
  `load_component_model(path)` (no `target=` argument needed) or `PDRun.from_path(...)`.

## Metrics

Configure training losses in `pd.loss_metrics` and extra eval-only metrics in `pd.eval_metrics`.
Keys are registered metric class names. Loss metrics must set `coeff`; they are evaluated
automatically, so do not repeat them under `eval_metrics`.

You can pass your own metrics by listing importable dotted modules in `pd.metric_modules`.
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
  eval_metrics:
    MyCustomEvalMetric: {}
```

Custom metric modules define a Pydantic config plus a metric class satisfying
`__init__(cfg, *, model, device)`, `reset()`, `update(ctx)`, and `compute()`. Use
`LossMetricConfig` for trainable losses and `MetricConfig` for eval-only metrics; see
[`param_decomp/metrics/base.py`](param_decomp/metrics/base.py).

## App And Post-Processing

```bash
make install-app
make app
```

The app reads post-processed artifacts. Run all post-processing stages from one config with:

```bash
pd-postprocess param_decomp/postprocess/pile.yaml
```

The stages are Harvest, Autointerp, Dataset attributions, Graph interpretation, and Clustering.

## Development

```bash
make check     # ruff format/lint + basedpyright
make type      # basedpyright only
make format    # ruff lint + format
make test      # tests not marked slow
make test-all  # all tests
```
