# `param_decomp_lab/eval_metrics/`

Batteries-included eval `Metric` set for the in-repo experiments, plus the YAML
dispatch wiring (`AnyEvalMetricConfig` + `EVAL_METRIC_CLASSES`).

## Why this lives in the lab (and not in core)

Eval metrics are **user-extensible** by design. We expect users to add their own eval
metrics for their own decomposition runs, so the metric set isn't part of the public
core API — anyone can instantiate a `Metric` subclass and pass it to
`EvalLoop(metrics=...)`.

This is the deliberate split from **loss metrics**, which are canonical and curated:
loss metrics live in `param_decomp/metrics/` and adding one is a core change. See
[`../../param_decomp/metrics/CLAUDE.md`](../../param_decomp/metrics/CLAUDE.md).

This dir is just the set of eval metrics *we* ship for the in-repo experiments.

## YAML dispatch

The in-repo experiments validate the YAML `eval.metrics` list via the
`AnyEvalMetricConfig` discriminated union (on `EvalConfig`, see
[`../experiments/CLAUDE.md`](../experiments/CLAUDE.md)), then instantiate each entry
with `EVAL_METRIC_CLASSES`:

```python
from param_decomp_lab.eval_metrics import EVAL_METRIC_CLASSES
metrics = [EVAL_METRIC_CLASSES[m.type](m) for m in cfg.eval.metrics]
```

Both pieces live in `__init__.py`.

## Adding a lab eval metric

1. Define `<Name>(Metric[<Name>Config])` + its `<Name>Config(BaseConfig)` in
   `<name>.py`. The config must carry a unique `type: Literal["<Name>"]` discriminator.
2. Append the config to `AnyEvalMetricConfig` in `__init__.py`.
3. Append the class to `EVAL_METRIC_CLASSES` in `__init__.py`.

The class extends `Metric` from `param_decomp.metrics.base`. Lifecycle is the same as
any other metric: `__init__(cfg)` → `bind(model, device)` → `update(ctx)` →
`compute()`.

## External / one-off eval metrics

If you're writing your own caller (not using the in-repo experiment runners), skip the
dispatch table entirely — instantiate your `Metric` subclasses directly and pass them
in `EvalLoop(metrics=...)`. Nothing in the core cares whether they came from a YAML
union or were constructed by hand.

## Note on `PGDReconLoss` + `StochasticHiddenActsReconLoss`

Both appear in `EVAL_METRIC_CLASSES` even though they're *loss* classes from core.
That's intentional: they're listed here so they can be added to YAML `eval.metrics`
purely for evaluation (without showing up as a training-loss coefficient). When used
as eval-only, their `coeff` is ignored.
