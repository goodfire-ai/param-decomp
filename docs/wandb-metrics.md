# W&B metric names

Existing configurations keep the legacy metric names. To opt a **new** decomposition
or pretraining run into two-level names, add the setting to its existing W&B block:

```yaml
wandb:
  project: your-project
  metric_schema: grouped
```

`legacy` and `grouped` are the only supported schemas. Grouped names contain exactly
one slash, separating the section from its chart, such as `train-loss/total` and
`train-lr/ppgd-src`. Site names sort by layer, then Q, K, V, O, gate, up, and down.

This setting changes W&B keys, including deferred-media step-axis bindings. Values,
logging cadence, local `metrics.jsonl` keys and console output retain their existing
meaning. A resumed W&B run must retain its saved schema; an older run without the
setting is legacy. Resumes validate the saved schema without rewriting the saved
configuration. Only fresh runs publish a new configuration. Existing
decomposition launch-config pins also continue to reject changes on resume.

## Implementation

- `param_decomp.metric_names` models keys, groups and rectangular grids.
- `param_decomp.metric_taxonomy` defines the pure naming grammar.
- `param_decomp.metric_schema` applies the selected schema and rejects name collisions,
  including collisions between separately logged records.
- Decomposition and pretraining sinks apply names only at the W&B boundary.

The grouped writer enumerates fixed diagnostic names, categories and supported model
site forms. Unknown entries raise; adding an emitter requires a mapping and a test.
Config-authored training-loss, auxiliary, capture and group labels remain open identifier
slots. Their spelling alone cannot establish membership in a particular run's config.

Custom-named evaluation probes are currently unsupported by grouped logging: their names
do not identify the probe type. Supporting them requires reading that identity from the
run configuration rather than guessing from a custom label.
