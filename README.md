# Parameter Decomposition

Training tools for parameter decomposition on neural networks. For a compact implementation of
the core method, see [`nano_param_decomp/`](nano_param_decomp/).

## References

- **VPD paper (April 2026):** https://www.goodfire.ai/research/interpreting-lm-parameters. [VPD Code Release](https://github.com/goodfire-ai/param-decomp/releases/tag/vpd-paper)
  Canonical 4L-pile run: `goodfire/spd/runs/s-55ea3f9b`.
- **SPD paper (June 2025):** https://arxiv.org/abs/2506.20790. [SPD Code Release](https://github.com/goodfire-ai/param-decomp/releases/tag/v1).

## Install

This repo contains the generic library and a thin private wrapper:

- `param-decomp`: the library, importing as `param_decomp` — enumerated layers as
  subpackages (`core` = the engine, `targets`, `pretrain`, `vendored_jax`, plus the
  composition/consumer layers: `experiments`, `harvest`, `autointerp`, `clustering`,
  `topology`, `adapters`, `infra`, …)
- `param-decomp-goodfire`: the Goodfire-internal cluster fit wrapping the library — the
  training launchers (`pd-lm` / `pd-pretrain`), the post-pipeline SLURM submitters
  (`pd-harvest`, `pd-autointerp`, `pd-intruder`, `pd-clustering`),
  and the dependency-chained pipeline (`pd-postprocess`)

```bash
make install-dev  # workspace dev install: library + wrapper + dev dependencies + pre-commit hooks
make install      # library only
```

## Run Experiments

The library installs the in-process `pd-*` commands (`pd-tms`, `pd-resid-mlp`,
`pd-cluster-merge`, `pd-cluster-distances`); every SLURM submitter (`pd-lm`,
`pd-pretrain`, `pd-harvest`, `pd-autointerp`, `pd-intruder`, `pd-clustering`,
`pd-postprocess`) comes from the private wrapper. A run is one
self-contained YAML; launch mode is config-driven (`runtime.launch: slurm | inline`),
no CLI flags:

```bash
pd-lm    param_decomp/core/configs/<config>.yaml
```

TMS and ResidualMLP now live only as JAX targets in `param_decomp/`
(`tms.py`, `resid_mlp.py`); the torch experiment dirs were deleted.

Training is the JAX single-pool trainer: the generic engine
(`param_decomp.core.run.run_decomposition_training`, a pure library) driven by the composition-side
composition root (`python -m param_decomp.experiments.lm.run`), launched via `pd-lm`. A
run is one self-contained YAML (the `param_decomp.experiments.config.ExperimentConfig`
schema over the core `param_decomp.core.configs` pieces). The torch trainer (`optimize()`, the
torch `Metric` impls, `RunSink`) was retired and is preserved at git tag `torch-oracle`. See
`param_decomp/core/CLAUDE.md` and `param_decomp/core/SPEC.md` for the trainer.

## Metrics

Training losses are configured in `pd.loss_metrics` as a list of `{type: "<ClassName>",
...}` entries; eval metrics in `eval.metrics`. Both are validated by the torch-free pydantic
schema in core (`param_decomp.core.configs`) and computed by the JAX trainer
(`param_decomp/core/losses.py`, `param_decomp/core/slow_eval.py`).

## Packaging

The root `pyproject.toml` builds the `param-decomp` library (the whole `param_decomp/`
package); the private Goodfire launchers build separately from
`param_decomp_goodfire/pyproject.toml`. Local development uses the uv workspace, so
absolute imports for both work after `make install-dev`.

## Development

```bash
make check     # ruff format/lint + basedpyright
make type      # basedpyright only
make format    # ruff lint + format
make test      # tests not marked slow
make test-all  # all tests
```
