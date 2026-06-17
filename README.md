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
pd-jax-lm    param_decomp_lab/experiments/lm/<wrapper>.yaml --nodes N
```

TMS and ResidualMLP now live only as JAX targets in `param_decomp_jax/jax_single_pool/`
(`tms.py`, `resid_mlp.py`); the torch experiment dirs were deleted.

Training is the JAX single-pool trainer (`param_decomp_jax/jax_single_pool/`, entry point
`jsp-train`), launched from the lab side via `pd-jax-lm`. A run is one self-contained YAML
(the `param_decomp_config` experiment schema). The torch trainer (`optimize()`, the torch
`Metric` impls, `RunSink`) was retired and is preserved at git tag `torch-oracle`. See
`param_decomp_jax/jax_single_pool/CLAUDE.md` and `SPEC.md` for the trainer.

## Metrics

Training losses are configured in `pd.loss_metrics` as a list of `{type: "<ClassName>",
...}` entries; eval metrics in `eval.metrics`. Both are validated by the torch-free
`param_decomp_config` schema and computed by the JAX trainer
(`param_decomp_jax/jax_single_pool/losses.py`, `slow_eval.py`).

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
