# Parameter Decomposition Lab

Lab package for the `param-decomp` repository. This distribution contains the in-repo
experiment glue, postprocessing pipelines, and SLURM tooling. It imports as
`param_decomp_lab` and depends on the core `param-decomp` package.

## Local Development

From the repository root:

```bash
make install-dev
```

This installs all workspace packages editably into the one venv, so the imports are
available:

```python
import param_decomp
import param_decomp_lab
```

## CLI Entrypoints

The lab package owns the `pd-*` commands:

```bash
pd-lm     param_decomp_lab/experiments/lm/<wrapper>.yaml --nodes N
pd-harvest    path/to/harvest_slurm_config.yaml
pd-autointerp <decomposition_id> --config path/to/autointerp_slurm_config.yaml --harvest_subrun_id h-YYYYMMDD_HHMMSS
```

The package also provides clustering, graph interpretation, dataset attribution, intruder,
and investigation CLIs declared in `param_decomp_lab/pyproject.toml`.
