# Parameter Decomposition Lab

Lab package for the `param-decomp` repository. This distribution contains the in-repo
experiments, visualization app, pretraining scripts, postprocessing pipelines, and SLURM
tooling. It imports as `param_decomp_lab` and depends on the core `param-decomp` package.

## Local Development

From the repository root:

```bash
make install-dev
```

This installs both workspace packages editably, so both imports are available:

```python
import param_decomp
import param_decomp_lab
```

## CLI Entrypoints

The lab package owns the `pd-*` commands:

```bash
pd-tms        param_decomp_lab/experiments/tms/tms_5-2_config.yaml
pd-resid-mlp  param_decomp_lab/experiments/resid_mlp/resid_mlp1_config.yaml
pd-lm         param_decomp_lab/experiments/lm/ss_llama_simple_mlp-2L.yaml
pd-pretrain   --config_path param_decomp_lab/experiments/lm/pretrain/configs/pile_llama_simple_mlp-4L-768.yaml
pd-harvest    path/to/harvest_slurm_config.yaml
pd-autointerp <decomposition_id> --config path/to/autointerp_slurm_config.yaml --harvest_subrun_id h-YYYYMMDD_HHMMSS
```

The package also provides clustering, graph interpretation, dataset attribution, intruder,
and investigation CLIs declared in `param_decomp_lab/pyproject.toml`.
