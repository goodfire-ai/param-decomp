# SPD - Stochastic Parameter Decomposition

**Note: The [spd-paper](https://github.com/goodfire-ai/param-decomp/tree/spd-paper) branch contains code used in the paper [Stochastic Parameter Decomposition](https://arxiv.org/abs/2506.20790). The main branch contains active work from Goodfire and collaborators since this paper's release. This is now an open source
research project. Please feel free to view the issues (or add to them) and make a PR!**

Weights and Bias [report](https://wandb.ai/goodfire/spd-tms/reports/SPD-paper-report--VmlldzoxMzE3NzU0MQ) accompanying the paper.

## Installation
From the root of the repository, run one of

```bash
make install-dev  # To install the package, dev requirements, pre-commit hooks, and create user files
make install  # To install the package (runs `pip install -e .`) and create user files
```

## Usage
Place your wandb information in a .env file. See .env.example for an example.

The repository consists of several `experiments`, each of which containing scripts to run SPD,
analyse results, and optionally a train a target model:
- `param_decomp/experiments/tms` - Toy model of superposition
- `param_decomp/experiments/resid_mlp` - Toy model of compressed computation and toy model of distributed
  representations
- `param_decomp/experiments/ih` - Small model trained on a toy induction head task.
- `param_decomp/experiments/lm` - Language model loaded from huggingface.
- There is also a toy MLP memorization experiment at `feature/memorization-experiments` which has
  not been merged to `main` (much less effort has gone in to validating it).

Note that the `lm` experiment allows for running SPD on any model from huggingface or that is
accessible to the environment, provided that you only need to decompose `nn.Linear`, `nn.Embedding`
or `transformers.modeling_utils.Conv1D` layers (other layer types are not yet supported, though
these should cover most modules). See `param_decomp/experiments/lm/ss_gpt2_config.yaml` for an example which
loads from huggingface and `param_decomp/experiments/lm/ss_gpt2_simple_config.yaml` for an example which
loads from https://github.com/goodfire-ai/simple_stories_train (with the model weights saved on
wandb).

## CLI Commands

The following CLI commands are available after installation:

### `param-decomp-local` - Local Execution

Run a single SPD experiment locally without SLURM:

```bash
param-decomp-local tms_5-2           # Run on single GPU (default)
param-decomp-local tms_5-2 --cpu     # Run on CPU
param-decomp-local tms_5-2 --dp 4    # Run on 4 GPUs (single node DDP)
```

### `param-decomp-run` - SLURM Cluster Execution

Run experiments on a SLURM cluster with git snapshots and W&B integration:

```bash
param-decomp-run --experiments tms_5-2                    # Run a specific experiment
param-decomp-run --experiments tms_5-2,resid_mlp1         # Run multiple experiments
param-decomp-run                                          # Run all experiments
param-decomp-run --experiments tms_5-2 --cpu              # Run on CPU
param-decomp-run --experiments ss_llama_simple --dp 4     # Data parallelism (4 GPUs)
```

For running hyperparameter sweeps:

```bash
param-decomp-run --experiments <experiment_name> --sweep --n_agents <n-agents> [--cpu]
```

**Sweep parameters:**
- Default sweep parameters are loaded from `param_decomp/scripts/sweep_params.yaml`
- You can specify a custom sweep parameters file by passing its path to `--sweep`
- Sweep parameters support both experiment-specific and global configurations:
  ```yaml
  # Params used for all experiments
  global:
    seed:
      values: [0, 1]
    loss_metric_configs:
      - classname: "ImportanceMinimalityLoss"
        coeff:
          values: [0.1, 0.2]

  # Experiment-specific parameters (override global)
  tms_5-2:
    seed:
      values: [100, 200]  # Overrides global seed
    task_config:
      feature_probability:
        values: [0.05, 0.1]
  ```

## Post-Processing Pipeline

After training an SPD model, you can run several post-processing steps which create artifacts
that can be viewed in the app.

Note, the default batch size for harvest and dataset attributions is 256, which works well for
models like `ss_llama_simple_mlp-2L-wide`. This is configurable with `--batch_size`. You should also
configure `--n_gpus` based on how many gpus you have available. You can use any number you'd like.

### 1. Harvest Component Statistics

First, collect component statistics (activation examples, correlations, token stats):

```bash
param-decomp-harvest goodfire/spd/runs/abc123 --n_gpus 24
```

### 2. Automated Interpretation

After harvesting, generate LLM interpretations of components:

```bash
param-decomp-autointerp goodfire/spd/runs/abc123
```

Requires `OPENROUTER_API_KEY` env var.

### 3. Dataset Attributions

Compute component-to-component attribution strengths:

```bash
param-decomp-attributions goodfire/spd/runs/abc123 --n_gpus 24
```

Can be run independently of harvest/autointerp.

### 4. Clustering (TODO)

Component clustering analysis is under development.

### Direct Script Execution

SPD can also be run by executing any of the `*_decomposition.py` scripts defined in the experiment
subdirectories, along with a corresponding config file:

```bash
python param_decomp/experiments/tms/tms_decomposition.py param_decomp/experiments/tms/tms_5-2_config.yaml
```

### Model Comparison

Use the model comparison script to analyse (post hoc) the geometric similarities between subcomponents of two different models:

```bash
python param_decomp/scripts/compare_models/compare_models.py param_decomp/scripts/compare_models/compare_models_config.yaml
```

See `param_decomp/scripts/compare_models/README.md` for detailed usage instructions.

## Development

Suggested extensions and settings for VSCode/Cursor are provided in `.vscode/`. To use the suggested
settings, copy `.vscode/settings-example.json` to `.vscode/settings.json`.

## App

This project contains an app for visualising and interpreting SPD decompositions. See the [README](./param_decomp/app/README.md).

### Contributing

Please read our [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to contribute to this project, including pull request requirements and review processes.

### Development Commands

There are various `make` commands that may be helpful.

```bash
make check  # Run pre-commit on all files (i.e. basedpyright, ruff linter, and ruff formatter)
make type  # Run basedpyright on all files
make format  # Run ruff linter and formatter on all files
make test  # Run tests that aren't marked `slow`
make test-all  # Run all tests
```
