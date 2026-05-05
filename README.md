# PD

**VPD paper (April 2025)**
- Paper: https://www.goodfire.ai/research/interpreting-lm-parameters
- Branch: main
- Wandb for run in paper: https://wandb.ai/goodfire/spd/runs/s-55ea3f9b

**SPD paper (June 2025)**
- Paper: https://arxiv.org/abs/2506.20790
- Branch: https://github.com/goodfire-ai/param-decomp/tree/spd-paper
- Wandb report: https://wandb.ai/goodfire/spd-tms/reports/SPD-paper-report--VmlldzoxMzE3NzU0MQ

## Installation

From the root of the repository, run one of:

```bash
make install-dev  # Install the package, dev requirements, pre-commit hooks
make install      # Install the package only (`pip install -e .`)
```

Place your wandb credentials in a `.env` file. See `.env.example` for an example.

## Canonical Example: 4-layer Llama on the Pile

The canonical decomposition for this codebase is a 4-layer Llama (MLP-only) trained on the Pile.
The decomposition run from the VPD paper is at
[`goodfire/spd/runs/s-55ea3f9b`](https://wandb.ai/goodfire/spd/runs/s-55ea3f9b), and corresponds to
the experiment registered as `pile_llama_simple_mlp-4L` with config
[`param_decomp/experiments/lm/pile_llama_simple_mlp-4L.yaml`](param_decomp/experiments/lm/pile_llama_simple_mlp-4L.yaml).

To reproduce it (locally on a single GPU, or on the SLURM cluster):

```bash
pd-local pile_llama_simple_mlp-4L              # Local, single GPU
pd-run --experiments pile_llama_simple_mlp-4L  # SLURM, with git snapshot + W&B view
```

`pd-run` additionally supports data parallelism (`--dp 4`), CPU-only execution (`--cpu`), and
hyperparameter sweeps (`--sweep --n_agents N`, parameters in
`param_decomp/scripts/sweep_params.yaml`).

Once the decomposition has trained, run the full post-processing pipeline (see below) to produce
the artifacts the app needs to visualise it.

## Other Experiments

The codebase ships configs for several other domains, all listed in
[`param_decomp/registry.py`](param_decomp/registry.py):

- **Language models** (`param_decomp/experiments/lm`): `pile_llama_simple_mlp-{2L,4L,12L}`,
  `ss_llama_simple{,_mlp}-{1L,2L}`, `ss_gpt2{,_simple{,_noln}}`, `gpt2`, `ts`.
- **Toy Model of Superposition** (`param_decomp/experiments/tms`): `tms_5-2`, `tms_5-2-id`,
  `tms_40-10`, `tms_40-10-id`.
- **Residual MLP** (`param_decomp/experiments/resid_mlp`): `resid_mlp{1,2,3}` — toy models of
  compressed computation and distributed representations.
- **Induction heads** (`param_decomp/experiments/ih`): a small model trained on a toy
  induction-head task.

The `lm` experiment can decompose any HuggingFace-loadable model, provided the target modules are
`nn.Linear`, `nn.Embedding`, or `transformers.modeling_utils.Conv1D` (other layer types are not yet
supported).

## App

This project ships a web app for visualising and interpreting decompositions — component
activations, dataset attributions, autointerp labels, and graph views. See the app's
[README](param_decomp/app/README.md) and [CLAUDE.md](param_decomp/app/CLAUDE.md) for details.

```bash
make app   # Launch backend + frontend dev servers
```

## Post-Processing Pipeline

After a decomposition has finished training, post-processing produces the artifacts the app reads:
component statistics, autointerp labels, dataset attributions, and graph-context interpretations.
Each stage is a separate CLI; `pd-postprocess` runs them all under one SLURM dependency graph from
a single config:

```bash
pd-postprocess param_decomp/postprocess/pile.yaml
```

The individual stages, with links to their docs:

- **Harvest** ([`pd-harvest`](param_decomp/harvest/CLAUDE.md)) — collect activation examples,
  correlations, and token statistics for each component.
- **Autointerp** ([`pd-autointerp`](param_decomp/autointerp/CLAUDE.md)) — generate LLM
  interpretations of components from harvested examples. Requires `OPENROUTER_API_KEY`.
- **Dataset attributions** ([`pd-attributions`](param_decomp/dataset_attributions/CLAUDE.md)) —
  compute component-to-component attribution strengths over the training distribution.
- **Graph interpretation** ([`pd-graph-interp`](param_decomp/graph_interp/CLAUDE.md)) —
  context-aware component labels that combine attributions and correlations.
- **Clustering** ([`pd-clustering`](param_decomp/clustering/CLAUDE.md)) — ensemble clustering of
  components.

Default batch sizes (256 for harvest and attributions) work for models like
`pile_llama_simple_mlp-4L`; tune via `--batch_size` / `--n_gpus` per stage.

## Development

Suggested VSCode/Cursor settings live in `.vscode/`. Copy `.vscode/settings-example.json` to
`.vscode/settings.json` to use them. See [CONTRIBUTING.md](CONTRIBUTING.md) for PR guidelines.

Useful `make` targets:

```bash
make check     # Run pre-commit on all files (basedpyright, ruff lint, ruff format)
make type      # basedpyright only
make format    # ruff lint + format
make test      # Tests not marked `slow`
make test-all  # All tests
```
