# Parameter Decomposition

This repo is for running parameter decomposition on neural networks.

**VPD paper (April 2025)**
- Paper: https://www.goodfire.ai/research/interpreting-lm-parameters
- Branch: main
- Wandb for run in paper: https://wandb.ai/goodfire/spd/runs/s-55ea3f9b
- Comparison CLTs/PLTs: https://github.com/bartbussmann/nn_decompositions/tree/vpd_paper

**SPD paper (June 2025)**
- Paper: https://arxiv.org/abs/2506.20790
- Branch: [spd-paper](https://github.com/goodfire-ai/param-decomp/tree/spd-paper)
- Wandb report: https://wandb.ai/goodfire/spd-tms/reports/SPD-paper-report--VmlldzoxMzE3NzU0MQ

## App

This project ships a web app for visualising and interpreting decompositions. You can point it at
any decomposed run, including ones we've already trained and stored on wandb (e.g. the canonical
`goodfire/spd/runs/s-55ea3f9b` below). At present, viewing a run still requires running the
**harvest** and **autointerp** post-processing stages yourself — these produce the artifacts the
app reads.

```bash
make install-app   # Install frontend dependencies (one-time)
make app           # Launch backend + frontend dev servers
```

See the app's [README](param_decomp/app/README.md) and
[CLAUDE.md](param_decomp/app/CLAUDE.md) for details.

## Nano Parameter Decomposition

[`nano_param_decomp/`](nano_param_decomp/) is a self-contained, single-file implementation of the
whole method. It deliberately omits alternative loss/CI/sigmoid types and various logging for brevity.

## Installation

From the root of the repository, run one of:

```bash
make install-dev  # Install the package, dev requirements, pre-commit hooks
make install      # Install the package only (`pip install -e .`)
```

## Experiments

Run an experiment locally with `pd-local <name>`, or on SLURM with `pd-run --experiments <name>`
(adds git snapshot + W&B view; also supports `--dp N`, `--cpu`, and `--sweep --n_agents N`). The
two main language-model decompositions:

- **`pile_llama_simple_mlp-4L`** — 4-layer Llama (MLP-only) on the Pile; the VPD paper run
  [`goodfire/spd/runs/s-55ea3f9b`](https://wandb.ai/goodfire/spd/runs/s-55ea3f9b)
  ([config](param_decomp/experiments/lm/pile_llama_simple_mlp-4L.yaml)).
- **`ss_llama_simple_mlp-2L`** — 2-layer Llama (MLP-only) on
  [SimpleStories](https://arxiv.org/abs/2504.09184); smaller and faster
  ([config](param_decomp/experiments/lm/ss_llama_simple_mlp-2L.yaml)).

Other registered experiments (TMS, ResidualMLP, induction heads, GPT-2 / TinyStories variants) are
listed in [`param_decomp/registry.py`](param_decomp/registry.py). The `lm` experiment can decompose
any HuggingFace-loadable model whose target modules are `nn.Linear`, `nn.Embedding`, or
`transformers.modeling_utils.Conv1D`.

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
`.vscode/settings.json` to use them. We are unlikely to be able to action new features, though
issue reports are greatly appreciated!

Useful `make` targets:

```bash
make check     # Run pre-commit on all files (basedpyright, ruff lint, ruff format)
make type      # basedpyright only
make format    # ruff lint + format
make test      # Tests not marked `slow`
make test-all  # All tests
```
