# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

**IMPORTANT**: Always activate the virtual environment before running Python or git operations:

```bash
source .venv/bin/activate
```
If working in a worktree, make sure there's a local `.venv` first by running `uv sync` in the worktree directory. Do NOT `cd` to the main repo — all commands (including git) should run in the worktree.

Repo requires `.env` file with WandB credentials (see `.env.example`)

## Project Overview

PD is a research framework for analyzing neural network components and their interactions through sparse parameter decomposition techniques.

- Target model parameters are decomposed as a sum of `parameter components`
- Parameter components approximate target model outputs despite differentiable stochastic masks
- Causal importance functions quantify how much each component can be masked on each datapoint
- Multiple loss terms balance faithfulness, output reconstruction quality, and component activation sparsity

The codebase supports three experimental domains: TMS (Toy Model of Superposition), ResidualMLP (residual MLP analysis), and Language Models.

The `lm` experiment can decompose any HuggingFace-loadable model whose target modules are
`nn.Linear`, `nn.Embedding`, or `transformers.modeling_utils.Conv1D`.

## Public API

The core PD framework exposes a single training entrypoint:

```python
from param_decomp import (
    optimize, PDConfig, RuntimeConfig, RunSink, Metric,
    MetricConfig, LossMetricConfig, RunBatch, ReconstructionLoss,
)
```

- `optimize(target_model, train_loader, eval_loader, *, run_batch, reconstruction_loss,
  pd_config, runtime_config, sink, eval_metrics, device)`: the only entrypoint. Caller
  supplies the target `nn.Module`, dataloaders, the run-batch / reconstruction callables,
  the two configs, a `RunSink` for outputs and cadence, and a list of pre-instantiated
  eval `Metric` objects. `optimize()` builds the `ComponentModel` internally and calls
  `Metric.bind(model, device)` on every eval metric before the loop.
- `PDConfig`: algorithm spec (CI fn, loss metrics, module patterns, optimizers, seed,
  tied weights, faithfulness warmup, …). Loss metrics live here as
  `loss_metrics: dict[str, LossMetricConfig]`.
- `RuntimeConfig`: substrate (autocast_bf16, device, dp).
- `RunSink`: output channels + cadence. Construct via `RunSink.local(out_dir, ...)`,
  `RunSink.with_wandb(out_dir, project=..., ...)`, or `RunSink.silent(...)` for tests.
  Frequency fields (`train_log_freq`, `eval_freq`, `slow_eval_freq`, `n_eval_steps`,
  `slow_eval_on_first_step`, `save_freq`) live on the sink. Cadence gating
  (`should_log_train`, `should_eval`, `should_run_slow_eval`, `should_save`) and side
  effects (`log`, `console`, `checkpoint`, `finish`) are methods on `RunSink`.
- `Metric` base class with `__init__(cfg)` and `bind(*, model, device)`. Built-in
  metrics live under `param_decomp/metrics/builtin/` and self-register via
  `@register_metric`. Loss metrics (subclasses of `LossMetricConfig`) are instantiated
  inside `optimize()` from `pd_config.loss_metrics`; eval metrics are caller-supplied.

### Adding a new experiment

Experiments are plain Python scripts, not drivers/subclasses. A "new experiment" is just
a `run.py` that builds the target model, dataloaders, eval metrics, configs, and sink,
then calls `optimize()`. The three in-repo experiments
(`param_decomp_lab/experiments/{tms,resid_mlp,lm}/run.py`) are the canonical references.
Shared YAML-parsing helpers live in `param_decomp_lab/experiments/utils.py`
(`load_yaml`, `build_eval_metrics`, `run_sink_from_logging_block`).

Per-experiment console entry points are declared in `pyproject.toml`:

```bash
pd-tms        path/to/config.yaml
pd-resid-mlp  path/to/config.yaml
pd-lm         path/to/config.yaml
```

For a brand-new experiment, drop a `run.py` next to a YAML config in your own package
and either call its `main(...)` directly or wire it up to a console script.

### Custom Metrics

Built-in metrics live under `param_decomp/metrics/builtin/` and self-register via
`@register_metric` (see `param_decomp/metrics/registry.py`). External users can register
their own metrics by listing import targets in `pd.metric_modules`:

```yaml
pd:
  metric_modules:
    - my_pkg.my_metrics                   # dotted module name, importable from the env
  loss_metrics:
    MyLoss:
      coeff: 1.0
      my_param: 0.5
```

The user's module imports `register_metric` and `LossMetricConfig` / `MetricConfig` /
`MetricContext` from `param_decomp.metrics`, defines a `@register_metric` Metric class
with a `config_type` ClassVar pointing at their pydantic config, and that's it. A
`@model_validator(mode="before")` on `PDConfig` imports these modules before
`loss_metrics` is validated.

For **eval** metrics, the experiment `run.py` instantiates them directly (using
`build_eval_metrics(eval_metrics_dict)` from `experiments.utils` if loading from a YAML
dict-of-configs) and passes the list to `optimize(eval_metrics=...)`.

### Configs

The PD trainer is configured by two pydantic configs plus a `RunSink`:

- **`PDConfig`** — algorithm: seed, ci_config, loss_metrics, optimizers, module_info,
  tied_weights, faithfulness warmup. Flipping any field here changes what algorithm runs.
- **`RuntimeConfig`** — compute substrate: autocast_bf16, device, dp. Perturbs numerics
  without changing the algorithm.
- **`RunSink`** — observation + outputs: train/eval/save cadence, on-disk `out_dir`, and
  optional W&B session. Sink methods own all side effects.

### Saved run layout

```
PARAM_DECOMP_OUT_DIR/decompositions/<run_id>/
  model_<step>.pth           # PD checkpoints (written by RunSink.checkpoint(...))
  <step>.json                # local logs (written by RunSink.log(...))
```

## Research Papers

This repository implements methods from two key research papers on parameter decomposition:

**Stochastic Parameter Decomposition (SPD)**

- [`papers/Stochastic_Parameter_Decomposition/spd_paper.md`](papers/Stochastic_Parameter_Decomposition/spd_paper.md)
- A version of this repository was used to run the experiments in this paper. But we continue to develop on the code, so it no longer is limited to the implementation used for this paper.
- Introduces the core SPD framework
- Details the stochastic masking approach and optimization techniques used throughout the codebase
- Useful reading for understanding the implementation details, though may be outdated.

**Attribution-based Parameter Decomposition (APD)**

- [`papers/Attribution_based_Parameter_Decomposition/apd_paper.md`](papers/Attribution_based_Parameter_Decomposition/apd_paper.md)
- This paper was the precursor to SPD.
- It introduced the concept of linear parameter decomposition.
- Contains theoretical foundations, broader context, and high-level conceptual insights of parameter decomposition methods.
- Useful for understanding the conceptual framework and motivation behind SPD

## Development Commands

**Setup:**

- `make install-dev` - Install package with dev dependencies and pre-commit hooks
- `make install` - Install package only (`pip install -e .`)

**Code Quality:**

- `make check` - Run full pre-commit suite (basedpyright, ruff lint, ruff format)
- `make type` - Run basedpyright type checking only
- `make format` - Run ruff linter and formatter

**Testing:**

- `make test` - Run tests (excluding slow tests)
- `make test-all` - Run all tests including slow ones
- `python -m pytest tests/test_specific.py` - Run specific test file
- `python -m pytest tests/test_specific.py::test_function` - Run specific test

**Running the App:**

- `make app` - Launch the PD visualization app (backend + frontend)

## Architecture Overview

**Core PD Framework:**

- `param_decomp/optimize.py` - The PD optimization loop (`optimize(...)`). The sole core entrypoint.
- `param_decomp/configs.py` - `PDConfig` (algorithm) + `RuntimeConfig` (substrate) + their nested
  helpers (`ScheduleConfig`, `OptimizerConfig`, `LayerwiseCiConfig`, etc.).
- `param_decomp/run_sink.py` - `RunSink`: output channels (local + wandb) + cadence + checkpointing.
- `param_decomp/eval.py` - `evaluate(...)` over a `dict[name, Metric]`.
- `param_decomp/models/component_model.py` - Core ComponentModel that wraps target models.
- `param_decomp/models/components.py` - Component types (LinearComponent, EmbeddingComponent, etc.).
- `param_decomp/models/batch_and_loss_fns.py` - `RunBatch` / `ReconstructionLoss` protocols +
  `run_batch_*` / `recon_loss_*` helpers.
- `param_decomp/metrics/` - Self-registering `Metric` classes (losses, eval metrics, figures).
  `metrics/base.py` defines the `Metric` ABC with `__init__(cfg)` + `bind(model, device)`.
  `metrics/builtin/*.py` ship the in-tree implementations.

**In-repo experiment scripts** (`param_decomp_lab/experiments/{tms,resid_mlp,lm}/run.py`) build the
target model + dataloaders + eval metrics + configs + sink and call `optimize()`. They share
YAML-parsing helpers in `param_decomp_lab/experiments/utils.py` (`load_yaml`, `build_eval_metrics`,
`run_sink_from_logging_block`).

**Terminology: Sources vs Masks:**

- **Sources** (`adv_sources`, `PPGDSources`, `self.sources`): The raw values that PGD optimizes adversarially. These are interpolated with CI to produce component masks: `mask = ci + (1 - ci) * source`. Used in both regular PGD (`param_decomp/metrics/pgd_utils.py`) and persistent PGD (`param_decomp/persistent_pgd.py`).
- **Masks** (`component_masks`, `RoutingMasks`, `make_mask_infos`, `n_mask_samples`): The materialized per-component masks used during forward passes. These are produced from sources (in PGD) or from stochastic sampling, and are a general PD concept across the whole codebase.

**Experiment Structure:**

Each experiment (`param_decomp_lab/experiments/{tms,resid_mlp,lm}/`) contains:

- `run.py` - Composition root: parses YAML, builds target/loaders/metrics/configs/sink, calls `optimize()`.
- `*_config.yaml` - Built-in YAML configs.
- `models.py` (TMS/ResidMLP) / `data.py` (LM) - Model/data helpers.
- `train_*.py` (TMS/ResidMLP) - Target-model pretraining scripts.
- `plotting.py` (TMS/ResidMLP) - Visualization utilities.

**Key Data Flow:**

1. The experiment script (`python -m param_decomp_lab.experiments.<kind>.run config.yaml`) reads
   the YAML, validates `PDConfig` / `RuntimeConfig` / per-experiment `target` + `data` blocks,
   and builds the target `nn.Module`, train/eval dataloaders, and eval `Metric` list.
2. The script builds a `RunSink` (local + optional wandb) from the YAML `logging:` block.
3. It calls `optimize(...)` with all of the above. `optimize()` constructs `ComponentModel`,
   binds eval metrics, instantiates loss metrics from `pd_config.loss_metrics`, and runs the
   training loop. Side effects (logging, checkpoints) flow through `RunSink`.

**Configuration System:**

- YAML experiment configs define parameters under `pd:`, `runtime:`, `target:`, `data:`, `logging:`.
- Pydantic models provide type safety and validation.
- WandB integration for experiment tracking and model storage (via `RunSink.with_wandb(...)`).

**Output Directory (`PARAM_DECOMP_OUT_DIR`):**

- Defined in `param_decomp/settings.py`.
- On cluster: `/mnt/polished-lake/artifacts/mechanisms/param-decomp/`.
- Off cluster: `~/param_decomp_out/`.
- Contains decompositions/ and pretrain run outputs.

## Directory Structure

```
<repo-root>/
├── papers/                          # Research papers (SPD, APD)
├── scripts/                         # Standalone utility scripts
├── tests/                           # Test suite
├── param_decomp/                    # Core library (the only thing externals import)
│   ├── metrics/                     # Self-registering Metric classes (losses, eval metrics, figures)
│   ├── models/                      # ComponentModel + components + batch_and_loss_fns
│   ├── utils/                       # Distributed / wandb / general helpers
│   ├── configs.py                   # PDConfig, RuntimeConfig (+ ScheduleConfig/OptimizerConfig/...)
│   ├── optimize.py                  # optimize() — the core entrypoint
│   ├── run_sink.py                  # RunSink (output + cadence)
│   ├── eval.py                      # evaluate(instances, ...)
│   └── settings.py                  # PARAM_DECOMP_OUT_DIR, SLURM_LOGS_DIR, SBATCH_SCRIPTS_DIR
├── param_decomp_lab/                # Lab tooling — experiments, post-processing, app
│   ├── experiments/
│   │   ├── tms/, resid_mlp/, lm/    # Each: run.py + YAMLs + per-experiment helpers
│   │   ├── spec.py                  # ExperimentSpec (compositional dispatch dataclass)
│   │   ├── utils.py                 # load_yaml / build_eval_metrics / run_sink_from_logging_block / save_run_meta
│   │   └── __init__.py              # EXPERIMENTS registry (name → ExperimentSpec)
│   ├── pretrain/                    # Target model pretraining (see pretrain/CLAUDE.md)
│   ├── harvest/                     # Statistics collection (see harvest/CLAUDE.md)
│   ├── autointerp/                  # LLM interpretation (see autointerp/CLAUDE.md)
│   ├── clustering/                  # Component clustering (see clustering/CLAUDE.md)
│   ├── dataset_attributions/        # Dataset attributions (see dataset_attributions/CLAUDE.md)
│   ├── graph_interp/                # Context-aware interpretation (see graph_interp/CLAUDE.md)
│   ├── postprocess/                 # Unified postprocessing pipeline
│   ├── investigate/                 # Agent investigation (see investigate/CLAUDE.md)
│   ├── app/                         # Web visualization app (see app/CLAUDE.md)
│   ├── topology/, adapters/, editing/  # Model-topology utilities
│   ├── scripts/                     # alpha_sweep, prompt_utils
│   └── saved_run.py                 # SavedRun: reload a PD run via its ExperimentSpec
├── Makefile                         # Dev commands (make check, make test)
└── pyproject.toml                   # Package config (both packages exported)
```

## Quick Navigation

### CLI Entry Points

| Command | Entry Point | Description |
|---------|-------------|-------------|
| `pd-tms` | `param_decomp_lab/experiments/tms/run.py` | Run the TMS experiment for the given YAML config |
| `pd-resid-mlp` | `param_decomp_lab/experiments/resid_mlp/run.py` | Run the ResidMLP experiment for the given YAML config |
| `pd-lm` | `param_decomp_lab/experiments/lm/run.py` | Run the LM experiment for the given YAML config |
| `pd-pretrain` | `param_decomp_lab/pretrain/scripts/run_slurm.py` | Pretrain target models |
| `pd-harvest` | `param_decomp_lab/harvest/scripts/run_slurm_cli.py` | Submit harvest SLURM job |
| `pd-autointerp` | `param_decomp_lab/autointerp/scripts/run_slurm_cli.py` | Submit autointerp SLURM job |
| `pd-attributions` | `param_decomp_lab/dataset_attributions/scripts/run_slurm_cli.py` | Submit dataset-attribution SLURM job |
| `pd-postprocess` | `param_decomp_lab/postprocess/cli.py` | Unified postprocessing pipeline |
| `pd-graph-interp` | `param_decomp_lab/graph_interp/scripts/run_slurm_cli.py` | Submit graph-interpretation SLURM job |
| `pd-clustering` | `param_decomp_lab/clustering/scripts/run_pipeline.py` | Clustering ensemble pipeline |
| `pd-cluster-harvest` | `param_decomp_lab/clustering/scripts/run_harvest.py` | Harvest activations → membership snapshot |
| `pd-cluster-merge` | `param_decomp_lab/clustering/scripts/run_merge.py` | Merge from snapshot (CPU-only) |
| `pd-intruder` | `param_decomp_lab/harvest/scripts/run_intruder_slurm_cli.py` | Submit intruder eval job |
| `pd-investigate` | `param_decomp_lab/investigate/scripts/run_slurm_cli.py` | Submit agent-investigation job |

### Files to Skip When Searching

Use `param_decomp/` as the search root (not repo root) to avoid noise.

**Always skip:**

- `.venv/` - Virtual environment
- `__pycache__/`, `.pytest_cache/`, `.ruff_cache/` - Build artifacts
- `node_modules/` - Frontend dependencies
- `.git/` - Version control
- `.data/` - Runtime data/caches
- `notebooks/` - Analysis notebooks (unless explicitly relevant)
- `wandb/` - WandB local files

**Usually skip unless relevant:**

- `tests/` - Test files (unless debugging test failures)
- `papers/` - Research paper drafts

### Common Call Chains

**Running Experiments:**

- `python -m param_decomp_lab.experiments.<kind>.run path/to/config.yaml` (also exposed as
  `pd-tms` / `pd-resid-mlp` / `pd-lm`) → reads YAML → builds target/loaders/metrics →
  calls `optimize(...)`.

## Common Usage Patterns

### Running Experiments

In-place experiment scripts read a YAML and call `optimize()`:

```bash
pd-tms       param_decomp_lab/experiments/tms/tms_5-2_config.yaml
pd-resid-mlp param_decomp_lab/experiments/resid_mlp/resid_mlp1_config.yaml
pd-lm        param_decomp_lab/experiments/lm/ss_llama_simple_mlp-2L.yaml
```

For a new experiment, drop a new `run.py` next to your YAML and either invoke it
directly with Python or wire it up to a console script in `pyproject.toml`.

### Cluster Usage Guidelines

- DO NOT use more than 8 GPUs at one time
- This includes not setting off multiple sweeps/evals that total >8 GPUs
- Monitor jobs with: `squeue --format="%.18i %.9P %.15j %.12u %.12T %.10M %.9l %.6D %b %R" --me`

## Coding Guidelines & Software Engineering Principles

**This is research code, not production. Prioritize simplicity and fail-fast over defensive programming.**

Core principles:

- **Fail fast** - assert assumptions, crash on violations, don't silently recover
- **No legacy support** - delete unused code, don't add fallbacks for old formats or migration shims
- **Narrow types** - avoid `| None` unless null is semantically meaningful; use discriminated unions over bags of optional fields
- **No try/except for control flow** - check preconditions explicitly, then trust them
- **YAGNI** - don't add abstractions, config options, or flexibility for hypothetical futures

```python
# BAD - defensive, recovers silently, wide types
def get_config(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return None

config = get_config(path)
if config is not None:
    value = config.get("key", "default")

# GOOD - fail fast, narrow types, trust preconditions
def get_config(path: Path) -> Config:
    assert path.exists(), f"config not found: {path}"
    with open(path) as f:
        data = json.load(f)
    return Config(**data)  # pydantic validates

config = get_config(path)
value = config.key
```

### Tests

- The point of tests in this codebase is to ensure that the code is working as expected, not to prevent production outages - there's no deployment here. Therefore, don't worry about lots of larger integration/end-to-end tests. These often require too much overhead for what it's worth in our case, and this codebase is interactively run so often that issues will likely be caught by the user at very little cost.

### Assertions and error handling

- If you have an invariant in your head, assert it. Are you afraid to assert? Sounds like your program might already be broken. Assert, assert, assert. Never soft fail.
- Do not write: `if everythingIsOk: continueHappyPath()`. Instead do `assert everythingIsOk`
- You should have a VERY good reason to handle an error gracefully. If your program isn't working like it should then it shouldn't be running, you should be fixing it.
- Do not write `try-catch` blocks unless it definitely makes sense
- **Write for the golden path.** Never let edge cases bloat the code. Before handling them, just raise an exception. If an edge case becomes annoying enough, we'll handle it then — but write first and foremost for the common case.

### Control Flow

- Keep I/O as high up as possible. Make as many functions as possible pure.
- Prefer `match` over `if/elif/else` chains when dispatching on conditions - more declarative and makes cases explicit
- If you either have (a and b) or neither, don't make them both independently optional. Instead, put them in an optional tuple

### Types, Arguments, and Defaults

- Write your invariants into types as much as possible.
- Use jaxtyping for tensor shapes (though for now we don't do runtime checking)
- Always use the PEP 604 typing format of `|` for unions and `type | None` over `Optional`.
- Use `dict`, `list` and `tuple` not `Dict`, `List` and `Tuple`
- Don't add type annotations when they're redundant. (i.e. `my_thing: Thing = Thing()` or `name: str = "John Doe"`)
- Differentiate no data from empty collections. Often it's important to differentiate `None` from `[]`
- Don't use bare dictionaries for structures whose values aren't homogenous
  - good: {<id>: <val>}
  - bad: {"tokens": …, "loss": …}
- Default args are rarely a good idea. Avoid them unless necessary. You should have a very good reason for having a default value for an argument, especially if it's caller also defaults to the same thing
- This repo uses basedpyright (not mypy)
- Keep defaults high in the call stack.
- Don't use `from __future__ import annotations` — use string quotes for forward references instead.

### Tensor Operations

- Try to use einops by default for clarity.
- Assert shapes liberally
- Document complex tensor manipulations

### Comments

- Comments hide sloppy code. If you feel the need to write a comment, consider that you should instead
  - name your functions more clearly
  - name your variables more clearly
  - separate a chunk of logic into a function
  - separate an inlined computation into a meaningfully named variable
- Don’t write dialogic / narrativised comments or code. Instead, write comments that describe
  the code as is, not the diff you're making. Examples of narrativising comments:
  - `# the function now uses y instead of x`
  - `# changed to be faster`
  - `# we now traverse in reverse`
- Here's an example of a bad diff, where the new comment makes reference to a change in code, not just the state of the code:

```
95 -      # Reservoir states
96 -      reservoir_states: list[ReservoirState]
95 +      # Reservoir state (tensor-based)
96 +      reservoir: TensorReservoirState
```

### Other Important Software Development Practices

- Don't add legacy fallbacks or migration code - just change it and let old data be manually migrated if needed.
- Delete unused code.
- If an argument is always x, strongly consider removing as an argument and just inlining
- **Update CLAUDE.md files** when changing code structure, adding/removing files, or modifying key interfaces. Update the CLAUDE.md in the same directory (or nearest parent) as the changed files.

### GitHub

- To view github issues and PRs, use the github cli (e.g. `gh issue view 28` or `gh pr view 30`).
- When making PRs, use the github template defined in `.github/pull_request_template.md`.
- Before committing, ALWAYS ensure you are on the correct branch and do not use `git add .` to add all unstaged files. Instead, add only the individual files you changed, don't commit all files.
- Use branch names `refactor/X` or `feature/Y` or `fix/Z`.
- NEVER use `--no-verify` to skip pre-commit hooks. They are there for a good reason. If pre-commit hooks fail, you MUST fix the underlying problem.
