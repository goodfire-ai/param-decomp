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

## Package Layout

This repo contains two flat-layout Python distributions:

- `param-decomp` is configured by the root `pyproject.toml`, installs only
  `param_decomp`, and exposes the core public API.
- `param-decomp-lab` is configured by `param_decomp_lab/pyproject.toml`, installs
  `param_decomp_lab`, depends on `param-decomp`, and owns all `pd-*` CLI entry points.

Local development uses the uv workspace in the root `pyproject.toml`; `make install-dev`
syncs all workspace packages editably so both `import param_decomp` and
`import param_decomp_lab` work.

## Public API

Import every name from where it is defined — there are no package-level re-exports. The
core training entrypoint and the configs/protocols it consumes:

```python
from param_decomp.optimize import optimize
from param_decomp.configs import PDConfig, RuntimeConfig
from param_decomp.run_sink import RunSink
from param_decomp.metrics.base import LossMetricConfig, Metric
from param_decomp.batch_and_loss_fns import RunBatch, ReconstructionLoss
```

- `optimize(target_model, train_loader, eval_loader, *, run_batch, reconstruction_loss,
  pd_config, runtime_config, sink, eval_metrics, device)`: the only entrypoint. Caller
  supplies the target `nn.Module`, dataloaders, the run-batch / reconstruction callables,
  the two configs, a `RunSink` for outputs and cadence, and a list of pre-instantiated
  eval `Metric` objects. `optimize()` builds the `ComponentModel` internally and calls
  `Metric.bind(model, device)` on every eval metric before the loop.
- `PDConfig`: algorithm spec (CI fn, loss metrics, module patterns, optimizers, seed,
  tied weights, faithfulness warmup, …). Loss metrics live here as
  `loss_metrics: list[AnyLossMetricConfig]` — a pydantic discriminated union over each
  metric's `type` literal.
- `RuntimeConfig`: substrate (autocast_bf16, device, dp).
- `RunSink`: a `Protocol` in `param_decomp/run_sink.py` describing the cadence
  gates (`should_log_train`, `should_eval`, `should_run_slow_eval`, `should_save`)
  and side-effect methods (`log`, `console`, `checkpoint`) that `optimize()` calls.
  Concrete implementations live with the caller. The lab ships
  `param_decomp_lab.run_sink.RunSink` (local files + wandb + non-main-rank no-op)
  with `.local(...)`, `.with_wandb(...)`, and `.silent(...)` constructors; tests
  and in-repo experiments use it.
- `Metric` base class with `__init__(cfg)` and `bind(*, model, device)`. Loss metrics ship
  in `param_decomp/metrics/`. Pydantic validates each `pd_config.loss_metrics` entry
  directly via the `AnyLossMetricConfig` discriminated union; `optimize()` then
  instantiates the matching `Metric` subclass using the type→class dispatch table
  `param_decomp.metrics.loss_metrics.LOSS_METRIC_CLASSES`. Eval metrics are
  caller-supplied — the lab ships its own set under `param_decomp_lab/eval_metrics/`,
  exposed via `param_decomp_lab.eval_metrics.EVAL_METRICS` for the in-repo experiments'
  YAML wiring.

### Adding a new experiment

Experiments are plain Python scripts, not drivers/subclasses. A "new experiment" is just
a `run.py` that builds the target model, dataloaders, eval metrics, configs, and sink,
then calls `optimize()`. The three in-repo experiments
(`param_decomp_lab/experiments/{tms,resid_mlp,lm}/run.py`) are the canonical references.
Shared YAML-parsing helpers live in `param_decomp_lab/experiments/utils.py`
(`load_yaml`, `build_eval_metrics`, `run_sink_from_logging_block`).

Per-experiment console entry points are declared in `param_decomp_lab/pyproject.toml`:

```bash
pd-tms        path/to/config.yaml
pd-resid-mlp  path/to/config.yaml
pd-lm         path/to/config.yaml
```

For a brand-new experiment, drop a `run.py` next to a YAML config in your own package
and either call its `main(...)` directly or wire it up to a console script.

### Metrics

Loss metrics ship in `param_decomp/metrics/`. Each `LossMetricConfig` carries a
`type: Literal["<ClassName>"]` discriminator. `configs.py` assembles them into
`AnyLossMetricConfig` (the discriminated union) so pydantic validates every
`pd.loss_metrics` entry without a custom validator. The runtime dispatch from `type`
literal to `Metric` subclass lives in
`param_decomp/metrics/dispatch.py::LOSS_METRIC_CLASSES`. Adding a new loss metric
means: (1) define the `Metric` subclass + its `LossMetricConfig` with a unique `type`
literal, (2) append the config to the `AnyLossMetricConfig` union in `configs.py`, and
(3) append the class to `LOSS_METRIC_CLASSES`.

Eval metrics are caller-supplied. Users instantiate `Metric` objects in their `run.py`
and pass them to `optimize(eval_metrics=...)`. The in-repo experiments share a YAML-driven
helper `build_eval_metrics(eval_metrics_dict)` in `experiments/utils.py`, backed by
`param_decomp_lab.eval_metrics.EVAL_METRICS` — adding a new lab eval metric means dropping the
class in `param_decomp_lab/eval_metrics/` and appending it there.

### Configs

The PD trainer is configured by two pydantic configs plus a `RunSink`:

- **`PDConfig`** — algorithm: seed, ci_config, loss_metrics, optimizers, decomposition targets,
  tied_weights, faithfulness warmup. Flipping any field here changes what algorithm runs.
- **`RuntimeConfig`** — compute substrate: autocast_bf16, device, dp. Perturbs numerics
  without changing the algorithm.
- **`RunSink`** (caller-supplied, Protocol in core) — observation + outputs: train/eval/save cadence and
  optional W&B session. Sink methods own all side effects.

`PDConfig`, `RuntimeConfig`, `OptimizerConfig`, and `AnyLossMetricConfig` live in
`configs.py`. Configs that have a clear implementation home live next to that
implementation, and callers import them from that implementation module directly:

- `ScheduleConfig` → `param_decomp.schedule` (next to `get_scheduled_value`)
- `DecompositionTargetConfig` → `param_decomp.decomposition_targets` (next to
  `DecompositionTarget` and `resolve_decomposition_targets`)
- `CiConfig` and friends (`LayerwiseCiConfig`, `AttnConfig`,
  `GlobalSharedTransformerCiConfig`, `GlobalCiConfig`) → `param_decomp.ci_fns`
  (next to the CI-fn `nn.Module` classes they configure)
- `masks.py` configs (`SamplingType`, `SubsetRoutingType` + members) → the
  Router implementations and runtime mask payload helpers in the same file
- Each metric's `LossMetricConfig` subclass → its `Metric` class in `metrics/<name>.py`

#### Config placement and import cycles

The rule used to decide where each config lives:

1. **Default:** keep the config in `configs.py`.
2. **Move the config next to its implementation when leaving it in `configs.py`
   would close an import cycle.** Concretely: if module `M` defines the
   implementation that consumes the config and is also (transitively) imported
   by `configs.py` — usually via the metric union — then `M → configs` closes
   the loop. Put the config in `M` instead and update callers to import it from
   `M` directly. Do not add a re-export to `configs.py`.
3. **Never use `if TYPE_CHECKING:` + forward-reference strings to paper over an
   import cycle.** If you find yourself reaching for it, the placement is wrong;
   apply rule 2 instead.

#### No package-level re-exports

`__init__.py` files are bare. Every name is imported from the module that
defines it (e.g. `from param_decomp.schedule import ScheduleConfig`, not
`from param_decomp.configs import ScheduleConfig` and not `from param_decomp
import optimize`). This keeps the import path canonical and avoids stale
intermediary indirection.

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

- `make install-dev` - Install all workspace packages with dev dependencies and pre-commit hooks
- `make install` - Install the core `param-decomp` package only
- `make install-lab` - Install core + lab packages without dev dependencies

**Code Quality:**

- `make check` - Run full pre-commit suite (basedpyright, ruff lint, ruff format)
- `make type` - Run basedpyright type checking only
- `make format` - Run ruff linter and formatter

**Testing:**

- `make test` - Run tests (excluding slow tests)
- `make test-all` - Run all tests including slow ones
- `python -m pytest param_decomp/tests/test_specific.py` - Run a specific core test file
- `python -m pytest param_decomp_lab/tests/test_specific.py::test_function` - Run a specific lab test

**Running the App:**

- `make app` - Launch the PD visualization app (backend + frontend)

## Architecture Overview

**Core PD Framework:**

- `param_decomp/optimize.py` - The PD optimization loop (`optimize(...)`). The sole core entrypoint.
- `param_decomp/configs.py` - `PDConfig` (algorithm) + `RuntimeConfig` (substrate) +
  `OptimizerConfig` + `AnyLossMetricConfig` discriminated union. Configs that live
  in other modules (`ScheduleConfig`, `DecompositionTargetConfig`, the `CiConfig`
  family, etc.) are imported here only to wire up `PDConfig` / `AnyLossMetricConfig`
  — callers import them from their implementation modules directly. See "Config
  placement and import cycles" above for the placement rule.
- `param_decomp/masks.py` - Runtime mask payloads (`ComponentsMaskInfo`,
  `RoutingMasks`, `WeightDeltaAndMask`, `make_mask_infos`) plus `SamplingType` /
  `SubsetRoutingType` configs, Router implementations, and stochastic mask helpers.
- `param_decomp/run_sink.py` - `RunSink` Protocol: the contract `optimize()` calls (cadence + log/console/checkpoint).
- `param_decomp_lab/run_sink.py` - Concrete `RunSink` for in-repo use: local files + wandb, with `.local`/`.with_wandb`/`.silent` constructors and rank-aware no-op fan-out.
- `param_decomp/component_model.py` - Core ComponentModel that wraps target models.
- `param_decomp/components.py` - Component types (LinearComponent, EmbeddingComponent, etc.) + `init_param_` + `make_components` / `get_module_input_dim` factories.
- `param_decomp/ci_fns.py` - CI configs (`CiConfig` family + `AttnConfig`) + CI-fn modules (`MLPCiFn`, `VectorMLPCiFn`, `VectorSharedMLPCiFn`, `GlobalSharedMLPCiFn`, `GlobalSharedTransformerCiFn`) + wrappers (`LayerwiseCiFnWrapper`, `GlobalCiFnWrapper`) + `make_ci_fn_wrapper` factory.
- `param_decomp/ci_nn_blocks.py` - Generic transformer building blocks used by the CI fns (`Linear`, `ParallelLinear`, `RoPEEmbedding`, `SelfAttention`, `TransformerBlock`).
- `param_decomp/sigmoids.py` - Sigmoid variants (`SIGMOID_TYPES`) used by CI fn output squashing.
- `param_decomp/batch_and_loss_fns.py` - `RunBatch` / `ReconstructionLoss` protocols
  + `move_batch_to_device`. The concrete `run_batch_*` / `recon_loss_*` helpers live in
  `param_decomp_lab/batch_and_loss_fns.py` (caller-supplied; `optimize()` doesn't import them).
- `param_decomp/decomposition_targets.py` - `DecompositionTargetConfig` + `resolve_decomposition_targets` for fnmatch-driven target selection, plus the `Identity` shim + `insert_identity_operations_` that registers it on target modules before decomposition.
- `param_decomp/metrics/` - Loss `Metric` classes + their `LossMetricConfig`s (one per file).
  `metrics/base.py` defines the `Metric` ABC with `__init__(cfg)` + `bind(model, device)`.
  `metrics/context.py` defines `MetricContext` (the per-step bundle passed to every
  `Metric.update(ctx)`). `metrics/dispatch.py` exposes `LOSS_METRIC_CLASSES`, the
  `type` literal → class dispatch table that `optimize()` uses to instantiate metrics
  from each `pd_config.loss_metrics` entry. PPGD's state machine lives in
  `metrics/persistent_pgd_state.py`; its configs + metric classes live in
  `metrics/persistent_pgd_recon.py`. Eval metrics ship in `param_decomp_lab/eval_metrics/`.

**In-repo experiment scripts** (`param_decomp_lab/experiments/{tms,resid_mlp,lm}/run.py`) build the
target model + dataloaders + eval metrics + configs + sink and call `optimize()`. They share
YAML-parsing helpers in `param_decomp_lab/experiments/utils.py` (`load_yaml`, `build_eval_metrics`,
`run_sink_from_logging_block`).

**Terminology: Sources vs Masks:**

- **Sources** (`adv_sources`, `PPGDSources`, `self.sources`): The raw values that PGD optimizes adversarially. These are interpolated with CI to produce component masks: `mask = ci + (1 - ci) * source`. Used in both regular PGD (`param_decomp/metrics/pgd_utils.py`) and persistent PGD (`param_decomp/metrics/persistent_pgd_state.py`).
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

- Defined in `param_decomp_lab/infra/settings.py`.
- On cluster: `/mnt/polished-lake/artifacts/mechanisms/param-decomp/`.
- Off cluster: `~/param_decomp_out/`.
- Contains decompositions/ and pretrain run outputs.

## Directory Structure

```
<repo-root>/
├── papers/                          # Research papers (SPD, APD)
├── scripts/                         # Standalone utility scripts
├── param_decomp/                    # Core library: training loop, configs, ComponentModel, loss metrics
│   ├── metrics/                     # Loss Metric classes (cfg+impl per file) + dispatch.py (LOSS_METRIC_CLASSES)
│   ├── tests/                       # Core-library test suite
│   ├── configs.py                   # PDConfig + RuntimeConfig + OptimizerConfig + AnyLossMetricConfig
│   ├── masks.py                     # Runtime mask payloads, Router impls, SamplingType / SubsetRoutingType configs, stochastic mask helpers
│   ├── optimize.py                  # optimize() — the core entrypoint (also holds loop_dataloader + collect_metric_outputs)
│   ├── run_sink.py                  # RunSink Protocol — contract optimize() calls
│   ├── component_model.py           # ComponentModel wrapper (target model + components + CI fn)
│   ├── components.py                # Components ABC + LinearComponents / EmbeddingComponents + make_components + get_module_input_dim
│   ├── ci_fns.py                    # CI configs (CiConfig family) + CI-fn nn.Modules + wrappers + make_ci_fn_wrapper
│   ├── ci_nn_blocks.py              # Generic NN scaffolding consumed by ci_fns (Linear, ParallelLinear, TransformerBlock, RoPE)
│   ├── sigmoids.py                  # Sigmoid variants used by CI fn output squashing
│   ├── distributed.py               # DistributedState + read-only state + reduce/gather collectives
│   ├── schedule.py                  # ScheduleConfig + get_scheduled_value
│   ├── torch_helpers.py             # bf16_autocast, get_obj_device, etc.
│   ├── batch_and_loss_fns.py        # RunBatch / ReconstructionLoss Protocols + move_batch_to_device
│   ├── decomposition_targets.py     # Decomposition target config + module-pattern resolution + Identity shim + insert_identity_operations_
│   ├── log.py                       # `logger` shared across the library
│   └── base_config.py               # Pydantic `BaseConfig` with YAML/JSON load/save, plus `Probability` annotated float and `runtime_cast`
├── param_decomp_lab/                # Lab tooling — experiments, post-processing, app, infra
│   ├── infra/                       # Cross-subsystem plumbing: settings, paths, slurm, wandb, sqlite, git, run_files, markdown, pydantic
│   ├── experiments/
│   │   ├── tms/, resid_mlp/, lm/    # Each: run.py + YAMLs + per-experiment helpers
│   │   ├── utils.py                 # load_yaml / build_eval_metrics / run_sink_from_logging_block / save_run_meta
│   │   ├── loadable_module.py       # LoadableModule ABC used by lab experiment models
│   │   └── __init__.py              # EXPERIMENTS registry (name → experiment module)
│   ├── eval_metrics/                # Eval Metric classes + EVAL_METRICS lookup table
│   ├── pretrain/                    # Target model pretraining (see pretrain/CLAUDE.md)
│   ├── harvest/                     # Statistics collection: pipeline + accumulator (see harvest/CLAUDE.md)
│   ├── autointerp/                  # LLM interpretation (see autointerp/CLAUDE.md)
│   ├── clustering/                  # Component clustering (see clustering/CLAUDE.md)
│   ├── dataset_attributions/        # Dataset attributions: pipeline + accumulator (see CLAUDE.md)
│   ├── graph_interp/                # Context-aware interpretation (see graph_interp/CLAUDE.md)
│   ├── postprocess/                 # Unified postprocessing pipeline
│   ├── investigate/                 # Agent investigation (see investigate/CLAUDE.md)
│   ├── app/                         # Web visualization app (see app/CLAUDE.md)
│   ├── topology/, adapters/, editing/  # Model-topology utilities
│   ├── models/                      # batch_and_loss_fns helpers (run_batch_*, recon_loss_*, calc_kl_divergence_lm) + component_model_utils (from_checkpoint, get_all_component_acts)
│   ├── utils/                       # seed.py (set_seed), data.py (SparseFeatureDataset, DatasetGeneratedDataLoader), distributed.py (init/cleanup/log0/get_device/ensure_cached_and_call)
│   ├── scripts/                     # alpha_sweep, prompt_utils
│   ├── tests/                       # Lab test suite
│   ├── target_ci.py                 # TargetCIPattern (Identity/Dense), TargetCISolution — for toy-model eval
│   ├── _linear_sum_assignment.py    # Vendored Hungarian algorithm (impl detail of target_ci)
│   ├── run_sink.py                  # Concrete RunSink (local files + wandb + rank-aware no-op)
│   └── saved_run.py                 # SavedRun: reload a PD run via its experiment module
├── Makefile                         # Dev commands (make check, make test)
├── pyproject.toml                   # Core param-decomp package + workspace config
└── param_decomp_lab/pyproject.toml  # Lab param-decomp-lab package + pd-* entry points
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

- `param_decomp/tests/`, `param_decomp_lab/tests/` - Test files (unless debugging test failures)
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

For a new lab experiment, drop a new `run.py` next to your YAML and either invoke it
directly with Python or wire it up to a console script in `param_decomp_lab/pyproject.toml`.

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
