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

Built-in experiments are auto-discovered from YAML configs under `param_decomp/experiments/<kind>/`
(TMS variants, ResidualMLP variants, and MLP-only Llama variants). See
`param_decomp/experiments/discovery.py` or run `pd-run --help` for the live list.

The `lm` experiment can decompose any HuggingFace-loadable model whose target modules are
`nn.Linear`, `nn.Embedding`, or `transformers.modeling_utils.Conv1D`.

## Public API

The core PD framework exposes these entrypoints, re-exported from `param_decomp/__init__.py`:

```python
from param_decomp import run_pd, load_component_model, PDConfig, PDTarget, PDRun, Run, ExperimentDriver
```

- `run_pd(config, logging_config, runtime_config, target, train_loader, eval_loader, device, *, run=None, artifacts=...)`:
  trains a decomposition. `PDConfig` carries algorithm/training settings; `PDTarget` bundles the
  target model + `run_batch` + reconstruction loss + optional tied weights. Core PD does not know
  about LM/TMS/etc. Helpers for the two `PDTarget` callables live in
  `param_decomp/models/batch_and_loss_fns.py`: `run_batch_passthrough`,
  `run_batch_first_element`, `make_run_batch(output_extract)`; and `recon_loss_mse`,
  `recon_loss_kl`. Callers can pass their own functions instead. `run` is a `Run`
  written to `run_metadata.yaml`; `artifacts` is a `{filename: data}` mapping for extra files
  saved beside the checkpoint.
- `load_component_model(path, *, target=None)`: reload a saved run as a `ComponentModel`. When `target` is
  omitted the run's driver reconstructs the target from the saved `Run`; pass `target=...`
  explicitly for runs produced via direct `run_pd` (no driver).
- `PDRun.from_path(path)`: handle to a saved run. Exposes `run` (`Run`),
  `pd_config`, `load_target()`, `load_dataloaders(...)`, and `load_model(target=None)`.

### Experiment Drivers

Experiments are open-world drivers, not a closed discriminated union in core code. A driver owns a
pure Pydantic `Run` subclass and converts it to runtime objects:

```python
class MyRun(Run):
    target: MyTargetConfig
    data: MyDataConfig

class MyDriver:
    name = "my_exp"                  # ClassVar[str] — wandb tag
    config_type = MyRun              # ClassVar[type[Run]]

    def build_target(self, run: MyRun) -> PDTarget: ...
    def build_dataloaders(self, run: MyRun, *, train_batch_size, eval_batch_size, ...): ...
```

`build_target` and `build_dataloaders` always fetch from upstream (wandb pretrain run, HF, …);
reload calls them exactly like a fresh run. Saved PD runs therefore depend on their upstream
continuing to exist — the wandb run path / HF model name in the config is the pin.

Built-in runtime definitions live in `param_decomp/experiments/{lm,tms,resid_mlp}/experiment.py`.
Custom users can run without editing core code by declaring the driver at the top of their YAML:

```yaml
driver_path: my_pkg.my_exp:MyDriver
pd: {...}
# ...
```

```bash
pd-run --config_path my_config.yaml
```

The driver import path is part of the saved `Run`, so reloading the run via `load_component_model(path)` can reconstruct the target without an explicit `target=` argument.

Callers can also bypass drivers entirely and call `run_pd` directly with their own `PDTarget` and
dataloaders — the right choice for notebook/script-driven use where `pd-run`, sweeps, and
post-processing tooling are not needed. Those runs reload with `load_component_model(path, target=...)`. The
README's "Custom experiments" section walks through both routes side-by-side.

### Per-experiment `Run` subclasses

Built-in YAML configs are pure `Run` configs nested under `driver_path:`, `pd:`,
`logging:`, `runtime:`, `target:`, and `data:`:

- `LMRun(driver_path, pd, logging, runtime, target: LMTargetConfig, data: LMDataConfig)`
- `TMSRun(driver_path, pd, logging, runtime, target, data)`
- `ResidMLPRun(driver_path, pd, logging, runtime, target, data)`

The three configs split by **what they affect**:

- **`PDConfig`** — algorithm specification: seed, ci_config, losses, optimizers,
  module_info. Flipping any field here changes what algorithm runs.
- **`RuntimeConfig`** — compute substrate: autocast_bf16, device, dp; future home for
  NCCL flags, gradient accumulation, fp8 variants. Perturbs numerics without changing
  the algorithm. **Config-only — no CLI overrides.** Edit the YAML (or copy it) to
  change substrate; you can't silently run "the same experiment" on different hardware.
  Cluster topology (GPUs per node) is `settings.GPUS_PER_NODE`, overridable via
  `PARAM_DECOMP_GPUS_PER_NODE` env var.
- **`LoggingConfig`** — observation: cadence (`*_freq`), `eval_batch_size`,
  `ci_alive_threshold`, eval-only metrics, plus `wandb_run_name` / `view_meta`. Never
  touches the optimizer. **Note:** the W&B *project* lives outside the `Run` (it's a
  deploy-time parameter — which account/team to log to) and is passed via `--project`
  to `pd-run` or `wandb_project=` to `run_pd`.

`Run` configs should not perform I/O. Put target loading and dataloader construction in
the driver.

### Saved run layout

```
PARAM_DECOMP_OUT_DIR/decompositions/<run_id>/
  run_metadata.yaml          # Run: driver_path + pd + logging + runtime + target + data
  model_<step>.pth           # PD checkpoints

PARAM_DECOMP_OUT_DIR/sweeps/<launch_id>/
  spec.yaml                  # SweepSpec snapshot (sweep launches only; single runs don't write this)
```

`Run` is a Pydantic model (defined in `param_decomp/run.py`):

```yaml
driver_path: "param_decomp.experiments.lm.experiment:Driver"   # null for notebook/custom runs
pd: {...}
logging:
  wandb_run_name: "seed=0_lr=1e-3"        # null lets W&B auto-name
  view_meta:                               # free-form labels (populated by sweep generators)
    lr_ratio: 0.1
    size: medium
  # ...other observation-only fields...
runtime: {...}                              # compute substrate
target: {...}                               # driver-specific (on the Run subclass)
data: {...}                                 # driver-specific (on the Run subclass)
```

`view_meta` is surfaced to W&B under a `view_meta/` prefix in `wandb.config` so the
UI can group/color runs by researcher-facing axes (not raw config fields). Populate
it from your sweep generator.

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
- `make install-app` - Install frontend dependencies (`npm install` in `param_decomp/app/frontend/`)

**Code Quality:**

- `make check` - Run full pre-commit suite (basedpyright, ruff lint, ruff format)
- `make type` - Run basedpyright type checking only
- `make format` - Run ruff linter and formatter

**Frontend (when working on `param_decomp/app/frontend/`):**

- `make check-app` - Run frontend checks (format, type check, lint)
- Or run individually from `param_decomp/app/frontend/`:
  - `npm run format` - Format code with Prettier
  - `npm run check` - Run Svelte type checking
  - `npm run lint` - Run ESLint

**Testing:**

- `make test` - Run tests (excluding slow tests)
- `make test-all` - Run all tests including slow ones
- `python -m pytest tests/test_specific.py` - Run specific test file
- `python -m pytest tests/test_specific.py::test_function` - Run specific test

**Running the App:**

- `make app` - Launch the PD visualization app (backend + frontend)

## Architecture Overview

**Core PD Framework:**

- `param_decomp/run_pd.py` - Main PD optimization logic called by all experiments
- `param_decomp/configs.py` - Core PD config and loss/metric config classes
- `param_decomp/experiments/runner.py` - `pd-run` CLI: submits to SLURM by default, runs in-process with `--local`
- `param_decomp/experiments/_worker.py` - Internal worker entrypoint each SLURM array task invokes
- `param_decomp/experiments/*/experiment.py` - Experiment configs and drivers that prepare targets,
  dataloaders, and artifacts
- `param_decomp/experiments/discovery.py` - Auto-discovery of built-in experiments from `experiments/<kind>/*.yaml`
- `param_decomp/models/component_model.py` - Core ComponentModel that wraps target models
- `param_decomp/models/components.py` - Component types (LinearComponent, EmbeddingComponent, etc.)
- `param_decomp/losses.py` - PD loss functions (faithfulness, reconstruction, importance minimality)
- `param_decomp/metrics.py` - Metrics for logging to WandB (e.g. CI-L0, KL divergence, etc.)
- `param_decomp/figures.py` - Figures for logging to WandB (e.g. CI histograms, Identity plots, etc.)

**Terminology: Sources vs Masks:**

- **Sources** (`adv_sources`, `PPGDSources`, `self.sources`): The raw values that PGD optimizes adversarially. These are interpolated with CI to produce component masks: `mask = ci + (1 - ci) * source`. Used in both regular PGD (`param_decomp/metrics/pgd_utils.py`) and persistent PGD (`param_decomp/persistent_pgd.py`).
- **Masks** (`component_masks`, `RoutingMasks`, `make_mask_infos`, `n_mask_samples`): The materialized per-component masks used during forward passes. These are produced from sources (in PGD) or from stochastic sampling, and are a general PD concept across the whole codebase.

**Experiment Structure:**

Each experiment (`param_decomp/experiments/{tms,resid_mlp,lm}/`) contains:

- `models.py` - Experiment-specific model classes and pretrained loading
- `experiment.py` - Pydantic experiment config, target/data builders, and driver
- `train_*.py` - Training script for target models
- `*_config.yaml` - Configuration files
- `plotting.py` - Visualization utilities

**Key Data Flow:**

1. `pd-run` (`experiments/runner.py`) resolves the input source — either a built-in experiment name, `--config_path`, `--rerun`, or `--sweep_generator_path` — into a `Run` (single launch) or a `SweepSpec` (many `Run`s sharing one driver and substrate). Every YAML/saved Run declares its driver via a top-level `driver_path:` field. With `--local` it dispatches in-process; otherwise `scripts/run_slurm.py:launch_slurm` submits a plain SLURM job (`Run`) or an array — one task per run — (`SweepSpec`).
2. Each worker invocation (`experiments/_worker.py`, called as `python -m param_decomp.experiments._worker` from SLURM tasks, or directly from runner.py in --local mode) loads the driver, checks that the parsed `Run` matches the driver's `config_type`, and calls the driver to build `PDTarget` plus train/eval loaders.
3. The worker passes the typed `Run` through to `run_pd`.
4. `run_pd` saves the `Run` / artifacts and trains a `ComponentModel` via `optimize()` with config-driven losses.
5. Post-processing reloads runs through `PDRun.load_target()` / `PDRun.load_dataloaders(...)` (or just `PDRun.load_model()` / `load_component_model(path)`).

**Configuration System:**

- YAML experiment configs define parameters under `pd:`, `target:`, and `data:`
- Pydantic models provide type safety and validation
- WandB integration for experiment tracking and model storage
- Supports both local paths and `wandb:project/runs/run_id` format for model loading
- Built-in experiments are auto-discovered from YAML configs in `param_decomp/experiments/<kind>/`; custom
  experiments declare their driver in the YAML (`driver_path: module:MyDriver`) and run via
  `pd-run --config_path config.yaml`

**Harvest, Autointerp & Dataset Attributions Modules:**

- `param_decomp/harvest/` - Offline GPU pipeline for collecting component statistics (correlations, token stats, activation examples)
- `param_decomp/autointerp/` - LLM-based automated interpretation of components
- `param_decomp/dataset_attributions/` - Multi-GPU pipeline for computing component-to-component attribution strengths aggregated over training data
- `param_decomp/graph_interp/` - Context-aware component labeling using graph structure (attributions + correlations)
- Data stored at `PARAM_DECOMP_OUT_DIR/{harvest,autointerp,dataset_attributions,graph_interp}/<run_id>/`
- See `param_decomp/harvest/CLAUDE.md`, `param_decomp/autointerp/CLAUDE.md`, `param_decomp/dataset_attributions/CLAUDE.md`, and `param_decomp/graph_interp/CLAUDE.md` for details

**Output Directory (`PARAM_DECOMP_OUT_DIR`):**

- Defined in `param_decomp/settings.py`
- On cluster: `/mnt/polished-lake/artifacts/mechanisms/param-decomp/`
- Off cluster: `~/param_decomp_out/`
- Contains: runs, SLURM logs, sbatch scripts, clustering outputs, harvest data, autointerp results

**Experiment Logging:**

- Uses WandB for experiment tracking and model storage
- All runs generate timestamped output directories with configs, models, and plots

## Directory Structure

```
<repo-root>/
├── papers/                          # Research papers (SPD, APD)
├── scripts/                         # Standalone utility scripts
├── tests/                           # Test suite
├── param_decomp/                             # Main source code
│   ├── investigate/                 # Agent investigation (see investigate/CLAUDE.md)
│   ├── app/                         # Web visualization app (see app/CLAUDE.md)
│   ├── autointerp/                  # LLM interpretation (see autointerp/CLAUDE.md)
│   ├── clustering/                  # Component clustering (see clustering/CLAUDE.md)
│   ├── dataset_attributions/        # Dataset attributions (see dataset_attributions/CLAUDE.md)
│   ├── harvest/                     # Statistics collection (see harvest/CLAUDE.md)
│   ├── postprocess/                 # Unified postprocessing pipeline (harvest + attributions + autointerp)
│   ├── graph_interp/                # Context-aware interpretation (see graph_interp/CLAUDE.md)
│   ├── pretrain/                    # Target model pretraining (see pretrain/CLAUDE.md)
│   ├── experiments/                 # Experiment implementations
│   │   ├── runner.py                # `pd-run` CLI (public)
│   │   ├── _worker.py               # internal worker each SLURM task invokes
│   │   ├── discovery.py             # built-in experiment auto-discovery
│   │   ├── driver.py                # ExperimentDriver protocol + load_driver
│   │   ├── tms/                     # Toy Model of Superposition
│   │   ├── resid_mlp/               # Residual MLP
│   │   └── lm/                      # Language models
│   ├── metrics/                     # Metrics - both for use as losses and as eval metrics
│   ├── models/
│   │   ├── component_model.py       # ComponentModel.from_checkpoint(...)
│   │   └── components.py            # LinearComponent, EmbeddingComponent, etc.
│   ├── scripts/
│   │   └── run_slurm.py             # launch_slurm (SLURM submit) — called by pd-run
│   ├── sweeps/                      # SweepSpec / SweepGenerator protocol + cartesian helper + example sweep
│   ├── utils/
│   │   └── slurm.py                 # SlurmConfig, submit functions
│   ├── configs.py                   # Core PD configs (PDConfig, ModuleInfo, loss configs, etc.)
│   ├── run_pd.py                             # Main optimization loop
│   ├── losses.py                    # Loss functions (faithfulness, reconstruction, etc.)
│   ├── figures.py                   # WandB figure generation
│   └── settings.py                  # PARAM_DECOMP_OUT_DIR, SLURM_LOGS_DIR, SBATCH_SCRIPTS_DIR
├── Makefile                         # Dev commands (make check, make test)
└── pyproject.toml                   # Package config
```

## Quick Navigation

### CLI Entry Points

| Command | Entry Point | Description |
|---------|-------------|-------------|
| `pd-run` | `param_decomp/experiments/runner.py` | Run a PD experiment. SLURM by default; `--local` runs in-process. Entry points are mutually exclusive: `<experiment>`, `--config_path`, `--rerun`, or `--sweep_generator_path /abs/path/file.py:func`. Every config (built-in YAML, user YAML, saved Run) declares its driver via a top-level `driver_path:` field. Compute substrate (`device`/`dp`) is declared in the experiment YAML's `runtime:` block. |
| `pd-harvest` | `param_decomp/harvest/scripts/run_slurm_cli.py` | Submit harvest SLURM job |
| `pd-autointerp` | `param_decomp/autointerp/scripts/run_slurm_cli.py` | Submit autointerp SLURM job |
| `pd-attributions` | `param_decomp/dataset_attributions/scripts/run_slurm_cli.py` | Submit dataset attribution SLURM job |
| `pd-postprocess` | `param_decomp/postprocess/cli.py` | Unified postprocessing pipeline (harvest + attributions + interpret + evals) |
| `pd-graph-interp` | `param_decomp/graph_interp/scripts/run_slurm_cli.py` | Submit graph interpretation SLURM job |
| `pd-clustering` | `param_decomp/clustering/scripts/run_pipeline.py` | Clustering ensemble pipeline |
| `pd-cluster-harvest` | `param_decomp/clustering/scripts/run_harvest.py` | Harvest activations → membership snapshot |
| `pd-cluster-merge` | `param_decomp/clustering/scripts/run_merge.py` | Merge from snapshot (CPU-only) |
| `pd-pretrain` | `param_decomp/pretrain/scripts/run_slurm_cli.py` | Pretrain target models |
| `pd-investigate` | `param_decomp/investigate/scripts/run_slurm_cli.py` | Launch investigation agent |

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

- `pd-run --local`: `runner.py` → `_worker.run_experiment` → `run_pd` (in-process)
- `pd-run` (SLURM, default): `runner.py` → `scripts/run_slurm.py:launch_slurm` → `utils/slurm.py` → SLURM array → `python -m param_decomp.experiments._worker` (one per task) → `run_pd`

**Harvest Pipeline:**

- `pd-harvest` → `param_decomp/harvest/scripts/run_slurm_cli.py` → `param_decomp/utils/slurm.py` → SLURM array → `param_decomp/harvest/scripts/run_worker.py` → `param_decomp/harvest/harvest.py`, then merge job → `param_decomp/harvest/scripts/run_merge.py`

**Autointerp Pipeline:**

- `pd-autointerp` → `param_decomp/autointerp/scripts/run_slurm_cli.py` → `param_decomp/utils/slurm.py` → `param_decomp/autointerp/interpret.py`

**Dataset Attributions Pipeline:**

- `pd-attributions` → `param_decomp/dataset_attributions/scripts/run_slurm_cli.py` → `param_decomp/utils/slurm.py` → SLURM array → `param_decomp/dataset_attributions/harvest.py`

**Clustering Pipeline:**

- `pd-clustering` → `param_decomp/clustering/scripts/run_pipeline.py` → `param_decomp/utils/slurm.py` → `param_decomp/clustering/scripts/run_clustering.py`

**Investigation Pipeline:**

- `pd-investigate` → `param_decomp/investigate/scripts/run_slurm_cli.py` → `param_decomp/utils/slurm.py` → SLURM → `param_decomp/investigate/scripts/run_agent.py` → Claude Code

## Common Usage Patterns

### Running Experiments (`pd-run`)

`pd-run` is the only CLI for running PD experiments. By default it submits a SLURM job (with a
git snapshot for reproducibility). Pass `--local` to run in this process instead — no SLURM, no
snapshot — useful for quick checks. Off-cluster `pd-run` fails fast unless `--local` is set.

```bash
pd-run tms_5-2                                                            # one SLURM job
pd-run --sweep_generator_path /abs/path/my_sweep.py:my_sweep --n_agents 4 # SLURM array sweep
pd-run tms_5-2                                                            # CPU/GPU/dp determined by YAML's runtime: block
pd-run --config_path my.yaml                                              # custom config (declares driver_path: in YAML)
pd-run --rerun <path-or-wandb-url>                                        # rerun from a saved run_metadata.yaml
pd-run tms_5-2 --local                                                    # in-process; no SLURM
```

One experiment per invocation. Multi-experiment campaigns are no longer a built-in workflow —
compose your own deploy script if you need that.

### Web App for Visualization

The PD app provides interactive visualization of component decompositions and attributions:

```bash
make app              # Launch backend + frontend dev servers
# or
python -m param_decomp.app.run_app
```

The app has its own detailed documentation in `param_decomp/app/CLAUDE.md` and `param_decomp/app/README.md`.

### Harvesting Component Statistics (`pd-harvest`)

Collect component statistics (activation examples, correlations, token stats) for a run:

```bash
pd-harvest <wandb_path> --n_batches 1000 --n_gpus 8    # Submit SLURM job to harvest statistics
```

See `param_decomp/harvest/CLAUDE.md` for details.

### Automated Component Interpretation (`pd-autointerp`)

Generate LLM interpretations for harvested components:

```bash
pd-autointerp <wandb_path>            # Submit SLURM job to interpret components
```

Requires `OPENROUTER_API_KEY` env var. See `param_decomp/autointerp/CLAUDE.md` for details.

### Agent Investigation (`pd-investigate`)

Launch a Claude Code agent to investigate a specific question about a PD model:

```bash
pd-investigate <wandb_path> "How does the model handle gendered pronouns?"
pd-investigate <wandb_path> "What components are involved in verb agreement?" --time 4:00:00
```

Each investigation:

- Runs in its own SLURM job with 1 GPU
- Starts an isolated app backend instance
- Investigates the specific research question using PD tools via MCP
- Writes findings to append-only JSONL files

Output: `PARAM_DECOMP_OUT_DIR/investigations/<inv_id>/`

For parallel investigations, run the command multiple times with different prompts.

See `param_decomp/investigate/CLAUDE.md` for details.

### Unified Postprocessing (`pd-postprocess`)

Run all postprocessing steps for a completed PD run with a single command. The CLI takes a
positional path to a `PostprocessConfig` YAML (the wandb run is specified inside the config):

```bash
pd-postprocess config.yaml                    # Submit the pipeline defined by this config
pd-postprocess config.yaml --dependency 123   # Wait for SLURM job 123 before starting
pd-postprocess config.yaml --dry_run          # Print the resolved config without submitting
```

The config schema is `PostprocessConfig` in `param_decomp/postprocess/config.py`. Set any optional
section to `null` to skip it:

- `attributions: null` — skip dataset attributions
- `autointerp: null` — skip autointerp entirely (interpret + evals)
- `autointerp.evals: null` — skip evals but still run interpret
- `intruder: null` — skip intruder eval
- `graph_interp: null` — skip context-aware graph interpretation (requires `attributions`)

SLURM dependency graph:

```
harvest (GPU array → merge)
├── intruder eval    (CPU, depends on harvest merge, label-free)
└── autointerp       (depends on harvest merge)
    ├── interpret    (CPU, LLM calls)
    │   ├── detection (CPU, depends on interpret)
    │   └── fuzzing   (CPU, depends on interpret)
attributions (GPU array → merge, parallel with harvest)
graph_interp         (CPU, depends on harvest merge + attributions merge)
```

**Metrics and Figures:**

Metrics and figures are defined in `param_decomp/metrics.py` and `param_decomp/figures.py`. These files expose dictionaries of functions that can be selected and parameterized in the config of a given experiment. This allows for easy extension and customization of metrics and figures, without modifying the core framework code.

### Sweeps

A sweep generator is a zero-arg callable returning a `SweepSpec`. The generator
loads whatever base config it wants, builds per-run configs, and declares the
driver in the returned spec — so `--sweep_generator_path` is a self-contained
entry point, mutually exclusive with `<experiment>`, `--config_path`, and `--rerun`.

```bash
pd-run --sweep_generator_path /abs/path/my_sweep.py:my_sweep --n_agents 4
```

The framework ships `example_cartesian_sweep` in `param_decomp/sweeps/cartesian.py`
as a reference — a TMS 5-2 seed sweep that uses the public `cartesian_product`
helper. Copy that file to start a new sweep.

**How It Works:**

1. `pd-run` imports the file at the given absolute path and looks up the named function.
2. The generator is called (no args) and returns a `SweepSpec` carrying
   `description` and `runs: list[Run]`. Each `Run` is self-describing: a
   `driver_path`, `pd` / `logging` / `runtime` configs, and `target` /
   `data` from the driver's `Run` subclass. `logging.wandb_run_name` and
   `logging.view_meta` are typically populated by the generator. The W&B
   project is not part of the `Run`; it's supplied via `--project` at launch.
   All runs in one sweep must share a `driver_path` and a `runtime` block
   (asserted by `SweepSpec.__post_init__`).
3. The materialized `SweepSpec` is written to
   `PARAM_DECOMP_OUT_DIR/sweeps/<launch_id>/spec.yaml` for reproducibility.
4. A SLURM array is submitted, capped at `--n_agents` concurrent tasks. Per-config
   validation happens worker-side at launch.
5. A git snapshot is created so all tasks run the same code, regardless of later edits.

This is **not** a W&B sweep agent — W&B sees independent runs sharing a `launch_id` tag.
`view_meta` from each run is surfaced under `view_meta/<key>` in `wandb.config` so you
can group/color by it in the UI.

**Cartesian helper (the 80% case):**

```python
# my_sweep.py
import yaml
from param_decomp.sweeps import SweepSpec
from param_decomp.sweeps.cartesian import cartesian_product

def my_sweep() -> SweepSpec:
    with open("/abs/path/to/base_config.yaml") as f:
        base = yaml.safe_load(f)
    return cartesian_product(
        base_config=base,
        grid={
            "pd.seed": [0, 1, 2],
            "pd.loss_metrics.importance_minimality.coeff": [0.1, 0.2, 0.5],
        },
        description="lr x recon coeff sweep",
        driver_path="param_decomp.experiments.tms.experiment:Driver",
    )
```

Dot-paths in the grid address into the base config. Each axis is recorded in
`view_meta` so the W&B UI can pivot on it.

**Custom sweeps** can build their `SweepSpec` however they like — there is no class
hierarchy. The function just has to return a `SweepSpec`; conformance to the
`SweepGenerator` protocol is structural.

**Rerunning runs from before the `Run` refactor:** `pd-run --rerun` will fail
pydantic validation on `run_metadata.yaml` files written before this refactor.
Old files have `driver:` / `config:` / top-level `wandb_*` / `view_meta`; the
new shape is top-level `driver_path:` / `pd:` / `logging:` / `runtime:` /
`target:` / `data:`, with `wandb_run_name` / `view_meta` nested under
`logging:`. (W&B project is no longer recorded on the `Run` — pass `--project`
explicitly when rerunning if you want anything other than the default project.)
Edit the saved YAML to match the new shape before rerunning.

**Logs:** `~/slurm_logs/slurm-<job_id>_<task_id>.out`

### Loading Models from WandB

Load trained PD models from wandb or local paths using these methods:

```python
from param_decomp import load_component_model, PDRun

# Common case: path → ComponentModel. The driver reconstructs the target from the saved run spec.
model = load_component_model("wandb:entity/project/runs/run_id")

# Manual/custom runs (no driver in the run spec): pass your own target.
target = ...
model = load_component_model("wandb:entity/project/runs/run_id", target=target)

# When you also need run/config access, use PDRun directly:
pd_run = PDRun.from_path("wandb:entity/project/runs/run_id")
print(pd_run.run)                        # Run (the driver-specific subclass, e.g. LMRun)
print(pd_run.pd_config)                  # PDConfig
model = pd_run.load_model()              # equivalent to load_component_model(path)
```

**Path Formats:**

- WandB: `wandb:entity/project/run_id` or `wandb:entity/project/runs/run_id`
- Local: Direct path to checkpoint file (config must be in same directory as `run_metadata.yaml`)

Downloaded runs are cached in `PARAM_DECOMP_OUT_DIR/runs/<project>-<run_id>/`.

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
