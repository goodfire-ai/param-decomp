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

SPD (Stochastic Parameter Decomposition) is a research framework for analyzing neural network components and their interactions through sparse parameter decomposition techniques.

- Target model parameters are decomposed as a sum of `parameter components`
- Parameter components approximate target model outputs despite differentiable stochastic masks
- Causal importance functions quantify how much each component can be masked on each datapoint
- Multiple loss terms balance faithfulness, output reconstruction quality, and component activation sparsity

The codebase supports three experimental domains: TMS (Toy Model of Superposition), ResidualMLP (residual MLP analysis), and Language Models.

**Available experiments** (defined in `spd/registry.py`):

- **TMS (Toy Model of Superposition)**:
  - `tms_5-2` - TMS with 5 features, 2 hidden dimensions
  - `tms_5-2-id` - TMS with 5 features, 2 hidden dimensions (fixed identity in-between)
  - `tms_40-10`
  - `tms_40-10-id`
- **ResidualMLP**:
  - `resid_mlp1` - 1 layer
  - `resid_mlp2` - 2 layers
  - `resid_mlp3` - 3 layers
- **Language Models**:
  - `ss_llama_simple`, `ss_llama_simple-1L`, `ss_llama_simple-2L` - Simple Stories Llama variants
  - `ss_llama_simple_mlp`, `ss_llama_simple_mlp-1L`, `ss_llama_simple_mlp-2L` - Llama MLP-only variants
  - `ss_gpt2`, `ss_gpt2_simple`, `ss_gpt2_simple_noln` - Simple Stories GPT-2 variants
  - `ss_gpt2_simple-1L`, `ss_gpt2_simple-2L` - GPT-2 simple layer variants
  - `pile_llama_simple_mlp-2L`, `pile_llama_simple_mlp-4L`, `pile_llama_simple_mlp-12L` - Pile Llama MLP-only variants
  - `pile_gpt2_simple-2L_global_reverse` - Pile GPT-2 with global reverse
  - `gpt2` - Standard GPT-2
  - `ts` - TinyStories

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
- `make install-app` - Install frontend dependencies (`npm install` in `spd/app/frontend/`)

**Code Quality:**

- `make check` - Run full pre-commit suite (basedpyright, ruff lint, ruff format)
- `make type` - Run basedpyright type checking only
- `make format` - Run ruff linter and formatter

**Frontend (when working on `spd/app/frontend/`):**

- `make check-app` - Run frontend checks (format, type check, lint)
- Or run individually from `spd/app/frontend/`:
  - `npm run format` - Format code with Prettier
  - `npm run check` - Run Svelte type checking
  - `npm run lint` - Run ESLint

**Testing:**

- `make test` - Run tests (excluding slow tests)
- `make test-all` - Run all tests including slow ones
- `python -m pytest tests/test_specific.py` - Run specific test file
- `python -m pytest tests/test_specific.py::test_function` - Run specific test

**Running the App:**

- `make app` - Launch the SPD visualization app (backend + frontend)

## Architecture Overview

**Core SPD Framework:**

- `spd/run_spd.py` - Main SPD optimization logic called by all experiments
- `spd/configs.py` - Pydantic config classes for all experiment types
- `spd/registry.py` - Centralized experiment registry with all experiment configurations
- `spd/models/component_model.py` - Core ComponentModel that wraps target models
- `spd/models/components.py` - Component types (LinearComponent, EmbeddingComponent, etc.)
- `spd/losses.py` - SPD loss functions (faithfulness, reconstruction, importance minimality)
- `spd/metrics.py` - Metrics for logging to WandB (e.g. CI-L0, KL divergence, etc.)
- `spd/figures.py` - Figures for logging to WandB (e.g. CI histograms, Identity plots, etc.)

**Terminology: Sources vs Masks:**

- **Sources** (`adv_sources`, `PPGDSources`, `self.sources`): The raw values that PGD optimizes adversarially. These are interpolated with CI to produce component masks: `mask = ci + (1 - ci) * source`. Used in both regular PGD (`spd/metrics/pgd_utils.py`) and persistent PGD (`spd/persistent_pgd.py`).
- **Masks** (`component_masks`, `RoutingMasks`, `make_mask_infos`, `n_mask_samples`): The materialized per-component masks used during forward passes. These are produced from sources (in PGD) or from stochastic sampling, and are a general SPD concept across the whole codebase.

**Experiment Structure:**

Each experiment (`spd/experiments/{tms,resid_mlp,lm}/`) contains:

- `models.py` - Experiment-specific model classes and pretrained loading
- `*_decomposition.py` - Main SPD execution script
- `train_*.py` - Training script for target models
- `*_config.yaml` - Configuration files
- `plotting.py` - Visualization utilities

**Key Data Flow:**

1. Experiments load pretrained target models via WandB or local paths
2. Target models are wrapped in ComponentModel with specified target modules
3. SPD optimization runs via `spd.run_spd.optimize()` with config-driven loss combination
4. Results include component masks, causal importance scores, and visualizations

**Configuration System:**

- YAML configs define all experiment parameters
- Pydantic models provide type safety and validation
- WandB integration for experiment tracking and model storage
- Supports both local paths and `wandb:project/runs/run_id` format for model loading
- Centralized experiment registry (`spd/registry.py`) manages all experiment configurations

**Harvest, Autointerp & Dataset Attributions Modules:**

- `spd/harvest/` - Offline GPU pipeline for collecting component statistics (correlations, token stats, activation examples)
- `spd/autointerp/` - LLM-based automated interpretation of components
- `spd/dataset_attributions/` - Multi-GPU pipeline for computing component-to-component attribution strengths aggregated over training data
- `spd/graph_interp/` - Context-aware component labeling using graph structure (attributions + correlations)
- Data stored at `SPD_OUT_DIR/{harvest,autointerp,dataset_attributions,graph_interp}/<run_id>/`
- See `spd/harvest/CLAUDE.md`, `spd/autointerp/CLAUDE.md`, `spd/dataset_attributions/CLAUDE.md`, and `spd/graph_interp/CLAUDE.md` for details

**Output Directory (`SPD_OUT_DIR`):**

- Defined in `spd/settings.py`
- On cluster: `/mnt/polished-lake/artifacts/mechanisms/spd/`
- Off cluster: `~/spd_out/`
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
├── spd/                             # Main source code
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
│   │   ├── tms/                     # Toy Model of Superposition
│   │   ├── resid_mlp/               # Residual MLP
│   │   ├── lm/                      # Language models
│   │   └── ih/                      # Induction heads
│   ├── metrics/                     # Metrics - both for use as losses and as eval metrics
│   ├── models/
│   │   ├── component_model.py       # ComponentModel, SPDRunInfo, from_pretrained()
│   │   └── components.py            # LinearComponent, EmbeddingComponent, etc.
│   ├── scripts/                     # CLI entry points (spd-run, spd-local)
│   ├── utils/
│   │   └── slurm.py                 # SlurmConfig, submit functions
│   ├── configs.py                   # Pydantic configs (Config, ModuleInfo, etc.)
│   ├── registry.py                  # Experiment registry (name → config)
│   ├── run_spd.py                   # Main optimization loop
│   ├── losses.py                    # Loss functions (faithfulness, reconstruction, etc.)
│   ├── figures.py                   # WandB figure generation
│   └── settings.py                  # SPD_OUT_DIR, SLURM_LOGS_DIR, SBATCH_SCRIPTS_DIR
├── Makefile                         # Dev commands (make check, make test)
└── pyproject.toml                   # Package config
```

## Quick Navigation

### CLI Entry Points

| Command | Entry Point | Description |
|---------|-------------|-------------|
| `spd-run` | `spd/scripts/run.py` | SLURM-based experiment runner |
| `spd-local` | `spd/scripts/run_local.py` | Local experiment runner |
| `spd-harvest` | `spd/harvest/scripts/run_slurm_cli.py` | Submit harvest SLURM job |
| `spd-autointerp` | `spd/autointerp/scripts/run_slurm_cli.py` | Submit autointerp SLURM job |
| `spd-attributions` | `spd/dataset_attributions/scripts/run_slurm_cli.py` | Submit dataset attribution SLURM job |
| `spd-postprocess` | `spd/postprocess/cli.py` | Unified postprocessing pipeline (harvest + attributions + interpret + evals) |
| `spd-graph-interp` | `spd/graph_interp/scripts/run_slurm_cli.py` | Submit graph interpretation SLURM job |
| `spd-clustering` | `spd/clustering/scripts/run_pipeline.py` | Clustering pipeline |
| `spd-pretrain` | `spd/pretrain/scripts/run_slurm_cli.py` | Pretrain target models |
| `spd-investigate` | `spd/investigate/scripts/run_slurm_cli.py` | Launch investigation agent |

### Files to Skip When Searching

Use `spd/` as the search root (not repo root) to avoid noise.

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

- `spd-run` → `spd/scripts/run.py` → `spd/utils/slurm.py` → SLURM → `spd/run_spd.py`
- `spd-local` → `spd/scripts/run_local.py` → `spd/run_spd.py` directly

**Harvest Pipeline:**

- `spd-harvest` → `spd/harvest/scripts/run_slurm_cli.py` → `spd/utils/slurm.py` → SLURM array → `spd/harvest/scripts/run.py` → `spd/harvest/harvest.py`

**Autointerp Pipeline:**

- `spd-autointerp` → `spd/autointerp/scripts/run_slurm_cli.py` → `spd/utils/slurm.py` → `spd/autointerp/interpret.py`

**Dataset Attributions Pipeline:**

- `spd-attributions` → `spd/dataset_attributions/scripts/run_slurm_cli.py` → `spd/utils/slurm.py` → SLURM array → `spd/dataset_attributions/harvest.py`

**Clustering Pipeline:**

- `spd-clustering` → `spd/clustering/scripts/run_pipeline.py` → `spd/utils/slurm.py` → `spd/clustering/scripts/run_clustering.py`

**Investigation Pipeline:**

- `spd-investigate` → `spd/investigate/scripts/run_slurm_cli.py` → `spd/utils/slurm.py` → SLURM → `spd/investigate/scripts/run_agent.py` → Claude Code

## Common Usage Patterns

### Running Experiments Locally (`spd-local`)

For collaborators and simple local execution, use `spd-local`:

```bash
spd-local tms_5-2           # Run on single GPU (default)
spd-local tms_5-2 --cpu     # Run on CPU
spd-local tms_5-2 --dp 4    # Run on 4 GPUs (single node DDP)
```

This runs experiments directly without SLURM, git snapshots, or W&B views/reports.

### Web App for Visualization

The SPD app provides interactive visualization of component decompositions and attributions:

```bash
make app              # Launch backend + frontend dev servers
# or
python -m spd.app.run_app
```

The app has its own detailed documentation in `spd/app/CLAUDE.md` and `spd/app/README.md`.

### Harvesting Component Statistics (`spd-harvest`)

Collect component statistics (activation examples, correlations, token stats) for a run:

```bash
spd-harvest <wandb_path> --n_batches 1000 --n_gpus 8    # Submit SLURM job to harvest statistics
```

See `spd/harvest/CLAUDE.md` for details.

### Automated Component Interpretation (`spd-autointerp`)

Generate LLM interpretations for harvested components:

```bash
spd-autointerp <wandb_path>            # Submit SLURM job to interpret components
```

Requires `OPENROUTER_API_KEY` env var. See `spd/autointerp/CLAUDE.md` for details.

### Agent Investigation (`spd-investigate`)

Launch a Claude Code agent to investigate a specific question about an SPD model:

```bash
spd-investigate <wandb_path> "How does the model handle gendered pronouns?"
spd-investigate <wandb_path> "What components are involved in verb agreement?" --time 4:00:00
```

Each investigation:

- Runs in its own SLURM job with 1 GPU
- Starts an isolated app backend instance
- Investigates the specific research question using SPD tools via MCP
- Writes findings to append-only JSONL files

Output: `SPD_OUT_DIR/investigations/<inv_id>/`

For parallel investigations, run the command multiple times with different prompts.

See `spd/investigate/CLAUDE.md` for details.

### Unified Postprocessing (`spd-postprocess`)

Run all postprocessing steps for a completed SPD run with a single command:

```bash
spd-postprocess <wandb_path>                              # Run everything with default config
spd-postprocess <wandb_path> --config custom_config.yaml  # Use custom config
```

Defaults are defined in `PostprocessConfig` (`spd/postprocess/config.py`). Pass a custom YAML/JSON config to override. Set any section to `null` to skip it:

- `attributions: null` — skip dataset attributions
- `autointerp: null` — skip autointerp entirely (interpret + evals)
- `autointerp.evals: null` — skip evals but still run interpret
- `intruder: null` — skip intruder eval

SLURM dependency graph:

```
harvest (GPU array → merge)
├── intruder eval    (CPU, depends on harvest merge, label-free)
└── autointerp       (depends on harvest merge)
    ├── interpret    (CPU, LLM calls)
    │   ├── detection (CPU, depends on interpret)
    │   └── fuzzing   (CPU, depends on interpret)
attributions (GPU array → merge, parallel with harvest)
```

### Running on SLURM Cluster (`spd-run`)

For the core team, `spd-run` provides full-featured SLURM orchestration:

```bash
spd-run --experiments tms_5-2                    # Run a specific experiment
spd-run --experiments tms_5-2,resid_mlp1         # Run multiple experiments
spd-run                                          # Run all experiments
```

All `spd-run` executions:

- Submit jobs to SLURM
- Create a git snapshot for reproducibility
- Create W&B workspace views

A run will output the important losses and the paths to which important figures are saved. Use these
to analyse the result of the runs.

**Metrics and Figures:**

Metrics and figures are defined in `spd/metrics.py` and `spd/figures.py`. These files expose dictionaries of functions that can be selected and parameterized in the config of a given experiment. This allows for easy extension and customization of metrics and figures, without modifying the core framework code.

### Sweeps

Run hyperparameter sweeps on the GPU cluster:

```bash
spd-run --experiments <experiment_name> --sweep --n_agents <n-agents> [--cpu]
```

Examples:

```bash
spd-run --experiments tms_5-2 --sweep --n_agents 4            # Run TMS 5-2 sweep with 4 GPU agents
spd-run --experiments resid_mlp2 --sweep --n_agents 3 --cpu   # Run ResidualMLP2 sweep with 3 CPU agents
spd-run --sweep --n_agents 10                                 # Sweep all experiments with 10 agents
spd-run --experiments tms_5-2 --sweep custom.yaml --n_agents 2 # Use custom sweep params file
```

**Supported Experiments:** All experiments in `spd/registry.py` (run `spd-local --help` to see available options)

**How It Works:**

1. Creates a WandB sweep using parameters from `spd/scripts/sweep_params.yaml` (or custom file)
2. Deploys multiple SLURM agents as a job array to run the sweep
3. Each agent runs on a single GPU by default (use `--cpu` for CPU-only)
4. Creates a git snapshot to ensure consistent code across all agents

**Sweep Parameters:**

- Default sweep parameters are loaded from `spd/scripts/sweep_params.yaml`
- You can specify a custom sweep parameters file by passing its path to `--sweep`
- Sweep parameters support both experiment-specific and global configurations:

  ```yaml
  # Global parameters applied to all experiments
  global:
    seed:
      values: [0, 1, 2]
    lr_schedule:
      start_val:
        values: [0.001, 0.01]

  # Experiment-specific parameters (override global)
  tms_5-2:
    seed:
      values: [100, 200] # Overrides global seed
    task_config:
      feature_probability:
        values: [0.05, 0.1]
  ```

**Logs:** logs are found in `~/slurm_logs/slurm-<job_id>_<task_id>.out`

### Loading Models from WandB

Load trained SPD models from wandb or local paths using these methods:

```python
from spd.models.component_model import ComponentModel, SPDRunInfo

# Option 1: Load model directly (simplest)
model = ComponentModel.from_pretrained("wandb:entity/project/runs/run_id")

# Option 2: Load run info first, then model (access config before loading)
run_info = SPDRunInfo.from_path("wandb:entity/project/runs/run_id")
print(run_info.config)  # Inspect config before loading model
model = ComponentModel.from_run_info(run_info)

# Local paths work too
model = ComponentModel.from_pretrained("/path/to/checkpoint.pt")
```

**Path Formats:**

- WandB: `wandb:entity/project/run_id` or `wandb:entity/project/runs/run_id`
- Local: Direct path to checkpoint file (config must be in same directory as `final_config.yaml`)

Downloaded runs are cached in `SPD_OUT_DIR/runs/<project>-<run_id>/`.

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
