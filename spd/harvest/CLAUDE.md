# Harvest Module

Offline GPU pipeline that collects component statistics in a single pass over training data. Produces data consumed by the autointerp module (`spd/autointerp/`) and the app (`spd/app/`).

## Usage (SLURM)

```bash
# Process specific number of batches
spd-harvest <wandb_path> --n_batches 2000 --n_gpus 24

# Process entire training dataset (omit --n_batches)
spd-harvest <wandb_path> --n_gpus 24

# With optional parameters
spd-harvest <wandb_path> --n_batches 1000 --n_gpus 8 \
    --batch_size 256 --ci_threshold 1e-6 --time 24:00:00 --job_suffix 30m
```

The command:
1. Creates a git snapshot branch for reproducibility (jobs may be queued)
2. Submits a SLURM job array with N tasks (one per GPU)
3. Each task processes batches where `batch_idx % world_size == rank`
4. Submits a merge job (depends on array completion) that combines all worker results

**Note**: `--n_batches` is optional. If omitted, the pipeline processes the entire training dataset.

## Usage (non-SLURM)

For environments without SLURM, run the worker script directly:

```bash
# Single GPU (defaults from HarvestConfig, auto-generates subrun ID)
python -m spd.harvest.scripts.run <wandb_path>

# Single GPU with config file
python -m spd.harvest.scripts.run <wandb_path> --config_path path/to/config.yaml

# Multi-GPU (run in parallel via shell, tmux, etc.)
# All workers and the merge step must share the same --subrun_id
SUBRUN="h-$(date +%Y%m%d_%H%M%S)"
python -m spd.harvest.scripts.run <path> --config_json '{"n_batches": 1000}' --rank 0 --world_size 4 --subrun_id $SUBRUN &
python -m spd.harvest.scripts.run <path> --config_json '{"n_batches": 1000}' --rank 1 --world_size 4 --subrun_id $SUBRUN &
python -m spd.harvest.scripts.run <path> --config_json '{"n_batches": 1000}' --rank 2 --world_size 4 --subrun_id $SUBRUN &
python -m spd.harvest.scripts.run <path> --config_json '{"n_batches": 1000}' --rank 3 --world_size 4 --subrun_id $SUBRUN &
wait

# Merge results after all workers complete
python -m spd.harvest.scripts.run <path> --merge --subrun_id $SUBRUN
```

Each worker processes batches where `batch_idx % world_size == rank`, then the merge step combines all partial results.

## Data Storage

Each harvest invocation creates a timestamped sub-run directory. `HarvestRepo` automatically loads from the latest sub-run.

```
SPD_OUT_DIR/harvest/<run_id>/
├── h-20260211_120000/          # sub-run 1
│   ├── harvest.db              # SQLite DB: components table + config table (WAL mode)
│   ├── component_correlations.pt
│   ├── token_stats.pt
│   └── worker_states/          # cleaned up after merge
│       └── worker_*.pt
├── h-20260211_140000/          # sub-run 2
│   └── ...
```

Legacy layout (pre sub-run, `activation_contexts/` + `correlations/`) is no longer supported.

## Architecture

### SLURM Launcher (`scripts/run_slurm.py`, `scripts/run_slurm_cli.py`)

Entry point via `spd-harvest`. Submits array job + dependent merge job.

**Intruder evaluation** (`spd/harvest/intruder.py`) evaluates the quality of the *decomposition itself* — whether component activation patterns are coherent — without relying on LLM-generated labels. Intruder scores are stored in `harvest.db`, not `interp.db`. Intruder eval is submitted as a top-level postprocess stage (via `spd-postprocess`), not as part of the harvest pipeline.

### Worker Script (`scripts/run.py`)

Internal script called by SLURM jobs. Accepts config via `--config_path` (file) or `--config_json` (inline JSON). Supports:
- `--config_path`/`--config_json`: Provide `HarvestConfig` (defaults used if neither given)
- `--rank R --world_size N`: Process subset of batches
- `--merge`: Combine per-rank results into final files
- `--subrun_id`: Sub-run identifier (auto-generated if not provided)

### Config (`config.py`)

`HarvestConfig` (tuning params) and `HarvestSlurmConfig` (HarvestConfig + SLURM params). `wandb_path` is a runtime arg, not part of config.

### Harvest Logic (`harvest.py`)

Main harvesting functions:
- `harvest_activation_contexts(wandb_path, config, output_dir, ...)`: Process batches for a single rank
- `merge_activation_contexts(output_dir)`: Combine worker results from `output_dir/worker_states/` into `output_dir`

### Harvester (`harvester.py`)

Core class that accumulates statistics in a single pass:
- **Correlations**: Co-occurrence counts between components (for precision/recall/PMI)
- **Token stats**: Input token associations (hard counts) and output token associations (probability mass)
- **Activation examples**: Reservoir sampling for uniform coverage across dataset

Key optimizations:
- Reservoir sampling: O(1) per add, O(k) memory, uniform random sampling from stream
- Subsampling: Caps firings per batch at 10k (plenty for k=20 examples per component)
- All accumulation on GPU, only moves to CPU for final `build_results()`

### Storage (`storage.py`)

`CorrelationStorage` and `TokenStatsStorage` classes for loading/saving harvested data.

### Database (`db.py`)

`HarvestDB` class wrapping SQLite for component-level data. Two tables:
- `components`: keyed by `component_key`, stores layer/idx/mean_ci + JSON blobs for activation examples and PMI data
- `config`: key-value store for harvest config (ci_threshold, etc.)

Uses WAL mode for concurrent reads. Serialization via `orjson`.

### Repository (`repo.py`)

`HarvestRepo` provides read-only access to all harvest data for a run. Automatically resolves the latest sub-run directory (by lexicographic sort of `h-YYYYMMDD_HHMMSS` names). Falls back to legacy layout if no sub-runs exist. Used by the app backend.

## Key Types (`schemas.py`)

```python
ActivationExample     # Token window + CI values around a firing
ComponentData         # All harvested info for one component
ComponentTokenPMI     # Top/bottom tokens by PMI
```

## Analysis (`analysis.py`)

Query functions for exploring harvested data:
- Component correlations (precision, recall, Jaccard, PMI)
- Token statistics lookup
- Activation example retrieval