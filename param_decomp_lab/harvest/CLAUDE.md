# Harvest Module

Offline GPU pipeline that collects component statistics in a single pass over training data. Produces data consumed by the autointerp module (`param_decomp_lab/autointerp/`) and the app (`param_decomp_lab/app/`).

## Checkpoint formats

Harvest loads both component-model checkpoint formats via `component_model_io.py`: the
core `ComponentModel` (single-pool, and pre-`e8ff5a64` 3-pool) and the vendored
`LMComponentModel` (post-`e8ff5a64` 3-pool, wrapped in `VendoredHarvestModel`). Both
satisfy the `HarvestableComponentModel` protocol; `SavedThreePoolLMRun.load_model`
sniffs the state-dict key prefixes (`detect_checkpoint_format`) and dispatches.

## JAX runs (`scripts/run_worker_jax.py`)

A JAX single-pool run (`param_decomp_jax`, orbax checkpoint) is harvested **natively** —
no torch component model, no `jsp-export` safetensors bridge. The run is opened with
`jax_single_pool.load_run.open_jax_run` (the reusable "open a JAX run for consumption"
pattern — see below); its frozen forward-only pass (lower-leaky CI + ‖U‖·(x@V) component
acts + clean-logit softmax) is converted into the SAME `HarvestBatch` the torch
`ParamDecompHarvestFn` produces, fed to the SAME `Harvester`, and written via the SAME
`HarvestRepo.save_results`. The harvest *output* (DB + `token_stats.pt`) is byte-identical
to the torch contract, so consumers read it unchanged. Run *metadata* (tokenizer, layer
descriptions) is the one place that differs: a JAX run has no torch `.pth`, so
`adapter_from_id` routes it (via `adapters.jax_pd.is_jax_run` — detects the `pd-jax-lm`
wrapper's `torch_config:` key) to `JaxPDAdapter`, which reads metadata from the pinned
config and builds only the target *architecture* (no orbax restore). The torch `PDAdapter`
stays the torch-run path.

```bash
python -m param_decomp_lab.harvest.scripts.run_worker_jax \
    --run_dir runs/p-761bc061 --n_batches 50 --batch_size 16
```

The forward runs in jax (CPU or one GPU); the accumulator stays torch — this worker is
the one place both stacks meet (the pure JAX package never imports torch). Pre-tokenized
parquet is served by the trainer's own `ShardServer` (never streamed from HF). On CPU the
dense component co-occurrence (`O(C²)` + `O(C·vocab)` per batch) dominates; pass
`--no_cooccurrence` for a quick spot-check (drops only `component_correlations.pt`).

### The reusable run-loading pattern (for clustering / autointerp / slow-eval / app)

`jax_single_pool.load_run.open_jax_run(run_dir, step=None) -> LoadedJaxRun` is the single
entry point any consumer of a JAX run should use. It rebuilds the frozen target +
`DecomposedLM` from the run's pinned config (`load_run_dir_config`), restores the orbax
checkpoint onto a reference `TrainState`, and exposes `run.forward(token_ids) ->
HarvestForward` plus the metadata torch consumers key on (`layer_activation_sizes`,
`vocab_size`, `site_names`). Pure JAX, torch-free, single-device-friendly. Add the
forward outputs a new consumer needs there; don't re-open checkpoints ad hoc.

## Usage (SLURM)

```bash
pd-harvest path/to/harvest_slurm_config.yaml
pd-harvest path/to/harvest_slurm_config.yaml --job_suffix v2
```

`HarvestSlurmConfig` (`config.py`) wraps a `HarvestConfig` plus SLURM knobs (`n_gpus`,
`partition`, `time`, `merge_time`, `merge_mem`). The decomposition target is specified
inside `config.method_config.wandb_path` — there is no separate positional
`<wandb_path>` argument anymore.

The launcher:
1. Creates a git snapshot branch for reproducibility
2. Submits a SLURM array (one task per GPU); each task runs `run_worker.py` and processes
   batches where `batch_idx % world_size == rank`
3. Submits a merge job (`run_merge.py`) that depends on the array

`HarvestConfig.n_batches` may be `"whole_dataset"` to consume the entire training set.

## Usage (non-SLURM)

```bash
# Single GPU (auto-generates subrun ID)
python -m param_decomp_lab.harvest.scripts.run_worker --config_json '{"method_config": {...}, "n_batches": 1000}'

# Multi-GPU: all workers + merge must share the same --subrun_id
SUBRUN="h-$(date +%Y%m%d_%H%M%S)"
CFG='{"method_config": {...}, "n_batches": 1000}'
for r in 0 1 2 3; do
  python -m param_decomp_lab.harvest.scripts.run_worker --config_json "$CFG" --rank $r --world_size 4 --subrun_id $SUBRUN &
done
wait
python -m param_decomp_lab.harvest.scripts.run_merge --subrun_id $SUBRUN --config_json "$CFG"
```

## Data Storage

Each harvest invocation creates a timestamped sub-run directory. `HarvestRepo` automatically loads from the latest sub-run.

```
PARAM_DECOMP_OUT_DIR/runs/<run_id>/harvest/
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

Entry point via `pd-harvest`. Submits array job + dependent merge job.

**Intruder evaluation** (`param_decomp_lab/harvest/intruder.py`) evaluates the quality of the *decomposition itself* — whether component activation patterns are coherent — without relying on LLM-generated labels. Intruder scores are stored in `harvest.db`, not `interp.db`. Intruder eval is submitted as a top-level postprocess stage (via `pd-postprocess`), not as part of the harvest pipeline.

### Worker Script (`scripts/run_worker.py`)

Internal worker invoked per SLURM array task. Args:
- `--config_json`: Inline JSON of `HarvestConfig` (required)
- `--rank R --world_size N`: Process the batches where `batch_idx % N == R`
- `--subrun_id`: Sub-run identifier (auto-generated `h-YYYYMMDD_HHMMSS` if omitted)

### Merge Script (`scripts/run_merge.py`)

Combines `worker_states/*.pt` from each rank into the final harvest artefacts. Args:
- `--config_json`, `--subrun_id` (must match the workers').

### Config (`config.py`)

`HarvestConfig` (tuning params, plus a `method_config` discriminated union that carries
`wandb_path` and method-specific options) and `HarvestSlurmConfig` (HarvestConfig + SLURM
params).

### Pipeline (`pipeline.py`)

- `harvest(...)`: Run a single rank's pass over a dataloader, writing partial state.
- `merge_harvest(output_dir, config)`: Combine all `worker_states/` into the final outputs.

### Accumulator (`accumulator.py`)

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
