# Harvest Module

Offline pipeline that collects component statistics in a single pass over training data.
Produces data consumed by the autointerp module (`param_decomp_lab/autointerp/`) and the
app (`param_decomp_lab/app/`). The whole module is **torch-free**: the only decomposition
runs it harvests are JAX single-pool runs (`run_worker_jax.py`), and the accumulator is
NumPy.

## JAX runs (`scripts/run_worker_jax.py`)

A JAX single-pool run (`param_decomp_jax`, orbax checkpoint) is harvested natively. The
run is opened with `jax_single_pool.load_run.open_jax_run` (the reusable "open a JAX run
for consumption" pattern — see below); its frozen forward-only pass (lower-leaky CI +
‖U‖·(x@V) component acts + clean-logit softmax) is converted (`np.asarray`) into a
`HarvestBatch` of NumPy arrays, fed to the `Harvester`, and written via
`HarvestRepo.save_results`. Run *metadata* (tokenizer, layer descriptions) comes through
`adapter_from_id`, which routes a JAX run (via `adapters.jax_pd.is_jax_run` — detects the
`pd-jax-lm` wrapper's `torch_config:` key) to `JaxPDAdapter`, reading metadata from the
pinned config and building only the target *architecture* (no orbax restore).

```bash
# single process
python -m param_decomp_lab.harvest.scripts.run_worker_jax \
    --run_dir runs/p-761bc061 --n_batches 50 --batch_size 16

# one rank of a sharded run (saves worker_states/worker_<rank>.npz; merge combines them)
python -m param_decomp_lab.harvest.scripts.run_worker_jax \
    --run_dir runs/p-761bc061 --n_batches 50 --batch_size 16 \
    --rank 0 --world_size 4 --subrun_id h-20260617_120000
```

The forward runs in jax (CPU or one GPU); the accumulator is NumPy. The pure JAX package
never imports torch and neither does this module. Pre-tokenized parquet is served by the
trainer's own `ShardServer` (never streamed from HF); `--rank/--world_size` shard via
`ShardServer`'s `process_index`/`process_count` (one slice of every global batch per
rank). On CPU the dense component co-occurrence (`O(C²)` + `O(C·vocab)` per batch)
dominates; pass `--no_cooccurrence` for a quick spot-check (drops only
`component_correlations.npz`).

### The reusable run-loading pattern (for clustering / autointerp / slow-eval / app)

`jax_single_pool.load_run.open_jax_run(run_dir, step=None) -> LoadedJaxRun` is the single
entry point any consumer of a JAX run should use. It rebuilds the frozen target +
`DecomposedModel` from the run's pinned config (`load_run_dir_config`), restores the orbax
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
2. Submits a SLURM array (one task per GPU); each task runs `run_worker_jax.py` as
   `--rank R --world_size N`, serving its `process_index=R` slice of every global batch
3. Submits a merge job (`run_merge.py`) that depends on the array

`HarvestConfig.n_batches` may be `"whole_dataset"` to consume the entire training set.

## Usage (non-SLURM)

```bash
# Single process (auto-generates subrun ID)
python -m param_decomp_lab.harvest.scripts.run_worker_jax \
    --run_dir runs/<run_id> --n_batches 1000 --batch_size 16

# Multi-rank: all workers + merge must share the same --subrun_id
SUBRUN="h-$(date +%Y%m%d_%H%M%S)"
for r in 0 1 2 3; do
  python -m param_decomp_lab.harvest.scripts.run_worker_jax \
      --run_dir runs/<run_id> --n_batches 1000 --batch_size 16 \
      --rank $r --world_size 4 --subrun_id $SUBRUN &
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
│   ├── component_correlations.npz
│   ├── token_stats.npz
│   └── worker_states/          # cleaned up after merge
│       └── worker_*.npz
├── h-20260211_140000/          # sub-run 2
│   └── ...
```

The tensor artefacts (`*.npz`, worker states) are NumPy `np.savez` archives.

## Architecture

### SLURM Launcher (`scripts/run_slurm.py`, `scripts/run_slurm_cli.py`)

Entry point via `pd-harvest`. Submits array job + dependent merge job.

**Intruder evaluation** (`param_decomp_lab/harvest/intruder.py`) evaluates the quality of the *decomposition itself* — whether component activation patterns are coherent — without relying on LLM-generated labels. Intruder scores are stored in `harvest.db`, not `interp.db`. Intruder eval is submitted as a top-level postprocess stage (via `pd-postprocess`), not as part of the harvest pipeline.

### Worker Script (`scripts/run_worker_jax.py`)

The only worker. Opens a JAX run, runs its frozen forward, accumulates into the NumPy
`Harvester`. Args:
- `--run_dir`: the JAX run dir (`runs/<run_id>`) (required)
- `--n_batches`, `--batch_size`, `--activation_threshold`
- `--rank R --world_size N`: serve `process_index=R`'s slice of every global batch; save
  to `worker_states/worker_<R>.npz`. Omit both for a single-process run that writes the
  final results directly.
- `--subrun_id`: Sub-run identifier (auto-generated `h-YYYYMMDD_HHMMSS` if omitted)
- `--no_cooccurrence`: skip the dense component co-occurrence matrix

### Merge Script (`scripts/run_merge.py`)

Combines `worker_states/*.npz` from each rank into the final harvest artefacts. Args:
- `--config_json`, `--subrun_id` (must match the workers').

### Config (`config.py`)

`HarvestConfig` (tuning params, plus a `method_config` discriminated union that carries
`wandb_path` and method-specific options) and `HarvestSlurmConfig` (HarvestConfig + SLURM
params).

### Pipeline (`pipeline.py`)

- `merge_harvest(output_dir, config)`: Combine all `worker_states/` into the final outputs.

### Accumulator (`accumulator.py`)

Core class that accumulates statistics in a single pass, as NumPy arrays on the host
(counts/co-occurrence use int64, probability-mass accumulators use float64):
- **Correlations**: Co-occurrence counts between components (for precision/recall/PMI)
- **Token stats**: Input token associations (hard counts) and output token associations (probability mass)
- **Activation examples**: Reservoir sampling for uniform coverage across dataset

Key optimizations:
- Reservoir sampling: O(1) per add, O(k) memory, uniform random sampling from stream
- Subsampling: caps examples kept per component per batch (`max_examples_per_batch_per_component`)

### Storage (`storage.py`)

`CorrelationStorage` and `TokenStatsStorage` classes for loading/saving harvested data
as `np.savez` archives.

### Database (`db.py`)

`HarvestDB` class wrapping SQLite for component-level data. Two tables:
- `components`: keyed by `component_key`, stores layer/idx/mean_ci + JSON blobs for activation examples and PMI data
- `config`: key-value store for harvest config (ci_threshold, etc.)

Uses WAL mode for concurrent reads. Serialization via `orjson`.

### Repository (`repo.py`)

`HarvestRepo` provides read-only access to all harvest data for a run. Automatically resolves the latest sub-run directory (by lexicographic sort of `h-YYYYMMDD_HHMMSS` names). Returns `None` if no sub-run exists. Used by the app backend.

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
