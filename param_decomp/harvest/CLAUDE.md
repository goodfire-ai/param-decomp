# Harvest Module


> This package is the in-job compute for this stage: worker module mains you run directly, inside whatever allocation your scheduler gave you. Nothing here submits or schedules a job.

Offline pipeline that collects component statistics in a single pass over training data.
Produces data consumed by the autointerp module (`param_decomp/autointerp/`). The
whole module is **torch-free**: the only decomposition
runs it harvests are JAX single-pool runs (`run_worker.py`), and the accumulator is
NumPy.

## JAX runs (`scripts/run_worker.py`)

A JAX single-pool run (`param_decomp`, orbax checkpoint) is harvested natively. The
run is opened with `param_decomp.experiments.lm.load_run.open_jax_run` (the reusable "open a JAX run
for consumption" pattern — see below); its frozen forward-only pass (lower-leaky CI +
‖U‖·(x@V) component acts + clean-logit softmax) is converted (`np.asarray`) into a
`HarvestBatch` of NumPy arrays, fed to the `Harvester`, and written via
`HarvestRepo.save_results`. Run *metadata* (tokenizer, layer descriptions) comes through
`adapter_from_id`, which routes a JAX run (via `adapters.pd.is_jax_run` — detects the
orbax `ckpts/` dir beside the run's single `config.yaml`) to `PDAdapter`, reading
metadata from the pinned config and building only the target *architecture* (no orbax
restore).

```bash
# single process
python -m param_decomp.harvest.scripts.run_worker \
    --run_dir runs/p-761bc061 --n_batches 50 --batch_size 16

# one rank of a sharded run (saves worker_states/worker_<rank>.npz; merge combines them)
python -m param_decomp.harvest.scripts.run_worker \
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

`param_decomp.experiments.lm.load_run.open_jax_run(run_dir, step=None) -> LoadedJaxRun` is the single
entry point any consumer of a JAX run should use. It rebuilds the frozen target +
`DecomposedModel` from the run's pinned config (`load_run_dir_config`), restores the orbax
checkpoint onto a reference `TrainState`, and exposes `run.forward(token_ids) ->
HarvestForward` plus the metadata torch consumers key on (`layer_activation_sizes`,
`vocab_size`, `site_names`). Pure JAX, torch-free, single-device-friendly. Add the
forward outputs a new consumer needs there; don't re-open checkpoints ad hoc.

## Usage

A harvest is N worker ranks over one decomposition, then one merge over their saved
states. Run the ranks however your allocation allows — concurrently across N GPUs, or
serially on one — then merge. The decomposition target is named inside the config, by
`method_config.wandb_path`; `HarvestConfig.n_batches` may be `"whole_dataset"` to
consume the entire training set.

```bash
# Single process (auto-generates subrun ID)
python -m param_decomp.harvest.scripts.run_worker \
    --run_dir runs/<run_id> --n_batches 1000 --batch_size 16

# Multi-rank: all workers + merge must share the same --subrun_id
SUBRUN="h-$(date +%Y%m%d_%H%M%S)"
for r in 0 1 2 3; do
  python -m param_decomp.harvest.scripts.run_worker \
      --run_dir runs/<run_id> --n_batches 1000 --batch_size 16 \
      --rank $r --world_size 4 --subrun_id $SUBRUN &
done
wait
python -m param_decomp.harvest.scripts.run_merge --subrun_id $SUBRUN --config_json "$CFG"
```

## Data Storage

Each harvest invocation creates a timestamped sub-run directory. `HarvestRepo` automatically loads from the latest sub-run.

```
<data_root>/runs/<run_id>/harvest/
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

### Intruder evaluation (`intruder.py`, `scripts/run_intruder.py`)

Evaluates the quality of the *decomposition itself* — whether component activation
patterns are coherent — without relying on LLM-generated labels. Intruder scores are
stored in `harvest.db`, not `interp.db`. It is a separate entry point over a finished
harvest, not a stage of the harvest pipeline:

```bash
python -m param_decomp.harvest.scripts.run_intruder <decomposition_id> \
    --config_json '{...IntruderEvalConfig...}' --harvest_subrun_id h-YYYYMMDD_HHMMSS
```

### Worker Script (`scripts/run_worker.py`)

The only worker. Opens a JAX run, runs its frozen forward, accumulates into the NumPy
`Harvester`. Args:
- `--run_dir`: the JAX run dir (`runs/<run_id>`) (required)
- `--data_root`: the required output root the harvest writes under
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

`HarvestConfig` (tuning params, plus a `method_config` that carries `wandb_path` and
method-specific options) and `IntruderEvalConfig`.

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

`HarvestRepo` provides read-only access to all harvest data for a run. Automatically resolves the latest sub-run directory (by lexicographic sort of `h-YYYYMMDD_HHMMSS` names). Returns `None` if no sub-run exists.

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
