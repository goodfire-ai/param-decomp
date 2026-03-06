# Clustering Module

Hierarchical clustering of SPD components based on coactivation patterns. Runs ensemble clustering experiments to discover stable groups of components that behave similarly.

## Usage

**`spd-clustering` / `run_pipeline.py`**: Runs multiple clustering runs (ensemble) with different seeds, then runs `calc_distances` to compute pairwise distances between results. Use this for ensemble experiments.

**`run_clustering.py`**: Runs a single clustering run. Useful for testing or when you only need one clustering result.

```bash
# Run clustering pipeline via SLURM (ensemble of runs + distance calculation)
spd-clustering --config spd/clustering/configs/pipeline_config.yaml

# Run locally instead of SLURM
spd-clustering --config spd/clustering/configs/pipeline_config.yaml --local

# Single clustering run (usually called by pipeline)
python -m spd.clustering.scripts.run_clustering --config <clustering_run_config.json>
```

## Data Storage

Data is stored in `SPD_OUT_DIR/clustering/` (see `spd/settings.py`):

```
SPD_OUT_DIR/clustering/
├── runs/<run_id>/                       # Single clustering run outputs
│   ├── clustering_run_config.json
│   └── history.zip                      # MergeHistory (group assignments per iteration)
├── ensembles/<pipeline_run_id>/         # Pipeline/ensemble outputs
│   ├── pipeline_config.yaml
│   ├── clustering_run_config.json       # Copy of the config used
│   ├── ensemble_meta.json               # Component labels, iteration stats
│   ├── ensemble_merge_array.npz         # Normalized merge array
│   ├── distances_<method>.npz           # Distance matrices
│   └── distances_<method>.png           # Distance distribution plot
└── ensemble_registry.db                 # SQLite DB mapping pipeline → clustering runs
```

## Architecture

### Pipeline (`scripts/run_pipeline.py`)

Entry point via `spd-clustering`. Submits clustering runs as SLURM job array, then calculates distances between results. Key steps:
1. Creates `ExecutionStamp` for pipeline
2. Generates commands for each clustering run (with different dataset seeds)
3. Submits clustering array job to SLURM
4. Submits distance calculation jobs (depend on clustering completion)

### Single Run (`scripts/run_clustering.py`)

Performs one clustering run:
1. Load decomposed model from WandB
2. Compute component activations:
   - **LM tasks**: Uses `n_tokens` and `n_tokens_per_seq` parameters. Iterates through batches of size `batch_size`, picks `n_tokens_per_seq` random token positions per sequence, collects CI values until `n_tokens` samples gathered. Result: `(n_tokens, C)` per layer.
   - **resid_mlp tasks**: Single batch of size `batch_size`, no sequence dimension. Uses `component_activations()` directly.
3. Run merge iteration (greedy MDL-based clustering)
4. Save `MergeHistory` with group assignments per iteration

### Merge Algorithm (`merge.py`)

Greedy hierarchical clustering using MDL (Minimum Description Length) cost:
- Computes coactivation matrix from component activations
- Iteratively merges pairs with lowest cost (via `compute_merge_costs`)
- Supports stochastic merge pair selection (`merge_pair_sampling_method`)
- Tracks full merge history for analysis

### Distance Calculation (`scripts/calc_distances.py`)

Computes pairwise distances between clustering runs in an ensemble:
- Normalizes component labels across runs (handles dead components)
- Supports multiple distance methods: `perm_invariant_hamming`, `matching_dist`
- Runs in parallel using multiprocessing

## Key Types

### Configs

```python
ClusteringPipelineConfig  # Pipeline settings (n_runs, distances_methods, SLURM config)
ClusteringRunConfig       # Single run settings (model_path, batch_size, n_tokens, merge_config)
MergeConfig               # Merge algorithm params (alpha, iters, activation_threshold)
```

### Data Structures

```python
MergeHistory              # Full merge history: group assignments at each iteration
MergeHistoryEnsemble      # Collection of histories for distance analysis
GroupMerge                # Current group assignments (component -> group mapping)
```

### Type Aliases (`consts.py`)

```python
ActivationsTensor         # Float[Tensor, "samples n_components"]
ClusterCoactivationShaped # Float[Tensor, "k_groups k_groups"]
MergesArray               # Int[np.ndarray, "n_ens n_iters n_components"]
DistancesArray            # Float[np.ndarray, "n_iters n_ens n_ens"]
```

## Math Submodule (`math/`)

- `merge_matrix.py` - `GroupMerge` class for tracking group assignments
- `merge_distances.py` - Distance computation between clustering results
- `perm_invariant_hamming.py` - Permutation-invariant Hamming distance
- `matching_dist.py` - Optimal matching distance via Hungarian algorithm
- `merge_pair_samplers.py` - Strategies for selecting which pair to merge

## Utility Scripts

**`get_cluster_mapping.py`**: Extracts cluster assignments at a specific iteration from a clustering run, outputs JSON mapping component labels to cluster indices (singletons mapped to `null`).

```bash
python -m spd.clustering.scripts.get_cluster_mapping /path/to/clustering_run --iteration 299
```

## App Integration

To make a cluster mapping available in the app's dropdown for a run, add its path to `CANONICAL_RUNS` in `spd/app/frontend/src/lib/registry.ts` under the corresponding run's `clusterMappings` array.

## Config Files

Configs live in `spd/clustering/configs/`:
- Pipeline configs: `*.yaml` files with `ClusteringPipelineConfig`
- Run configs: `crc/*.json` files with `ClusteringRunConfig`

Example pipeline config:
```yaml
clustering_run_config_path: "spd/clustering/configs/crc/ss_llama_simple_mlp-1L.json"
n_runs: 10
distances_methods: ["perm_invariant_hamming"]
wandb_project: "spd"
```
