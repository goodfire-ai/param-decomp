# Clustering Module

Hierarchical clustering of PD components based on coactivation patterns. Discovers stable
groups of components that behave similarly.

**Zero torch.** The whole subsystem is numpy/scipy/numba. The JAX worker reads a JAX
single-pool run, samples CI, and streams numpy arrays into the accumulator; the merge is
pure numpy. (The forward itself runs in jax inside the worker.)

## Workflow: harvest (JAX) → merge (CPU)

GPU-light activation collection is separated from CPU-only merging. Harvest once, merge
many times with different configs.

```bash
# 1. Harvest a JAX single-pool run (orbax checkpoint) into a membership snapshot.
python -m param_decomp_lab.clustering.scripts.run_worker \
    --run_dir runs/p-761bc061 --n_tokens 50000 --batch_size 16 --n_tokens_per_seq 16
# → PARAM_DECOMP_OUT_DIR/clustering/harvests/ch-<id>/

# 2. Merge from the snapshot (CPU-only).
pd-cluster-merge /path/to/ch-<id>/ merge_config.json --run-id c-<id> --seed 0 [--plot]
# → PARAM_DECOMP_OUT_DIR/clustering/runs/c-<id>/
```

`pd-cluster-merge` seeds the stdlib `random` the stochastic merge-pair samplers draw from,
so distinct `--seed`s give independent merge trajectories over the same snapshot. `--plot`
emits per-run diagnostics (`plots/cluster_sizes.png`, periodic `plots/iter_*.png`).

## Workflow: ensemble → consensus (the headline `pd-clustering`)

A single clustering run is noisy; cluster *stability* is read off an ensemble of seeded
runs. `pd-clustering` fans one decomposition out into `n_runs` seeded `harvest → merge`
members, then a consensus job normalizes their labels and computes per-iteration pairwise
distances + a stability plot. Three dependency tiers:

```
harvest array (N × 1 GPU, seeded dataset)            run_worker
   └─ merge array (N × CPU, seeded sampler)           run_merge        [afterok harvest]
         └─ consensus job per distance method         calc_distances   [afterok merge]
```

```bash
# SLURM (seeded array + dependent merge array + dependent consensus):
pd-clustering --config param_decomp_lab/clustering/configs/ensemble_example.yaml
# In-process (sequential), for a small example / debugging:
pd-clustering --config <ensemble.yaml> --local
# Re-run consensus alone over existing member runs:
pd-cluster-distances --ensemble-id e-<id> --clustering-run-ids c-a,c-b,c-c \
    --distances-method perm_invariant_hamming
```

`ClusteringEnsembleConfig` (`scripts/run_pipeline.py`) is flat: `harvest` (`HarvestConfig`)
+ `merge` (`MergeConfig`) + `n_runs` + `base_seed` (member i uses `base_seed + i` for both
the harvest dataset and the merge sampler) + `distances_methods` + slurm fields. Output:

```
PARAM_DECOMP_OUT_DIR/clustering/ensembles/<ensemble_id>/
├── ensemble_config.yaml / harvest_config.json / merge_config.json
├── ensemble_meta.json                # normalization metadata (calc_distances)
├── ensemble_merge_array.npz          # normalized merge array
├── distances_<method>.npz            # per-iteration distance tensor
└── plots/distances_<method>.png      # stability plot
```

The consensus engine is the surviving `MergeHistoryEnsemble.normalized()` (unions member
labels, putting each member's missing/dead components into singleton groups) +
`math.compute_distances` (`perm_invariant_hamming` / `matching_dist` / `matching_dist_vec`,
multiprocessing over iterations). `calc_distances.py` is only the driver + plot; it does
not reimplement the math. Distance matrices are strict-lower-triangular (diag/upper `NaN`),
matching the per-iteration `perm_invariant_hamming_matrix` convention.

The run is opened with `param_decomp_lab.experiments.lm.load_run.open_jax_run` (the reusable JAX
"open a run for consumption" pattern, shared with `harvest`); the lower-leaky CI from its
frozen forward is sampled per token position (`flatten_lm_activations`) and streamed — as
a numpy-array dict — into the `MembershipBuilder`, producing a `ProcessedMemberships`
snapshot. `pd-cluster-merge` reads it unchanged.

- `HarvestConfig` (`harvest_config.py`): model_path, n_tokens, activation_threshold, etc.
- `MergeConfig` (`merge_config.py`): alpha, iters, merge_pair_sampling_method, etc.

## Data Storage

Stored under `PARAM_DECOMP_OUT_DIR/clustering/` (see `param_decomp_lab/infra/settings.py`):

```
PARAM_DECOMP_OUT_DIR/clustering/
├── harvests/<harvest_id>/               # Membership snapshots (run_worker)
│   ├── harvest_config.json
│   ├── memberships.npz                  # Sparse CSC matrix (scipy)
│   └── metadata.json                    # labels, n_samples, n_components
├── runs/<run_id>/                       # Merge outputs (pd-cluster-merge)
│   ├── merge_config.json
│   └── history.zip                      # MergeHistory (group assignments per iteration)
```

## Architecture

### Membership accumulation (`memberships.py`)

`MembershipBuilder` streams thresholded boolean memberships from numpy activation batches
without materializing the full dense `[n_samples, n_components]` matrix. Per-module max/sum
stats drive dead-component filtering; alive components are packed into a sparse CSC matrix,
then each column is stored as a `CompressedMembership` (sparse indices or bitset, whichever
is cheaper). `flatten_lm_activations` samples token positions from `(B, T, C)` CI.

`activations.py` is the dense reference path (`process_activations`,
`filter_dead_components`) the streaming builder is checked against in tests — not used in
production harvest.

### Merge Algorithm (`merge.py`, `compute_costs.py`)

Greedy hierarchical clustering using MDL (Minimum Description Length) cost. Builds the
coactivation matrix (`X.T @ X` over the sparse sample-by-component CSR), then iteratively
merges the pair with lowest cost (`compute_merge_costs`), recomputing affected
coactivations from compressed memberships (`recompute_coacts_merge_pair_memberships`,
numba-accelerated row scan). Supports stochastic merge-pair selection
(`math/merge_pair_samplers.py`: range / mcmc / exp_rank, all numpy + stdlib `random`).
Tracks full merge history.

### Distance analysis (`math/`)

`MergeHistoryEnsemble.normalized()` aligns component labels across an ensemble of histories
(handling differing dead components); `math/merge_distances.compute_distances` computes
pairwise distances via `perm_invariant_hamming` (numpy + scipy.optimize) or `matching_dist`
(numpy). Multiprocessing over iterations.

## Key Types

```python
MergeConfig               # Merge algorithm params (alpha, iters, sampling method, ...)
MergeHistory              # Group assignments at each iteration (BatchedGroupMerge, int32)
MergeHistoryEnsemble      # Collection of histories for distance analysis
GroupMerge                # Current group assignments (component -> group mapping)
```

### Type Aliases (`types.py`) — all numpy

```python
ActivationsArray          # Float[np.ndarray, "samples n_components"]
ClusterCoactivationShaped # Float[np.ndarray, "k_groups k_groups"]
GroupIdxsArray            # Int[np.ndarray, " n_components"]
MergesArray               # Int[np.ndarray, "n_ens n_iters n_components"]
DistancesArray            # Float[np.ndarray, "n_iters n_ens n_ens"]
```

`MergeHistory` stores group/pair indices as **int32** — int16 overflows above 32767
components (numpy raises; torch silently truncated).

## Plotting Submodule (`plotting/`)

Torch-free (numpy + matplotlib) diagnostics:

- `plot_coact_and_costs` — per-iteration coactivation/cost heatmaps, wired to the merge's
  `LogCallback` by `run_merge --plot`.
- `plot_merge_history_cluster_sizes` — per-run cluster-size-vs-iteration scatter.
- `plot_dists_distribution` — the ensemble stability plot (`calc_distances`).

## Math Submodule (`math/`)

- `merge_matrix.py` - `GroupMerge` / `BatchedGroupMerge` (component→group assignments)
- `merge_distances.py` - distance computation between clustering results
- `perm_invariant_hamming.py` - permutation-invariant Hamming distance
- `matching_dist.py` - optimal matching distance
- `merge_pair_samplers.py` - merge-pair selection strategies

## Utility Scripts

**`get_cluster_mapping.py`**: cluster assignments at a given iteration, as JSON mapping
component labels to cluster indices (singletons → `null`).

```bash
python -m param_decomp_lab.clustering.scripts.get_cluster_mapping /path/to/clustering_run --iteration 299
```

## Run ID Prefixes

`RUN_TYPE_ABBREVIATIONS` in `param_decomp_lab/infra/run_files.py`: `c` (clustering/runs),
`ch` (clustering/harvests), `e` (clustering/ensembles).
