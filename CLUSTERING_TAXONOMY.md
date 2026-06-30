# Clustering Pipeline — Feature Taxonomy (pre-migration vs HEAD)

Read at the `torch-oracle` git tag (full original pipeline) and cross-referenced against
`HEAD` (post torch→JAX migration). Purpose: inform a restore decision. **Read-only
analysis — no code was modified.**

Status tags per capability:
- **✅ kept** — present in `HEAD`
- **❌ dropped** — present only at `torch-oracle`
- **♻ replaced** — the original was dropped but a JAX-native equivalent exists in `HEAD`

The headline: the **core merge engine + the harvest→merge data path survived** (now
JAX-native at harvest); everything that turned single merges into an **ensemble +
consensus stability analysis** (the multi-run orchestrator, the distance/consensus
computation, all WandB logging, and the entire `plotting/` dir) was **dropped**.

---

## 1. End-to-end flow / stages

The original pipeline had two distinct entry modes over the same core:

```
                                  ┌─────────────── ENSEMBLE MODE (run_pipeline / pd-clustering) ──────────────┐
                                  │                                                                            │
  decomposed PD run               │  N seeded clustering runs (SLURM array, 1 GPU each)                        │
        │                         │        each: harvest → merge → MergeHistory                               │
        ▼                         │                          │                                                 │
  ┌───────────┐   ┌───────────┐   │                          ▼                                                 │
  │  HARVEST  │──▶│   MERGE   │───┼──▶  calc_distances: normalize labels → distance matrix per iter → plot     │
  └───────────┘   └───────────┘   │        (consumes N MergeHistories, emits ensemble consensus artifacts)     │
   activations →   greedy MDL      └────────────────────────────────────────────────────────────────────────┘
   memberships     clustering →
   snapshot        MergeHistory
        │
        └────────────── HARVEST-THEN-MERGE MODE (pd-cluster-harvest + pd-cluster-merge, CPU sweeps) ──────────
```

| Stage | What | Key file(s) | Status |
|---|---|---|---|
| **Harvest** | Run the decomposed model over tokens, sample per-token CI, accumulate sparse component-activity memberships into a compressed snapshot (`memberships.npz` + `metadata.json` + optional `preview`) | `scripts/run_harvest.py` (torch); `memberships.py`, `activations.py`, `sample_membership.py`, `harvest_config.py` | torch driver ❌ / engine ✅ / ♻ JAX driver |
| **Harvest (JAX-native)** | Same snapshot, opened via `open_jax_run`, CI sampled from the JAX frozen forward — no torch component model, no safetensors bridge | `scripts/run_worker_jax.py` | ✅ kept (new) |
| **Merge** | Greedy MDL hierarchical clustering over the coactivation matrix; emits `MergeHistory` (group assignments + selected pair per iteration) as `history.zip` | `merge.py`, `compute_costs.py`, `merge_config.py`, `merge_history.py`, `math/merge_matrix.py` | ✅ kept |
| **Merge CLI** | `pd-cluster-merge <snapshot> <merge_config>` — CPU-only, re-mergeable many times per harvest | `scripts/run_merge.py` | ✅ kept |
| **Single clustering run (harvest+merge+WandB)** | One orchestrated harvest→merge with full WandB logging (scalars, tensors, artifacts, plots) | `scripts/run_clustering.py` | ❌ dropped |
| **Ensemble orchestration** | Submit N seeded clustering runs as a SLURM array + a dependent distance job; WandB workspace view | `scripts/run_pipeline.py` (`pd-clustering`) | ❌ dropped |
| **Distances / consensus** | Load N histories, normalize labels across runs, compute per-iteration pairwise distance matrices, plot distribution | `scripts/calc_distances.py`, `merge_history.py::MergeHistoryEnsemble`, `math/merge_distances.py` | driver ❌ / `MergeHistoryEnsemble` + `compute_distances` ✅ (kept but now unused) |
| **Cluster mapping extraction** | Dump component→cluster JSON at a chosen iteration (singletons → null) for app consumption | `scripts/get_cluster_mapping.py` | ✅ kept |

What each stage consumes/produces:
- **Harvest** consumes a decomposed run + a data loader; produces a membership snapshot dir under `clustering/harvests/ch-<id>/`.
- **Merge** consumes a snapshot + a `MergeConfig`; produces `clustering/runs/c-<id>/{merge_config.json, history.zip}`.
- **calc_distances** consumes K `history.zip`s; produces `clustering/ensembles/e-<id>/{ensemble_meta.json, ensemble_merge_array.npz, distances_<method>.npz, plots/distances_<method>.png}`.

---

## 2. Ensemble / consensus logic — **the headline dropped capability** ❌

The whole reason to run the pipeline rather than a single merge: measure **clustering
stability** across stochastic re-runs. Mechanism, detailed:

1. **Seeded re-runs.** `run_pipeline.py` pre-allocates `n_runs` run IDs and emits one
   `run_clustering` command per run, each with `seed_offset = idx`. The offset perturbs
   `harvest.dataset_seed` (`run_clustering.main`), so each ensemble member sees a
   different token batch → potentially different dead components and stochastic merge
   pair draws. *(Status: ❌ — the orchestrator that fans these out is gone.)*

2. **Label normalization across runs** (`MergeHistoryEnsemble.normalized()`,
   `merge_history.py`). Because members can have *different alive-component sets*, raw
   group-index arrays aren't comparable. Normalization:
   - takes the **union** of all component labels across members (sorted → canonical index space, `c_components`);
   - remaps each member's `group_idxs` into that union index space;
   - puts every label **missing** from a member into **its own singleton group** in that member;
   - records `overlap_stats` (per-member alive/union ratio) and rich `merge_meta`.
   Produces a dense `MergesArray [n_ens, n_iters_min, c_components]`.
   *(Status: ✅ the method survives in the kept `merge_history.py`, but ❌ nothing in HEAD calls it.)*

3. **Per-iteration distance matrices** (`compute_distances`, `math/merge_distances.py`).
   For each merge iteration, compute the `n_ens × n_ens` pairwise distance between
   members' (normalized) clusterings → `DistancesArray [n_iters, n_ens, n_ens]`.
   Parallelized across iterations with a multiprocessing `Pool`.
   *(Status: ✅ function kept, ❌ caller dropped.)*

4. **Consensus artifacts + stability plot.** `calc_distances.py` saves the normalized
   merge array, ensemble metadata, the distance tensor, and a distance-distribution plot
   (distance vs iteration — low/stable = robust clustering structure).
   *(Status: ❌ dropped.)*

This is the capability with **no replacement** in the JAX path: the JAX worker→merge
route produces *single* runs only.

---

## 3. Distance metrics + merge algorithm (the math)

### Merge algorithm — greedy MDL hierarchical clustering ✅ kept
`merge.py::merge_iteration_memberships` + `compute_costs.py`.
- Builds a coactivation matrix from compressed memberships (CSR), GPU if it fits (`_choose_coact_device`).
- Each iteration: compute pairwise **merge costs**, **sample** a pair to merge, recompute coactivations/memberships for the merged pair, record into `MergeHistory`.
- **MDL objective** (`compute_mdl_cost`): `MDL = Σ_i s_i (log₂ k + α·r(P_i))` — coactivation-mass-weighted bits-to-encode + an **`alpha` rank penalty** on group size.
- **Merge cost** (`compute_merge_costs`): closed-form ΔMDL for merging each pair `(i,j)`.
- Early-stops when `k_groups ≤ 3`.

### Merge-pair samplers (`math/merge_pair_samplers.py`) ✅ kept
Stochastic alternatives to pure-greedy selection (the source of ensemble diversity):
| Sampler | What | Key kwarg |
|---|---|---|
| `range` | Uniform over all pairs whose cost ≤ `min + threshold·(max−min)` (threshold=0 ⇒ greedy) | `threshold` (default 0.05) |
| `mcmc` | Softmax over costs, `P ∝ exp(−cost/T)` | `temperature` |
| `exp_rank` | Rank-ordered, `P(rank r) ∝ exp(−decay·r)`; inverse-CDF sampled (dodges multinomial's 2²⁴ limit) | `decay` |

### Distance metrics ✅ kept (engine), ❌ dropped (only caller was the pipeline)
| Method | What | Key file |
|---|---|---|
| `perm_invariant_hamming` | Best label-permutation alignment via Hungarian (`scipy.linear_sum_assignment`) on a label co-occurrence matrix; distance = `n_components − matched` | `math/perm_invariant_hamming.py` |
| `matching_dist` | Co-membership-agreement distance: compares same-cluster indicator matrices, summed lower-triangle abs diff (loop form) | `math/matching_dist.py` |
| `matching_dist_vec` | Vectorized XOR-of-co-membership form of the above | `math/matching_dist.py` |

### Supporting math ✅ kept
- `math/merge_matrix.py` — `GroupMerge` (canonical component→group assignment; `merge_groups`, `to_matrix`, `identity`, `random`, `all_downstream_merged`) + `BatchedGroupMerge` (per-iteration stack used inside `MergeHistory`).
- `math/semilog.py` — `semilog()` signed log-ish transform for WandB metric readability (only consumer was `run_clustering`'s callback ❌).

---

## 4. Config surface

| Config | What's tunable | File | Status |
|---|---|---|---|
| `HarvestConfig` | `model_path`, `batch_size`, `n_tokens`, `n_tokens_per_seq` / `use_all_tokens_per_seq`, `dataset_seed`, `activation_threshold`, `filter_dead_threshold`, `filter_dead_stat` (max/mean), module-name filter | `harvest_config.py` | ✅ kept |
| `MergeConfig` | `alpha` (rank penalty), `iters` (None ⇒ `n_components−1`), `merge_pair_sampling_method`, `merge_pair_sampling_kwargs`; `stable_hash` for sweep dirs | `merge_config.py` | ✅ kept |
| `ClusteringRunConfig` | bundles `harvest` + `merge` + `ensemble_id` + `LoggingIntervals` (stat/tensor/plot/artifact) + WandB project/entity | `clustering_run_config.py` | ❌ dropped |
| `ClusteringPipelineConfig` | `clustering_run_config_path`, `n_runs`, `distances_methods`, SLURM (job-name prefix / partition / mem), WandB project/entity, `calc_distances` toggle, `create_git_snapshot` | `scripts/run_pipeline.py` | ❌ dropped |
| Pipeline YAMLs | `configs/pipeline_config.yaml`, `configs/pipeline-dev-simplestories.yaml` | `configs/*.yaml` | ❌ dropped |
| Run-config JSONs/example | `configs/crc/*.json`, `configs/crc/example.yaml`, `configs/README.md` | `configs/crc/*` | ❌ dropped |

---

## 5. CLIs

| CLI / entry | Drives | Status |
|---|---|---|
| `pd-clustering` (`run_pipeline.py`) | Whole **ensemble**: pre-allocate run IDs + git snapshot, submit SLURM array of clustering runs, submit dependent `calc_distances` jobs (one per method), create WandB workspace view; `--local` runs serially/in-parallel locally | ❌ dropped |
| `run_clustering.py` (`python -m …`) | **One** harvest→merge run with full WandB logging (scalars + tensor heatmaps + merge-history artifacts + per-iter plots); the array task body | ❌ dropped |
| `calc_distances.py` (`python -m …`) | Load N histories → `MergeHistoryEnsemble.normalized()` → `compute_distances` → save npz + distribution plot | ❌ dropped |
| `pd-cluster-harvest` (`run_harvest.py`) | torch harvest → membership snapshot | ❌ dropped (torch); ♻ replaced by `run_worker_jax.py` |
| `run_worker_jax.py` (`python -m …`) | JAX-native harvest → identical snapshot via `open_jax_run` | ✅ kept (new) |
| `pd-cluster-merge` (`run_merge.py`) | CPU-only merge from a snapshot → `history.zip`; the sweep workhorse | ✅ kept |
| `get_cluster_mapping.py` (`python -m …`) | Extract component→cluster JSON at a chosen iteration | ✅ kept |

---

## 6. Plotting / visualization ❌ entire `plotting/` dir dropped

| Plot | Of what | File |
|---|---|---|
| `plot_merge_iteration` | 3-panel per-iteration: merge matrix + coactivations + costs heatmaps (module-boundary tick labels) | `plotting/merge.py` |
| `plot_merge_matrix` | Group×component membership matrix with optional row-sum sidebar | `plotting/merge.py` |
| `plot_merge_history_cluster_sizes` | Cluster size vs iteration (log-y scatter) | `plotting/merge.py` |
| `plot_dists_distribution` | **Ensemble consensus plot** — pairwise distance vs iteration (points or min/median/mean/quartile bands, symlog y) | `plotting/merge.py` |
| `plot_activations` | Raw + concatenated (+ optional greedy-sorted) activation heatmaps, coactivation (lin/log) heatmaps, per-all/per-component/per-sample histograms | `plotting/activations.py` |
| `add_component_labeling` | Shared axis helper: module-boundary major ticks from `"module:idx"` labels | `plotting/activations.py` |

All were wired into WandB via `run_clustering._log_callback` — the `LogCallback`
protocol survives in `merge.py` but **no caller in HEAD passes one** (`run_merge` calls
`merge` with `log_callback=None`).

---

## 7. Outputs / artifacts per stage

| Stage | Artifacts | Dir | Status |
|---|---|---|---|
| Harvest | `harvest_config.json`, `memberships.npz`, `metadata.json`, optional `preview` | `clustering/harvests/ch-<id>/` | ✅ (♻ JAX driver) |
| Merge | `merge_config.json`, `history.zip` (`MergeHistory`) | `clustering/runs/c-<id>/` | ✅ kept |
| Ensemble | `pipeline_config.yaml`, `ensemble_meta.json`, `ensemble_merge_array.npz`, `distances_<method>.npz`, `plots/distances_<method>.png` | `clustering/ensembles/e-<id>/` | ❌ dropped |
| WandB | per-run scalars, tensor heatmaps, merge-history artifacts, plot images, ensemble workspace view | (WandB) | ❌ dropped |
| Cluster mapping | component→cluster JSON at an iteration | (stdout/file) | ✅ kept |

---

## Synthesis — what's most worth restoring

The merge engine and the harvest→merge data path are intact and JAX-native; the gap is
everything that made clustering an **ensemble stability study** rather than a single
greedy merge. Top candidates:

1. **Ensemble orchestration over the JAX worker (highest value).** Restore the
   `run_pipeline` fan-out idea but re-pointed at `run_worker_jax.py` instead of the torch
   `run_clustering`. The dropped SLURM-array + seeded-seed-offset logic in
   `run_pipeline.py` is mostly framework-agnostic; the only torch coupling lived in the
   `run_clustering`/`run_harvest` body, which the JAX worker already replaces. Entails: a
   thin launcher that submits N `run_worker_jax → pd-cluster-merge` jobs at seeded
   `dataset_seed`s + a dependent distance job. *Not obviated by the JAX path* — the JAX
   path gives single runs only.

2. **The consensus computation + stability plot (high value, low cost).** The hard part
   already survives: `MergeHistoryEnsemble.normalized()`, `compute_distances`, and all
   three distance metrics are kept and just need a caller. Restoring `calc_distances.py`
   (load K `history.zip`s → normalize → distances → npz) + `plot_dists_distribution` is a
   small, near-mechanical re-add of a thin driver + one plotting function. This is the
   payoff of (1).

3. **Per-run diagnostic plots (medium value).** `plot_activations`,
   `plot_merge_iteration`, `plot_merge_history_cluster_sizes` + the `LogCallback` wiring.
   The `LogCallback` hook is already kept in `merge.py`; restoring `plotting/` and a
   logging callback gives back the live merge/coactivation/cluster-size views. Lower
   priority than (1)+(2): useful for understanding a single run, not for the headline
   stability question. WandB-specific glue (`semilog`, `wandb_log_tensor`) would come
   with it.

**Obviated by the JAX worker→merge path:** the torch `run_harvest.py` /
`run_clustering.py` / `pd-cluster-harvest` bodies and `clustering_run_config.py`'s
harvest+merge+WandB bundling — `run_worker_jax.py` + `pd-cluster-merge` already cover
single-run harvest+merge cleanly. A restore should rebuild **ensemble + consensus** on
top of those, not resurrect the torch single-run drivers.
