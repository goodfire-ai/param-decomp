# Dataset Attributions Module

Multi-GPU pipeline for computing component-to-component attribution strengths aggregated over the training dataset. Unlike prompt attributions (single-prompt, position-aware), dataset attributions answer: "In aggregate, which components typically influence each other?"

## Usage (SLURM)

```bash
pd-attributions <wandb_path> --config attr_slurm_config.yaml --harvest_subrun_id h-YYYYMMDD_HHMMSS
pd-attributions <wandb_path> --config attr_slurm_config.yaml --harvest_subrun_id h-... --job_suffix v2
```

The YAML/JSON config is an `AttributionsSlurmConfig` — wraps `DatasetAttributionConfig`
(which carries `wandb_path`, `n_batches`, `batch_size`, `ci_threshold`) plus SLURM knobs
(`n_gpus`, `partition`, `time`, `merge_time`, `merge_mem`). The harvest subrun supplies the
alive-mask set the attribution pass restricts to.

The command:
1. Creates a git snapshot branch for reproducibility
2. Submits a SLURM job array (one per GPU)
3. Each task processes batches where `batch_idx % world_size == rank`
4. Submits a merge job (depends on array completion)

## Usage (non-SLURM)

```bash
# Single GPU
python -m param_decomp.dataset_attributions.scripts.run_worker <wandb_path> \
    --config_json '{"wandb_path": "<wandb_path>", "n_batches": 1000}' \
    --harvest_subrun_id h-YYYYMMDD_HHMMSS --rank 0 --world_size 1 --subrun_id da-xxx

# Multi-GPU
SUBRUN="da-$(date +%Y%m%d_%H%M%S)"
CFG='{"wandb_path": "<wandb_path>", "n_batches": 1000}'
for r in 0 1 2 3; do
  python -m param_decomp.dataset_attributions.scripts.run_worker <wandb_path> \
    --config_json "$CFG" --harvest_subrun_id h-... --rank $r --world_size 4 --subrun_id $SUBRUN &
done
wait
python -m param_decomp.dataset_attributions.scripts.run_merge --wandb_path <wandb_path> --subrun_id $SUBRUN
```

## Data Storage

```
PARAM_DECOMP_OUT_DIR/dataset_attributions/<run_id>/
├── da-20260223_183250/                    # sub-run (latest picked by repo)
│   ├── dataset_attributions.pt            # merged result
│   └── worker_states/
│       └── dataset_attributions_rank_*.pt
```

`AttributionRepo.open(run_id)` loads the latest `da-*` subrun that has a `dataset_attributions.pt`.

## Attribution Metrics

Two metrics: `AttrMetric = Literal["attr", "attr_abs"]`

| Metric | Formula | Description |
|--------|---------|-------------|
| `attr` | E[∂y/∂x · x] | Signed mean attribution |
| `attr_abs` | E[∂\|y\|/∂x · x] | Attribution to absolute value of target (2 backward passes) |

Naming convention: modifier *before* `attr` applies to the target (e.g. `attr_abs` = attribution to |target|).

## Architecture

### Storage (`storage.py`)

`DatasetAttributionStorage` stores four structurally distinct edge types:

| Edge type | Fields | Shape | Has abs? |
|-----------|--------|-------|----------|
| component → component | `regular_attr`, `regular_attr_abs` | `dict[target, dict[source, (tgt_c, src_c)]]` | yes |
| embed → component | `embed_attr`, `embed_attr_abs` | `dict[target, (tgt_c, vocab)]` | yes |
| component → unembed | `unembed_attr` | `dict[source, (d_model, src_c)]` | no |
| embed → unembed | `embed_unembed_attr` | `(d_model, vocab)` | no |

All layer names use **canonical addressing** (`"embed"`, `"0.glu.up"`, `"output"`).

Unembed edges are stored in residual space (d_model dimensions). `w_unembed` is stored alongside the attribution data, so output token attributions are computed on-the-fly internally — callers never need to provide the projection matrix. No abs variant for unembed edges because abs is a nonlinear operation incompatible with residual-space storage.

**Normalization**: `normed[t, s] = raw[t, s] / source_denom[s] / target_rms[t]`. Component sources use `ci_sum[s]` as denominator, embed sources use `embed_token_count[s]` (per-token occurrence count). This puts both source types on comparable per-occurrence scales.

Key methods: `get_top_sources(key, k, sign, metric)`, `get_top_targets(key, k, sign, metric)`. Both return `[]` for nonexistent components. `merge(paths)` classmethod for combining worker results via weighted average by n_tokens.

### Harvester (`harvester.py`)

Accumulates attributions using gradient × activation. Uses **concrete module paths** internally (talks to model cache/CI). Four accumulator groups mirror the storage edge types. Key optimizations:
1. Sum outputs over positions before gradients (reduces backward passes)
2. Output-residual storage (O(d_model) instead of O(vocab))
3. `scatter_add_` for embed sources, vectorized `.add_()` for components (>14x faster than per-element loops)

### Harvest (`harvest.py`)

Orchestrates the pipeline: loads model, builds gradient connectivity, runs batches, translates concrete→canonical at storage boundary via `topology.target_to_canon()`.

### Scripts

- `scripts/run_worker.py` — worker entrypoint (single GPU)
- `scripts/run_merge.py` — merge entrypoint (CPU only, needs ~200G RAM)
- `scripts/run_slurm.py` — SLURM launcher (array + merge jobs)
- `scripts/run_slurm_cli.py` — CLI wrapper for `pd-attributions`

### Config (`config.py`)

- `DatasetAttributionConfig`: n_batches, batch_size, ci_threshold
- `AttributionsSlurmConfig`: adds n_gpus, partition, time, merge_time, merge_mem (default 200G)

### Repository (`repo.py`)

`AttributionRepo.open(run_id)` → loads latest subrun. Returns `None` if no data.

## Query Methods

All query methods take `metric: AttrMetric` (`"attr"` or `"attr_abs"`).

| Method | Description |
|--------|-------------|
| `get_top_sources(target_key, k, sign, metric)` | Top sources → target |
| `get_top_targets(source_key, k, sign, metric)` | Top targets ← source |

Key format: `"embed:{token_id}"`, `"0.glu.up:{c_idx}"`, `"output:{token_id}"`.

Note: `attr_abs` returns empty for output targets (unembed edges have no abs variant).
