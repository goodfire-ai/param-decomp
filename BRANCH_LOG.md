# Layerwise Parallel SPD Training

## What this branch is

A new entry point — `pd-run-layerwise` — that parallelises SPD training across
target modules. Given an orchestrator YAML (a fully-normalised `Config` whose
`module_info` enumerates every target module), the script fans out one
independent SLURM task per module: each task is a separate training run that
decomposes a single module. Per-module runs are independent — they share only
the frozen target model — and runs cluster under a single `wandb_group` for
UI grouping.

This is a structural factoring of SPD training: when the only cross-layer
coupling is whole-model PPGD, each layer's CI fn + components is a genuinely
independent optimization. Splitting unlocks per-module independent
hyperparameters, isolated failures, and trivial cluster scheduling.

## Conversation arc & decisions

### 1. Design discussion
- Identified that with only layerwise reconstruction losses, training is fully
  parallelisable per module — each module's CI fn / components is an
  independent optimization problem sharing only the frozen target.
- Chose a "two-phase curriculum" approach over an async sidecar:
  - **Phase 1 (this branch):** N independent per-module runs, no whole-model
    cross-layer pressure.
  - **Phase 2 (future, task #3):** merge component checkpoints + short PPGD
    finetune for cross-layer adversarial pressure.
- Rejected: async sidecar for continuous whole-model PPGD pressure
  (complexity, gradient staleness, coordinator SPOF).

### 2. Orchestrator design
- Orchestrator YAML is a complete `Config` with a `wandb_group` set and
  `module_info` enumerating concrete modules (no wildcards). No "overrides"
  / "deltas" — the orchestrator represents the full normalised config.
- Generator does two things per run: filter `module_info` to one entry,
  set `wandb_run_name` to the module path.
- Module patterns are listed manually since C values are per-module and
  paths aren't standardised across architectures.

### 3. Validator (`run_layerwise.py:_validate_orchestrator`)
Rejects orchestrators that would defeat the layerwise split:
- `wandb_group` must be set (used for UI clustering).
- `module_info` patterns must be concrete (no `*` wildcards) and unique.
- Forbidden losses: `StochasticReconLoss`, `StochasticReconSubsetLoss`,
  `PersistentPGDReconSubsetLoss` — these require whole-model joint training.
- `PersistentPGDReconLoss` is allowed (single-module PPGD is coherent — it
  just acts on the one module's mask).

### 4. WandB group auto-suffixing
- Orchestrator's `wandb_group` field is the *base name*.
- Actual group is `{base}-{launch_id_timestamp}`, e.g.
  `jose-layerwise-v1-20260515_092220`.
- Prevents collisions across reruns of the same orchestrator.
- The base name is also written into `extra_wandb_config.group_base` for
  filtering in the WandB UI.

### 5. WandB tagging & config promotion
- New `wandb_tags: list[str]` field on `Config` — extra tags merged with the
  launch/experiment tags.
- New `extra_wandb_config: dict[str, str]` field — keys promoted to top-level
  `wandb.config`, so they're available as group-by axes in WandB chart panels
  (tags can't be used for group-by).
- Generator auto-populates `block` (e.g. `L0`) and `module_type`
  (e.g. `mlp.c_fc`) as both tags and config properties so you can overlay the
  same layer type across blocks, or all modules per block.

### 6. Profiling iteration
- **py-spy first** (5 min sample on a running task). Misleading — sync waits
  showed up as "Python time" in `calc_l0` and `.item()` calls. Reported them
  as bottlenecks; they're 0.1% each.
- **CUDA events** (`PD_PROFILE=1` env-var gated, 60-step run with 10-step
  warmup, added to `run_param_decomp.py`). Truth:
  ```
  data            165.2 ms  (38.7%)   ← only real bottleneck
  backward        102.1 ms  (23.9%)   ← real GPU compute
  compute_losses   87.1 ms  (20.4%)   ← real GPU compute
  target_fwd       51.2 ms  (12.0%)
  ci_compute       19.0 ms  ( 4.5%)
  optimizer_step    1.3 ms  ( 0.3%)
  loss_items        0.2 ms  ( 0.1%)
  calc_l0           0.2 ms  ( 0.1%)
  ```
- Fix: `num_workers=2, pin_memory=True, persistent_workers=True` on the
  `DataLoader` (`data.py:260`). Hardcoded for now; can be hoisted to
  `DatasetConfig` if needed.
- Re-profile post-fix: `data: 0.04 ms`, total `426→261 ms` (1.63×
  throughput, ~24h → ~14.5h per task).

### 7. Cluster trip-ups
- **Snapshot branches:** `pd-run-layerwise` git-snapshots the live tree per
  launch, so code changes only take effect for *new* submissions, not
  in-flight ones.
- **HF Hub transient FileNotFoundError:** at one point the streaming dataset
  failed to fetch parquet shards (likely a brief HF Hub hiccup amplified
  by 48 tasks × 2 workers hitting the Hub simultaneously). Cancelled and
  resubmitted under the new auto-suffixed groups; bottom of queue currently.

## Where things are at end-of-session

- **Code:** ready, `basedpyright` and `ruff` clean.
- **Snapshots / runs:**
  - **Successful runs from yesterday:** 11 / 24 v1 modules completed full
    200k-step training in WandB group `jose-layerwise-v1` (modules `4` and
    `15–24`: `h.0.attn.k_proj` + all of layers 2 & 3 except `h.2.mlp.{c_fc,
    down_proj}`). These pre-date the eval-config fix and don't have
    `CEandKLLosses` / `PGDReconLoss` evals.
  - **Latest queued arrays:** 21411 (v1) and 21412 (v2-ppgd) — both
    PENDING in the cluster queue. New groups:
    - `jose-layerwise-v1-20260515_092220`
    - `jose-layerwise-v2-ppgd-20260515_092232`
  - **Eval configs corrected:** both yamls now include `CEandKLLosses`
    (master sanity eval) and `PGDReconLoss` (per-module adversarial mask
    robustness).
- **Background monitor:** `b3p5jy3lb`, polling `sacct` every 30 min, fires
  when both arrays complete.

## File map

| Path | What |
|---|---|
| `param_decomp/scripts/run_layerwise.py` | Orchestrator loader, validator, per-module splitter, SLURM array submitter. Hardcodes `lm_decomposition.py` as the decomp script (the layerwise split is fundamentally an LM concern). |
| `param_decomp/scripts/run_layerwise_cli.py` | Fire CLI entry. Resolves orchestrator path (absolute, relative-to-cwd, relative-to-REPO_ROOT). |
| `param_decomp/experiments/lm/jose_layerwise.yaml` | Vanilla orchestrator (24 modules, no PPGD, layerwise stochastic recon @ coeff 1.0). |
| `param_decomp/experiments/lm/jose_layerwise_v2.yaml` | PPGD-included variant (same as v1 + `PersistentPGDReconLoss` @ 0.5). |
| `param_decomp/configs.py` | Adds `wandb_group`, `wandb_tags`, `extra_wandb_config` fields to `Config`. |
| `param_decomp/utils/wandb_utils.py` | `init_wandb` now accepts `group=`, threads through to `wandb.init`. `extra_wandb_config` keys promoted to top-level `wandb.config`. |
| `param_decomp/run_param_decomp.py` | (a) Passes `wandb_group` to `init_wandb`. (b) `wandb_tags` merged with launch/experiment tags. (c) `PD_PROFILE=1` env-var gates CUDA-event phase timing for diagnostic runs (no-op otherwise). |
| `param_decomp/data.py` | `DataLoader` now uses `num_workers=2, pin_memory=True, persistent_workers=True`. Eliminated the 165ms/step dataloader stall. |
| `pyproject.toml` | New entry point `pd-run-layerwise`. |

## Key design diffs vs base `pile_llama_simple_mlp-4L.yaml`

| field | base | layerwise v1 |
|---|---|---|
| `wandb_group` | — | `jose-layerwise-v1` (base; auto-suffixed at launch) |
| `ci_config.simple_transformer_ci_cfg.d_model` | 2048 | 1024 |
| `ci_config…n_blocks` | 8 | 4 |
| `ci_config…mlp_hidden_dim` | [8192] | [4096] |
| `ci_config…attn_config.n_heads` | 16 | 8 |
| `module_info` | 6 wildcard patterns (`h.*.X`) | 24 concrete entries (`h.{0..3}.X`) |
| `StochasticReconSubsetLoss` (0.5) | present | **removed** |
| `PersistentPGDReconLoss` (0.5) | present | **removed** (v1) / **kept** (v2) |
| `StochasticReconLayerwiseLoss` (1.0) | — | **added** |
| `FaithfulnessLoss`, `ImportanceMinimalityLoss` | kept | kept |
| `steps` | 400000 | 200000 |
| `CI_L0.groups` | per-layer + total | total only |

CI fn shrunk ~8× because the network only sees one module's activations.
Module patterns concretised because the generator filters one per run.
Whole-model losses dropped from v1 since each task only sees its module's
gradient; v2 re-adds PPGD to test whether single-module adversarial pressure
helps. Steps halved on the bet that per-module optimisation converges sooner
than joint — pure guess, can extend if needed.

## Open follow-ups

- Task #3: PPGD finetune phase after the 24 per-module runs complete.
  Needs (a) a "compose 24 component checkpoints into one ComponentModel"
  loader, (b) a finetune entrypoint with PPGD + small baseline coefficients,
  (c) a WandB artifact recording the source run IDs.
- The `num_workers=2` is hardcoded in `data.py`. If we want per-experiment
  control, hoist it to `DatasetConfig`.
- The dropped `CEandKLLosses` / `PGDReconLoss` mistake: the 11 v1 modules
  that finished yesterday don't have these eval metrics. If we want them
  re-evaluated post-hoc on those 11 checkpoints, that's a separate eval-only
  pass.
