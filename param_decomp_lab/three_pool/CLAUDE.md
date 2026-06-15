# `param_decomp_lab/three_pool/`

The 3-pool training subsystem — sibling of `param_decomp.optimize.Trainer` for
splitting a decomposition run across three rank pools (CI fn, chunkwise V/U,
PPGD adversary). See `DESIGN.md` for the per-step comm graph and the module
docstring in `optimize.py` for the data-handling contract.

The chunkwise pool shards the decomposed *sites* into **chunks**; each chunk is a
site-group replicated across `chunk_dp` DDP ranks. ("Chunkwise" — not "layerwise":
a chunk owns an arbitrary slice of sites, not a model layer. The core-library
*layerwise recon loss* — `StochasticReconLayerwiseLoss`, reconstruct each decomposed
layer's output independently — is a separate, stable concept the chunkwise pool
*uses*; it keeps the name "layerwise".)

| File | What it covers |
|---|---|
| `optimize.py` | `ThreePoolTrainer` + `optimize_three_pool`; the training loop, `snapshot`/`from_snapshot`. Consumes `ThreePoolConstrainedPDConfig` (reads `pd.losses.*` directly). Config constraints are type-level (see `pd_config.py`) + a load-time validator on `ThreePoolLMExperimentConfig`; the topology is resolved into ranks/chunks in `_build_runtime` (which also runs the site-coverage check, needs the loaded model) |
| `pd_config.py` | `ThreePoolConstrainedPDConfig` + the typed `ThreePoolLosses(faith, imp, stoch, ppgd, recon_plan)` struct — the 3-pool-constrained `PDConfig`. Lives with the subsystem (not `experiments/lm/`) so the subsystem is self-contained: `optimize` imports it from its own package, no back-dependency into `experiments/lm/` |
| `config.py` | `ThreePoolTopology` (`ci` / `ppgd` / `chunkwise` `PoolSpec`s of per-rank batch + `sites_per_chunk`) + `resolve(ordered_sites, batch_size) -> ResolvedLayout`. Authors per-rank batch, NOT rank ids; the resolver derives ranks/chunks/world_size in canonical order. Parse-time validation = cross-divisibility of the three per-rank batches |
| `layout.py` | `World` topology + the runtime `Chunk` (`.ranks` / `.sites`); `build_world` constructs every process group (threading `pg_timeout` into each); `BatchEdge` — symmetric per-edge batch-slice geometry (CI↔chunk, CI↔PPGD) answering routing for both fan directions |
| `checkpoint.py` | offline state_dict assembly from on-disk partials (`assemble_model_state_dict_from_partials`) + the leader key-partition helpers (`owned_model_state_keys` / `ci_fn_state_keys`) |
| `consolidate.py` | `consolidate_step` — async, off-train-loop, **streaming** assembly of `model_<step>.pth` + `training_<step>.pth` from a step's small (parameter-shaped) scratch partials; prunes old `training_*.pth` + `ppgd_*/`; deletes the scratch dir. The data-shaped PPGD sources are NOT here — each adversary rank writes its own `ppgd_<step>/rank_<r>.pth` at snapshot time, in parallel (see "Checkpoint save" below). `load_ppgd_shard` reads a rank's shard on resume; `unconsolidated_steps` lists recoverable steps |
| `consolidate_cli.py` | `python -m …consolidate_cli <run> [--step N]` — manual CPU-only recovery for a failed/preempted async consolidation (separate module to avoid an import cycle with `experiments.lm.run`) |
| `role.py` | `PoolRole = CIRole \| ChunkRole \| PPGDRole` — this rank's pool role; per-pool fields are union variants, not optional attrs |
| `context.py` | `PoolContext = CIContext \| ChunkContext \| PPGDContext` — `world` + `role` + this pool's portals; the trainer matches on it to dispatch step fns |
| `portals.py` | Cross-pool exchanges as typed objects — one class per DAG edge (pack layout + routing + dtype + PG in one place) |
| `step_{ci,chunkwise,ppgd}.py` | per-pool step functions |
| `recon_plan.py` | `ReconPlan` (`PerSitePlan` \| `SubsetReconPlan`) — how each chunk turns its sites into a list of recon forwards |
| `eval_step.py` | 3-pool eval pass (PPGD pool runs metrics; others barrier through) |
| `reductions.py` | cross-pool log reductions: losses (`aggregate_losses_to_rank0`), peak memory (`aggregate_max_memory_to_rank0`), grad norms (`aggregate_grad_norms_to_rank0` + `per_param_grad_norms`) |
| `SUM_GRAD_CONVENTION.md` | the gradient-assembly scaling convention (proposal) |

## 2-pool variant (`two_pool_*.py` + `step_pool_a.py`)

The 2-pool variant **merges the CI and PPGD pools into one Pool A** (adversary + CI fn
co-located on the same ranks, same batch slice); the chunkwise pool (Pool B) is
unchanged. Because masks are produced where the adversary consumes them, the entire
CI↔PPGD edge (mask send AND g_CI return) disappears — the adversary's g_CI is the LOCAL
`.grad` of the CI forward's own `lower_leaky`. (That edge is also what deadlocked
seq-2048 fan-out runs, so deleting it is a structural fix.) The only surviving cross-pool
edge is Pool A ↔ chunk: masks out, g_CI back, V/U grads out, updated V/U in.

The **gradient assembly is identical** to the 3-pool (see `SUM_GRAD_CONVENTION.md`): the
CI-fn grad seed is `g_CI_chunk + g_CI_adversary + imp_min` (adversary half now local
instead of received), SUM-reduced over the Pool A group; V/U grad = chunkwise(owner) +
adversary(replica, contribute-once on the chunk leader). Validated by
`tests/test_two_pool_grad_check_distributed.py` (non-square `n_a=4` / `chunk_dp=2`, all
loss terms, worst rel err 4.04e-7 vs the SAME single-process reference the 3-pool uses).

### PPGD source scopes (2-pool)

The 2-pool supports two PPGD source scopes (`losses.ppgd.scope`):

- `bsc` — an independent source per (batch element, position). Pool A's
  per-rank batch slice is self-contained, so no cross-rank source sync; the final source
  step uses this rank's own grads. This is the long-standing default.
- `sc` — ONE source shared across the whole **global** batch (shape
  `(1, S, C)` per site instead of `(B, S, C)`), a ~1000× storage/memory win that makes
  large-batch + full-model runs feasible. The shared source is **replicated** across the
  Pool A data-parallel ranks: `PersistentPGDState` broadcast-inits it from the Pool A
  group leader and **AVG-reduces** its grads over the Pool A group (`world.ci_pool_group`)
  on every PGD step (warmup inner loop + the final `(N+1)`'th step). Identical init +
  deterministic AVG + identical optimizer step keeps the replicas bit-identical without
  per-step re-broadcast.

The reduction is **AVG over the Pool A group** (equivalently SUM ÷ group size). Each rank's
per-rank source grad is `∂(sum_loss_local / n_examples_local)/∂source`; with uniform batch
slicing `mean_r(sum_loss_local / n_examples_local) == sum_loss_global / n_examples_global`,
so AVG reproduces the shared-source full-batch grad. Confirmed by a real one-backward grad
check (RNG pinned) — `tests/test_two_pool_grad_check_distributed.py` is parametrized over
both scopes, each compared to its own single-process reference (broadcast: one shared
source; per-batch: per-rank slices stitched), worst rel err **4.04e-7** for both.
`PersistentPGDState.__init__` takes a `replica_sync_group` (the Pool A group when the scope
needs sync, else `None`); the metric path (`persistent_pgd_recon.py`) passes the active
reduction group, so the single-pool whole-world DP path for replicated scopes is owned by
the state machine too. The 3-pool path is unchanged (`bsc` only).

Key reuse trick (`two_pool_layout.build_two_world`): the 2-pool world is the 3-pool
`World` with `ci_ranks == ppgd_ranks == pool_a_ranks` and one Pool A all-reduce group
serving as both the CI-pool and PPGD-pool group. With that identity, **the chunkwise pool
is byte-for-byte the 3-pool chunkwise pool** (same `ChunkContext` / `step_chunkwise` /
portals — its four cross-pool edges all land on Pool A), and the surviving portal classes
(`CiValuesToChunkwise` / `GradCiFromChunkwise` / `GradVuFromPPGD` / `UpdatedVuToPPGD`)
work unchanged keyed on `ci_chunk_edge`. A `PoolARole` presents `.as_ci()` / `.as_ppgd()`
views for the two portal families.

**Cross-pool send/recv order is load-bearing** in `step_pool_a`: CI and the adversary are
on ONE rank, so the chunkwise step's send-g_CI / recv-g_VU pair (serviced on two
different ranks concurrently in the 3-pool) must be serviced here in the SAME order
chunkwise issues them — recv g_CI FIRST, then send g_VU — or the two pools deadlock.

| File | What it covers |
|---|---|
| `two_pool_config.py` | `TwoPoolTopology` (`pool_a` / `chunkwise` `PoolSpec`s) + `resolve` → `TwoPoolResolvedLayout` (canonical order: chunks first, then Pool A) |
| `two_pool_layout.py` | `build_two_world` — constructs a `World` with `ci_ranks == ppgd_ranks == pool_a_ranks` directly (bypasses `build_world`'s disjoint assertion) |
| `two_pool_role.py` | `PoolARole` (with `.as_ci()` / `.as_ppgd()`) `\| ChunkRole` |
| `two_pool_context.py` | `PoolAContext \| ChunkContext`; `PoolAPortals` (the four A↔chunk edges) |
| `step_pool_a.py` | the merged Pool A step (CI fwd + imp + adversary + fused CI backward) |
| `two_pool_optimize.py` | `TwoPoolTrainer` + `optimize_two_pool`. Reuses `ThreePoolConstrainedPDConfig` + `_ThreePoolRuntime` (ci_ranks == ppgd_ranks). **Checkpoint/resume implemented** — `snapshot`/`from_snapshot` reuse the 3-pool partial format + `ThreePoolTrainingState` (the Pool A partial carries CI fn + ci-fn optimizer; each Pool A rank writes its PPGD sources to `ppgd_<step>/rank_<r>.pth` at snapshot time, in parallel; `consolidate.py` routes ci-fn optimizer via a `"pool_a"` case, and `from_snapshot` takes a `ppgd_shard_dir`). Saves on `cadence.save_every` + a forced final snapshot |
| `two_pool_eval_step.py` | 2-pool eval pass (Pool A builds the full `MetricContext` locally — no cross-pool CI ship; chunkwise barriers through) |
| `two_pool_reductions.py` | 2-pool cross-pool log reductions (Pool A emits imp/ppgd/l0; chunkwise emits faith/stoch) |

Run a 2-pool LM run via `pd-lm-2pool` (`experiments/lm/two_pool_run.py`,
`TwoPoolLMExperimentConfig`). The launch path mirrors the 3-pool's: `--dp N` pushes a
per-run git snapshot ref and submits the SLURM job (single-node N≤8, multi-node N>8 a
multiple of 8); the async consolidate+slow-eval job fires on `on_save`
(`submit_slurm_async_consolidate_and_eval`, passing `--variant two_pool` so `async_eval`
loads the parent config as a `TwoPoolLMExperimentConfig`). `--resume <resume.yaml>` rebuilds
via `TwoPoolTrainer.from_snapshot`. A local training-only smoke runs via
`torchrun --standalone --nproc_per_node=N -m param_decomp_lab.experiments.lm.two_pool_run <cfg>`
(but save→consolidate→eval needs `--dp`, like the 3-pool — see "Launch path" below).

## wandb logging parity with single-pool

The 3-pool `train/` + `eval/` keys are kept identical to `param_decomp.optimize.Trainer`'s
so a 3-pool run and a single-pool run overlay on the same wandb panels:

- **Losses** (`_log_train_metrics` in `optimize.py`): the four losses log as
  `train/loss/<MetricClassName>` (e.g. `train/loss/FaithfulnessLoss`), not the internal
  short names. The class name comes from each loss config's `type` literal, carried on
  `_ThreePoolRuntime.log_name_{faith,imp,stoch,ppgd}`. Plus `train/loss/total`.
- **Grad norms**: per-param `train/grad_norms/components/<site>.<param>` and
  `train/grad_norms/ci_fns/<param>`, plus `train/grad_norms/summary/{components,ci_fns,total}`
  — same layout as single-pool's `component_grad_norms`. Params are pool-sharded, so each
  owning rank computes its **pre-clip** norms (`per_param_grad_norms`, after the in-pool
  SUM-reduce so they're the true global grad, before the clip) and stashes them in
  `metrics["grad_norms/..."]`; `aggregate_grad_norms_to_rank0` all-gathers component norms
  within the chunkwise pool and ships the CI leader's ci-fn norms to rank 0 (object collectives,
  log-steps only). PPGD owns no trained params. Summaries are derived on rank 0.
- **Per-loss grad norms** (3-pool-only diagnostic for coeff rebalancing):
  `train/grad_norms/components/by_loss/{FaithfulnessLoss,StochasticReconLayerwiseLoss,PersistentPGDReconLoss}`
  — each loss term's contribution to the global V/U grad. faith + ppgd are contribute-once
  (chunk-leader-only), stoch is each rank's partial; one chunk SUM-all-reduce of
  `[faith, ppgd, total]` recovers each term's global grad (`stoch = total - faith - ppgd`),
  leader takes sum-sq (`_component_grad_sumsq_by_loss` in `step_chunkwise.py`), and
  `aggregate_component_grad_by_loss_to_rank0` SUMs across chunks. CI-fn split (imp vs recon)
  not done — the fused CI backward would have to be unfused. Log-steps only.
- **LR schedules**: both `train/schedules/lr/components` and `train/schedules/lr/ci_fn`
  (rank 0 computes the ci-fn LR from the schedule directly — no cross-pool comm).
- **3-pool extras** (no single-pool equivalent): `train/perf/step_ms` (MAX over chunkwise),
  `train/mem/{chunk,ci,ppgd}_peak_gb`, `train/metrics/mean_l0` (batch-mean active CI components
  per token across all sites, threshold 0 — the live sparsity readout; CI pool computes
  it from `lower_leaky` and ships it via the loss-reduction path, see `reductions.py`).
- **Eval**: in-train fast eval logs `eval/<k>`; slow eval is async (`experiments/lm/async_eval.py`)
  under `slow_eval/<k>` on a `slow_eval/step` axis. The single-pool path now also runs its
  slow metrics in-train under the same `slow_eval/` namespace (`_build_eval_loop(..., include_slow=True)`
  for 1-pool, `False` for 3-pool); the `slow_eval/step` axis is defined in `init_wandb`.

## Gradient-assembly scaling: the SUM convention

See `SUM_GRAD_CONVENTION.md` for the full derivation. Summary: every
data-parallel gradient reduction is **SUM** (`all_reduce_ci_fn_grads`,
`all_reduce_grads_in_chunk`, and PPGD's V/U reduce). Each producer emits a
*partial sum* normalized only by the honest GLOBAL count — NO `n_ci` /
`chunk_dp` transport factor. `SUM(partials) = total`, so no producer needs a
pool's size. The REPLICATED contributions are handled structurally rather than by
a replica-count divide: faith + broadcast-PPGD V/U **contribute once** (emitted
on the chunk leader only), and imp-min uses the **detached-global-residual** trick
(`S = local + (all_reduce_sum(local.detach()) - local.detach())`) so its backward
is a local partial. The grad-clip `n_replicas` is unchanged — it counts distinct
params for the global norm, independent of the grad-reduce op. Validated by
`tests/test_three_pool_grad_check_distributed.py` (non-square, all loss terms).

## Checkpoint save: partials on the loop, consolidation off it

The save path is split so the train loop never blocks on a multi-GB read.

`snapshot()` (on the train loop, all ranks collective): each rank writes a
**self-contained partial** to `scratch_dir/step_<S>/rank_<r>.pth` — its owned
model params (chunk leaders → chunk-sites V/U; CI pool leader → CI fn) and its
optimizer state (name-keyed). Adversary ranks (PPGD pool in 3-pool; Pool A in
2-pool) ALSO write their data-shaped PPGD sources, **in parallel**, straight to
the stable shard `out_dir/ppgd_<S>/rank_<r>.pth` (`out_dir = scratch_dir.parent`)
— each rank writes its own, so this is free (the same bytes the partial used to
carry, just split into the file resume reads directly). Rank 0 also writes
`meta.pth` (configs + fingerprint + `c_per_site` / `all_sites`). There is **one
pre-write barrier** (so rank 0's `mkdir` + `meta` write land before others write
into the dir) and **one post-write rejoin barrier** (so all ranks leave
`snapshot()` together). All writes are cheap and parallel — no rank does a 100 GB
read on the loop, and no rank serializes a TB of PPGD shards — so neither barrier
can overrun the watchdog. NO model NCCL gather, NO rank-0 assembly here.

`consolidate_step()` (`consolidate.py`, async SLURM job, off the loop): **streams**
a step's small (parameter-shaped) partials one at a time (peak RAM ≈ one partial +
the assembled CPU model, never the full partial set) → assembles the full
`ComponentModel` state_dict + `ThreePoolTrainingState` → writes `model_<S>.pth` +
`training_<S>.pth` → prunes old `training_*.pth` to the last
`DEFAULT_KEEP_LAST_N_TRAINING` (=3; **all `model_*.pth` are kept**) → prunes old
`ppgd_*/` → deletes `step_<S>/`. It runs inside the async slow-eval job
(`experiments/lm/async_eval.py`), as a CPU-only phase BEFORE the eval pass, so the
assembled `model_<S>.pth` exists before the eval loads it. It reads **only** the
small partials — the PPGD shards were already written at snapshot time, so
consolidation never reads or rewrites the data-shaped sources. (This was the I/O
bottleneck before: rank 0 streamed every multi-GB partial AND rewrote every PPGD
shard sequentially — TBs at 80-GPU scale — overrunning the async waiter timeout.)

**PPGD sources are NOT in `training_<S>.pth`, and NOT in the scratch partials.**
They're `bsc`-scoped (sized by `batch × seq × n_components`) — the only
persisted state that's data-shaped rather than parameter-shaped, so aggregating it
onto rank 0 doesn't scale (~2.3 TB at batch 1280, OOMs any node) and streaming it
through rank 0 in consolidation is an I/O bottleneck. Instead each adversary rank
writes its own sources straight to `ppgd_<S>/rank_<r>.pth` (`ppgd_shard_dirname`)
inside `snapshot()`, in parallel with every other rank's write;
`from_snapshot` takes a `ppgd_shard_dir` and each adversary rank reads its own
shard (`load_ppgd_shard`). A missing shard ⇒ that rank's adversary re-warms via
`n_warmup` (graceful). This ties a *PPGD* resume to the same pool layout (the V/U +
optimizer state stays topology-agnostic). `ppgd_<S>/` dirs are pruned (in
`consolidate_step`) to the last `DEFAULT_KEEP_LAST_N_PPGD` (=1 — they're huge);
resume uses the newest checkpoint. Idempotent: a no-op if `training_<S>.pth`
already exists (and it cleans any leftover scratch in that case); the scratch dir
is deleted only on success.

This is the fix for how run 34446 (p-a5b667e9) died: the old synchronous rank-0
read held the other ranks at a barrier past the NCCL watchdog. There is no
on-loop read to outrun the watchdog now.

### Consolidation reliability (the async job's contract)

`async_eval.main` runs consolidation via `_consolidate_or_wait`, which is
engineered to **never hang** (a hung job holds GPUs forever — the one
unacceptable failure mode for an off-loop job):

  * **CPU-only, post-init.** Assembly builds a full `ComponentModel` buffer; it
    must run on CPU (under `torch.device("cpu")`) — building it on the job's
    selected, shared GPU hangs. It runs AFTER `init_distributed` (so
    `build_target`'s `ensure_cached_and_call` has its distributed state) but the
    buffer never touches the GPU. `assemble_model_state_dict_from_partials`
    freezes the target (`eval()` + `requires_grad_(False)`) before constructing
    the buffer — `build_target` only `.eval()`s it, and `ComponentModel` asserts
    a frozen target (an unfrozen target was the original "hang": rank 0 hit the
    assertion and died while the other rank waited).
  * **Rank 0 assembles; others file-wait, fail-fast.** Non-rank-0 ranks poll the
    shared FS for `training_<S>.pth` (written last) rather than an NCCL barrier
    (so the multi-second CPU read never interleaves with a GPU collective). On
    any consolidation error rank 0 writes a `.consolidate_failed_<S>` sentinel
    and re-raises; the waiters bail out (raising) on the sentinel OR a bounded
    timeout. A failed child **errors and releases its GPUs**, it does not wedge.
  * **Partials persist on failure → re-runnable.** `consolidate_step` deletes
    `step_<S>/` only after a successful write, so a failed/preempted
    consolidation leaves the partials intact. Recover with the CLI:

        python -m param_decomp_lab.three_pool.consolidate_cli <run_id|out_dir> [--step N]

    With no `--step` it consolidates every `unconsolidated_steps(out_dir)` (a step
    with partials but no `training_<step>.pth`). The prune is concurrency-safe
    (`unlink(missing_ok=True)`) since multiple per-step children may prune the
    same old file at once.

## Launch path & the git snapshot (don't smoke save/eval via raw torchrun)

A real run goes through **`pd-lm-3pool --dp N`** (`experiments/lm/three_pool_run.py`), which
pushes a per-run **git snapshot ref** (`refs/runs/snapshot/<run_id>`) and submits the SLURM job
against a *clone* of that ref. The async consolidation + slow-eval job
(`submit_slurm_async_consolidate_and_eval`, fired by `on_save`) **clones that same snapshot ref**
— so it sees exactly the committed+snapshotted code, NOT your working tree.

Consequences when iterating:

  * **Uncommitted code must be committed before `pd-lm-3pool --dp`** or the snapshot won't
    contain it. (The snapshot is git-based; untracked/dirty files are excluded.)
  * **Training-only smokes can run against the live tree** via
    `srun ... torchrun --standalone --nproc_per_node=N -m
    param_decomp_lab.experiments.lm.three_pool_run <cfg>` — fast for iterating on uncommitted
    changes. But the forced final-step checkpoint still fires `on_save`, whose consolidation job
    will **fail to clone** `refs/runs/snapshot/<run_id>` (it was never pushed). That failure is
    expected for a raw-torchrun smoke and does **not** mean training or the model is broken — it
    only means save→consolidate→eval is untested on that path. To exercise save/resume/eval, use
    `pd-lm-3pool --dp` with committed code.

## Process-group timeout

`build_world` takes a `pg_timeout` and threads it into every `dist.new_group`
call. **This is load-bearing:** `new_group` does NOT inherit the timeout passed
to `init_process_group` — with `timeout=None` it silently uses the 10-min NCCL
library default. The 3-pool runs all of its real collectives on these subgroups
(never the default group), so the timeout must be set explicitly or a slow
collective trips the watchdog.

The default is **10 min** (`_DEFAULT_PG_TIMEOUT` in `optimize.py`), brought down
from 30 min once consolidation moved off the loop. The invariant is now: **the
PG timeout must exceed the worst-case on-loop collective gap**, which is the
in-train (fast) eval pass plus a checkpoint partial-write barrier — minutes, not
the old ~10-min rank-0 read. Override (seconds) via `PD_3POOL_PG_TIMEOUT_S` —
used by the watchdog-safe-at-low-timeout test to force a tight bound.

When chunkwise `torch.compile` is on (the default — see below), the timeout widens to
**20 min** (`_COMPILE_PG_TIMEOUT`), because step 0 pays a one-time ~minutes
compilation while the other pools wait at the first cross-pool collective. The
widening is uniform across ranks (the flag is global), and steady-state collectives
are still sub-second.

## Compile / checkpoint toggles (config fields, default on)

All four pool optimisations are authored on `PooledRuntimeConfig` (in `three_pool/config.py`):
`compile_chunkwise`, `compile_ci_fn`, `compile_ppgd`, `checkpoint_ci_fn` — all `bool`,
default `True`. `ThreePoolRuntimeConfig` / `TwoPoolRuntimeConfig` (lab) subclass it and add
the authored `topology`, so the toggles have a single home and the trainers read them off
`self.runtime_config` (no env vars, no back-dependency into `experiments/lm/`). Disable one
in a YAML with e.g. `runtime.compile_ppgd: false`. Any compile path widens the step-0 PG
timeout (`_resolve_pg_timeout(compiling=...)`). Single-pool (`pd-lm`) doesn't compile, so
these live only on the pooled config.

On the **2-pool**, the adversary's masked forward lives on Pool A (not a separate PPGD
pool), so `compile_ppgd` compiles `component_model.model` on the Pool A rank — alongside
the CI fn (`compile_ci_fn`), since Pool A holds both. The compiled artifact + fused
`autograd.grad` are the same ones the 3-pool PPGD pool validated; Pool A's model has
activation checkpointing off (autograd.grad recompute is nondeterministic under ckpt), so
it's a plain forward compile, no checkpointed-RNG-op partitioner concern. Isolated probe at
production width: **~1.2× on the Pool A step** (the matmuls are already tensor-core-saturated
at training token counts; the win is compile fusing the memory-bound cast/mask/norm tail).

**Gotcha — recompiles scale with decomposed-site count.** The masked forward specializes per
mask-key, so dynamo recompiles roughly once per decomposed site (plus a couple for the
no-grad-target vs grad-recon grad_mode split). The production Llama target (one MLP =
**3 sites**: gate/up/down) converges in ~3–5 recompiles, well under dynamo's default
`recompile_limit` (8), and stays compiled (verified: 0 steady-state recompiles, 1.21×).
A many-site config (e.g. GPT2 `h.*.attn.{q,k}_proj` = 24 sites) **blows the limit and
silently falls back to eager** — the run still completes, it just loses the compile win. If
a future PPGD config decomposes >~6 sites, raise `torch._dynamo.config.recompile_limit`
above the site count (cost: front-loaded one-time recompiles at startup).

## Chunkwise torch.compile (`compile_chunkwise`, default on)

The chunkwise pool's component model is `torch.compile`d **whole-model** (the model's
block-loop `checkpoint(block, …)` lives *inside* the compiled region) — ~**2.74×** on the
chunkwise step (the throughput pole), 0 graph breaks, validated clean at 160-GPU distributed
scale. Chunkwise-only (PPGD/CI have slack; PPGD's `autograd.grad` is unvalidated under compile).

**Requires torch >= 2.11.** With the checkpoint inside the compiled region the AOT min-cut
partitioner sees the checkpointed flash-SDPA as a must-recompute nondeterministic-seeded op.
On torch <= 2.10 its `functionalize_rng_ops` then `KeyError`'d on the DCE'd SDPA RNG op
(`partitioners.py` `has_recomputable_rng_ops` → `functionalize_rng_ops`) — but **only** in the
distributed run (not reproducible single-GPU; manifests at scale). torch 2.11 added the guard
that skips that op instead of erroring, so whole-model compile is clean. (The earlier
workaround — compile per-block with the checkpoint left eager — gave 2.42× and is gone.)
Per-rank inductor/triton cache dirs are kept as a defensive measure against shared-cache
contention across the 160 concurrent compilers.

## CI-fn checkpoint + compile (`checkpoint_ci_fn` / `compile_ci_fn`, both default on)

The CI pool activation-checkpoints the CI-fn transformer blocks
(`GlobalSharedTransformerCiFn.enable_activation_checkpointing`) and then `torch.compile`s the
**whole CI-fn forward** with the checkpoint loop inside the compiled region (`ci_fn.compile()`)
— same torch >= 2.11 pattern as the chunkwise pool. Checkpoint recomputes the 16384-wide MLP / attn intermediates in
backward, saving ~**15 GB** of block-activation high-water on the CI rank; whole-forward
compile turns the checkpoint's +12.9% step-time cost into a net **−9.2%** vs baseline (1-GPU
B200 probe; whole-region beats per-block's −4.6%). The CI pool is compute-idle (PPGD is the
long pole), so even the bare ckpt cost would be free on the critical path. Whole-forward
compile + ckpt + flash-SDPA is validated on real 2-GPU DDP/NCCL. Any compile path
(chunkwise, CI, or PPGD) widens the step-0 PG timeout.

## PPGD torch.compile (`compile_ppgd`, default on)

The PPGD pool `torch.compile`s the **same `component_model.model` masked forward** the chunkwise
pool compiles — the warmup PGD inner loop and the final recon forward both run it. Because it's
the identical compiled artifact, the forward-at-scale risk is already retired by the chunkwise
pool's 160-GPU validation; the only PPGD-specific path is the **fused `torch.autograd.grad`**
(not `.backward()`) over V/U + CI + sources, plus the `n_warmup` source-only backwards.

That fused-`autograd.grad`-under-compile combo was the reason PPGD was previously left
uncompiled. It's now 1-GPU-validated (vendored-Llama proxy, ckpt + flash-SDPA): the **isolated**
single fused backward is numerically correct (fp32 grad rel-err **8e-7**; bf16 ~1.4%, benign
autocast reordering), and the **warmup loop runs recompile/graph-break-free**. ~2–3× on PPGD
*compute* (proxy). Note the end-to-end PPGD step grad is intrinsically ~**27% run-to-run
nondeterministic** (CUDA `atomicAdd` in the flash-attn/`kl_div` backward, amplified by the chaotic
adversarial loop), so it can't be tight-tolerance grad-checked end-to-end — and compile's delta
sits far below that floor. **Not yet validated at real 8B scale** (the chunkwise compile retires
most of that risk, but the autograd.grad path at scale is unexercised); the next real run is the
de-risk. Probes (frozen on the `lore-artifacts` branch):
[ppgd_compile_probe.py](https://github.com/goodfire-ai/param-decomp/blob/e7ccf47bfd218cb68eb9d4e9ebca964ccccf2630/lore_artifacts/ppgd_compile_probe.py)
(isolated) and
[ppgd_full_compile_probe.py](https://github.com/goodfire-ai/param-decomp/blob/e7ccf47bfd218cb68eb9d4e9ebca964ccccf2630/lore_artifacts/ppgd_full_compile_probe.py)
(full path).

## CI value wire dtype split (`portals.py`)

The cross-pool wire dtype is split by payload. **CI value masks** (`lower_leaky` /
`upper_leaky`) ship as **fp16** (`CI_VALUE_WIRE_DTYPE`) — they're bounded in ≈[0, 1]
(leaky-hard sigmoid), so fp16's 10 mantissa bits give ~8× finer resolution near 1.0 than
bf16's 7 at identical 2 bytes. **Gradients** (CI grads, V/U grads), **V/U weights**, and the
unbounded **`pre_sigmoid`** logit keep **bf16** (`CI_GRAD_WIRE_DTYPE` / `WIRE_DTYPE`) for the
exponent range. Received fp16 values upcast to fp32 on the consume side
(`_releaf_ci_fp32_for_grads`), so downstream math is unchanged — grad check worst rel err
4.04e-7. The eval-only `CiOutputsEvalToPPGD` packet bundles `pre_sigmoid` with the masks, so
that whole packet stays bf16 rather than splitting the buffer by dtype (off the critical path).

## Resume (`from_snapshot`)

3-pool runs persist a `ThreePoolTrainingState` (not the single-pool
`TrainingState`), written by the **async consolidation job**, not the train loop.
`resolve_step` / `read_training_snapshot` pick the latest consolidated
`training_<step>.pth`. If a run dies after a save but before that step's
consolidation finishes, resume from the previous consolidated step — at most one
save-interval of lost progress (the scratch partials for the unconsolidated step
are left on disk; they can be consolidated manually via `consolidate_step` if
that interval matters).

`from_snapshot` validates the saved topology against the current one, but the
comparison runs on EVERY rank, so it compares only the **rank-invariant** core
(`world_size` / `ci_ranks` / `ppgd_ranks` / `n_chunks`) via
`_rank_invariant_fingerprint_core` — never a rank-local view. The per-chunk
ranks→sites mapping is re-derived from the snapshot's `three_pool_config`.

### SLURM-requeue self-resume

`_submit_slurm` (2-pool and 3-pool) sets `SlurmConfig.requeue=True`, emitting
`#SBATCH --requeue` so SLURM re-runs the *same* `... <config> --run_id <id>` command
on node failure / opportunistic preemption. That command has no `--resume`, so the
worker entry (`_fresh_or_requeue_main`) checks `latest_checkpoint_step(out_dir)` for
the run's own dir: a consolidated `training_<step>.pth` ⇒ resume in place (same
`run_id`, same `out_dir`, `resume_wandb=True` → continuous wandb curves) via
`_resume_in_place` → `_run_resume`; no checkpoint ⇒ fresh. The cross-run
`--resume <resume.yaml>` path (different `run_id`, new wandb run) is unchanged and
shares `_run_resume` with `resume_wandb=False`. `resume` is threaded
`init_pd_run` → `with_wandb` → `init_wandb` (`wandb.init(resume="allow")`).
Validated end-to-end at 4 GPU by `.scratch_smoke_logs/requeue_resume_test.sbatch`
(re-running the identical command resumes at the saved step, not step 0).

Repro/fault-injection env knobs (never set in production):
`PD_3POOL_PG_TIMEOUT_S`, `PD_3POOL_SNAPSHOT_RANK0_SLEEP_S` (sleeps rank 0 inside
`snapshot()` AFTER the partial write — proves the sleep no longer stalls the
loop now that the read is async), `PD_3POOL_DISABLE_REJOIN_BARRIER` (drops the
post-write rejoin barrier). Regression tests:
`param_decomp_lab/tests/test_three_pool_pg_timeout.py`.

## Chunkwise recon plan (`recon_plan.py`)

The chunkwise pool's reconstruction unit is parameterised by `losses.recon_plan`
(`ThreePoolLosses`, default `PerSitePlan`). Each chunk turns its `sites`
into a **list of recon forwards** via `plan.generate(sites, mask_shape,
device)`; `step_chunkwise` runs one masked forward+backward per entry.

- `PerSitePlan` — one forward per site, that site routed everywhere. The
  original "swap one matrix at a time" loop; bit-exact with the pre-routing path.
- `SubsetReconPlan(routing, n_samples)` — `n_samples` forwards, each over *all*
  the chunk's sites with a freshly-drawn per-position routing. `routing=all` →
  joint "swap everything at once" (`n_samples=1` → one forward instead of N → ~N×
  less chunkwise compute); `routing=uniform_k_subset` / `static_probability` →
  per-position subset recon (à la core `StochasticReconSubsetLoss`). Reuses the
  core `masks.py` routers via `get_subset_router`.

The cross-pool DAG is unchanged — it's all keyed on the chunk's `sites`, and every
site still gets exactly one re-leafed CI tensor whose `.grad` accumulates across the
forward list and ships back once.

**Per-forward fresh deltas (streaming-backward invariant).** `recon_one_forward` takes a
`WeightDeltasFn` (`make_weight_deltas_fn(component_model)`), not a precomputed delta dict,
and calls it once per forward so each forward owns an independent `target − VU` autograd
subgraph. This is load-bearing for any multi-forward plan (`SubsetReconPlan(n_samples>1)`,
or any plan that reconstructs a site in more than one forward): the streaming loop
backwards per forward and frees that forward's graph, so a *shared* delta tensor would be
backward'd through twice → "backward through the graph a second time". The fn keeps
placement caller-owned (plain for the 3-pool, DTensor-aware `calc_weight_deltas` for the
flat FSDP path) and is called while V/U is in its native sharded state (FSDP reshards
after each forward's backward). The flat single-pool twin (`ChunkwiseSubsetReconLoss`,
`param_decomp_lab/metrics/chunkwise_subset_recon.py`) shares the SAME body/fn but
accumulates forward-only into one loss for the trainer's single `total_loss.backward()` —
bit-identical V/U + CI grads to the streaming per-forward backward (proven RNG-pinned in
`tests/test_chunkwise_subset_recon_metric.py`). The other flat loss terms (faith / imp /
PPGD) are proven equal to the 2-pool's pool-step helpers in
`tests/test_flat_vs_two_pool_loss_terms.py`.

**Gradient scaling.** The only scaling knob is `N_est` = the global total of chunkwise
recon forwards per step (`runtime.n_est = Σ over chunks of
plan.n_forwards(sites)`), computed once at `_build_runtime`. It replaces the
old `n_sites_total` in the stoch denominator
(`step_chunkwise._run_routing_forwards`):

    stoch_grad_denom = n_positions * N_est * chunk_dp / n_ci

For the default per-site plan `N_est == n_sites_total`, so this is bit-exact with
the old path. The grad check `tests/test_three_pool_recon_plan.py` proves (real
`.grad`, one backward, RNG pinned — per
[[feedback_grad_scaling_needs_real_grad_check]]) that at `n_ci=chunk_dp=1` the
denom collapses to the textbook single-pool normalisation `sum_loss / n_examples`
(`n_examples = n_forwards * n_positions`) for every plan.

## Cross-pool batch divisibility (bidirectional)

Each cross-pool edge (CI↔chunk, CI↔PPGD) requires the two batch arities to be
**cross-divisible** — one divides the other, EITHER direction (`n_ci | n_down`
OR `n_down | n_ci`). Ragged pairs where neither divides the other are rejected.
The three "per-rank batch divides `pd.batch_size`" constraints still hold.

`BatchEdge` (layout.py) owns the geometry for one edge and answers every routing
question symmetrically:

  * **CI coarse** (`n_ci <= n_down`): one CI rank fans a sub-slice to `fanout`
    downstream ranks; grads stitch back fanout→one. One downstream rank ↔ one CI
    rank.
  * **CI fine / inverted** (`n_ci > n_down`): one downstream rank gathers CI
    from `fanout` CI ranks (`PendingCiValues` holds the `fanout` packets and
    stitches them) and scatters grads back to those CI ranks. One CI rank ↔ one
    downstream rank.

The six portal exchange methods + the eval CI ship consume `world.ci_chunk_edge` /
`world.ci_ppgd_edge` and never branch on the regime. Unit tests:
`param_decomp_lab/tests/test_three_pool_batch_edge.py`.

## Topology config schema (`config.py`)

The topology is authored as per-rank batches + a site→chunk split, NOT rank ids.
The resolver derives every rank in canonical order (chunks first → rank 0 is the
chunk-0 leader by construction, then CI, then PPGD), so overlap / dup / gaps /
sum≠world / non-uniform DP / the rank-0 convention are all unrepresentable.

```yaml
runtime:
  topology:
    ci:        { per_rank_batch: 16 }   # n_ci   = batch / 16
    ppgd:      { per_rank_batch: 16 }   # n_ppgd = batch / 16
    chunkwise:
      per_rank_batch: 16                # chunk_dp = batch / 16 (DDP per chunk)
      sites_per_chunk: null             # null = all decomposed sites in one chunk
    use_fused_kl: true
```

`runtime.dp` is dropped from authored 3-pool configs: the world size is derived
from the resolved topology and asserted == the torchrun world in `build_world`.
`ThreePoolTopology.resolve(ordered_sites, batch_size)` returns a frozen
`ResolvedLayout` (`ci_ranks`, `ppgd_ranks`, per-chunk `(ranks, sites)`, `world_size`)
that `optimize._build_runtime` turns into runtime `Chunk` objects.
