# `param_decomp_lab/three_pool/`

The 3-pool training subsystem — sibling of `param_decomp.optimize.Trainer` for
splitting a decomposition run across three rank pools (CI fn, layerwise V/U,
PPGD adversary). See `DESIGN.md` for the per-step comm graph and the module
docstring in `optimize.py` for the data-handling contract.

| File | What it covers |
|---|---|
| `optimize.py` | `ThreePoolTrainer` + `optimize_three_pool`; the training loop, `snapshot`/`from_snapshot`. Consumes `ThreePoolConstrainedPDConfig` (reads `pd.losses.*` directly). Config constraints are type-level (see `pd_config.py`) + a load-time validator on `ThreePoolLMExperimentConfig`; only site-coverage validation remains here (`_build_runtime`, needs the loaded model) |
| `pd_config.py` | `ThreePoolConstrainedPDConfig` + the typed `ThreePoolLosses(faith, imp, stoch, ppgd, routing_plan)` struct — the 3-pool-constrained `PDConfig`. Lives with the subsystem (not `experiments/lm/`) so the subsystem is self-contained: `optimize` imports it from its own package, no back-dependency into `experiments/lm/` |
| `layout.py` | `World` topology; `build_world` constructs every process group (threading `pg_timeout` into each); `BatchEdge` — symmetric per-edge batch-slice geometry (CI↔LW, CI↔PPGD) answering routing for both fan directions |
| `checkpoint.py` | offline state_dict assembly from on-disk partials (`assemble_model_state_dict_from_partials`) + the leader key-partition helpers (`owned_model_state_keys` / `ci_fn_state_keys`) |
| `consolidate.py` | `consolidate_step` — async, off-train-loop assembly of `model_<step>.pth` + `training_<step>.pth` from a step's scratch partials; prunes old `training_*.pth`; deletes the scratch dir. `unconsolidated_steps` lists recoverable steps |
| `consolidate_cli.py` | `python -m …consolidate_cli <run> [--step N]` — manual CPU-only recovery for a failed/preempted async consolidation (separate module to avoid an import cycle with `experiments.lm.run`) |
| `config.py` | `ThreePoolConfig` + topology validation |
| `role.py` | `PoolRole = CIRole \| LWRole \| PPGDRole` — this rank's pool role; per-pool fields are union variants, not optional attrs |
| `context.py` | `PoolContext = CIContext \| LWContext \| PPGDContext` — `world` + `role` + this pool's portals; the trainer matches on it to dispatch step fns |
| `portals.py` | Cross-pool exchanges as typed objects — one class per DAG edge (pack layout + routing + dtype + PG in one place) |
| `step_{ci,layerwise,ppgd}.py` | per-pool step functions |
| `routing_plan.py` | `RoutingPlan` (`PerSitePlan` \| `SubsetRoutingPlan`) — how each LW block turns its owned sites into a list of recon forwards |
| `eval_step.py` | 3-pool eval pass (PPGD pool runs metrics; others barrier through) |
| `reductions.py` | cross-pool log reductions: losses (`aggregate_losses_to_rank0`), peak memory (`aggregate_max_memory_to_rank0`), grad norms (`aggregate_grad_norms_to_rank0` + `per_param_grad_norms`) |
| `SUM_GRAD_CONVENTION.md` | the gradient-assembly scaling convention (proposal) |

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
  within the LW pool and ships the CI leader's ci-fn norms to rank 0 (object collectives,
  log-steps only). PPGD owns no trained params. Summaries are derived on rank 0.
- **Per-loss grad norms** (3-pool-only diagnostic for coeff rebalancing):
  `train/grad_norms/by_loss/{FaithfulnessLoss,StochasticReconLayerwiseLoss,PersistentPGDReconLoss}/components`
  — each loss term's contribution to the global V/U grad. faith + ppgd are contribute-once
  (block-leader-only), stoch is each rank's partial; one block SUM-all-reduce of
  `[faith, ppgd, total]` recovers each term's global grad (`stoch = total - faith - ppgd`),
  leader takes sum-sq (`_component_grad_sumsq_by_loss` in `step_layerwise.py`), and
  `aggregate_component_grad_by_loss_to_rank0` SUMs across blocks. CI-fn split (imp vs recon)
  not done — the fused CI backward would have to be unfused. Log-steps only.
- **LR schedules**: both `train/schedules/lr/components` and `train/schedules/lr/ci_fn`
  (rank 0 computes the ci-fn LR from the schedule directly — no cross-pool comm).
- **3-pool extras** (no single-pool equivalent): `train/perf/step_ms` (MAX over LW),
  `train/mem/{lw,ci,ppgd}_peak_gb`, `train/metrics/mean_l0` (batch-mean active CI components
  per token across all sites, threshold 0 — the live sparsity readout; CI pool computes
  it from `lower_leaky` and ships it via the loss-reduction path, see `reductions.py`).
- **Eval**: in-train fast eval logs `eval/<k>`; slow eval is async (`experiments/lm/async_eval.py`)
  under `slow_eval/<k>` on a `slow_eval/step` axis. The single-pool path now also runs its
  slow metrics in-train under the same `slow_eval/` namespace (`_build_eval_loop(..., include_slow=True)`
  for 1-pool, `False` for 3-pool); the `slow_eval/step` axis is defined in `init_wandb`.

## Gradient-assembly scaling: the SUM convention

See `SUM_GRAD_CONVENTION.md` for the full derivation. Summary: every
data-parallel gradient reduction is **SUM** (`all_reduce_ci_fn_grads`,
`all_reduce_grads_in_block`, and PPGD's V/U reduce). Each producer emits a
*partial sum* normalized only by the honest GLOBAL count — NO `n_ci` /
`n_per_block` transport factor. `SUM(partials) = total`, so no producer needs a
pool's size. The REPLICATED contributions are handled structurally rather than by
a replica-count divide: faith + broadcast-PPGD V/U **contribute once** (emitted
on the block leader only), and imp-min uses the **detached-global-residual** trick
(`S = local + (all_reduce_sum(local.detach()) - local.detach())`) so its backward
is a local partial. The grad-clip `n_replicas` is unchanged — it counts distinct
params for the global norm, independent of the grad-reduce op. Validated by
`tests/test_three_pool_grad_check_distributed.py` (non-square, all loss terms).

## Checkpoint save: partials on the loop, consolidation off it

The save path is split so the train loop never blocks on a multi-GB read.

`snapshot()` (on the train loop, all ranks collective): each rank writes a
**self-contained partial** to `scratch_dir/step_<S>/rank_<r>.pth` — its owned
model params (LW block leaders → owned-sites V/U; CI pool leader → CI fn),
its optimizer state (name-keyed), and (PPGD) its sources. Rank 0 also writes
`meta.pth` (configs + fingerprint + `c_per_site` / `all_sites`). There is **one
pre-write barrier** (so rank 0's `mkdir` + `meta` write land before others write
into the dir) and **one post-write rejoin barrier** (so all ranks leave
`snapshot()` together). Both barriers are cheap — no rank does a 100 GB read on
the loop anymore — so neither can overrun the watchdog. NO model NCCL gather, NO
rank-0 assembly here.

`consolidate_step()` (`consolidate.py`, async SLURM job, off the loop): reads all
of a step's partials → assembles the full `ComponentModel` state_dict +
`ThreePoolTrainingState` → writes `model_<S>.pth` + `training_<S>.pth` → prunes
old `training_*.pth` to the last `DEFAULT_KEEP_LAST_N_TRAINING` (=3; **all
`model_*.pth` are kept**) → deletes `step_<S>/`. It runs inside the async
slow-eval job (`experiments/lm/async_eval.py`), as a CPU-only phase BEFORE the
eval pass, so the assembled `model_<S>.pth` exists before the eval loads it.
Idempotent: a no-op if `training_<S>.pth` already exists (and it cleans any
leftover scratch in that case); the scratch dir is deleted only on success.

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

When LW `torch.compile` is on (the default — see below), the timeout widens to
**20 min** (`_COMPILE_PG_TIMEOUT`), because step 0 pays a one-time ~minutes
compilation while the other pools wait at the first cross-pool collective. The
widening is uniform across ranks (the flag is global), and steady-state collectives
are still sub-second.

## LW torch.compile (default on; `PD_DISABLE_LW_COMPILE=1` to disable)

The LW pool's component model is `torch.compile`d **whole-model** (the block-loop's
`checkpoint(block, …)` lives *inside* the compiled region) — ~**2.74×** on the LW step (the
throughput pole), 0 graph breaks, validated clean at 160-GPU distributed scale. LW-only
(PPGD/CI have slack; PPGD's `autograd.grad` is unvalidated under compile).

**Requires torch >= 2.11.** With the checkpoint inside the compiled region the AOT min-cut
partitioner sees the checkpointed flash-SDPA as a must-recompute nondeterministic-seeded op.
On torch <= 2.10 its `functionalize_rng_ops` then `KeyError`'d on the DCE'd SDPA RNG op
(`partitioners.py` `has_recomputable_rng_ops` → `functionalize_rng_ops`) — but **only** in the
distributed run (not reproducible single-GPU; manifests at scale). torch 2.11 added the guard
that skips that op instead of erroring, so whole-model compile is clean. (The earlier
workaround — compile per-block with the checkpoint left eager — gave 2.42× and is gone.)
Per-rank inductor/triton cache dirs are kept as a defensive measure against shared-cache
contention across the 160 concurrent compilers.

## CI-fn checkpoint + compile (both default on)

The CI pool activation-checkpoints the CI-fn transformer blocks
(`GlobalSharedTransformerCiFn.enable_activation_checkpointing`, `PD_DISABLE_CI_CKPT=1` to
disable) and then `torch.compile`s the **whole CI-fn forward** with the checkpoint loop
inside the compiled region (`ci_fn.compile()`, `PD_DISABLE_CI_COMPILE=1` to disable) — same
torch >= 2.11 pattern as LW. Checkpoint recomputes the 16384-wide MLP / attn intermediates in
backward, saving ~**15 GB** of block-activation high-water on the CI rank; whole-forward
compile turns the checkpoint's +12.9% step-time cost into a net **−9.2%** vs baseline (1-GPU
B200 probe; whole-region beats per-block's −4.6%). The CI pool is compute-idle (PPGD is the
long pole), so even the bare ckpt cost would be free on the critical path. Whole-forward
compile + ckpt + flash-SDPA is validated on real 2-GPU DDP/NCCL. Either compile path (LW or
CI) widens the step-0 PG timeout.

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
(`world_size` / `ci_ranks` / `ppgd_ranks` / block count) via
`_rank_invariant_fingerprint_core` — never a rank-local view. That helper also
tolerates the pre-fix rank-local fingerprint format baked into existing
production checkpoints (p-a5b667e9). The per-block ranks→sites mapping is
re-derived from the snapshot's `three_pool_config`.

Repro/fault-injection env knobs (never set in production):
`PD_3POOL_PG_TIMEOUT_S`, `PD_3POOL_SNAPSHOT_RANK0_SLEEP_S` (sleeps rank 0 inside
`snapshot()` AFTER the partial write — proves the sleep no longer stalls the
loop now that the read is async), `PD_3POOL_DISABLE_REJOIN_BARRIER` (drops the
post-write rejoin barrier). Regression tests:
`param_decomp_lab/tests/test_three_pool_pg_timeout.py`.

## LW recon routing plan (`routing_plan.py`)

The LW pool's reconstruction unit is parameterised by `losses.routing_plan`
(`ThreePoolLosses`, default `PerSitePlan`). Each block turns its `owned_sites`
into a **list of recon forwards** via `plan.generate(owned_sites, mask_shape,
device)`; `step_layerwise` runs one masked forward+backward per entry.

- `PerSitePlan` — one forward per owned site, that site routed everywhere. The
  original "swap one matrix at a time" loop; bit-exact with the pre-routing path.
- `SubsetRoutingPlan(routing, n_samples)` — `n_samples` forwards, each over *all*
  owned sites with a freshly-drawn per-position routing. `routing=all` →
  joint "swap everything at once" (`n_samples=1` → one forward instead of N → ~N×
  less LW compute); `routing=uniform_k_subset` / `static_probability` →
  per-position subset recon (à la core `StochasticReconSubsetLoss`). Reuses the
  core `masks.py` routers via `get_subset_router`.

The cross-pool DAG is unchanged — it's all keyed on `owned_sites`, and every owned
site still gets exactly one re-leafed CI tensor whose `.grad` accumulates across the
forward list and ships back once.

**Gradient scaling.** The only scaling knob is `N_est` = the global total of LW
recon forwards per step (`runtime.n_est = Σ over blocks of
plan.n_forwards(owned_sites)`), computed once at `_build_runtime`. It replaces the
old `n_sites_total` in the stoch denominator
(`step_layerwise._run_routing_forwards`):

    stoch_grad_denom = n_positions * N_est * n_per_block / n_ci

For the default per-site plan `N_est == n_sites_total`, so this is bit-exact with
the old path. The grad check `tests/test_three_pool_routing_plan.py` proves (real
`.grad`, one backward, RNG pinned — per
[[feedback_grad_scaling_needs_real_grad_check]]) that at `n_ci=n_per_block=1` the
denom collapses to the textbook single-pool normalisation `sum_loss / n_examples`
(`n_examples = n_forwards * n_positions`) for every plan.

## Cross-pool batch divisibility (bidirectional)

Each cross-pool edge (CI↔LW, CI↔PPGD) requires the two batch arities to be
**cross-divisible** — one divides the other, EITHER direction (`n_ci | n_down`
OR `n_down | n_ci`). Ragged pairs where neither divides the other are rejected.
The three "arity divides `pd.batch_size`" constraints still hold.

`BatchEdge` (layout.py) owns the geometry for one edge and answers every routing
question symmetrically:

  * **CI coarse** (`n_ci <= n_down`): one CI rank fans a sub-slice to `fanout`
    downstream ranks; grads stitch back fanout→one. One downstream rank ↔ one CI
    rank.
  * **CI fine / inverted** (`n_ci > n_down`): one downstream rank gathers CI
    from `fanout` CI ranks (`PendingCiValues` holds the `fanout` packets and
    stitches them) and scatters grads back to those CI ranks. One CI rank ↔ one
    downstream rank.

The six portal exchange methods + the eval CI ship consume `world.ci_lw_edge` /
`world.ci_ppgd_edge` and never branch on the regime. Unit tests:
`param_decomp_lab/tests/test_three_pool_batch_edge.py`.
