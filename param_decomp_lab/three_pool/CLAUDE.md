# `param_decomp_lab/three_pool/`

The 3-pool training subsystem — sibling of `param_decomp.optimize.Trainer` for
splitting a decomposition run across three rank pools (CI fn, layerwise V/U,
PPGD adversary). See `DESIGN.md` for the per-step comm graph and the module
docstring in `optimize.py` for the data-handling contract.

| File | What it covers |
|---|---|
| `optimize.py` | `ThreePoolTrainer` + `optimize_three_pool`; the training loop, `snapshot`/`from_snapshot`, config validation |
| `layout.py` | `World` topology; `build_world` constructs every process group (threading `pg_timeout` into each); `BatchEdge` — symmetric per-edge batch-slice geometry (CI↔LW, CI↔PPGD) answering routing for both fan directions |
| `checkpoint.py` | offline state_dict assembly from on-disk partials (`assemble_model_state_dict_from_partials`) + the leader key-partition helpers (`owned_model_state_keys` / `ci_fn_state_keys`) |
| `consolidate.py` | `consolidate_step` — async, off-train-loop assembly of `model_<step>.pth` + `training_<step>.pth` from a step's scratch partials; prunes old `training_*.pth`; deletes the scratch dir |
| `config.py` | `ThreePoolConfig` + topology validation |
| `role.py` | `PoolRole = CIRole \| LWRole \| PPGDRole` — this rank's pool role; per-pool fields are union variants, not optional attrs |
| `context.py` | `PoolContext = CIContext \| LWContext \| PPGDContext` — `world` + `role` + this pool's portals; the trainer matches on it to dispatch step fns |
| `portals.py` | Cross-pool exchanges as typed objects — one class per DAG edge (pack layout + routing + dtype + PG in one place) |
| `step_{ci,layerwise,ppgd}.py` | per-pool step functions |
| `eval_step.py` | 3-pool eval pass (PPGD pool runs metrics; others barrier through) |

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
`model_*.pth` are kept**) → deletes `step_<S>/`. It runs as the first phase of
the existing async slow-eval job (`experiments/lm/async_eval.py`, rank 0 only,
then a barrier) so the assembled `model_<S>.pth` exists before the eval loads it.
Idempotent: a no-op if `training_<S>.pth` already exists or the scratch dir is
gone.

This is the fix for how run 34446 (p-a5b667e9) died: the old synchronous rank-0
read held the other ranks at a barrier past the NCCL watchdog. There is no
on-loop read to outrun the watchdog now.

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
