# `param_decomp_lab/three_pool/`

The 3-pool training subsystem — sibling of `param_decomp.optimize.Trainer` for
splitting a decomposition run across three rank pools (CI fn, layerwise V/U,
PPGD adversary). See `DESIGN.md` for the per-step comm graph and the module
docstring in `optimize.py` for the data-handling contract.

| File | What it covers |
|---|---|
| `optimize.py` | `ThreePoolTrainer` + `optimize_three_pool`; the training loop, `snapshot`/`from_snapshot`, config validation |
| `layout.py` | `World` topology; `build_world` constructs every process group (threading `pg_timeout` into each) |
| `checkpoint.py` | `gather_full_state_dict_to_rank0` — rebuilds the full model state on rank 0 |
| `config.py` | `ThreePoolConfig` + topology validation |
| `role.py` | `PoolRole = CIRole \| LWRole \| PPGDRole` — this rank's pool role; per-pool fields are union variants, not optional attrs |
| `context.py` | `PoolContext = CIContext \| LWContext \| PPGDContext` — `world` + `role` + this pool's portals; the trainer matches on it to dispatch step fns |
| `portals.py` | Cross-pool exchanges as typed objects — one class per DAG edge (pack layout + routing + dtype + PG in one place) |
| `step_{ci,layerwise,ppgd}.py` | per-pool step functions |
| `eval_step.py` | 3-pool eval pass (PPGD pool runs metrics; others barrier through) |

## Checkpoint-save invariant (do not break)

`snapshot()` is collective. The ordering is: all ranks gather the model to
rank 0, all ranks write their per-rank partial to the shared-FS scratch dir,
barrier, then **rank 0 alone** serially reads every partial (~100 GB at XL) to
assemble the canonical `ThreePoolTrainingState`. The other ranks have nothing
to do during that read.

There is a **mandatory rejoining `dist.barrier(group=cross_pool_p2p_group)`
after rank 0 finishes reading.** Without it, the non-rank-0 ranks race ahead
into the next training step while rank 0 is still reading; the next step's
collectives (PPGD V/U all-reduce, cross-pool sends) block on rank 0 and trip
the NCCL collective-timeout watchdog, aborting the whole job. This is exactly
how run 34446 (p-a5b667e9) died at its first checkpoint. The barrier makes all
ranks resume in lock-step.

## Process-group timeout (do not lower)

`build_world` takes a `pg_timeout` and threads it into every `dist.new_group`
call. **This is load-bearing:** `new_group` does NOT inherit the timeout passed
to `init_process_group` — with `timeout=None` it silently uses the 10-min NCCL
library default. The 3-pool runs all of its real collectives on these subgroups
(never the default group), so the timeout must be set explicitly or a slow
checkpoint save / eval pass trips the 10-min watchdog.

The default is 30 min (`_DEFAULT_PG_TIMEOUT` in `optimize.py`). The invariant is
simply: **the PG timeout must exceed the worst-case rank-0 checkpoint read
time.** Override (seconds) via `PD_3POOL_PG_TIMEOUT_S` — used by the
save-watchdog repro (`scripts/repro_3pool_save_watchdog.sbatch`) to force the
bug at small scale.

## Resume (`from_snapshot`)

3-pool runs persist a `ThreePoolTrainingState` (not the single-pool
`TrainingState`); `read_training_snapshot` returns either and the caller
narrows. `from_snapshot` validates the saved topology against the current one,
but the comparison runs on EVERY rank, so it compares only the
**rank-invariant** core (`world_size` / `ci_ranks` / `ppgd_ranks` /
block count) via `_rank_invariant_fingerprint_core` — never a rank-local view.
That helper also tolerates the pre-fix rank-local fingerprint format baked into
existing production checkpoints (p-a5b667e9). The per-block ranks→sites mapping
is re-derived from the snapshot's `three_pool_config`.

Repro/fault-injection env knobs (never set in production):
`PD_3POOL_PG_TIMEOUT_S`, `PD_3POOL_SNAPSHOT_RANK0_SLEEP_S` (inject a sleep into
rank-0's read), `PD_3POOL_DISABLE_REJOIN_BARRIER` (reproduces the pre-fix race).
Regression tests: `param_decomp_lab/tests/test_three_pool_pg_timeout.py`.
