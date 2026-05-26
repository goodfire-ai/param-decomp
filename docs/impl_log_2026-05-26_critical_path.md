# Critical-path measurement on GPT-2 XL Q/K 3-pool — impl log (2026-05-26)

Picks up where `handoff_2026-05-26_3pool_perf.md` left off. Two cherry-pick
regressions repaired, instrumentation wired up properly, and the irreducible
serial bound measured directly from in-production GPU stream time.

## Cherry-pick regressions found and fixed

The 3-pool perf-investigation commits were cherry-picked from
`origin/archive/3pool-shard-resumption` onto `feature/multipool`. The restore
commit `34d641c5` brought back four scaffolding pieces; these were missed:

1. **`param_decomp/ci_fns.py` — `GlobalCiFnWrapper.forward`** reverted to
   `self.components[layer_name]` (was `.get(layer_name)` in baseline). CI pool
   calls `drop_components()` to free V/U memory; the lookup KeyError'd on every
   CI rank. Fixed by restoring `.get()` plus the explanatory comment.
2. **`param_decomp/three_pool/layout.py` — `flush_nccl_event_timings()`**
   called `torch.cuda.synchronize()` (full device sync) at the end of every
   step. That drains the NCCL stream holding PPGD's pending cross-step async
   broadcast → deadlock. `optimize.py:run`'s comment at the end-of-step sync
   already explained why a full device sync is unsafe there; the flush ignored
   it. Fixed by per-event `post.synchronize()` (only waits on the specific
   event, doesn't drain other streams).

A subagent audit found no other runtime regressions (only a missing
`_maybe_enable_memory_profile` feature in `lm/run.py` and the corresponding
offline analyzer `scripts/analyze_mem_profile.py` — both pre-date the
perf-investigation set; restore separately if needed).

## PhaseProfiler wiring

`dac30cd5`'s profiling-toolchain commit refactored `PhaseProfiler.phase()` from
emitting exit traces inline to buffering CUDA events and requiring an explicit
`flush_pending_gpu_events()` call. But `param_decomp_lab/experiments/lm/run.py`
never instantiates a `PhaseProfiler`; the per-step-function fallback
(`PhaseProfiler(enabled=False)` inside each `step_*`) only creates a local
dummy whose buffer never flushes. Result: entry lines (`phase: X cur=...gb`)
fire but exit lines (`phase: X end ... cpu=... gpu=... wait=...`) never do —
on either branch.

Fixed in `param_decomp/three_pool/optimize.py:ThreePoolTrainer.run`: when no
profiler is passed in, default to a `PhaseProfiler(enabled=False)` so it
threads through to `step_*` functions and gets flushed at end-of-step. With
`enabled=False`, the torch.profiler integration stays off (CUPTI is broken at
this scale per the doc) but the cpu/gpu/wait emission works.

## Critical-path analyzer

`scripts/critical_path.py` was weighting nodes by CPU wall, which double-counts
cross-stream wait time. Switched to GPU stream time when the new exit-line
`gpu=` field is present (falls back to CPU wall when absent). Added an
`irreducible_gpu` total to the path summary. Parser changes live in
`scripts/gantt_step.py` (shared parser).

## NCCL event timing vs. phase exit timing

Both are GPU stream times measured around the relevant region. They agree on
synchronous comms but diverge on async-pipelined comms:

| Phase exit | Phase gpu | Matching NCCL event | NCCL gpu |
|---|---|---|---|
| `ci/5_recv_g_ci_from_lw` | 532.1 ms | `recv_g_ci_from_layerwise:wait` | 532.0 ms ✓ |
| `ci/6_recv_g_ci_from_ppgd` | 146.9 ms | `recv_g_ci_from_ppgd:wait` | 146.9 ms ✓ |
| `ci/9_in_pool_allreduce` | 30.9 ms | `all_reduce_ci_fn_grads` | 30.8 ms ✓ |
| `lw/D5_recv_g_vu_from_ppgd` | 142.6 ms | `recv_g_vu_from_ppgd:recv` | 142.2 ms ✓ |
| `lw/D2_wait_ci_recv` | **162 ms** | `async_recv_ci_from_ci_pool` | **0.2 ms** ✗ |

The mismatch is informative: the NCCL event for `async_recv_ci_from_ci_pool`
times the irecv *post* (the kickoff, which is ~free). The actual GPU stream
wait is realised much later, inside the phase that blocks on completion
(`lw/D2_wait_ci_recv`). For async-pipelined cross-pool comms, the phase wrap
is the truth; the NCCL event line undercounts.

**Implication**: per-phase cpu/gpu/wait emission is the primary diagnostic for
this codebase. NCCL event timing is a corroborating layer (great for synchronous
ops; insufficient on its own for async patterns). Standalone repros are not
needed to measure distributed overhead — the phase data is direct. They remain
useful for one separate question ("can per-pool compute itself be made cheaper
on its own").

## Per-pool GPU breakdown (step 5, steady-state)

Per-phase GPU stream time, summed by category. Each pool runs in parallel; the
slowest's step time is the system's step time.

| Pool | Step | Real compute | Cross-stream waits | "Real compute" share |
|---|---|---|---|---|
| LW | 951 ms | **640 ms** (D3_layerwise: 580) | 235 ms (D2 wait CI: 162, D5 wait PPGD: 73) | 67% |
| PPGD | 890 ms | **750 ms** (D3_warmup: 383, D4 recon: 92, D5 bwd: 100, D6 reduce: ~110) | ~100 ms (D2 wait CI) | 84% |
| CI | 951 ms | **274 ms** (ci/1 fwd: 38, ci/4: 58, ci/8a bwd: 55, ci/8b: 67, ci/9: 31, ci/10: 17) | **679 ms** (ci/5 wait LW: 532, ci/6 wait PPGD: 147) | 29% |

Confirms the doc's claim that CI's "ci/8a wall ~600 ms" was misattributed wait
— actual GPU work in the CI bwd is 55 ms. The `bf16` autocast win is real.

## Irreducible serial

The slowest pool's real-compute sum is the lower bound on step time without
changing the algorithm:

- **Floor: ~750 ms** (PPGD's compute) — that's how fast a step could be if all
  cross-pool waits were perfectly overlapped.
- **Observed: 951 ms**.
- **Gap: ~200 ms** of recoverable overhead (about 21% of step time).

This is the answer to "how close are we to the irreducible serial." We're at
~80% of the floor. To get further requires either reducing PPGD's actual
compute (D3_warmup 383 ms is the single biggest target), reducing LW's D3
(580 ms — currently the second-biggest single compute node, and would matter
because LW's compute floor 640 ms isn't far below PPGD's 750 ms), or removing
the cross-pool waits that don't overlap with compute.

## Working numbers (reference)

- Smoke launcher: `PD_PHASE_TRACE=1 python scripts/gpt2_xl_qk_production.py --smoke --ci-bwd-profile --nccl-event-timing`
- Reference log for the numbers above: `/mnt/data/artifacts/mechanisms/param-decomp/slurm_logs/slurm-33994.out`
- Analyzer: `python scripts/critical_path.py --step 5 --ranks 0:LW,96:CI,100:PPGD <log>`
