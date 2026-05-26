# 3-Pool training perf — handoff (2026-05-26)

One session's worth of investigation into step-time bottlenecks on the
GPT-2 XL Q/K 3-pool smoke (112 GPUs, 96 sites, ~2.64B-param CI fn). Captures
what landed, what we learned, what's still partly believed, and the next
moves.

## TL;DR

Step time **2235 ms → 810 ms (-64%, 2.76× faster)** over 4 commits. The big
single win was autocasting the CI fn forward in bf16; the rest was the cost
of *understanding why* the rest looked slow but wasn't. We finished the day
knowing:

- The CI fn backward is **not** the bottleneck — it runs in ~34 ms of real
  GPU work. The ~600 ms we were measuring was cross-stream wait time
  misattributed to the backward phase by `with p.phase(...):`.
- The real bottleneck for compute is **LW's `D3_layerwise`** (~750 ms per
  step, ~82% real work in standalone).
- We also shipped a 5-piece **profiling toolchain** so this class of
  misattribution is one-glance-catchable next time instead of one-day.

## Commits landed today (on `feature/resumption`)

| SHA | What | Effect |
|---|---|---|
| `61ae4ef1` | bf16 autocast on CI fn forward (`ci/1` phase) | 2235 → 847 ms (-62%). Was running fp32 / TF32 because `_target_fwd_and_cache` upcasts inputs to fp32 and no autocast wrapped `calc_causal_importances`. |
| `731ee481` | Per-stage CI fn bwd CUDA-event instrumentation (`PD_CI_FN_BWD_PROFILE=1`). Plus a global pre-step barrier fix so `PD_TORCH_PROFILE_RANKS` doesn't deadlock. | Diagnostic only. Revealed total GPU bwd = 34 ms vs `ci/8a` phase wall = 626 ms. |
| `f8dc6180` | Single sigmoid+assert on unsplit CI fn output, not per-site loop (96 sigmoids → 1). Tests updated for the new `Tensor` (not `dict`) return type. | 847 → 824 ms. `ci/1` 34 → 11 ms. Cleaner code more than perf win. |
| `dac30cd5` | Profiling toolchain (5 pieces) + defer per-step `.item()` syncs to log-only steps + per-site `.item()` in LW D3 → Tensor accumulator. | 824 → 810 ms. `lw/D3_layerwise` 803 → 748 ms. |

## What we proved today (with confidence)

| Claim | Confidence | Evidence |
|---|---|---|
| CI fn fwd ran fp32/TF32 not bf16, was massive compute waste. | **Very high** | bf16 autocast match expected FLOPS ratio; sole change cut step 62%. |
| CI fn bwd is **not** CPU-dispatch bound. | **Very high** | Standalone single-process repro at production-scale config: 33 ms wall = 33 ms GPU event. |
| The "626 ms ci/8a wall" was cross-stream wait, not bwd compute. | **Very high** | Adding `torch.cuda.synchronize()` before the bwd moved the 600 ms cleanly from `ci/8a` → `ci/6_recv_g_ci_from_ppgd`. CI was waiting for PPGD's send to land in GPU memory. The stream-aware allocator's cross-stream wait was billed to the consumer (bwd) instead of the comm. |
| LW's `D3_layerwise` is the real critical-path compute. | **High** | Standalone D3 ≈ 612 ms, production ≈ 748 ms (~82% real work). 8 sites × per-site fwd+bwd through the LW components vs target. |
| Cutting D3 alone won't dramatically reduce step time. | **Medium** | Critical-path analyzer says cut D3 → only -34 ms because CI's chain is right behind. **Caveat**: the analyzer currently uses CPU wall as node weight, which over-counts CI by the misattributed wait. Re-run with `gpu=` ms as weight (see TODO below) to confirm. |
| PPGD has ~40% slack (239 ms standalone vs 400 ms prod for `D3_warmup`). | **Medium** | Standalone repro vs production diff. The "slack" is likely cross-stream NCCL waits — would surface with the new `PD_NCCL_EVENT_TIMING=1` flag enabled. |

## What we ruled OUT today

These all had plausibility but the data didn't support them:

- **96-sigmoid-loop autograd dispatch overhead** — refactored to single sigmoid, saved only ~15 ms (not the ~290 ms we'd theorized). Per-autograd-node CPU cost is ~0.4 ms, not ~3 ms.
- **`zero_grad(set_to_none=True)` causing allocator churn for the 10.58 GB grad buffers** — flipping to `set_to_none=False` made it 14 ms *slower*, not faster. Allocator is not the bottleneck.
- **CUDA graphs / `torch.compile(mode="reduce-overhead")` would help** — since the bwd is GPU-bound (not CPU-bound), there's no per-kernel CPU dispatch overhead to graph away. Don't waste a day on this.
- **`torch.profiler` is usable at this scale** — we tried four variants and all deadlocked the moment CUPTI activated. Cause unconfirmed (possibly CUPTI ↔ NCCL incompatibility). The standalone repro approach sidesteps this entirely.

## Profiling toolchain shipped (commit `dac30cd5`)

Five orthogonal pieces. Each opt-in via env var so production runs aren't affected.

### 1. CPU/GPU/wait per phase
- **Where:** `param_decomp/three_pool/profiler.py`, `optimize.py`, `scripts/analyze_step_times.py`
- **Activation:** always on when `PD_PHASE_TRACE=1`
- **What it gives you:** every phase exit line now reads
  ```
  phase: ci/1_ci_fn_fwd end cpu=133.0ms gpu=120.5ms wait=+12.5ms
  ```
- **Why it matters:** `cpu - gpu == implicit cross-stream wait time`. The single most useful upgrade — would have caught today's misattribution in 30 seconds.

### 2. `PD_SYNC_DEBUG` env var
- **Where:** `param_decomp/three_pool/optimize.py`, `two_pool/optimize.py`
- **Activation:** `PD_SYNC_DEBUG=warn` (logs every implicit CPU↔GPU sync) or `PD_SYNC_DEBUG=error` (crashes on first, with traceback to culprit). Launcher flags: `--sync-debug`, `--sync-error`.
- **What it gives you:** automatic detection of `.item()` / `bool(tensor)` / `.cpu()` syncs.
- **When to use:** suspect a hidden sync is hurting perf.

### 3. Critical-path analyzer
- **Where:** `scripts/critical_path.py`
- **How:** parses the phase trace log, wires the known cross-pool send/recv pairs into a DAG, computes the longest weighted path through one step.
- **Output:** the path + "if you cut phase X, step drops by Y ms" estimates.
- **TODO (important):** currently uses CPU wall as node weight. Now that the per-phase logs include `gpu=` ms, switch to GPU time as weight. Otherwise it double-counts cross-stream waits (CI's chain is over-attributed by ~590 ms).

### 4. Standalone reproducers
- **Where:** `scripts/standalone_repros/{ci_fn_bwd,lw_d3_layerwise,pgd_d3_warmup}.py` + README
- **How:** each constructs the relevant nn.Module(s) at exact production scale, runs fwd+bwd in a single process on one GPU, dumps `key_averages` table + per-step ms.
- **Why it matters:** the standalone vs production gap is "distributed overhead"; the floor is "real compute". Today's punchline came from this.
- **Run:** `srun --gres=gpu:1 --time=20:00 python scripts/standalone_repros/<name>.py`. **Don't specify `--partition`.**

### 5. `PD_NCCL_EVENT_TIMING` env var
- **Where:** `param_decomp/three_pool/layout.py` (13 NCCL call sites wrapped)
- **Activation:** `PD_NCCL_EVENT_TIMING=1` env var, or `--nccl-event-timing` launcher flag.
- **What it gives you:** per-NCCL-op `cpu=X gpu=Y` trace lines. Large CPU + small GPU → "peer was slow to start". Large GPU → "wire transfer was slow".
- **Caveat:** records on default stream (lower bound for actual NCCL stream time). Avoids fragile internal torch APIs.

### Also: Gantt visualizer
- **Where:** `scripts/gantt_step.py`
- **What it gives you:** 3-pool ASCII waterfall for one step, with phase boundaries + legend. Useful for eyeballing send/recv handoffs across pools.

## `.item()` deferral (commit `dac30cd5`)

`step_ci`, `step_layerwise`, `step_ppgd` now take `should_log: bool` (= `step % cadence.train_log_every == 0`). When False:
- Metrics dict returns empty
- All `.item()` calls in the per-step metrics path skip
- NaN check + `_log_train_metrics` in `trainer.run` also gate on `should_log`

**Trade-off**: NaN detection now lags by up to `train_log_every - 1` steps. At our config (every 10 steps), the propagation distance is bounded; reversible by flipping the gate.

**Bigger win: per-site Tensor accumulator in LW D3.** The
`stoch_total_value += (loss_s / n_positions).item()` inside the for-loop
was 8 syncs per step (one per site), forcing each site's bwd to drain on
GPU before the next could begin. Replaced with `stoch_total_t =
stoch_total_t + loss_s.detach() / n_positions` accumulated on GPU, then
`.item()`'d once at the end if logging. Cut `lw/D3_layerwise` wall from
803 → 748 ms.

Provenance comments in `step_*.py` explain why each `.item()` was costly
and what's reversible. Search for `.item()` in those files.

## Working tree state at end of session

- ✅ All my work committed. Branch is `feature/resumption` (32 commits ahead of origin).
- 🟡 `stash@{0}`: safety snapshot from before the profiling-toolchain commit.
- 🟡 `stash@{1}`: safety snapshot from earlier today.
- 🟡 Resumption WIP files NOT committed (uncommitted modifications in `param_decomp_lab/resumption/`, `param_decomp_lab/experiments/lm/run.py`, plus untracked `param_decomp_lab/resumption/check.py`). These are the user's in-progress work, untouched today.
- 🟡 Some dead torch.profiler infra is in the uncommitted lm/run.py — wired in but doesn't work at this scale (CUPTI deadlocks). Either remove or note as known-broken in a future commit.

## Next moves, prioritized

In order of leverage-per-effort:

1. **First thing next session: run a smoke with the new instrumentation.**
   ```bash
   PD_PHASE_TRACE=1 python scripts/gpt2_xl_qk_production.py \
     --smoke --ci-bwd-profile --nccl-event-timing
   ```
   Then analyze with the per-phase `cpu/gpu/wait` columns + NCCL event timing. First time we'll have a clean per-phase work-vs-wait breakdown end-to-end. This is **information**, not optimization — purely diagnostic.

2. **Update `scripts/critical_path.py` to use `gpu=` ms as node weight** (currently CPU wall, which over-counts CI's chain by ~590 ms). One-line change once #1's data is in hand.

3. **Then look at LW D3** with the new tools:
   - What's the `cpu`/`gpu`/`wait` split inside D3? If wait is significant, fix the upstream comm that's making CI wait.
   - Is autocast bf16 active inside D3? It should be (`with p.phase("lw/D3_layerwise"), autocast_bf16(cfg.bf16_autocast):`) — confirm in the log.
   - Per-site fwd+bwd is currently sequential. Pipelining sites within a block could overlap GPU work.
   - Component count / size — is each site doing more work than necessary?

4. **PPGD's ~40% slack** — standalone (239 ms) vs prod (400 ms) gap. With `PD_NCCL_EVENT_TIMING=1` we'd see if PPGD is waiting on cross-stream NCCL ops. Likely lower-leverage than LW D3 since PPGD is off the critical path most of the time, but ~160 ms is real.

5. **Stale-gradient pipelining** (architectural, last resort) — give CI 1-step-old gradients so it doesn't have to wait for LW. Big change but cuts a real dependency edge if all else is exhausted.

## Anti-patterns to avoid

These wasted time today; don't repeat:

- **Don't optimize a phase before checking if its wall time is real work or wait.** Always look at `gpu=` vs `cpu=` first. If `gpu << cpu`, your phase is waiting, not working — go upstream of the wait.
- **Don't trust a critical-path analysis that uses CPU wall as node weight.** Use GPU time. CPU wall double-counts cross-stream waits.
- **Don't reach for CUDA graphs / `torch.compile(mode="reduce-overhead")` without first verifying you're CPU-dispatch-bound.** We're not. The standalone repros prove the bwd is GPU-bound.
- **Don't try to make `torch.profiler` work at 112-GPU scale.** It deadlocks CUPTI ↔ NCCL. Use single-process standalone repros instead — gives you the same `key_averages` data without the headaches.
- **Don't add asserts that call `.all()` on GPU tensors in hot paths.** Each is a sync. The 96-site sigmoid asserts cost real time. Mathematical guarantees (clamp-based sigmoids are in range by construction) make many of these asserts pointless.

## Useful pointers

- **Log location:** `/mnt/home/oli/param_decomp_out/slurm_logs/slurm-<jobid>.out`
- **Reference run for current baseline:** `slurm-33943.out` (50-step smoke, ~810 ms steady-state with all today's wins applied)
- **Per-phase analyzer:** `python scripts/analyze_step_times.py --rank=<R> --skip-warmup=2 <log>`
- **Critical path:** `python scripts/critical_path.py <log> --step 5`
- **Gantt:** `python scripts/gantt_step.py <log> --step 5`
- **Standalone repros:** `srun --gres=gpu:1 --time=20:00 python scripts/standalone_repros/<name>.py`

Production launcher flags worth knowing:
```
python scripts/gpt2_xl_qk_production.py --smoke [...]
  --ci-bwd-profile        # per-stage CI fn bwd CUDA-event timing
  --nccl-event-timing     # per-NCCL-op cpu/gpu split
  --sync-debug            # warn on every implicit CPU↔GPU sync
  --sync-error            # crash on first one (traceback to culprit)
  --torch-profile         # (broken at scale, kept for completeness)
```
