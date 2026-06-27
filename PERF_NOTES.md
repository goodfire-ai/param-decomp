# full-32L pure-HSDP perf notes (overnight, branch `perf/hsdp-mfu`)

Goal: raise **MFU** of the full-32L Llama-8B pure-HSDP ÷N step (not the arbitrary 2–3s).
Intuition: <20% MFU = poor, 40–55% = well-tuned, 60% = excellent. This step is atypical
(several full-network forwards + 32B CI fn + PGD), so calibrate "good" lower — target ~35–50%.

## Established facts (verified this session)
- **Baseline step ~19–20s** (pure-HSDP ÷N, command buffers off, autotune off), 200+ clean steps.
- **autotune ON → ~12.5s/step** (confirmed stable over 6 steps; step-1 = ~300s one-time autotune search, cached). ~35% win, zero crashes. → should be the production default.
- **The `cuMemCreate FABRIC ... CUDA_ERROR_NOT_PERMITTED` is a BENIGN warning** (present 64× in the healthy 200-step run too). Cluster has no user-permitted NVLink fabric memory; XLA probes, fails, falls back. NOT a crash. (I wasted runs treating it as one — corrected.)
- **command buffers / TP both unusable**: command buffers were never actually tested (no STREAM_CAPTURE ever fired; killed on the benign warning) — but TP, when run honestly, is much SLOWER (lb1 tp4 = 38.5s for batch 8 vs 20s for batch 32 = ~7× worse throughput). **TP parked.**
- **cuDNN-flash for the CI-fn attention**: valid switch but negligible (<1–2%, the CI attention at B_local=1/T=512 is tiny). Hygiene fix only.
- **save+resume validated** for the ÷N layout (saved ckpt 100, resumed to 170, no OOM).

## The bottleneck (Codex-confirmed from the 4-node trace, p-81b07d93, autotune-off)
- Step ~20s = **~8.4s GPU-kernel-busy + ~11.4s GPU IDLE** (terminal, after kernels; all 4 nodes identical → not a straggler).
- The idle is **host-side / executable-internal** — host thread blocked in "Wait for LaunchOnDevice completion", zero GPU kernels, NOT bandwidth-bound (IB floor for the whole ÷N transfer is 0.51s).
- Striking related find: **~43,000 tiny AllGather invocations** during the active window (FSDP collective fragmentation). Leading hypothesis: NCCL CPU-side proxy progressing 43k tiny collectives lags the GPU → host blocks ~11s.
- **So the #1 MFU lever is the GPU idle (>50% of the step).** Killing it ≈ 2× MFU. Structural (coarsen/bucket gathers), not a flag.

## Plan
1. [done] autotune default, perf branch, profiled-autotune trace (130711) for the current-best breakdown.
2. Attack the idle: confirm the gather-fragmentation mechanism, then reduce the collective count.
3. Track MFU after each change (step-time + GPU-busy occupancy + est. FLOPs).
4. cuDNN one-liner + cosmetic TP tidy as cleanup.

## Running log
- (this file updated as I go)

## Metric I'll track (proxy, since no FLOP accounting yet)
- **Occupancy = GPU-busy / step-time** (baseline ~8.4/20 = 42%). Fixing the idle → ~90% occupancy ≈ 2× effective MFU. Absolute MFU via XLA cost_analysis later.

## Leading 11s fix to test: XLA collective-combine flags
- 43k tiny AllGathers → if un-combined, NCCL CPU-proxy overhead. XLA can MERGE small collectives:
  `--xla_gpu_all_gather_combine_threshold_bytes`, `--xla_gpu_reduce_scatter_combine_threshold_bytes`,
  `--xla_gpu_all_reduce_combine_threshold_bytes` (raise these to fuse the fragments).
- Cheap flag test (no code change); doesn't touch the allocator (so shouldn't re-trip the fabric path).
- Gate on the 130711 autotune trace first (confirm the 43k AllGathers + idle persist with autotune).

## Result: collective-combine thresholds — NO EFFECT (refuted)
- autotune + 1GB combine thresholds = 12.44–12.56s = identical to autotune-alone (12.5s).
- So the ~11s idle is NOT un-combined gather fragmentation. Either XLA can't merge them (data deps) or merging doesn't touch the idle.
- Reverted the combine flags. Next: the 130711 autotune trace to see the actual 12.5s GPU-busy-vs-idle split + the host-thread activity during the idle (Codex: host blocked in "Wait for LaunchOnDevice completion").

## Autotune trace breakdown (p-902af596, 12.6s step)
- GPU-busy 5.07s (kernels contiguous, done at 4.83s) + **7.55s TERMINAL IDLE** (host blocked, GPU idle, zero kernels) + 0.23 head. **Occupancy 40%.**
- autotune sped compute (8.4→5.07s) AND shrank idle (11.4→7.55s) as a side effect.
- 5399 AllGathers on GPU:0 (~43k total), 2.5s NCCL kernel time — but these are IN-SCAN (recon layer-loop + CI chunk-loop), so the combine pass can't merge them (refuted above).
- **The 7.55s tail idle is THE MFU killer.** Hypothesis: un-overlapped cross-node ÷N grad-reduce after the backward → test latency-hiding scheduler / pipelined collectives.

## Test: XLA latency-hiding scheduler + pipelined collectives (overlap the tail)

## Codex tail-idle diagnosis (p-902af596)
- The 7.55s tail = ONE unbroken PjRt "Wait for LaunchOnDevice completion" span, GPU idle, NO child events (no kernels, no visible NCCL proxy). The innermost scan `while.1897` closes at ~5.0s; then nothing until the outer span closes at ~12.4s.
- Late inter-node `SendRecv`/collective-permute under `pd_value_and_grad/pd_recon_masked_fwd` runs ~4.55–4.73s (just before quiescence). No labeled reduce-scatter/all-reduce.
- Best explanation: a late cross-node collective whose IB data movement completes OFF-GPU (proxy thread, untraced) → host blocks ~7.5s.
- Lever ranking: (1) latency-hiding scheduler [TESTING 130714], (2) reduce cross-node data volume, (3) hoist V/U reconstruction so it isn't re-done per recon chunk (XLA may already CSE it — need the all-gather count to confirm), (4) NCCL tuning (weakest — no proxy events visible).

## Result: latency-hiding + pipelined collectives — NO EFFECT (refuted)
- autotune + LHS + pipelined_{all_reduce,reduce_scatter,all_gather} = 12.55–12.91s = identical to autotune-alone.
- **Three scheduling-flag levers now refuted: collective-combine, latency-hiding, pipelined.** The 7.5s tail is NOT an XLA-scheduling problem — it's a genuine SERIAL cross-node op at the step end that can't be overlapped.
- Stepped back from flag-guessing. Codex now doing the definitive HLO-schedule analysis (p-902af596 dumps) to NAME + SIZE the serial tail op (grad reduce-scatter over `replicate`? recon-backward collective-permute?) and classify the fix (NCCL transport tuning vs reduce-volume/bf16-grad vs restructure).

## STATE for Oli (morning)
- **WIN banked: autotune → 12.5s/step (from ~20s), ~1.6×, occupancy 40%.** Should be the production default (XLA_FLAGS drop `--xla_gpu_autotune_level=0`; ~5min one-time autotune compile, cached).
- **Remaining: the 7.5s serial cross-node tail = 60% of the step = the MFU ceiling.** Flag-resistant. The fixes are STRUCTURAL (likely bf16-grad-reduce / gradient-overlap-bucketing / less cross-node recon data) — these touch numerics/semantics, so they're YOUR call, not a safe overnight unilateral change. Codex is pinning the exact op + the best fix class; I'll write up the options + recommendation.
- TP parked; cuDNN-flash = negligible (deferred); save+resume validated; fabric warning benign.

## ★ Codex breakthrough: the 7.5s tail = tiny loss-scalar all-reduce over the broken NVLS/fabric path
- Tail op = `%all-reduce-done.77`: a **128-byte f32[32] all-reduce of the scalar losses over ALL 32 devices** (axis_0, crosses IB), root-tuple dependency. Floor = nanoseconds; takes 7.5s EVERY step.
- Connection: full-32 group spans NVLink → NCCL tries NVLS (NVLink SHARP) → hits this cluster's broken fabric memory (the "benign" cuMemCreate FABRIC warning) → slow fallback every step. Grad all-reduces (over `replicate`/4-node only) avoid NVLS → fast.
- TEST: NCCL_NVLS_ENABLE=0 + NCCL_CUMEM_ENABLE=0 → skip the fabric path. Expect the 7.5s to vanish → ~5s step, ~80% occupancy.

## Note: the 64 fabric WARNINGS are XLA's allocator, NOT NCCL
- NCCL_NVLS/CUMEM=0 did NOT remove the cuMemCreate-FABRIC warnings (still 64) → those come from XLA's `cuda_vmm_allocator` at alloc time, independent of NCCL. (Benign — falls back to POSIX_FD.)
- The PERF tail is a separate thing: the NCCL all-reduce (%all-reduce-done.77). Whether NCCL_NVLS=0 fixes THAT is the open question — needs step-2+ time (step 1 is the ~300s autotune search).
- WATCHER GOTCHA (made twice): do NOT grep `CUDA_ERROR` / `CUDA_ERROR_NOT` as a failure — it matches the benign fabric warning. Real failures = STREAM_CAPTURE_INVALIDATED | Traceback | oom-kill | RESOURCE_EXHAUSTED | srun: error | sacct FAILED.

## Result: NCCL_NVLS=0 + NCCL_CUMEM=0 — NO EFFECT (refuted). 5th flag to fail.
- 12.7s = autotune-alone. The tail is NOT NCCL's NVLS/fabric path.
- Reverted ALL refuted flags. **Banked config = autotune-only (XLA_FLAGS="--xla_gpu_enable_command_buffer="): 12.5s, 40% occupancy, ~1.6× over 20s baseline.** Non-semantic, landable.
- **The 7.5s tail is flag-resistant (combine/LHS/pipelined/NVLS/CUMEM all failed) → structural.** A 128-byte loss all-reduce can't be 7.5s on its own → likely the GRAD reduce-scatter chain it transitively depends on. Codex deciding loss-vs-grad + the safe fix.
- **Production run NOT launched** (Oli's condition = a working perf fix; not met). 12.5s/40%-occ would be ~14-day tail-bound run. Hold for the fix.

## ★★ CRACKED (direct /proc evidence): the 7.5s tail is a per-step mprotect/mm-lock storm
- /proc/<pid>/task/*/stack during the tail: swarm of `ts_pool_worker` threads in `do_mprotect_pkey` + `lock_mm_and_find_vma` (mm-lock contention). A per-step mmap/mprotect storm, host-side.
- Mechanism: cuMemCreate(FABRIC) fails (no fabric on HGX) → VMM allocator retries with a different handle type → re-maps memory (mprotect) → dozens of threads serialize on the kernel mm_lock → ~7.5s. The FABRIC fallback IS the cost (not benign). Explains flag-immunity (not GPU/net/schedule) + fixed-per-step + invisible-to-compute.
- FIX TO TEST: XLA_PYTHON_CLIENT_ALLOCATOR=platform (cudaMalloc, no VMM/cuMem → no FABRIC → no per-step mprotect). Never cleanly tested (false-flagged on TP). Testing on pure-HSDP autotune now.

## Allocator tests for the mprotect-storm hypothesis
- **platform allocator: OOM** (57 GiB, no pooling → fragmentation) — inconclusive on the tail.
- **mprotect storm is INTERMITTENT** (2 of ~10 /proc samples), while the tail is every-step → it's a strong lead but NOT yet proven to be the 7.5s (could be periodic background). `tensorstore` IS installed → `ts_pool_worker` may be orbax/ckpt I/O (periodic), which would make the mprotect bursts NOT the per-step tail.
- **cuda_async allocator (pooling, non-cuMem): TESTING (130720)** — the clean test. If it drops the step to ~5s, the cuMem/VMM allocator (FABRIC→mprotect) WAS the tail. If not, the tail is non-allocator (and the mprotect storm is a red herring too).

## Honest standing assessment (for Oli)
- **WIN, landable now: autotune → 12.5s (from ~20s), 40% occupancy.** Config-only (XLA_FLAGS drop autotune-off; ~5min one-time autotune compile, cached). Recommend as production default regardless of the tail.
- **The 7.5s tail: extensively isolated, mechanism still not definitively pinned.** Ruled out: NVLS, socket (it's IB+GDR), the loss all-reduce, grad-transport-bandwidth, and 3 scheduling flags. Direct /proc evidence shows a mprotect/mm-lock storm but its per-step-causality is unconfirmed. This likely needs a dedicated NCCL/CUDA profiler (nsys) or CoreWeave input — beyond clean isolation tonight.
- **Production run: HELD** (condition = a working tail fix; not yet met). autotune-alone (12.5s/40%) would be a ~14-day run — your call whether that's worth launching vs waiting for the tail fix.

## ★ cuda_async allocator REFUTES the mprotect-storm hypothesis (130720: 12.5–13s)
- cuda_async (cudaMallocAsync pools, NON-cuMem, no FABRIC path) = 12.5–13.0s = identical to autotune-alone.
- If the 7.5s tail were the cuMem/VMM FABRIC→mprotect storm, a non-cuMem allocator would have killed it. It didn't.
- → **The allocator is NOT the tail cause.** The intermittent mprotect storm in /proc was the orbax/tensorstore ckpt-I/O red herring I'd flagged (it's periodic, not per-step). MALLOC_ARENA_MAX=2 (130722) attacks the same glibc-arena angle → expected no-op (running, for completeness).
- **Refutation tally: scheduling flags (combine/LHS/pipelined), NVLS/CUMEM, 3 allocators (platform OOM, cuda_async, default). The tail survives ALL env/flag/allocator A/B.**

## Definitive next step: py-spy --native on the live rank-0 process (in progress)
- py-spy IS available (~/.local/bin/py-spy, shared FS → on compute nodes). `dump --native` gives the merged Python+C stack of the training process DURING the 7.5s idle.
- Distinguishes: (a) native device/collective wait — Python frozen on the jitted step call, C stack in cuStreamSynchronize/NCCL → confirms a genuine serial cross-node op; vs (b) Python-side work — frames moving through orbax/logging/a host barrier → the "tail" is host bookkeeping, not a collective.
- This is the real observation (not flag-guessing) — sampling rank-0 across ~3 steps via `srun --overlap --jobid`.

## Git state RESOLVED (was the blocker)
- The pure-HSDP ÷N refactor (V/U + CI-fn _reconstruct_ci_compute_weights, 24 files) was UNCOMMITTED working-tree state — present in run snapshots (pd-lm snapshots the tree) but never on a branch; my earlier add-all + failed commit (unreachable-code from a grad-norm test edit) left it staged. FIXED: reverted the test edit, committed the working tree (b08dbe363). ÷N now durable; tree clean. Nothing lost.

## ★★★ BREAKTHROUGH (direct /proc thread-state sampling): the tail is HOST-SIDE, main thread CPU-bound
Sampled `/proc/<pid>/task/*/{status,wchan,stat}` of the live rank-0 python (PID, 266 threads) every ~0.3s across steps (ptrace-free — `status`/`wchan` readable same-user even at ptrace_scope=1; `stack` and py-spy are NOT — yama blocks attach to a non-descendant). Two clean phases per step:
- **Compute**: `main=S:futex_wait_queue` (sleeping), 8–17 `py_xla_execute` threads in State R (driving the GPU).
- **Tail**: `main=R` (on-CPU, RUNNING), **R=1 in 33/38 R-samples → single-threaded**; all 263 other threads asleep. NOT futex, NOT mprotect, NOT a syscall.
- **The 7.5s "GPU-idle tail" is ONE python thread burning ONE core.** This is why EVERY device-side lever failed (combine/LHS/pipelined/NVLS/CUMEM + 3 allocators): the cost was never on the device.

### Narrowing the host-side cost
- Model uses **scan-over-layers with STACKED params** (leading n_layer axis = scan xs) → LOW leaf count (~tens) → **pytree-flatten / eqx.filter_jit-partition is NOT the 7s** (it's O(n_leaves)=cheap).
- Remaining candidates: (a) **GC churn**, (b) **CPU spin-wait** — PjRt/NCCL busy-polling (State R) for a cross-node collective whose data moves off-GPU on the NCCL proxy/NIC, GPU idle. (b) reconciles host-bound-R with the earlier "late SendRecv + Wait-for-LaunchOnDevice" trace finding.
- **DISAMBIGUATOR LAUNCHED (130723, PD_TIME_STEPS=1)**: splits each step into `sample_batch / dispatch(py) / compute(dev=block_until_ready)`. compute(dev) large → device/collective (spin-waited), the fix is the collective; dispatch(py)/sample large → genuine host Python (GC/dispatch).

### Tooling notes
- py-spy IS installed (~/.local/bin) but ptrace_scope=1 + no passwordless sudo → "Permission Denied" attaching to the trainer (launched by a different srun tree). `/proc/.../status`+`wchan` are the ptrace-free fallback that worked.

## ★★★★ ROOT CAUSE (definitive, PD_TIME + PD_LEAVES, job 130723/130724): the step is DISPATCH-BOUND, not compute-bound
Per-step attribution at the full32L topology (dp=32, 4 nodes, autotune, steady state):
```
sample_batch = 0.02s   |   step_fn dispatch(py) = 12.5s   |   compute(dev)=block_until_ready = 0.02s
PD_LEAVES: state=2169 leaves   lm=13   eqx.partition(lm,state)=0.003s
```
- **The whole 12.5s is inside the `step_fn(...)` call. `step_fn` is SYNCHRONOUS (block-after = 0.02s → it internally waits for the device), so the 12.5s = device-kernel time + host DISPATCH, NOT a pure idle.** /proc resolves it into two per-step phases: a device-active phase (`main=S`, 8–17 XLA threads R) AND a multi-second SINGLE-THREADED dispatch phase (`main=R`, GIL-held main thread grinding per-array dispatch). Autotune's 20→12.5s win was on the device kernels; the residual single-threaded dispatch phase is the remaining lever and is leaf-count-bound. (PD_BENCH quantifies the dispatch portion directly.)
- **`state` has 2169 sharded array leaves** (224 sites × {V,U} = 448 params + Adam mu/nu mirror + CI-fn + sources). `eqx.partition` is 0.003s → NOT the Python partition. The 12.5s is jax's **C++ dispatch over 2169 sharded arrays on the 32-GPU multi-host mesh** (~5.7 ms/leaf: per-array sharding validation + transfer scheduling + output `Array` construction). A well-known jax "many small sharded arrays" cost.
- **Why every prior lever failed**: combine/LHS/pipelined/NVLS/CUMEM + 3 allocators all target the DEVICE/network. The cost was never there. The "7.5s tail" and "11s idle" were always host-side dispatch.
- **Why it appeared at full32L and not the smaller pile runs**: leaf count scales with sites. 4–9-layer LlamaSimpleMLP has ~tens of sites → dispatch ~sub-second, invisible. 32 layers × 7 kinds = 224 sites → 2169 leaves → 12.5s.

### THE FIX: store the trainable state STACKED per-kind (leading layer axis), mirroring the frozen model
- The frozen model (`lm`) is ALREADY scan-stacked → only 13 leaves. `targets/llama8b.py::_reconstruct_compute_weights` already STACKS the per-site `vu` dict into per-kind `[n_layer, ...]` arrays INSIDE the jit. The waste is that `state` is STORED as the per-site dict, so the dispatch boundary sees 2169 leaves.
- Change `DecompVU.vu: dict[str,(V,U)]` (448 leaves) → per-kind stacked arrays (7 kinds × {V,U} = 14 leaves; V:[n_layer,d_in,C], U:[n_layer,C,d_out] — layers within a kind are uniform, the scan precondition). The Adam state + CI-fn follow the same collapse. **state: 2169 → ~60 leaves (~35×).**
- Predicted: dispatch 12.5s → ~0.3–0.5s; step becomes device-bound. Enables much larger batch at high MFU (the real goal).
- **Touches**: `components.py` (DecompVU + shardings + init), `targets/llama8b.py` (masked-forward indexing `components.vu[name]` → stacked slice — much of `_reconstruct_compute_weights` becomes a no-op), the optimizer state, `ci_fn.py`, and the **orbax checkpoint format** (existing ckpts need migration or it's fresh-runs-only).
- **Numerics-PRESERVING** (same arrays, relayout) → verifiable via `tests/equivalence/`. But the checkpoint-format change makes this OLI'S CALL, not a safe overnight unilateral change (per the standing guidance). Isolation + proof done; refactor pending approval.

## ⚠️ CORRECTION (PD_BENCH + PD_ASYNC refute the dispatch-bound conclusion above)
The "dispatch-bound / stack the state" conclusion (the 2169-leaf section) is **WRONG**. Two controls killed it:
- **PD_BENCH**: identity-jit dispatch over the 448-leaf per-site vu dict = **0.018s** (stacked-14 = 0.002s). jax dispatch over hundreds of sharded arrays is milliseconds, not seconds → 2169 leaves ≈ 90ms, NOT 12.5s. Leaf count is not the cost. (Stacking the state is still a fine micro-opt — ~12× on a tiny absolute — but it is NOT the lever.)
- **PD_ASYNC** (3 unblocked dispatches + 1 block): call1=324s (compile), **call2=12.6s, call3=12.4s, final_block=0.017s**. Each step_fn blocks for the full step → the wall is **DEVICE EXECUTION (~12.5s/step)**, not host dispatch. The `main=R` single-threaded /proc phase was a **spin-wait** on the synchronous device step, not Python CPU work. (Lesson: `main=R` ≠ "python-bound"; jax block/dispatch busy-polls. Confirm device-vs-host with an unblocked-dispatch test, not /proc state.)

### Corrected picture — DEVICE-bound, overhead-dominated at tiny per-GPU batch
- Step ≈ 12.5s of device time. autotune's 20→12.5s was a device-kernel win (consistent).
- At **B=32 / 32 GPU = 1 seq/GPU**, useful FLOPs per step are tiny; the 12.5s is dominated by **fixed per-step device overhead**: the ÷N cross-`replicate` gather, the ~43k in-scan FSDP AllGathers (recon layer-loop + CI chunk-loop), recon recompute across the 4 chunks, and collective/launch bubbles. → intrinsically low MFU.
- **The two real MFU levers** (both device-side, the original investigation's targets):
  1. **Bigger batch** — amortize the fixed per-step overhead. First-order. (Oli wants B=64–128 anyway.) Measuring step-time vs batch finds the MFU knee.
  2. **Reduce fixed device overhead** — coarsen/bucket the 43k gathers, cut recon recompute. Second-order, structural.
- NEXT: batch-scaling sweep (B=32→64→128 at dp=32, step-time each) to quantify the amortization + locate the MFU-optimal batch; then a device profile of where the 12.5s goes.

## Batch-scaling sweep (130727/130728): BLOCKED by per-GPU memory — batch is NOT a usable MFU lever here
- **B=64 dp=32 (2 seq/GPU)**: compiled (remat floor 129.0GiB, "can't reduce below 123.97GiB") then OOM'd at runtime step 2 → one rank died → coordination-service cascade (grpc "connection refused" on the others is the SYMPTOM, not cause).
- **B=128 dp=32 (4 seq/GPU)**: remat floor **187.4GiB** (> B200 ~180GB) → OOM before stepping.
- **Conclusion**: at dp=32 the feasible batch is ~1 seq/GPU (B=32). The full32L step is memory-capped at ~1/GPU (matches the reference B=128 dp=128 = 1/GPU). Bigger GLOBAL batch needs more nodes (dp=64/128) at the SAME 1 seq/GPU → adds GPUs but does NOT raise per-GPU MFU (each GPU still does 1 seq, still overhead-bound). So **batch amortization is not available** to fix per-GPU MFU; the lever is reducing the per-step device overhead.

## HONEST STANDING ASSESSMENT (corrected, end of session)
What is now SOLID:
- The full32L step is **DEVICE-bound at ~12.5s/step** (PD_ASYNC: each synchronous step_fn call ≈12.5s, final block 0.017s). NOT host-dispatch-bound (PD_BENCH: 448 sharded leaves dispatch in 18ms; leaf count is a red herring). `main=R` /proc was a spin-wait.
- Device timeline (earlier trace p-902af596): ~5s GPU-busy + **~7.5s GPU-IDLE** per step. The 7.5s GPU-idle is the MFU ceiling (occupancy ~40%).
- **autotune banked**: 20→12.5s device-kernel win, config-only, landable.
- Per-GPU batch is memory-capped at ~1 seq/GPU → batch can't amortize the overhead.

What the 7.5s GPU-idle IS (best current understanding) and why it's hard:
- A per-step span where the GPU is idle and the host spin-waits (main=R) — consistent with waiting on a cross-node collective whose completion is off-GPU (NCCL proxy / IB), NOT a GPU kernel. The loss-scalar all-reduce was named earlier but is only 128B (can't be 7.5s on size) → likely a RENDEZVOUS/sync stall, not bandwidth.
- Flag/config/allocator-RESISTANT: refuted = collective-combine, latency-hiding, pipelined collectives, NVLS, CUMEM, 3 allocators (platform/cuda_async/default), MALLOC_ARENA_MAX. None moved the 12.5s.
- Remaining structural candidates (NOT yet tried; device-side): coarsen/bucket the ~43k in-scan FSDP AllGathers (recon layer-loop + CI chunk-loop), and check for a per-step straggler (sample /proc on ALL ranks simultaneously — is one rank lagging 7.5s, or do all idle at one collective?).
- **This needs nsys (not on login PATH; available on compute nodes?) or CoreWeave input** to name the exact stalling op and whether it's a straggler vs a genuinely slow collective. Beyond what flag/config A/B can reach.

Production run: **HELD** — Oli's condition was a working perf fix; autotune is banked but the 7.5s GPU-idle (the MFU ceiling) is unresolved. A 12.5s/step × 100k run is ~14.5 days. Don't launch until the idle is addressed.

## Gather-hoist lever investigated (code reading) — likely NOT the win
Question: is the cross-`replicate` (IB) ÷N V/U reconstruction redundantly recomputed per recon chunk?
- `targets/llama8b.py::_run_masked_forward` calls `_reconstruct_compute_weights(per_kind)` at line 541, BEFORE the scan. The remat is **per-LAYER** (line 596: `jax.checkpoint(block)` wraps only the scan body, NOT the whole forward). → **the reconstruction is OUTSIDE the remat region; it is NOT recomputed by remat within a forward.**
- Across the N recon chunks (N separate masked forwards), the reconstruction is a PURE function of `vu` (identical inputs; masks differ but are handled separately) and is not under remat → **XLA CSE is eligible to dedupe it to 1× per step.**
- → The "hoist the V/U gather out of the per-chunk recompute" idea (CLAUDE.md candidate) is largely a no-op: the structure already avoids the within-forward recompute, and cross-chunk is CSE-able. NOT the 7.5s lever.
- Step HLO (recent full-step dump) carries ~1089 all-gather-start + ~1790 all-reduce ops/step — dominated by the per-layer FSDP gathers (intra-node NVLink, cheap each), not the few big cross-node V/U gathers. Classifying them precisely (cross-node vs intra-node replica groups) needs proper HLO tooling / nsys — the XLA dump's replica_group format didn't grep cleanly here.

### Net: the 7.5s GPU-idle remains the MFU ceiling and is NOT addressable by the cheap/structural levers tried
Refuted now: all scheduling/transport flags, NVLS/CUMEM, 3 allocators, MALLOC_ARENA, batch-amortization (memory-capped), AND the gather-hoist (reconstruction already outside remat). The idle is a cross-node collective rendezvous/sync stall whose root cause needs **nsys** (compute-node only) or **CoreWeave** to pin (straggler vs slow-collective). autotune (20→12.5s) stands as the banked, landable win. Production held.

## ★ All-ranks straggler check (job 130731, clean 10Hz /proc across all 4 nodes) — NO straggler, main SLEEPS
Sampled the trainer python main-thread State on ALL 4 ranks simultaneously at 10Hz for 30s (4 parallel single-node `srun --overlap -w <node>`; the multi-node `-N4` overlap step only grabs 1 node — use 4 separate sruns):
```
node 065: 10 R / 290 S    node 107: 10 R / 290 S    node 129: 9 R / 291 S    node 131: 7 R / 293 S
```
- **No straggler**: all 4 ranks are statistically identical (~3% R, ~97% S). The 7.5s GPU-idle is a SYMMETRIC cost all ranks pay together → a genuine collective/rendezvous wait, NOT one slow rank.
- **Main thread SLEEPS (S/futex_wait) ~97%, does NOT spin (R).** This CORRECTS the earlier "main=R spin-wait" reading — that was an artifact of the slow ~1s/sample sampler (266-thread read overhead) + post-cancellation zombie samples. At clean 10Hz the main thread is overwhelmingly blocked-sleeping, waiting for the synchronous device step. Fully consistent with device-bound.
- **Net**: device-bound 12.5s; ~40% GPU occupancy; the 7.5s GPU-idle is a symmetric cross-node collective/rendezvous wait (all ranks block together, GPU idle, NCCL progresses off-GPU on the NIC). This is the "genuine slow collective" branch → needs nsys (compute-node) / CoreWeave to name the op + why it's slow. Not a straggler, not host-Python, not leaf-count, not allocator, not flags.

## ★★★★★ LIKELY CULPRIT NAMED (HLO collective census, step module): 32 cross-node all-reduces in the CI-FN FORWARD
nsys confirmed unavailable (not in cuda bin, nowhere on FS) → used the after_spmd_partitioner HLO. Mesh is 3-axis `[axis_0=fsdp=8, axis_1=tp=1(phantom), axis_2=replicate=4]`. Classified every collective by the axis it crosses:
- **axis_0 (fsdp, intra-node NVLink, cheap)**: 584 all-gather + 1005 all-reduce — the ~43k-class per-layer FSDP weight gathers. NOT the bottleneck (NVLink).
- **axis_2 (replicate, CROSS-NODE / IB, expensive)**: **49 all-reduce**, broken down:
  - 17× `f32[]` scalar (`reduce_sum`) — the loss-scalar reductions.
  - **32× `bf16[...]` weight-sized (`[4096,512]`,`[512,4096]`,`[4096,2048]`,…) inside `op_name=pd_ci_fn_fwd_main`** — the CI-fn FORWARD doing cross-node all-reduces.
- **Mechanism**: the CI fn (~31B, ÷N-sharded over the FULL mesh incl. the cross-node `replicate` axis) has a forward that CONTRACTS over a replicate-sharded dim → tensor-parallel-style **cross-node all-reduce over IB**, ×32/step. At ~150-200ms IB rendezvous+transfer each → ~6-7s ≈ the observed 7.5s GPU-idle. **The ÷N optimizer-state sharding (a memory win) leaked into the CI-fn COMPUTE, forcing per-forward cross-node reductions.**
- Consistent with: device-bound, symmetric across ranks (all ranks do the same 32 reduces), flag-resistant (combine can't merge data-dep collectives across the CI-fn matmul chain; LHS can't overlap a serial reduce→matmul→reduce dependency), and the earlier trace's "late SendRecv + one long host wait" (the host blocks while the 32 IB reduces drain on the NCCL proxy).

### FIX DIRECTION (top candidate for next session — structural, numerics-preserving)
- Audit `ci_fn._reconstruct_ci_compute_weights` + the CI-fn forward sharding: the reconstruction is meant to gather the `replicate` axis to a ÷fsdp compute layout BEFORE the per-chunk scan (like the target weights) so NO forward matmul contracts over `replicate`. The 32 axis_2 all-reduces in `pd_ci_fn_fwd_main` show a replicate-sharded contraction REMAINS — find which CI-fn weight/activation dim is still `replicate`-sharded at forward time and pin it replicate-LOCAL (gather once in entry, like llama8b does for V/U). Should remove the 32 cross-node reduces → kill most of the 7.5s.
- Numerics-preserving (pure relayout) → verify via `tests/equivalence/`. This is the concrete lever that replaces the earlier "needs nsys/CoreWeave" handoff: the culprit is named (CI-fn forward cross-node reduces from ÷N leak), the fix is a sharding pin.
- Caveat: HLO is from a recent full-model step module (mesh [8,1,4]=32, Llama-8B dims); if it's not the exact perf-branch config, the CI-fn-forward cross-node-reduce structure is architectural and still applies — confirm the axis_2 count on a fresh perf-branch HLO dump first.

## ✅ CONFIRMED on the perf-branch config + REFINED: 32 cross-node reduces are the CI-fn BACKWARD (remat'd, per-chunk)
Verified on p-adaf8a8e (the ACTUAL perf-branch run: dp=32, jax-full32L-HSDP-b32-dp32-PROFILE) — mesh `[axis_0=fsdp=8, axis_1=tp=1, axis_2=replicate=4]`, identical census: 49 axis_2 (cross-node) all-reduces/step, **32 of them under `pd_ci_fn_fwd`**. Caveat from the prior section is RESOLVED — not a wrong-config artifact.

REFINEMENT (op_name detail of the 32): they are NOT the plain forward — they're under
`pd_value_and_grad/transpose(jvp(pd_ci_fn_fwd_main))/while/body/closed_call/checkpoint/{abc,dc->abd | abc,cd->abd | ...a,ab->...b}`:
- `transpose(jvp(...))` = the **BACKWARD** (VJP) of the CI-fn forward.
- `while/body` = inside the **per-chunk scan**.
- `checkpoint` = under **remat** (the CI-fn per-chunk recompute).
- → The CI-fn BACKWARD, recomputed under remat per chunk, issues **32 cross-node (IB) all-reduces** contracting over a `replicate`-sharded dim. These are the IB collectives; at IB rendezvous+transfer latency × 32, ≈ the 7.5s GPU-idle, and they land in the backward — matching the trace's "late SendRecv + long host wait" phase.

### TWO fix candidates (next session; both numerics-preserving, perf-branch)
1. **Sharding audit**: find which tensor in the CI-fn backward is still `replicate`-sharded at contraction time. The forward weights ARE re-pinned to ÷fsdp by `_reconstruct_ci_compute_weights` (entry, before scan), so the leak is likely an ACTIVATION (a tap / intermediate) or a grad tensor that stays `replicate`-sharded into the backward matmul. Pin it replicate-local (`with_sharding_constraint` to ÷fsdp / replicated) so the backward contracts intra-node only (axis_0), not cross-node (axis_2). Expected: removes the 32 IB reduces → kills most of the 7.5s.
2. **Remat interaction**: the reduces are under `checkpoint` (CI-fn per-chunk remat). Test whether the recompute re-introduces the cross-node contraction that the entry reconstruction removed — i.e. the remat recomputes the forward WITHOUT the ÷fsdp pin (the pin is outside the checkpoint region). If so, either pin inside the remat region or mark the reconstructed weights as remat-saved (not recomputed). A/B: CI-fn remat off (memory cost) should drop the axis_2 count if remat is the cause.

This NAMES the lever concretely (CI-fn backward cross-node reduces from a ÷N sharding leak, remat-interacting) — a real structural fix, not an environmental dead-end. nsys/CoreWeave NOT required to proceed; a fresh perf-branch HLO dump confirms the axis_2 count after any fix.

## ★★★★★★ DEFINITIVE MECHANISM: CI-fn weight-grad cross-node reduce is INSIDE the per-chunk scan (runs n_chunks×)
Traced to the exact mechanism on the perf-branch HLO (p-adaf8a8e):
- The 32 axis_2 (replicate / cross-node IB) all-reduces are `pd_value_and_grad/transpose(jvp(pd_ci_fn_fwd_main))/**while/body**/closed_call/checkpoint/{...a,ab->...b | abc,dc->abd}/transpose` — i.e. the CI-fn BACKWARD, **inside the per-chunk `while` (scan) body**, under remat.
- **An all-reduce inside a while-body executes ONCE PER ITERATION at runtime** → 32 ops × n_chunks iterations = ~32·n_chunks cross-node IB reductions per step (the HLO shows 32 static ops; runtime count is n_chunks× that).
- **It's a WEIGHT gradient**: an all-reduce over `axis_2` = the `replicate` (DATA-parallel) axis in the backward is the data-parallel WEIGHT-grad reduction (activation grads are per-example, never reduce over the data axis). Root cause: `ci_batch_sharded` pins the CI batch over BOTH `("replicate","fsdp")`, so the CI-fn weights (shared across chunks via the scan) get their data-parallel grad reduced over `replicate` — and GSPMD placed that reduction INSIDE the scan body instead of on the final loop-accumulated grad.

### THE FIX (definitive, next session): defer the CI-fn weight-grad cross-replicate reduce to AFTER the scan
- The CI-fn weights are scan-shared; their grad is a loop-carried sum over chunks. The data-parallel all-reduce over `replicate` should run ONCE on the final accumulated grad AFTER the scan — not per chunk inside it.
- Implementation options: (a) accumulate the per-chunk weight grad LOCALLY (no cross-replicate reduce) inside the scan, then a single `psum`/all-reduce over `replicate` after the scan; (b) a sharding constraint / `jax.lax.scan` grad structuring that makes GSPMD hoist the data-axis reduction out of the loop body; (c) check whether the per-chunk remat (`checkpoint`) is what forces the reduce inside — if the reduce is in the recompute region, restructure so the grad reduction sits outside the checkpointed body.
- Expected win: ~n_chunks× fewer cross-node IB all-reduces in the CI-fn backward → removes most of the 7.5s GPU-idle. Numerics-preserving (sum is associative; reduce-once == reduce-per-chunk-then-sum). Verify via `tests/equivalence/` + confirm the axis_2 count drops ~n_chunks× on a fresh perf-branch HLO dump.
- This is the concrete, mechanistic root cause + fix for the MFU ceiling — reached entirely from HLO on the login node, no nsys/CoreWeave needed.

## ⚠️ CORRECTION to the mechanism above: CI-fn weights are PER-CHUNK (stacked), so the fix is BATCHING the reduce, not eliminating it
Re-read `ci_fn.py` CIBlock/ChunkTransformer `.shardings`: the CI-fn weights carry a **leading `n_chunks` axis** (`[nc, head_out, d_model_in]` etc., axis 0 unsharded) — they are **per-chunk stacked, NOT shared across chunks**. The scan iterates over `nc`, using a different weight slice per chunk. So:
- My prior "shared weights → accumulate-then-reduce-once" framing was WRONG. Each chunk has INDEPENDENT weights, so each chunk's weight-grad data-parallel reduce over `replicate` is legitimate — the n_chunks× cross-node reduces are NOT redundant work.
- BUT they are **un-batched**: GSPMD places the cross-`replicate` reduce of each chunk's weight-grad slice INSIDE the scan body (executed per iteration) instead of reducing the full stacked `[nc, ...]` weight grad in ONE collective AFTER the scan. Same total bytes, but n_chunks small IB rendezvous instead of 1 batched one.
- **Corrected fix**: hoist/batch the data-parallel reduce — reduce the stacked `[nc,...]` CI-fn weight grad ONCE over `replicate` after the scan (one big collective covering all chunks) rather than per-chunk inside. Numerics identical (each chunk's grad still reduced over the same data axis; concatenating then reducing == reducing per-slice). Win is from collective COUNT (rendezvous), ~n_chunks× fewer cross-node IB all-reduces.
- **Implementation is GSPMD-placement-dependent and needs HLO iteration**: try (a) a `with_sharding_constraint` on the post-scan stacked weight grad to force the data-axis reduction after the loop, (b) check if the per-chunk remat (`checkpoint`) is pinning the reduce inside the recompute region, (c) restructure so the scan emits an unreduced local grad stack and a single reduce follows. After each attempt, dump HLO and check the `axis_2` all-reduce count inside `while/body` drops. This is empirical GSPMD work — the honest next step, not a one-line certainty.

### Net (most accurate statement): 
The 7.5s GPU-idle ≈ the CI-fn per-chunk weight-grad cross-node (IB) all-reduces, placed un-batched inside the scan. autotune banked. The fix is to batch/hoist that data-parallel reduce out of the per-chunk loop — numerics-preserving, validated by `tests/equivalence/` + a before/after HLO `axis_2` count, no nsys needed. Whether GSPMD cooperates needs a fresh perf-branch run per attempt; that's the next session's empirical loop.

## Toy repro of the scan-grad reduce placement — did NOT cleanly reproduce; fix needs real-model HLO iteration
Built a minimal repro (scratchpad/scan_reduce_repro{,2}.py): `lax.scan` over stacked per-chunk
weights `W[nc,d,d]` (d ÷fsdp, input batch over full `(replicate,fsdp)` mesh), grad over W,
on a 4-device sim mesh (2×2). Swept {carry-accumulate vs emit-stacked-ys} × {remat vs no-remat}.
- Result: 3 data-parallel all-reduces, lowers to a `while` loop, but NONE clearly land inside the
  while body in the toy — i.e. a plain scan-over-stacked-DP-weights does NOT obviously reproduce
  the real CI-fn's inside-`while/body` reduce placement.
- → The real CI-fn's inside-scan placement depends on a feature the toy lacks (most likely the
  per-chunk remat RECOMPUTE interacting with the data-axis reduce, and/or the exact tap/CI
  activation sharding `ci_batch_sharded` pins). A faithful toy would need to mirror those.
- **Conclusion**: the fix can't be nailed from a toy — it needs an empirical loop on the REAL CI fn
  (change → launch perf-branch run → dump HLO → check `axis_2` all-reduce count inside `while/body`
  drops → confirm `tests/equivalence/`). That's the concrete next-session task; the diagnosis
  (CI-fn per-chunk weight-grad cross-node reduces, un-batched in the scan) is solid and points
  exactly where to work. Stopped the toy line here to avoid circling (per the "step back" guidance).

## Broadcast experiment (Oli's idea: drop the scan) — FITS but doesn't consolidate alone
PD_CI_BROADCAST=1 replaces the CI-fn chunk `lax.scan` with `vmap` (all 32 chunks at once).
- **Memory: FITS** (no OOM, surprising — per-chunk remat keeps the 32-chunk broadcast in budget). My OOM prediction was wrong; measured it.
- **Step time: UNCHANGED (~11.7–12.4s vs scan 12.5s)** — no speedup alone.
- **Why**: HLO shows ~32 cross-node syncs STILL present — broadcast UNROLLED them (out of the `while` body into the flat graph) but XLA did NOT auto-COMBINE them. 32 independent chunk computations → 32 independent grad syncs side-by-side instead of looped. So "drop the scan" is necessary (frees the syncs from the loop) but not sufficient.
- **Unlock**: earlier the collective-combine flags were a no-op BECAUSE the syncs were trapped in the `while` body (combiner can't reach into loops). Now that broadcast lifted them to the flat graph, the combiner SHOULD be able to merge them. → testing broadcast + raised all-reduce-combine threshold (1GB) [131175].

## ⚠️⚠️ DIAGNOSIS REFUTED: CI-fn cross-node syncs are NOT the bottleneck (broadcast+combine = no speedup)
broadcast + combine (1GB thresholds) HLO: the CI-fn axis_2 reduces are now **17 batched `bf16[32,4096,4096]`/`[32,4096,16384]`** — the leading `[32,...]` proves the combine CONSOLIDATED the 32 per-chunk syncs into one-per-weight-over-all-chunks. The mechanism worked.
- **Step time UNCHANGED: ~11.6–12.3s** (scan 12.5, broadcast-alone ~12, broadcast+combine ~12). Consolidating the CI-fn cross-node syncs → ZERO speedup.
- **→ The CI-fn weight-grad cross-node syncs were NEVER the 7.5s bottleneck.** The entire "CI-fn per-chunk reduce" diagnosis is refuted by direct measurement (changed the sync structure completely; step invariant). I over-fit the HLO collective census without confirming it was on the critical path.
- The 7.5s GPU-idle is something ELSE — candidates not yet isolated: the TARGET/recon path (chunkwise suffix forwards over 32 layers + their own collectives), the recon grid, or PGD. The 17 NON-ci_fn axis_2 reduces (target/recon) were untouched by the CI-fn broadcast.
- NEXT: real profiler trace (jax.profiler) of the current step to find what the GPU actually waits on in the 7.5s idle — stop inferring from HLO collective counts, look at the timeline.

## ★★★★★★★ REAL BOTTLENECK (perfetto trace, p-2bc5e6cd, scan/default): COMMUNICATION-bound, FSDP AllGather-dominated
Parsed the actual GPU timeline (scratchpad_trace_analyze.py on the chrome trace.json.gz). 37% occupancy (5.0s busy / 13.4s ≈ 3 steps). Top GPU kernels by total time:
- **ncclDevKernel_AllGather_RING_LL: 16,544 ms** (dominant — FSDP per-layer weight gathers)
- **ncclDevKernel_SendRecv: 5,318 ms** (cross-node collective-permute)
- gemm matmuls: ~15,000 ms summed
→ **The GPU spends MORE time in NCCL collectives (AllGather+SendRecv ≈ 22s) than in matmuls.** The step is COMMUNICATION-bound on FSDP weight all-gathers. (This is the ORIGINAL "collective-progression / 43k-gather fragmentation" hypothesis — correct all along; the CI-fn grad-reduce was a red herring I chased via HLO inference.)

### Collective time by phase (pd-scope):
- **AllGather pd_pgd_warmup_ascend: 7,110 ms** + SendRecv 1,021 ms  → PGD ascent ~8.1s
- **AllGather pd_value_and_grad: 6,900 ms** + SendRecv 3,660 ms  → backward ~10.6s
- AllGather pd_ci_fn_fwd_detached: 1,766; clean_fwd 348; read_taps 339
- **`pd_pgd_warmup_ascend` is PER-STEP** (adversary.py:9 — "each step runs n_warmup_steps supplemental ascents + one final"), NOT a startup artifact. n_warmup_steps=2 → 2 all-sites "route-all" recon forwards/step, each re-gathering ALL FSDP weights.

### Levers (to MEASURE, not assume — burned once already):
1. **PGD ascent (~8s)**: 2 all-sites route-all forwards/step re-gather all FSDP weights. Weights are FIXED during the ascent (only the source/mask updates) → gathering once and reusing across ascents could cut ~2-3× of those gathers. Or reduce n_warmup_steps (semantic). Quantify via ablation first.
2. **Backward (~10.6s)**: FSDP weight re-gathers during grad (remat recompute). Inherent-ish; overlap or gather-granularity.
3. General: combine/coarsen the FSDP AllGathers (the RING_LL many-small pattern); reduce cross-node SendRecv.

## ★ PGD ascent quantified: no-PGD ablation = 7.6s vs full 12.5s → PGD ascent is ~4.9s (~40% of step)
no-PGD config (PersistentPGDReconLoss removed) steady step = 7.62s (vs 12.5s full). **The PGD ascent costs ~4.9s/step — the single biggest lever.** Matches the trace (pd_pgd_warmup_ascend was the top AllGather consumer).
- Mechanism (train.py:279 `warmup_scoring_loss`): each ascent runs the full all-sites `masked_forward` (re-gathers ALL FSDP weights), with `components` FIXED and only `sources` (mask) varying. n_warmup_steps=2 supplemental + 1 final = ~3 gather-heavy all-sites forwards/step.
- The remaining 7.6s (no-PGD) is also gather-bound (backward FSDP re-gathers).
- **Root structural issue: FSDP re-gathers all 32 layers' weights for EVERY forward pass, and the step has many (clean + recon chunks + 2-3 PGD ascents + backward recompute).** That's the 16.5s AllGather.
- Levers: (a) reduce n_warmup_steps [SEMANTIC — adversary quality, Oli's call; measuring the cost curve via sweep], (b) gather-reuse across ascents [numerics-preserving but HARD: keeping all 32 layers gathered = the OOM FSDP avoids], (c) reduce per-forward gather cost / overlap.
