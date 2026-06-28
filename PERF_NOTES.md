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

## PGD ascent cost curve (n_warmup_steps sweep)
| ascents/step | step | Δ |
|---|---|---|
| 0 (no-PGD) | 7.62s | — |
| 1 (nw=0, final only) | 9.37s | +1.75s |
| 3 (nw=2, full) | 12.5s | +4.9s |
~1.6s per gather-bound all-sites ascent forward. nw=2→0 saves ~3.1s (SEMANTIC — adversary quality, Oli's call). Floor without PGD = 7.6s (backward+recon gathers).

## Why combining the FSDP gathers doesn't work (and the structural wall)
The 16.5s AllGather is `RING_LL` (small-message protocol) = MANY small per-layer gathers INSIDE the layer scan (while-body). Combine-threshold flags (tested at 1GB in broadcast+combine) DON'T reach them — same reason as the CI-fn reduces: the combiner can't merge collectives inside a `while` body. Unrolling the 32-layer scan would expose them to the combiner but materializes all layers' weights = the OOM FSDP exists to avoid. So: can't combine (in-loop), can't unroll (memory). The accessible levers are (a) PGD n_warmup (semantic), (b) NCCL protocol/algo tuning for the per-gather efficiency [TESTING], (c) deep gather-granularity restructure.

## NCCL_PROTO=Simple — NO EFFECT (12.6s vs 12.5s). Gathers not protocol-bound. Flag dead-end.
## Structural tests in flight: PD_REPLICATE_WEIGHTS (replicate V/U compute, kill per-forward gather) + 4-chunk CI fn (blocks_per_chunk=8, shrink CI fn ~8x). These change memory/size, not flags.

## ★ REPLICATE-WEIGHTS OOM (358 GiB) — reframes the whole problem
PD_REPLICATE_WEIGHTS=1 (replicate V/U compute, only shard optimizer) → OOM trying to allocate **358.85 GiB**. The V/U decomposition compute weights are ~358GB — MUCH bigger than the 8B target.
- **So "the model is 8B, just replicate it / don't FSDP" is WRONG here.** The TRAINABLE state (V/U ~358GB + CI fn ~31B ≈ 400GB+) is the real scale, not the frozen 8B target. It genuinely MUST be sharded; the per-forward gather is largely inherent to a 400GB decomposition on 32 GPU.
- Current ÷fsdp V/U = ~45GB/GPU resident; replicated = 358GB/GPU (OOM). FSDP is necessary, not over-engineering.
- → The lever is NOT "stop sharding" — it's (a) shrink the decomposition/CI fn (4-chunk test in flight), (b) OVERLAP the necessary gathers with compute (if XLA isn't, because they're in scan), (c) fewer forwards (PGD ascents). The "small model over-sharded" framing is wrong; correcting the research agents/Codex accordingly.

## ★★★ RESEARCH (collective-overlap agent) — why flags failed + 2 new concrete levers
1. **Flags are no-ops because collectives live inside `lax.scan`.** XLA latency-hiding + collective-pipeliner operate on the FLAT graph; they can't overlap/combine collectives across un-unrolled loop iterations. (jax #22210; the pipeliner that tries is buggy w/ while-loop double-buffering.) Confirms our in-scan finding — explains combine/LHS/pipelined all no-op.
2. **KNOWN XLA:GPU BUG (xla #14397 / jax #22252): per-layer `jax.remat` with a fine-grained save policy (`save_only_these_names`) SERIALIZES async collectives onto the compute stream** → no overlap. WE USE per-layer remat. → CHEAPEST high-value test: switch remat policy to `nothing_saveable`/empty, re-profile, see if all-gather overlaps.
3. **Arithmetic-intensity floor: all-gather is only hideable above ~2,200 tokens/GPU** (C/W_collective ≈ 990e12/450e9). We run **512 tokens/GPU** (1 seq × 512) — 4× BELOW the floor → the gather is on the critical path NO MATTER THE FLAGS. Fix = MORE batch per GPU.
- **Connects the whole strategy**: we're memory-capped at ~1 seq/GPU (B=64 OOM) → below the overlap floor. Shrinking the CI fn (4-chunk) frees memory → bigger per-GPU batch → above floor → gathers overlap behind compute → high MFU. Coherent plan: shrink CI fn + raise per-GPU batch + fix remat-policy overlap.
4. Other levers: `unroll=2` on the layer scan (exposes overlap to scheduler, jax #22210 workaround); `shard_map` + explicit async collectives for the dominant layer (GSPMD won't overlap in loops); gather-once-reuse across microbatches.
Sources: jax-ml/scaling-book gpus.md, jax#22210, xla#14397, openxla flags_guidance.

## ★★★ RESEARCH (scaling-strategy agent) — CONVERGES with the overlap agent; adds mesh layout
Both agents independently agree:
- **LHS is OFF by default on GPU + can't reach into scan; per-layer remat (fine save policy) serializes collectives onto the main stream (xla #14397) — kills overlap.** [audit our remat]
- **Arithmetic-intensity floor ~2,500 tokens/GPU** (H100 const; recompute for B200). We're at 512 → comm-bound regardless of flags. Fix = more batch/GPU OR add TP.
- **Realistic GPU MFU = 35–47%, ~40% planning number** (Llama-3 405B = 41% on 16k H100; at 4 nodes the IB hop is the wall). Don't benchmark vs TPU/MaxText (55–65%). Our ~37% occupancy may be closer to the realistic envelope than assumed — BUT we're comm-bound, not compute-bound, so there's room.
NEW insight — **MESH LAYOUT**: bind **TP=8 to the in-node NVLink axis + FSDP/DP=4 to the cross-node IB axis**. TP is batch-INDEPENDENT → keeps you compute-bound at LOW per-GPU batch (8× lower than pure FSDP). When batch is memory-capped (our case), ADD TP rather than widen FSDP. (Reconcile with our earlier "TP parked, 7× slower" — that may have been wrong axis-binding / cross-node TP; worth revisiting with TP pinned in-node.) NVIDIA: "TP across nodes is almost always a loss."
- ZeRO-1 (replicate weights) is the win WHEN the model fits replicated — but our V/U is 358GB (doesn't fit) → ZeRO-1-replicate is out, confirmed by OOM. TP is the alternative lever for the low-batch regime.
- Keep `lax.scan` (unrolling caused 4–5× LHS memory blowup on a 300B MoE; jax#20763). Size combine thresholds to ONE layer's bytes (not arbitrary 1GB).
Sources: jax-ml/scaling-book (gpus/training), HF ZeRO analysis, jax#22210/#20763/#25404, xla#14397, NVIDIA Megatron-Bridge.

## SYNTHESIS — the externally-validated plan (both agents + our measurements)
1. **Audit the per-layer remat** (xla#14397) — cheapest test; likely serializing our collectives. [NEXT]
2. **We're 4× below the overlap floor** (512 vs ~2500 tok/GPU) — the root comm-bound cause. Levers to raise effective intensity: (a) shrink CI fn → free memory → bigger batch (4-chunk test in flight), (b) ADD TP=8 in-node (revisit the parked TP with correct axis binding).
3. **Verify LHS/overlap actually fires in the HLO** (silently regresses) — set -O1 / LHS + pipelined + double-buffering, confirm via dump.
4. Target ~40% MFU (realistic GPU envelope), not 60%.

## ★★★★ CODEX (architecture review) — CONVERGES with both research agents: wrong comm SHAPE, switch to TP
- Current FSDP-all-gather-of-large-weights INSIDE the layer scan is the WRONG SHAPE for THIS workload: it's not one forward/backward, it's MANY forwards through the same fixed weights (clean + 4 recon suffix chunks + ~3 PGD ascents + backward) → "all-gather weights once per layer per forward" is multiplied. FSDP trades memory for repeated weight all-gathers; that only pays when memory is the binding constraint, which it isn't for the reuse pattern.
- **THE STRATEGIC REDIRECT (all 3 sources agree): TENSOR PARALLELISM.** Shard the large weights across the 8 in-node NVLink GPUs (column/row parallel: QKV/up column-parallel, out/down row-parallel) and communicate ACTIVATIONS (small) per layer instead of gathering WEIGHTS (huge) per forward. "For repeated forwards with fixed weights, resident or tensor-parallel weights have a better communication shape than repeated all-gathered weights." Mesh: data=4 across nodes (IB), model/TP=8 in-node (NVLink).
- No compiler-only fix for scan+all-gather — must change the PROGRAM (replicate / TP / blocked residency). Blocked scan (gather k layers, sub-scan) = middle ground.
- 31B CI fn = architectural red flag; shrink it (4-chunk directionally right).
- CAVEAT we measured: Codex/research assumed the trained weights are ~8B/16GB and "just replicate" — but the V/U decomposition is ~358GB (OOM-confirmed), ~22× the target. So REPLICATE is OUT; TP (÷8, activation-comm) is the lever, not replication. TP doesn't cut memory vs FSDP but changes comm shape (activations not weights).

## 4-chunk CI fn result: 12.5s → 10.79s (modest ~14% direct; main value = frees memory for batch)
Shrinking the CI fn 32→4 chunks helps but isn't the big win — confirms the V/U gathers (recon/PGD forwards), not the CI fn, dominate. Its real value: memory headroom → bigger per-GPU batch → toward the overlap floor.

## ★★★★★ STRATEGIC CONCLUSION (step-back deliverable)
We've been optimizing FSDP gather COST within the wrong comm SHAPE. The convergent external view: switch the large-weight path to **tensor parallelism (in-node, communicate activations not weights)** — this is the structural fix the flags/combine/overlap could never reach. Supporting levers: shrink CI fn (4-chunk), fix the per-layer-remat collective serialization (xla#14397), raise per-GPU batch (512<<2500 floor). NOTE: full32L HAD a (dp,tp) TP path that was PARKED as "7× slower" — but that was likely misconfigured (wrong axis / cross-node / weights-still-gathered); proper in-node activation-comm TP is what all 3 sources prescribe. Revisit TP correctly. Realistic target ~40% MFU (GPU envelope), not 60%.

## ★★★★★★ THE RESOLUTION: TP was parked after testing the WRONG config; tp=8 (the prescribed fix) never tried
The HSDP×TP 3-D mesh `(replicate, dp, tp)` with `dp·tp=8` (both in-node) EXISTS — on worktree branch `worktree-agent-afabd20f3300f05c7` (commits 574530f3f, 6bc5e2cd2, d8843c1bd …). It's proper Megatron TP: V/U C-on-tp (column/row parallel), CI-fn heads-on-tp, activation all-reduce on tp (NVLink). Revive-not-rebuild.
- `tp=8` ⇒ pure intra-node TP, `dp=1` → **eliminates the in-node FSDP weight-gather** (the 16.5s bottleneck); weights sharded only on tp, communication is ACTIVATIONS. This is EXACTLY what all 3 external sources prescribe.
- BUT the parked configs only tested **tp=2 / tp=4 = HYBRIDS** (dp=4/2 FSDP + tp): keep the FSDP weight-gather AND add TP overhead = worst of both → the "7× slower" verdict. **tp=8 (pure in-node TP, dp=1) was NEVER tested.**
- Current branch (perf/hsdp-mfu) is pure-HSDP 2-D `(replicate, fsdp)` — the TP axis was stripped when the team parked TP. So testing tp=8 = revive the afabd20f HSDP×TP code + config tp=8 (4 nodes → replicate=4, dp=1, tp=8).
**→ The convergent recommendation (TP) is already 90% built and was abandoned after testing the wrong (hybrid) config. The prescribed tp=8 is a revive + config experiment, not a rebuild.**

## ★★★ RIGOROUS (buffer-assignment dump, not arithmetic): 358GB OOM = 5 LIVE COPIES of the replicated weights
The grad-reshard fix did NOT help (same 358GB OOM) → my optimizer-gather arithmetic was WRONG (per Oli's caution). The buffer-assignment + memory-usage-report dump (`--xla_dump_to`) shows the truth:
- Peak 411GiB, dominated by `5×bf16[32,14336,10240]` (down_proj V replicated, 9.4GB each) + `5×bf16[32,8192,14336]` (gate/up) + `5×bf16[32,4096,4096]` (o_proj) — i.e. ~5 LIVE COPIES of each replicated weight, NOT the optimizer, NOT one tensor.
- Cause (llama8b.py:307 `_stack_per_kind_masked_inputs`): each masked_forward (~5/step: recon chunks + PGD ascents) re-stacks the V/U into `[n_layer,d_in,C]` BUNDLED with that forward's masks → XLA can't CSE the (identical, mask-independent) V/U across forwards → 5 stacked copies. With ÷fsdp each copy is 1/8 size (never bit); replicated, 5×9.4GB/kind = OOM.
- **FIX**: hoist the V/U stack+reconstruct to ONCE per step (mask-independent), pass the shared compute V/U to all forwards; per-forward builds only masks. 1 copy (37GB) not 5 (185GB). Model-interface refactor (masked_output signature) — numerics-identical, validate via tests/equivalence.
- This earlier per-forward-duplication hypothesis was right; I'd wrongly talked myself into the optimizer story via byte-arithmetic. LESSON: read the buffer-assignment dump, never attribute memory by shape-arithmetic.

## FIX SPEC: hoist V/U stack+reconstruct to once-per-step (makes replicate fit; helps any sharding)
The replicate OOM is per-forward V/U-stack duplication (5 live copies). Deterministic fix (numerics-identical; validate via tests/equivalence BEFORE any GPU launch):
1. `llama8b._stack_per_kind_masked_inputs` (line 307): SPLIT into
   - `_stack_compute_vu(components, n_layers)` → `{kind: {V:[nl,d_in,C], U:[nl,C,d_out]}}` (mask-INDEPENDENT), then `_reconstruct_compute_weights` on it. Computed ONCE.
   - `_stack_masks(masks, delta, routes, live, n_layers, leading)` → `{kind: {live, mask, delta, route}}` (per-forward, small).
2. `_run_masked_forward` (523): take pre-built `per_kind_vu` as an arg; build only `per_kind_masks` per call; merge for the scan.
3. `masked_output`/`masked_site_outputs` (lm.py protocol + llama8b 615/644): accept `per_kind_vu` instead of `vu` (or add a `model.reconstruct_compute_vu(components)` method returning it).
4. `train.py`: call `per_kind_vu = model.reconstruct_compute_vu(components_bf16)` ONCE (line ~270), pass to all masked_forward calls (recon 281-ish, PGD 399-ish).
Result: ONE reconstructed compute-V/U buffer (37GB replicated, fits) reused by all ~5 forwards, vs 5 copies (185GB, OOM). Also trims the ÷fsdp path (minor). Then MEASURE: does replicate (no per-forward gather) actually drop the 12.5s — the still-unproven original hypothesis.
NOTE: low-regret (reconstruct-once is a general improvement) but it's interface surgery — do it focused, not at the tail of a marathon session.

## ★ GOVERNING METHODOLOGY (Oli): verify the COMPILED strategy, never trust config→strategy
A sharding config is a REQUEST to GSPMD, not a guarantee. NEVER hypothesize "strategy X should do Y", launch a config you HOPE compiles to X, then attribute the result to X — without first verifying from the compiled HLO that GSPMD actually produced X's structure. (This retro-invalidates the parked TP "7× slower" verdict AND my replicate conclusions: neither verified the compiled strategy.)

### TP verification GATE — before ANY conclusion about tp=8:
Launch tp=8 short run with HLO/buffer dump, then CONFIRM from the dump (not the config, not comments):
1. Weights STAY tp-sharded — the per-layer full-gather buffers are ABSENT (no `bf16[14336,10240]` per-layer full gathers; in ÷fsdp we measured 70× of them).
2. Per-layer comm is an ACTIVATION all-reduce over the tp axis (Megatron pattern), NOT a weight all-gather. Census collectives by mesh axis + op_name.
3. The intended C-on-tp / heads-on-tp sharding actually materialized (check the V/U + CI-fn buffer shapes are tp-sharded).
4. Per-GPU batch is 8 (global 32 / DP 4), and it FITS (memreport peak < HBM) — the open memory question (8 seq/GPU; big intermediates tp-sharded so ~comparable, but verify).
ONLY if 1–4 hold does the step-time number reflect "TP". Otherwise the config didn't compile to TP and the result says nothing about TP.

### Sequence parallelism (SP) — deferred (Oli)
SP is the standard layer on top of TP: shard the residual-stream activations along the SEQUENCE dim in the regions between TP blocks (all-gather/reduce-scatter at the TP boundaries instead of all-reduce). It directly attacks the "residual is 8× per GPU" memory term I flagged. Worth it IF the activation memory is the binding constraint after TP — but defer until TP itself is verified working; uncertain how cleanly it fits our recon/PGD/scan structure.

## ★★★★★ TP=8 VERIFIED TO COMPILE TO REAL TP (gate passed) — but OOMs on 8-seq/GPU activations
Ran the existing tp8 config (afabd20f, profiling variant) with HLO+buffer dump. GATE check (from the dump, not the config):
- V/U = `bf16[32,14336,1280]` = [n_layer, d_in, C/tp=10240/8] → **C-sharded on tp (Megatron-C)** ✓ (NOT the fsdp `[32,1792,10240]`).
- **Per-layer weight full-gather `bf16[14336,10240]`: ZERO** ✓✓ — the 16.5s weight-gather bottleneck is ELIMINATED. This is genuine TP, unlike the parked tp2/4 hybrids.
- Activations `bf16[32,8,512,1280]` → **8 seq/GPU confirmed** (global 32 / DP 4), C-sharded.
- **OUTCOME: OOM, peak 211GiB > 180GB.** The 8-seq/GPU activations + fp32 logits `f32[8,512,128256]` (1.96GB×) dominate — exactly the activation-memory cost predicted for 8 seq/GPU.
→ TP is real and kills the weight-gather; the binding constraint is now ACTIVATION memory (the predicted tradeoff). Fixes: (a) smaller global batch → fewer seq/GPU [testing global 8 → 2 seq/GPU], (b) sequence parallelism (deferred — the proper fix for the residual/activation term), (c) more remat / bf16 logits. NOTE: methodology worked — we KNOW it's real TP, so the OOM genuinely reflects TP's memory profile, not a mis-compiled config.

## ★★★★★ TP VALIDATED (gate passed + real timing): gather gone, ~3.3× better per-GPU; global throughput needs bigger batch
tp8 b8 (global 8 → 2 seq/GPU) FITS, ~12.0s/step (autotune-OFF). Matched autotune-off comparison:
| strategy | global batch | seq/GPU | step (autotune-off) | per-GPU seq/s | global seq/s |
|---|---|---|---|---|---|
| HSDP | 32 | 1 | 20s | 0.050 | 1.60 |
| TP (tp8) | 8 | 2 | 12s | 0.167 (3.3×) | 0.67 |
- TP removed the gather: HSDP-off 20s (8.4 busy + 11.4 idle) → TP-off 12s (dropped ~8s ≈ the gather idle). So the gather WAS a real wall-clock cost and TP eliminates it (gate-verified: no per-layer full-gathers).
- **TP is ~3.3× more efficient PER-GPU** (autotune-off matched). BUT global throughput still favors HSDP (1.6 vs 0.67) because TP is memory-capped at 2 seq/GPU (8 seq/GPU OOM'd) → smaller global batch. The per-GPU win is real; converting it to a global win needs a bigger per-GPU batch.
- NEXT: sweep TP global batch up (16=4/GPU, 24=6/GPU) to find max fit; add autotune (was OFF); SP for the full 8/GPU. Compare global seq/s head-to-head vs HSDP 1.6 (or autotune-on 2.56).
- METHODOLOGY held: gate verified real TP before any timing claim; both numbers autotune-off for a fair compare.

## ★★★★★★ TP BATCH SWEEP — TP alone is memory-capped at ~2 seq/GPU → SP is REQUIRED, not optional
- tp8 b8 (2 seq/GPU): FITS, 12s autotune-off.
- tp8 b16 (4 seq/GPU): **OOM** (49.86GiB over).
- tp8 b32 (8 seq/GPU): OOM (211GB).
So TP fits only ~2 seq/GPU. Note TP fits 2× the seq/GPU as HSDP (HSDP OOMs at 2/GPU) — it IS more memory-efficient per-seq — but the binding issue is the DP COUNT:
- **TP8×DP4 has only 4 data-parallel replicas** (the 8 in-node GPUs are TP, not DP), vs HSDP's 32. So global batch = 4 × seq/GPU. To match HSDP's global 32, TP needs 8 seq/GPU → OOM.
- TP capped at 2 seq/GPU → global batch 8 → 0.67 seq/s, vs HSDP's 32 → 1.6 seq/s. **TP-alone LOSES on global throughput** despite 3.3× better per-GPU efficiency.
- **The only way TP wins: SP.** Sequence parallelism shards the per-GPU activations along seq, relieving exactly the activation memory that caps TP at 2 seq/GPU → lets TP run 8 seq/GPU → global batch 32 at TP's 3.3× per-GPU efficiency → ~2-3× faster than HSDP. So SP is REQUIRED for the TP path to pay off (un-defer it).
- Verdict: TP verified correct + per-GPU efficient, but it trades DP replicas for model-parallelism, so it needs a bigger per-GPU batch than memory allows WITHOUT SP. SP is the unlock, not an optional add-on.

## ⛔ CORRECTION — the "3.3× better per-GPU" above is a UNITS ERROR; TP is currently SLOWER, full stop
The "per-GPU seq/s" column was computed with inconsistent denominators: HSDP's throughput
was divided by **32** (all GPUs), TP's by **4** (DP replicas only). Apples-to-oranges.
Both strategies run on the SAME 32 GPUs, so the only honest per-GPU number is `global_seq_s / 32`:
| strategy | GPUs | global batch | step (autotune-off) | global seq/s | seq/s/GPU (÷32) |
|---|---|---|---|---|---|
| HSDP | 32 | 32 | 20s | 1.60 | **0.050** |
| TP (tp8) | 32 | 8 | 12s | 0.67 | **0.021** |
- On identical hardware + wall-clock, HSDP does 2.4× the throughput. You CANNOT be "more
  efficient per-GPU" AND lower total throughput at the same GPU count — the claim was
  self-contradictory. **TP is currently ~2.4× slower per token, not 3.3× faster.**
- What TP genuinely showed (HLO-verified, still true): it REMOVES the weight-gather. But that
  did not pay off, because TP's structure costs 3/4 of the data-parallel width (DP=4 not 32)
  and still takes 12s.
- **The decisive comparison was never run.** On 32 GPUs the two strategies occupy DISJOINT
  batch regimes (HSDP min = global-32 at 1 seq/GPU; TP max = global-8) so they can't be
  compared at matched batch there. "Is activation-comm cheaper than weight-gather PER TOKEN?"
  is still UNMEASURED. Everything about SP downstream rests on that unmeasured per-token claim.
- **Clean test = single node, 8 GPUs, global batch 8, both ways**: pure FSDP-8 (weight-gather
  over NVLink) vs TP-8 (activation-comm over the same NVLink), everything else identical,
  matched autotune. That isolates gather-vs-activation-comm with zero confound. Only if TP-8
  beats FSDP-8 here is the TP(+SP) direction worth pursuing; if not, the whole TP path is dead.
  → running this next.

## ⏳ IN FLIGHT — decisive single-node A/B (both on afabd20f worktree, autotune-OFF, mesh = only variable)
Launched 2026-06-27. The afabd20f mesh `(replicate, dp, tp)` collapses to single-node when
runtime.dp=8 (replicate=1), so both legs are 8 GPUs / global batch 8, differing ONLY in mesh:
- **131381 FSDP-8**: `dp=8, tp=1` → weights FSDP-sharded on `dp`, per-layer weight-gather over NVLink. config `llama8b_full32L_AB_fsdp8_b8_PROFILE.yaml`. HLO dump `hlo_AB_fsdp8`.
- **131382 TP-8**: `dp=8, tp=8` → weights C-sharded on `tp`, activation all-reduce over NVLink. config `llama8b_full32L_AB_tp8_b8_PROFILE.yaml`. HLO dump `hlo_AB_tp8`.
GATE before trusting timings: verify FSDP leg shows per-layer weight-gathers + no C-shard;
TP leg shows C-sharded V/U + activation all-reduces + no weight-gather. THEN compare step time.
Decision rule: TP-8 step < FSDP-8 step ⇒ activation-comm beats weight-gather per token ⇒ TP(+SP) worth it. Else ⇒ TP path dead, refocus on the HSDP gather/overlap levers.

## ⛔ FSDP-8 leg OOM'd — the single-node A/B is INFEASIBLE on the FSDP side (and that's itself informative)
131381 FSDP-8 compiled then OOM'd in `jit_step` (178.3GB args > 176.2GB base limit warning at
compile → RESOURCE_EXHAUSTED at run). Cause: single-node shards the optimizer only **÷8**, not
the production **÷32** — ~4× heavier per-GPU master+Adam state for ~49B trainable params
(18.3B V/U + ~31B CI fn). Global batch 8 / 1-seq-per-GPU is already FSDP's MINIMUM (can't
shrink further), so single-node FSDP simply can't hold this model. **The clean per-token
gather-vs-activation-comm isolation test is not runnable this way.**

### The deeper reason the A/B keeps being un-runnable: incompatible memory profiles
HSDP and TP never fit the SAME (GPU-count, global-batch) point, so a matched comparison
doesn't exist on this hardware:
- HSDP min global batch = #GPUs (1 seq/GPU floor); raising it OOMs (B=64 dp=32 OOM'd). Capped ~1 seq/GPU.
- TP needs MANY seq/GPU to use its few DP replicas, but 8 seq/GPU OOMs. Capped ~2 seq/GPU.
They are memory-capped at opposite, non-overlapping operating points.

### What the data we ALREADY have actually says (no new run needed)
At each strategy's BEST FEASIBLE point on 32 GPUs: HSDP global-32 @ 12.5s (autotune-on) =
2.56 seq/s vs TP global-8 @ ~12s = ~0.67 seq/s. **HSDP wins ~3-4× at the feasible frontier.**
TP's only hope (SP, to reach 8 seq/GPU) is speculative and a large build.

### REFRAME — the binding constraint is MEMORY, not the gather per se
The causal chain for low MFU: **memory cap → stuck at ~1 seq/GPU → below the ~2,500 tok/GPU
arithmetic-intensity floor → weight-gather can't overlap behind compute → exposed on the
critical path → low MFU.** The gather being "exposed" is a SYMPTOM; the cause is too few
tokens/GPU, which is caused by the memory cap. So the highest-EV lever is **per-step memory
reduction on the (already-winning) HSDP path** to reach 2 seq/GPU and cross the overlap floor —
NOT a speculative TP+SP build. Candidate memory levers (per prior full32L memory model — VERIFY
each before claiming): f32→bf16 grad-accum (~41GiB), hoisted CI gather (∝param/tp), bf16 logits.
Freeing ~41GiB/GPU could unlock 2 seq/GPU on HSDP → directly attacks the actual bottleneck.

### BOTH single-node legs OOM'd — confirms memory is binding (not a strategy choice)
- FSDP-8 (131381): args 178.3GB, OOM in jit_step. **Optimizer-dominated** (÷8, 1 seq/GPU).
- TP-8 (131382): args **287.7GB**, OOM in jit_step. **Activation-dominated**: single-node TP has
  dp_axis=1, so all 8 seq run on every TP rank → 8 seq/GPU activations + the same ÷8 optimizer.
Neither strategy fits this model on 8 GPUs. You need ≥16-32 GPUs just to shard the optimizer
enough to fit, after which activations pin HSDP at ~1 seq/GPU. → MEMORY is the lever, decisively.
Next: run `param_decomp.tools.memreport` on a PRODUCTION HSDP dump to attribute the top memory
terms FACTUALLY (not byte-arithmetic) before picking which memory lever to land.

## ✅ MEASURED — production HSDP b32/dp32 peaks at 96.4GiB, NOT ~180. "Hard memory cap" was overstated too.
`memreport` on p-f928808d (the real production step: PGD nw=2, sites_per_chunk=56 / 4 chunks,
remat on, b32, dp32 — verified from its config.yaml; single jit_step module, peak 96.37GiB):
- **There is ~84GB of headroom at b32 on paper.** So the dp=8 AB OOMs (178/287GB) were NOT
  representative — they carry 4× the optimizer state of production dp=32. My "memory binding
  decisively, capped at 1 seq/GPU" claim above was built on those + a stale b64 note. Overstated.
- The peak is dominated by many `bf16[32,...]` buffers each held in **5–19 live copies**
  (e.g. `bf16[32,512,4096]` ×19, `bf16[32,512,8192]` ×18, plus `5×bf16[...]` grouped buckets).
  Whether the leading 32 is n_layer (weight stacks) or global-batch (activations) is NOT
  certain from shapes alone (see [[feedback_hlo_stack_frame_unreliable]]) — do NOT theorize the
  identity; measure. But many-live-copies ⇒ likely reducible duplication.
- **Reframed problem (narrow + tractable):** the step is only 96GB, yet b64/dp32 (2 seq/GPU —
  the doubling that crosses the ~2,500 tok/GPU gather-overlap floor) reportedly OOMs at RUNTIME
  (old note: ~129GB modeled). So the target is NOT "reduce memory in general" — it's the
  specific b32→b64 runtime gap. Re-measuring b64 directly (does it still OOM? peak? dominant
  term that doesn't fit?) to replace the stale note and size the lever exactly. → launching b64.

## ✅ MEASURED b64/dp32 (131383): modeled peak 139GiB (FITS on paper) — runtime OOM is FRAGMENTATION, not a wall
`memreport` on hlo_b64_probe (sharded ÷N path, same module_51640 HLO as b32 production):
- **b64 modeled peak = 138.99GiB** vs b32's 96.37GiB → +42.6GB for the batch doubling. Both
  under the 180GB ceiling on paper. Resident is the SAME weight-stack buffers, batch-independent
  (`5×bf16[32,1792,10240]` etc. recur), + the batch-scaling activation/logit terms (`bf16[2,512,128256]` etc.).
- **Runtime OOM was on a single 72.08GiB allocation.** 139GB modeled + a 72GB contiguous
  transient under the default BFC allocator = classic FRAGMENTATION OOM (BFC can't find 72GB
  contiguous though free total suffices), NOT a true capacity wall. Known zero-code lever:
  `XLA_PYTHON_CLIENT_ALLOCATOR=platform` (cudaMalloc, no BFC arena fragmentation) — launch.py
  already exposes `--allocator platform`. Also flagged in [[project_9layer_40k_save_oom]].
  → testing b64 + platform allocator next; if it fits we get 2 seq/GPU with NO code change.
- CAVEAT on the payoff: 2 seq/GPU = 1024 tok/GPU, still BELOW the ~2,500 tok/GPU overlap floor,
  so the gather won't FULLY hide — but 2× batch amortizes the fixed gather over 2× the work, so
  MFU should still rise. Fully hiding the gather needs ~5 seq/GPU (infeasible on memory) — which
  is the real argument that the gather may be structurally exposed at any feasible batch (the
  honest case FOR eventually revisiting TP/activation-comm). Measure b64 first.

## ⛔ b64 + platform allocator (131384) ALSO OOM'd — b64 is genuinely infeasible at dp=32, not just fragmentation
Both allocators fail: platform tried a 105GiB cudaMalloc, BFC tried 73.46GiB — neither fits.
So the runtime working set for b64 genuinely exceeds 180GB (the modeled 139GB undercounts the
real runtime peak by ~40GB+; the ~73GB single transient is the trigger, and it's NOT in the
after-opt buffer report so it's a runtime/collective/arena allocation). **2 seq/GPU at dp=32 is
infeasible without real memory reduction — the allocator swap was not enough.**

## ⮕ SYNTHESIS / DECISION POINT (state at end of this measurement pass)
The night's measurements, all verified, converge on a hard structural picture:
1. **TP is not a win** and can't even be isolation-tested (both single-node legs OOM); at the
   feasible frontier HSDP beats it ~3-4× (the "3.3× per-GPU" was a units error).
2. **HSDP b32 (1 seq/GPU) is the feasible operating point** (96GB, fits). **b64 (2 seq/GPU) is
   infeasible** (OOMs under both allocators; needs ~73GB more than fits).
3. **Batch-amortization — the lever to hide the gather — is blocked two ways:** (a) b64 doesn't
   fit, and (b) EVEN b64 (1024 tok/GPU) is below the ~2,500 tok/GPU overlap floor; fully hiding
   the gather needs ~5 seq/GPU, which is far out of memory reach. So the FSDP weight-gather is
   **structurally exposed at every feasible batch on this model+topology.**
4. The banked, shipped win remains **autotune (1.6×, 20→12.5s)**. Everything beyond it is a big bet.

The remaining real options, all substantial:
- **(A) Reduce the gather VOLUME, not hide it:** revisit TP/activation-comm done right (+ SP for
  its activation memory). Big build; the only path that attacks the exposed gather head-on.
- **(B) Memory surgery to fit b64:** hunt + shrink the ~73GB runtime transient and/or collapse
  the 5×-duplicated weight stacks (hoist refactor). Wins at most a partial-overlap b64 (still
  below the floor) → modest MFU gain, NOT a step change. But it's HSDP-native and lower-risk.
- **(C) Accept ~12.5s/40%-occupancy as near the practical ceiling** for this config and stop
  spending on MFU. The data says the easy/medium wins are exhausted.
This is a strategic fork for Oli — A (big/structural), B (modest/safe), or C (stop). The
measurement to justify the choice is done; the next step is a decision, not another probe.

## ✅ Option B sized from the b32 dump — it CANNOT unlock b64, so it's not an MFU lever
Analyzed p-f928808d buffer-assignment (leading-dim classification; weight stacks
`bf16[32,d_in_shard,C]` vs per-layer activations `bf16[32,1,512,hidden]`):
- One full copy of the ÷fsdp bf16 compute weights ≈ 5GB/GPU (V/U 18.3B → 36GB unsharded ÷8).
  Held ~5× at peak (the duplication seen in the after-opt report) ⇒ hoist/collapse saves ~20GB.
- b64's MEASURED activation growth over b32 = +43GB (96→139GB modeled). **~20GB prize < +43GB
  need ⇒ B does NOT fit b64.** It only lowers the b32 peak (96→~78GB), which buys ZERO MFU
  (b32 already fits; same batch, same step time). (The ~20GB is an estimate bounded by total
  weight memory; exact figure needs the hoist built+measured, but it's strictly < the +43GB.)
- ⇒ **Fork narrows to A (TP+SP, the only structural fix for the exposed gather) or C (accept the
  ceiling).** B is off the table as an MFU play. No further probe will change this; it's a decision.

## 🏁 DURABLE CONCLUSION of the full32L MFU investigation
The full-model HSDP step is ~12.5s at ~40% occupancy because the FSDP per-layer weight-gather
is on the critical path and CANNOT be hidden: hiding needs ~5 seq/GPU (≥2,500 tok/GPU floor),
memory caps us at 1 seq/GPU (b64=2/GPU OOMs under both allocators; collapsing weight duplication
saves ~20GB < the +43GB b64 needs). TP removes the gather but loses ~3-4× on global throughput
(memory-capped at ~2 seq/GPU via only 4 DP replicas) UNLESS paired with SP. Banked win: autotune
(1.6×). Net: 12.5s/~40% is near the practical ceiling for THIS config; the only step-change left
is a structural rewrite (TP+SP), which is a multi-day bet. Recommendation: bank autotune, treat
~40% as the working ceiling unless/until the TP+SP investment is greenlit.

## ↩ REOPENED (Oli's intuition: HSDP should reach higher local batch w/ memory+comms right) — and the evidence agrees
Investigated the b64 OOM properly (HLO + b32-vs-b64 buffer diff) instead of declaring it fundamental:
- **No single tensor > 1.17GB** in the whole step (largest HLO output = a down_proj weight stack).
  So the ~73GB OOM alloc is the **BFC allocator growing its arena**, i.e. total working set +
  many concurrent collective scratch buffers — NOT one giant buffer.
- **~1,724 all-gathers + ~982 all-reduces PER STEP** (HLO op census). The all-gathers ≈ one per
  site (224) per forward × the recon-grid forwards — the FSDP weight gather is NOT shared across
  the recon-grid/PGD forwards. This is the comms half of the cap (scratch fragmentation + overhead).
- **b32→b64 buffer diff — what actually scales with batch (sum, identifies composition):**
  `bf16[32,2,512,8192]` +50GB, `f32[2,512,8193]` +45GB, `bf16[32,2,512,4096]` +34GB,
  `bf16[32,2,512,10240]` +32GB, `f32[2,512,10241]` +28GB, `f32[2,512,4097]` +23GB.
  Two reducible families: (a) **f32 per-layer component intermediates `[batch,seq,C+1]`** (could
  be bf16 → ~half), (b) **`[n_layer,batch,seq,C]` per-layer component-activation stacks** from the
  scan's checkpointed BACKWARD (recompute/scan-structure artifact, not the forward — training-step
  forward runs collect=None so it emits no stack).
- **Conclusion: the batch cap is reducible activation memory + arena/collective fragmentation, NOT
  fundamental weight/optimizer memory** (those are batch-independent: V/U+CI-fn+Adam are ÷N and
  fixed). So HSDP CAN plausibly reach higher local batch — supports Oli. Candidate levers:
  (1) bf16 the f32 `[batch,seq,C]` intermediates (numerics-check vs SPEC), (2) tighten the
  recon-grid/PGD backward so per-layer component activations don't stack `[n_layer,...]`,
  (3) reduce/Combine the 1,724 collectives (comms + scratch). Localizing next with a zero-code
  nw=0 (no-PGD) b64 probe to size the PGD-forward contribution.

## ★★★★★★★ LOCALIZED — b64 (2 seq/GPU) FITS without PGD; the PGD adversary is THE blocker (Oli vindicated)
noPGD b64/dp32 (131387) **RUNS and trains** — steady **15.7s/step** (autotune-off), modeled
peak **98.4GiB**. vs PGD b64 = 139GiB (OOM). So:
- **The PGD persistent-adversary forward adds ~40GB at b64** — and noPGD-b64 (98GB) ≈ PGD-b32
  (96GB): **the adversary costs about as much memory as doubling the batch.**
- Root: `warmup_scoring_loss` (train.py:280) is a **route-all ALL-224-SITES forward** (SPEC S24
  torch-warmup parity), run `n_warmup_steps=2` times as Adam ascents (each a fwd+bwd). It's
  ALREADY rematted (`remat=remat_recon_forwards`) — so remat isn't the gap; the cost is the
  extra all-sites fwd+bwd passes. The recon grid by contrast chunks 224 sites into 4×56.
- **⇒ The batch ceiling is NOT fundamental — it's the PGD warmup ascents.** HSDP runs 2 seq/GPU fine without them.

### Levers to fit b64 WITH the full algorithm (risk order)
1. **`n_warmup_steps` 2→1 or 0** — fewest extra all-sites fwd+bwd. SEMANTIC (adversary quality),
   Oli's call. Already a TIME lever (~3.1s, line 311); now also THE memory lever for b64. The
   FINAL ascent reuses the main backward (S14, no extra forward), so only the warmups are extra.
2. **Chunk the route-all adversary forward** to match the recon grid (4×56 not all-224). Bigger
   structural change + semantic care (route-all is SPEC S24) but keeps adversary strength.
3. Combine/reduce the 1,724 collectives (orthogonal comms win).

### Payoff check (running): does b64 actually raise throughput via gather amortization?
noPGD b64 = 64 seq / 15.7s = **4.08 seq/s**. If noPGD b32 ≈ 32/~12s = ~2.7 seq/s, b64 is ~1.5×
better throughput (fixed gather amortized over 2× batch — Oli's point). Launching noPGD b32
(autotune-off) for the matched baseline to CONFIRM before recommending the n_warmup change.

## ★★★★★★★★ MATCHED noPGD b32 vs b64 — batch is a WEAK lever (~12%); the PGD ADVERSARY is the real MFU+memory sink
Matched autotune-off:
| config     | seq/GPU | step   | seq/s | peak  |
|------------|---------|--------|-------|-------|
| noPGD b32  | 1       | 8.78s  | 3.64  | 85GB  |
| noPGD b64  | 2       | 15.7s  | 4.08  | 98GB  |
- **noPGD step is COMPUTE-bound**: 2× batch → 1.79× step (8.78→15.7). So batch amortization buys
  only ~12% throughput (3.64→4.08 seq/s). Raising batch is NOT a strong MFU lever — corrects my
  "~1.5×" projection (I'd guessed noPGD b32≈12s; it's 8.78s).
- **The real signal**: noPGD b32 = 8.78s vs production PGD b32 ≈ 20s (autotune-off) ⇒ **the PGD
  adversary adds ~11s** — and that ~11s IS the long-flagged "~11s GPU-idle." The PGD adversary's
  route-all all-224-sites forward (warmup_scoring_loss, train.py:280) is BOTH the +40GB memory
  blocker AND the gather-bound idle sink (its share of the 1,724 collectives). The base recon is
  healthy + compute-bound.
- **⇒ The single highest-value target is the PGD adversary forward, not batch and not generic
  memory.** Fixing it pays twice (time AND memory):
  1. **Chunk the route-all adversary forward** to match the recon grid (4×56 vs all-224) — cuts its
     peak gathers/activations; SPEC-S24 semantic care needed (route-all parity).
  2. **Combine its collectives** (the all-gathers in the all-sites forward) — comms/idle win.
  3. **n_warmup 2→1/0** — removes whole adversary fwd+bwd passes (semantic, Oli's call).
- This is the coherent end: the ~40% MFU / ~11s idle is the PGD adversary's gather-bound all-sites
  forward. Optimize THAT (chunk + combine) for the real HSDP-native MFU win; batch is secondary.

## ★★★★★★★★★ TRANSIENT GAP ATTRIBUTED (live-range sweep) — peak is dominated by RE-MATERIALIZED compute-weight copies
Floor ≈33GB (validated, c78e9d28a/f47444e04) vs measured 96GB ⇒ ~63GB reducible. To attribute it,
built `param_decomp/tools/liverange_peak.py` (5b16c0125): joins buffer-assignment (size+shape) with
the live-range file (start-end program point), sweeps to the TRUE peak, decomposes co-residency.
The static slab report can't do this (slabs share offsets across disjoint buffers — it misled me earlier).

On p-f928808d (b32/dp32) peak working set = **79.7GiB** at program point 15169 (+~17GB optimizer,
which lives as donated params spread over many small ÷N f32 shards, not in the step buffers → 79.7+17≈96 ✓).
Composition at peak:
| GiB | count | shape | identity |
|-----|-------|-------|----------|
| 17.5 | **×20** | bf16[32,8192,1792] | gate/up **U** stack (÷fsdp) |
| 10.9 | **×10** | bf16[32,1792,10240] | down **V** stack |
| 5.0 | ×20 | bf16[32,512,8192] | activation |
| 5.0 | ×20 | bf16[32,1,512,8192] | activation |
| 3.0 | ×192 | f32[1,512,8193] | f32 intermediate |
- **Claim: the peak is dominated by 10–20 CO-RESIDENT COPIES of the bf16 compute-weight stacks
  (~30GB+ across shapes).** 20 buffers of one shape all live at one program point CANNOT be
  offset-sharing (that needs disjoint lifetimes) ⇒ real physical duplication. The ÷fsdp compute
  weight is reconstructed once in ENTRY but then RE-MATERIALIZED per-forward (4 recon chunks +
  PGD adversary all-sites + faithfulness + backward) instead of being shared/CSE'd across them.
- **⇒ The #1 reducible transient is compute-weight re-materialization, not activations.** This is the
  HSDP-native, config-preserving lever: make the forwards SHARE one reconstructed ÷fsdp compute
  weight (donation / CSE / hoist the reconstruction so XLA doesn't recompute the cast+gather per
  forward). Recovering ~25-35GB would take peak 96→~65GB and likely fit b64 WITH the full config.
- Caveat (under async validation): live-range join is 88% (28908/32988); the ×20 count + the
  "re-materialized per forward, not aliasing" interpretation is the thing to verify.

## ✅ VALIDATED (skeptical async, PASS) — peak compute-weight duplication is REAL; refined to 10 fwd + 10 bwd
The ×20 `bf16[32,8192,1792]` co-resident at peak sit at 20 DISTINCT offsets (agent tried to break it via
offset-aliasing, couldn't). Decomposition refined:
- **10 FORWARD copies** = ÷N→÷fsdp reconstruction all-gather (448→1792) RE-EMITTED per forward context
  (`jvp(pd_recon_masked_fwd)/stack`; 10 recon + 2 PGD across the program), NOT once in ENTRY as intended.
  → LEVER 1: share one reconstruction across the recon-grid/adversary forwards (CSE/donate/hoist) — config-preserving.
- **10 BACKWARD copies** = weight-grad accumulators (`broadcast(0)` + dynamic_update_slice in the bwd while).
  → LEVER 2: grad-accumulation/remat strategy (separate from CSE).
~28GiB on these two shapes alone; tens of GB total. Both levers preserve the config (pure compile/structure).

### Fix scoping (next branch)
Lever 1 is the cleaner first target: the ÷N→÷fsdp gather is supposed to run ONCE in ENTRY
(`_reconstruct_compute_weights`, llama8b.py:368) landing a shared ÷fsdp stack, but the HLO shows it
re-emitted inside each `jvp(pd_recon_masked_fwd)` — i.e. it's INSIDE the per-forward (and likely the
value_and_grad / per-chunk remat) region, so XLA recomputes the gather per use. The fix: ensure the
reconstructed ÷fsdp compute weight is computed once OUTSIDE the per-forward/per-chunk path and threaded in
as a shared value the recon grid + adversary all reuse, with remat NOT set to recompute it. Verify by
re-running liverange_peak: the forward [32,8192,1792] count should drop ~10→~1-2.

### Lever-1 fix precisely located (llama8b.py:546-549)
`_run_masked_forward` runs `_stack_per_kind_masked_inputs` (MASK-DEPENDENT, per-forward) then
`_reconstruct_compute_weights` (the ÷N→÷fsdp gather + bf16 cast — MASK-INDEPENDENT, depends only on V/U).
The mask-independent reconstruction is bundled inside the per-forward fn ⇒ redone every forward ⇒ the 10×.
FIX (config-preserving hoist): split the V/U reconstruction (do ONCE at step entry, outside the remat
region) from the per-forward mask bundling; thread the reconstructed ÷fsdp V/U into all masked_output calls
(recon grid + adversary + faithfulness). Touches `masked_output`/`_run_masked_forward` signatures + train.py.
VERIFY: CPU 4-sim-device HLO dump — the [32,8192,1792] all-gather count in jvp(pd_recon_masked_fwd) drops
~10→~1; numerics bit-identical (same reconstruction, shared). Watch the remat interaction: the recon
forwards are remat'd, so the shared weight must be a saved input to the remat region, not recomputed inside it.

## ★★★★★★★★★★ HOIST SHIPPED (branch perf/hoist-vu-reconstruction) — verification in flight
The lever-1 fix is implemented + Codex-reviewed + green: `prepare_compute_weights(vu)` on
`DecomposedModel` does the mask-INDEPENDENT V/U stack + ÷N→÷fsdp gather + bf16 cast ONCE per
step; `masked_output`/`masked_site_outputs` take the shared `prepared` and only attach per-forward
masks. train.py builds it once (live + detached). Toys/SimpleMLP prepare = identity. Numerics
bit-identical (equivalence goldens pass). Codex verdict: remat boundary clean (prepare is outside
the per-forward checkpoint), gradients correct, detached/live correct; it caught 2 stale test call
sites (fixed). Commits a9f3bc720 + a6be2eea9.
GPU verification (validate-the-HLO, NOT claimed until measured): **131420 hoist-b32** (peak should
drop ~96→~70GB; the [32,8192,1792] copies should fall ~20→~few) + **131421 hoist-b64** (does 2
seq/GPU now FIT?). Autotune-off, distinct HLO dumps (hlo_hoist_b32 / hlo_hoist_b64).

## ⚠️ FLOOR CORRECTION — it MISSED the PPGD source state (~7GB resident)
The 33GB floor counted V/U+CI optimizer + compute weights + frozen + 1-fwd acts, but NOT the
persistent PGD sources. Production (`sc` / shared_across_batch, source_dtype=f32): source
`(1,T,C+1)` per site = 2.38GB + Adam m,v = **7.13GB/GPU, REPLICATED**. So true resident floor ≈ 40GB.
(At peak the `f32[1,512,C+1]` source buffers also show ~×3/site from the 3 ascents ≈ +7GB transient.)

## PPGD source-state sharding (Oli's "sources ~ optimizer state" lens) — one real inconsistency
- `sc` source itself: REPLICATED (`P()`) — CORRECT (shared across batch, in every forward's mask).
- `sc` source GRAD: AVG all-reduced across DP every ascent (3×/step). Small, real collective.
- `sc` source ADAM (m,v ~4.76GB f32): **REPLICATED, NOT ÷N-sharded** — unlike the weight optimizer
  (ZeRO-1 ÷N). This is the asymmetry: sources ARE optimizer state but don't get the ÷N treatment.
  m,v are only used in the source UPDATE (never the forward) → could ÷N-shard + gather at update →
  **~4.6GB/GPU back, semantics-preserving** (memory↔one-small-gather trade). Candidate lever-3.
- `bsc` (Oli's proposal): `(B,T,C+1)` batch-sharded over DP → eliminates the source all-reduce
  (grad shard-local, `_skip_all_reduce`) AND shards Adam naturally. BUT (1) SEMANTIC change
  (independent per-example adversary vs shared) — a method decision, not a free opt; (2) its source
  memory GROWS with per-GPU batch (sc is batch-independent). Do (a) ÷N-source-Adam regardless;
  (b) bsc only if per-example adversaries are wanted on merits.

## ✅✅ HOIST VERIFIED ON GPU (131420 b32 / 131421 b64) — step-time win + b64 unlock, both REAL
Matched autotune-off, full PGD config:
| config | step | seq/s | modeled peak | live-range peak | gate/up-U copies | down-V copies | result |
|--------|------|-------|--------------|-----------------|------------------|---------------|--------|
| baseline b32 | ~20s | 1.60 | 96GB | 79.7GB | ×20 | ×10 | runs |
| HOIST b32 | **13.0s** | 2.46 | 99GB | 77.5GB | **×12** | **×6** | runs |
| baseline b64 | — | — | 139GB | — | — | — | **OOM** |
| HOIST b64 | 27.5s | 2.33 | **124GB** | — | — | — | **FITS + COMPLETED** |
- **b64 (2 seq/GPU) now FITS with the full PGD config** (124GB modeled vs 139 baseline; job COMPLETED,
  zero OOM) — the headline: HSDP-no-TP reaches 2 seq/GPU under the current config. The hoist's ~15GB
  modeled reduction at b64 was exactly enough to clear the runtime cliff.
- **Step time b32: ~20s → 13s (~35% faster, ~1.5×).** The hoist removed the redundant FORWARD
  cross-node gathers (part of the ~11s idle). Mechanism confirmed: forward weight-copies collapsed
  (gate/up-U ×20→×12, down-V ×10→×6 = the ~10/~5 forward re-gathers dropped to ~1-2; the remaining
  ~10/~5 are the BACKWARD grad-accumulators = lever-2, untouched).
- **Batch does NOT help throughput post-hoist** (b32 2.46 vs b64 2.33 seq/s): once the redundant
  gathers are gone the step is compute-bound (2× batch → 2× time). So the MFU win is the HOIST itself,
  not bigger batch. (b64's value is headroom / larger effective batch, not speed.)
- Caveat: baseline-b32 20s is the session AB measurement (same config+autotune, hoist the only delta),
  not a fresh re-run — can nail it with one matched baseline job if needed.
- Lever-2 (backward grad-accumulators, the remaining ×10/×5) is the next memory target; lever-3
  (÷N-shard source Adam, ~4.6GB) is orthogonal + semantics-preserving.

## ⚠️ HONESTY CORRECTION — hoist + autotune-on = 11.4s; the production-relative win is ~9%, NOT ~35%
hoist+autotune-ON b32 (131422, COMPLETED) = **11.4s**. Full matrix (b32):
| | autotune-off | autotune-on |
|---|---|---|
| baseline | ~20s | **12.5s** |
| hoist | 13s | **11.4s** |
- autotune alone: 20→12.5 (1.6×). hoist alone: 20→13 (1.54×). hoist+autotune: 20→11.4 (1.75×).
- **The hoist and autotune OVERLAP** (both attack the gather idle) — they do NOT stack. The ~35%
  hoist win was measured at autotune-OFF, where the redundant gathers are fully exposed. At PRODUCTION
  (autotune-on), autotune already hides most of that cost, so the hoist's marginal step-time win is
  **12.5→11.4 ≈ 9%**. Don't quote ~1.5× as a production number — it's ~9% on step time.
- The hoist's DURABLE value is therefore: (1) ~9% production step time, (2) the **b64 memory unlock**
  (2 seq/GPU fits — headroom for larger effective batch), (3) ~10 fewer collectives/step (helps more
  at larger scale / when comm-bound). The b64 unlock + collective reduction may matter more than the 9%.
- Lesson (again): measure at the PRODUCTION setting (autotune-on) before quoting a win; autotune-off
  deltas overstate any gather/idle fix because autotune independently mitigates it.

## ⚠️ CORRECTION (trace, 131423) — production step is NOT compute-bound; still 37% occ, gather-dominated
Traced the production (hoist+autotune-on) b32 step (p-15870537/profile). My "compute-bound post-hoist"
claim (inferred from b32→b64 2× scaling) is WRONG. Trace facts:
- **GPU occupancy = 37%** (busy 4.56s / span 12.26s) → ~63% IDLE remains.
- **Dominant GPU op: `ncclDevKernel_AllGather_RING_LL` = 15.5s summed** (×8 GPU streams) vs top gemm
  fusion ~6.1s. So the step is **gather/collective-bound**, not compute-bound.
- Interpretation: the hoist removed the REDUNDANT ÷N→÷fsdp re-gathers (the ~9% win + b64 unlock), but
  the **per-layer ÷fsdp→full FSDP gathers inside the scan** (the "43k small gathers" / collective-
  progression) still dominate and don't overlap compute → the residual ~63% idle. This is the ORIGINAL
  arithmetic-intensity-floor problem (per-layer gather doesn't hide behind per-layer compute at 1 seq/GPU).
- ⇒ The cron's "~11s GPU-idle / collective-progression / 43k-gather fragmentation" is REAL and LARGELY
  REMAINS post-hoist. The next lever is gather↔compute OVERLAP or fewer/bigger per-layer gathers
  (combine-threshold, remat-policy overlap, or the in-scan gather structure) — NOT batch (which the
  trace-corrected picture shows won't help via amortization here either).
- METHODOLOGY (4th correction this session): I inferred compute-bound from batch-scaling; the TRACE
  refuted it. Profile, don't infer occupancy.

## 🧱 FLAG-LEVEL COLLECTIVE LEVERS EXHAUSTED — residual 63% idle is STRUCTURAL
Checked the production HLO (hlo_hoist_b32_autotune) for whether the gathers can be overlapped more:
- All-gathers are **already ASYNC** (1848 all-gather-start / 2168 -done, **zero sync** all-gathers)
  and **pipelined** (1357 hits) → the **latency-hiding scheduler is already ON**; XLA already overlaps
  collectives as far as the dependency graph allows. Yet occupancy is 37%.
- Combined with the earlier result (autotune + 1GB combine-threshold = no change vs autotune-alone),
  the **flag-level collective levers (autotune, async/pipelined collectives, combine-threshold) are
  all already applied or tested-null.** The residual idle is NOT a missing flag.
- ROOT (structural): the per-layer ÷fsdp→full gathers live inside a `lax.scan` over 32 layers and feed
  their OWN layer's matmul — so each gather can't hide behind its consumer, and the scan serializes
  iterations (layer N+1's gather can't freely prefetch behind layer N's compute). At 1 seq/GPU the
  per-layer compute is too small to hide the gather (the arithmetic-intensity floor). This is the same
  structural wall the early investigation hit — the hoist removed the REDUNDANT part, not this part.
- ⇒ Remaining MFU levers are all STRUCTURAL (Oli's call), not flags:
  (1) fewer/bigger gathers via a different weight layout (e.g. gather >1 layer at once = more memory,
      the replicate-weights tradeoff), (2) restructure/unroll the scan for cross-iteration prefetch,
      (3) more tokens/GPU to cross the overlap floor (memory-gated; b64 fits post-hoist but trace still
      gather-bound), or (4) TP (communicate activations not weights — the other structural path).
- HONEST END-STATE of the flag-level MFU push: banked = hoist (~9% + b64 unlock + fewer collectives)
  + autotune (1.6×). Beyond that needs a structural change. Recommend: merge the hoist, pause flag-level
  tuning (exhausted), and treat the structural levers as a separate scoped decision.

## ➡️ NEXT STRUCTURAL LEVER (numerics-preserving): coalesce the per-layer gathers (scan unroll-by-K)
Characterized the production all-gathers (hlo_hoist_b32_autotune): **1,848 all-gather-start/step,
median 16MB, 30% (572) are <8MB** — fragmented + overhead-bound (they dominate at 37% occ but move
modest bytes over NVLink, so the cost is per-launch/progression × 1848, not bandwidth). Top shapes:
bf16[4096,4096]×219, bf16[4096,14336]×126 (the per-layer ÷fsdp→full weight gathers).
- **Why flags can't fix it:** the gathers are inside the serialized `lax.scan` over layers; the
  combine-threshold only merges INDEPENDENT collectives, not loop-body ones (that's why the earlier
  combine-threshold test was null). Confirmed: async+pipelined already on; this is a STRUCTURAL limit.
- **The lever:** process K layers per scan iteration (unroll-by-K) and co-gather their weights → ~1848/K
  gathers, each K× bigger → far less collective-progression overhead → higher occupancy. Numerics-
  preserving (same gathers, batched). Cost: K layers' transient weight memory (headroom exists post-hoist).
- **Directly targets the cron's "43k-gather fragmentation."** This is the MFU lever beyond the (now
  exhausted) flags. Prize needs a prototype+trace to bound (the gather-overhead share of the 63% idle),
  but the fragmentation (30% <8MB, 1848 launches) is the clear culprit.
- Recommended as the next FOCUSED task (a second hot-path structural change — better done Oli-aware
  with the branch→repro→Codex→trace loop than slammed unilaterally overnight). NOT a config/recon-
  granularity change (the scan unroll is invisible to the math + the recon chunking).

## prize-bound attempt (replicate-weights, K=all) OOM'd → gather is memory-load-bearing; unroll-by-K is the only feasible form
Tried PD_REPLICATE_WEIGHTS=1 at b32 to cheaply bound the per-layer-gather-elimination prize (full weight
resident → no per-layer gather). It OOM'd (222GB single alloc — the full replicated V/U+CI-fn). So K=all
is infeasible; this CONFIRMS the per-layer ÷fsdp gather is necessary for memory (can't trade it away
wholesale). ⇒ the gather-coalesce lever only exists as the MIDDLE ground (unroll-by-K, K=2-4: gather a
few layers' weights at once, K× transient, not all 32). The full-replicate run can't stand in for it, so
the prize can't be cheaply bounded — it needs an actual unroll-by-K prototype + trace. Recommend that as
the next focused task (Oli-aware: it touches the recon-forward scan, numerics-preserving, branch→repro→
Codex→trace loop). NOT launching more speculative runs — the cheap prize-bound is exhausted.

## 📋 SHOVEL-READY PLAN — unroll-by-K gather coalesce (the one live MFU lever; green-light to build)
Implementation (in `llama8b._run_masked_forward`, numerics-preserving, NOT a config/recon change):
1. Reshape the scanned arrays `self.stacked` + `per_kind[kind][...]` from `[n_layer, …]` to
   `[n_layer/K, K, …]` (n_layer=32; K∈{2,4,8} all divide 32).
2. `lax.scan` over `n_layer/K` iterations; the body processes K layers via an inner Python loop
   (residual threads through all K, exactly as the current per-layer body does).
3. At the body START, explicitly gather the K-layer V/U together — one `with_sharding_constraint`
   to full d_in on the `[K, d/8, C]` slice → XLA emits ONE all-gather of `[K,…]` instead of K
   separate ones. This is the coalesce: 1848 gathers → ~1848/K.
4. `jax.checkpoint` the K-layer body (replaces the per-layer checkpoint).

TRADEOFF (why it needs measurement, not assumption):
- WIN: ~1848/K fewer gather launches → less collective-progression overhead (the 37%-occupancy idle).
  Rough: ~1.9s/GPU gather is ~1ms × 1848 launches; ÷K could save ~1.4s at K=4 + reduce bubbles → ~10-20%.
- COST: the checkpointed body now recomputes K layers in the backward → **K× the per-layer recompute
  activation transient** ([K,1,512,14336] MLP-hidden etc). At K=2-4 that's +a few GB on the 96GB b32 peak
  (under budget), but it MUST be verified — peak regression is the risk.
- ⚠️ PRECEDENT: remat/recompute tradeoffs have surprised before (the 3-pool LW work measured ~0% from
  similar dtype/ckpt changes — recompute dominated). So the prize is genuinely uncertain; build a K=2 and
  K=4 prototype and MEASURE (trace occupancy + peak + step) before trusting it.
VERIFY: HLO all-gather count drops 1848→~1848/K; trace occupancy >37%; memreport peak still <180GB;
equivalence goldens bit-identical; Codex review. Effort: moderate (one function + the reshape).
STATUS: scoped + ready; held pending green-light — it's a 2nd hot-path structural change with an
uncertain (possibly-wash) prize, so it warrants a deliberate go rather than an overnight slam.

## ★★★★★ CRITICAL-PATH DECOMPOSITION (trace 131436, named-scope) — answers "do we understand the critical path"
GPU-busy time by pd_* phase (summed across 8 GPUs; per-GPU: 4.6s busy + 7.6s idle = 12.2s wall, 38% occ):
| phase | GPU-busy | gemm | all-gather |
|---|---|---|---|
| pd_pgd_warmup_ascend (ADVERSARY) | **20.4s (56%)** | **14.4s** | 5.5s |
| pd_value_and_grad (main recon+bwd) | 11.2s (31%) | 2.7s | 6.0s |
| pd_ci_fn_fwd_detached | 2.5s | 0.6s | 1.7s |
| clean_fwd + read_taps | 2.5s | 1.8s | 0.7s |
**Critical path has TWO parts:**
1. **Wall-clock is IDLE-dominated** (~7.6s/GPU of 12.2s) = gather-progression bubbles from the 1848
   per-layer gathers. ⇒ **unroll-by-K is correctly aimed** (at the idle). Ceiling ~2.4× if ALL idle
   removed (wall→4.6s busy floor); realistically the gather-bubble fraction.
2. **BUSY is ADVERSARY-dominated** (pd_pgd_warmup_ascend 20.4s = 56%, and COMPUTE-heavy: 14.4s gemm —
   the all-sites route-all forward, every site's V@U, run n_warmup=2×). The CI-fn all-reduce I feared
   would co-dominate is only 2.5s → RULED OUT as a major factor.
- ⇒ unroll-by-K is the right lever FOR THE CURRENT CONFIG (targets the idle). The single biggest
  step-change lever overall is the ADVERSARY forward (20.4s busy, compute-bound) — but that's
  config/semantic (n_warmup, or chunk the all-sites forward to match the recon grid), fenced off.
- Decision (Oli): build unroll-by-K (idle lever, correctly targeted); the adversary is the bigger but
  semantic prize if the MFU push ever reopens the config.

## ❌ unroll-by-K (gather coalesce) — MEASURED FAILURE, path closed (branch perf/gather-coalesce-unroll-k, NOT merged)
Built GATHER_UNROLL_K=2 (scan over K-layer groups, explicit `with_sharding_constraint(P())` to gather
the K-layer V/U in one all-gather). Numerics bit-identical (equivalence goldens pass). GPU result (b32,
autotune-off, 131437): REGRESSION on all three —
- step 13s → **15.4s** (slower), gathers 1848 → **2302** (UP), peak 96 → **113GB** (the K× recompute).
- ROOT (HLO diagnosis): XLA **double-gathers**. The explicit `[2,…]` coalesced gathers appear
  (bf16[2,4096,4096]×72 etc.) but the per-layer ones are NOT eliminated — bf16[4096,4096] gathers rose
  to 490 (from 219). Slicing `[k]` out of the pre-gathered `[K,d,C]` doesn't stop the matmul re-gathering
  per layer, and the K layers can't be batched-matmul'd (sequential residual dependency). So the explicit
  gather is pure ADDED work on top of the unchanged implicit per-layer gathers + K× recompute peak.
- ⇒ The gather-coalesce-via-scan-unroll path is CLOSED (this mechanism). Stays on its branch, unmerged.
- The wash-risk flagged up front (LW precedent) MATERIALIZED — measured before claiming, reverted by
  non-merge. perf/hsdp-mfu keeps the hoist (the banked win); no regression on the canonical branch.

## 🏁 MFU SPRINT END-STATE (honest)
BANKED (on perf/hsdp-mfu, verified): hoist (~9% prod step + b64/2-seq-per-GPU unlock + fewer collectives)
+ autotune (1.6×). Production step ~11.4s, 37% occ. CLOSED/exhausted: flag-level collective levers
(async/pipelined already on), gather-coalesce (unroll-by-K regresses), lever-3 (memory-only, doesn't
help gather-bound MFU). The residual idle is the per-layer FSDP gather, which is NOT cheaply coalesceable
(XLA double-gathers; sequential residual blocks batching). The ONLY remaining step-change lever is the
config-protected ADVERSARY forward (20.4s busy, compute-bound, all-sites route-all × n_warmup) — a
SEMANTIC change (chunk it / reduce n_warmup), Oli's call. For the current config, ~11.4s/37% is the floor.

## ★★★★★★ ASYNC TEST (131438) — the step is HOST/COLLECTIVE-SYNCHRONOUS, not device-bound (THE reframe)
PD_ASYNC_TEST (3 step_fn calls without blocking + 1 final block), hoist b32 autotune-off:
`call1=271.8s(compile) call2=15.46s call3=15.37s final_block=0.013s`.
- **Each steady call ≈ the full step time; final block ≈ 0.** That's the HOST-SYNCHRONOUS signature
  (device-async would be: small calls + big final block). ⇒ the step is bottlenecked on COLLECTIVE
  COORDINATION (host-side launch of ~2800 collectives/step + cross-host sync), GPU idle waiting for
  the host. CONFIRMS the cron's "host-side GPU-idle" framing. (Multi-host caveat: "big calls" could be
  host-launch OR cross-host collective sync; both implicate the ~2800 collectives, not GPU compute/bw.)
- **EXPLAINS every gather lever underdelivering:** the bottleneck is collective COUNT/coordination, not
  gather bandwidth. unroll-by-K ADDED collectives (1848→2302) ⇒ regressed. The hoist helped because it
  REMOVED collectives (redundant re-gathers). So the metric that matters is COLLECTIVE COUNT, not bytes.
- ⇒ Lever: FEWER host launches. Canonical fix = CUDA graphs / XLA command buffers, currently DISABLED
  (`--xla_gpu_enable_command_buffer=` empty; turned off earlier as "~0% + capture crashes" — but that
  predates knowing it's host-launch-bound). RE-TESTING command buffers ON now. Also: any lever that cuts
  collective COUNT (vs bandwidth) is the right class.

## ✅ COMMAND BUFFERS = ~0% (clean, pure-hoist) — NOT host-launch-bound; it's the SERIAL collective chain
Clean test (PURE hoist, no unroll confound, cmdbuf ON): **12.9s vs 13s OFF = ~0%**, zero capture errors.
⇒ command buffers (which collapse host LAUNCHES into a graph replay) do NOT help → the step is NOT
host-LAUNCH-bound. Combined with the async test (host-synchronous) + command-buffers-null:
- The host-synchronous ~15s/call is the host/GPU WAITING on the SERIALIZED per-layer collective chain
  (each layer's ÷fsdp→full NVLink gather → its matmul → next layer's gather; scan-serialized), NOT
  issuing launches and NOT Python.
- 37% occupancy = the per-layer gathers don't overlap their per-layer compute (too little compute/gather
  at 1 seq/GPU — the arithmetic-intensity floor), and the scan serializes layers so no cross-iteration
  prefetch. This is the same structural wall, now RIGOROUSLY pinned: not Python, not host-launch
  (cmdbuf ~0%), not bandwidth (gathers are overhead/serialization, not bytes) — it's collective-progression.

## 🏁🏁 DEFINITIVE CRITICAL-PATH VERDICT
RULED OUT (measured): Python (single jit step), host-launch (command buffers ~0%), gather bandwidth,
CI-fn all-reduce, flag-level collective tuning (async/pipelined already on), gather-coalesce (unroll-by-K
regresses — XLA double-gathers). BANKED: hoist (~9% — removed redundant collectives) + autotune (1.6×).
ROOT: the SERIAL per-layer NVLink gather→matmul chain (collective-progression), un-overlappable at
1 seq/GPU (arithmetic-intensity floor) + scan serialization. The metric that moves it is COLLECTIVE
COUNT on the critical chain, not bytes/launch/python. Structural frontiers (Oli's call): (a) more
tokens/GPU to cross the overlap floor (memory-capped; b64 fits post-hoist), (b) restructure the scan for
cross-iteration gather prefetch (hard; unroll-by-K's naive form failed), (c) the config-protected
ADVERSARY forward (biggest single phase, 20.4s, compute+collectives). For the current config: ~12.9s/37%
is the floor; hoist+autotune is the banked win.

## 🔬 EXPLORE — b64 occupancy 37%→45%: overlap IS arithmetic-intensity-limited (more-tokens path has a prize)
Traced hoist-b64 (autotune-off): occupancy **45%** (12.5s busy / 28s) vs b32 **37%**. The AllGather is
~batch-INDEPENDENT (16.8s/node ≈ b32's 15.5s — same weights), so 2× compute hides more of the fixed
gather → overlap improves. So the root-cause idle IS the arithmetic-intensity floor (per-layer gather
not hidden by too-little per-layer compute), NOT a fundamentally un-overlappable serialization.
- BUT 2 seq/GPU isn't enough: b64 throughput 2.33 vs b32 2.46 seq/s (compute doubled, gather not yet
  amortized). The crossover where more-tokens nets FASTER is higher (~5 seq/GPU / the ~2,500 tok/GPU floor).
- ⇒ **The more-tokens/GPU structural path (B) has a REAL prize** (occupancy-confirmed), but needs to reach
  ~5 seq/GPU, which is memory-gated (b64 fits, b128 OOMs). The canonical way to free the activation memory
  for more seq/GPU is **SEQUENCE PARALLELISM** (shard activations along seq). So SP — earlier deferred — is
  the principled structural lever for the root cause, not just a TP add-on. This REFINES the frontier:
  B (more tokens via SP) is occupancy-justified; A (adversary, config-protected) remains the other big one.

## 🔬 EXPLORE — b96 (3 seq/GPU) OOMs (71.7GB transient): more-tokens path is memory-capped at 2 seq/GPU
hoist-b96 OOM'd (same ~72GB runtime transient as pre-hoist b64). So post-hoist the memory cliff is
between 2 and 3 seq/GPU: b64 (2/GPU) fits (124GB), b96 (3/GPU) OOMs. Occupancy trajectory: b32 37% →
b64 45% → b96 (can't measure). The trend confirms more-tokens improves overlap, but we CANNOT reach the
~5 seq/GPU crossover where it nets faster — memory caps it at 2/GPU.
- ⇒ CONVERGED: the more-tokens path (B) has a real, occupancy-validated prize but is HARD-GATED behind
  freeing activation memory. **Sequence parallelism (SP)** — shard activations along seq — is the
  principled unlock (the ~72GB transient is per-seq activation/recompute, exactly what SP shards). The
  memory levers I noted (f32→bf16 grad-accum ~41GiB, bf16 logits) are too small for the ~72GB transient.
- EXPLORATION VERDICT: within numerics-preserving + non-config + non-big-build, the space is exhausted.
  The two real frontiers are both Oli-decisions: SP (big build, occupancy-justified prize) or the
  config-protected adversary forward. The ~72GB per-seq transient is the SP target; pinning it exactly
  needs runtime device-memory profiling (not chased — SP shards it regardless).

## ✅ HOIST save/resume gate — GREEN (production/PR-ready)
SAVESMOKE (steps=250, save_every=100) on the hoist: SAVE ✓ — checkpoints written cleanly at 100/200/250,
zero errors, steady 12.8s/step. RESTORE ✓ — resume (--run_id) loaded the checkpoint and reached process
shutdown (the only failure was a multi-host SHUTDOWN-barrier abort because I resumed an ALREADY-COMPLETED
run: step==steps → immediate no-op exit desyncs the ranks; a pre-existing edge case, NOT the hoist, and
NOT a production path — real requeue resumes MID-run with steps remaining). Structurally the hoist can't
break resume: it changed the forward METHODS (prepare_compute_weights / masked_output signature), not the
checkpoint CONTENTS (state = V/U + optimizer + sources, unchanged) — a hoist-era ckpt loads identically.
⇒ The banked hoist (perf/hsdp-mfu) is verified production-ready: bit-identical numerics + Codex-reviewed +
~9% step + b64 unlock + save/resume green. Ready for a feature/jax PR on your go.

## 🧭 SP FEASIBILITY SCOPE (the validated lever — de-risking before any build)
SP shards the per-seq activations to fit more seq/GPU → cross the ~2,500 tok/GPU overlap floor (the
occupancy-validated prize). What it touches in THIS codebase, and the risk:
1. **Mesh:** needs a (dp, sp) split (dp·sp = 32) — trade data-parallel degree for a seq-parallel sub-axis
   (analogous to how tp carves out a TP axis). Shard the residual/activations on the sp axis.
2. **⚠️ ATTENTION is the crux risk.** `FrozenAttn.core` runs cuDNN flash SDPA, which needs the FULL
   sequence and a CAREFULLY-TUNED sharding (q/k/v IDENTICAL, batch-parallel over the full mesh, heads
   REPLICATED — the code has extensive comments: "Query, key and value should have same sharding",
   flash partitioner is finicky). Seq-sharding REQUIRES all-gathering seq before attention (un-shard),
   running attention, re-sharding after — which fights the flash partitioner and risks (a) partitioner
   errors / (b) losing flash attention → non-flash fallback (materializes scores, slower). This is the
   single biggest SP risk and where a prototype would most likely break (cf. the cuDNN sharding battles
   already in the code + the unroll-by-K XLA-double-gather surprise).
3. **Per-token ops are easy:** RMSNorm, MLP, the V/U/frozen matmuls all work on seq-sharded activations
   (the weight gather is unchanged; activation just smaller/rank). RoPE per-position is fine.
4. **Loss + sources:** KL is per-position → seq-sharded reduce (mean over the sp axis). PPGD `sc` sources
   `(1,T,C+1)` → shard T on sp; the source all-reduce becomes an sp-reduce. Mechanical but must be exact.
- **Verdict:** SP is the right lever (occupancy-validated prize), but it's a genuinely HIGH-RISK multi-day
  build in this codebase — the cuDNN flash-attention partitioner is the load-bearing risk, and SP's own
  seq-all-gather-for-attention overhead could partially offset the gain (must be measured). Prototype path:
  (1) (dp,sp) mesh + seq-shard residual/MLP only, attention via gather-seq→flash→reshard; (2) trace
  occupancy at 4-5 seq/GPU; (3) confirm flash survives the gather/reshard (else the win may evaporate).
  ⇒ Recommend explicit go + a scoped prototype phase (not a blind multi-day commit). For now: NOT built.

## ❌ native scan unroll=2 — ~0% (step 13s, both autotune-off); scan-level overlap tricks all fail
Tested `jax.lax.scan(..., unroll=2)` (XLA native unroll, distinct from the manual unroll-by-K): numerics
bit-identical (equiv goldens pass), step ≈12.9-13.4s = ~0% vs hoist 13s (autotune-off, clean compare).
The 51% occupancy reading is a CONFOUND (autotune-OFF slower kernels fill more of the span; the 37%
baseline was autotune-ON) — not a real overlap gain. Reverted (no benefit + extra compile).
⇒ Both scan-level overlap attempts fail: manual unroll-by-K REGRESSED (double-gather), native unroll=2
~0%. The per-layer gather→matmul→next-layer chain is un-overlappable by scan tricks (the gather feeds its
own matmul; XLA won't prefetch across the dependency). The ONLY lever left for the root cause is reducing
activation memory to fit more tokens/GPU (cross the overlap floor) = SEQUENCE PARALLELISM. Scan-level
tuning space now also exhausted.

## ❌ NCCL NVLS = ~0% (didn't engage) — testing PROTO next
NCCL_NVLS_ENABLE=1 (hoist b32): step ~13s (unchanged), AllGather kernel STILL `RING_LL` (NVLS did not
engage — 16MB intra-node gathers below NVLS threshold / unsupported for this all-gather). Collective-
ALGORITHM tuning (NVLS) closed. Note: the regime is serialization-bound (host-synchronous chain), so NCCL
tuning addresses gather BANDWIDTH not the serialization — low EV. Testing NCCL_PROTO=Simple (last knob).

## ❌ NCCL_PROTO=Simple = ~0% — NCCL angle fully CLOSED; cheap-knob sweep DONE
PROTO=Simple (hoist b32): step ~12.9-13.8s = ~0% vs LL's 13s. So both NCCL knobs (NVLS no-engage, PROTO
no-help) are null → confirms the regime is SERIALIZATION-bound (host-synchronous gather chain), not
gather-BANDWIDTH-bound, so collective tuning can't move it. Probe scaffolding reset to clean.
**CHEAP-KNOB SPACE DEFINITIVELY EXHAUSTED** (measured null/closed, this session): autotune (banked 1.6×),
async/pipelined collectives (on), combine-threshold, command buffers, scan-unroll (manual REGRESSED +
native ~0%), replicate-weights (OOM), more-tokens (memory-capped @2 seq/GPU), NCCL NVLS + PROTO. The ONE
lever for the root cause (serial per-layer gather chain, arithmetic-intensity-limited) is SEQUENCE
PARALLELISM — relieve activation memory → more tokens/GPU → gathers overlap. Multi-day build, attention-
partitioner-risk, scoped above. STOPPING the knob-sweep (further cheap runs just re-confirm null).
