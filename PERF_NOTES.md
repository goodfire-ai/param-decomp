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
