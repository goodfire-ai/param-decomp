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
