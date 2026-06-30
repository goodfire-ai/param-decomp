---
name: jax-gpu-perf
description: Debugging/optimizing JAX+XLA GPU training perf (FSDP weight-gather overlap, memory/peak attribution, collective scheduling, roofline). Use when investigating step-time, comms overlap, OOM, or MFU on the param-decomp JAX trainer (or any JAX/GSPMD GPU workload). Carries hard-won lessons + validated analysis tools.
---

# JAX/XLA GPU performance debugging

Lessons and tools accumulated debugging the full-32L Llama-8B VPD step. Read the lesson that
fits, then use the matching tool in `tools/`. **Meta-rule: go to the authoritative artifact
first; don't reverse-engineer from a text format you have to guess at.**

## Lesson 1 — Memory/peak: sum the buffer-assignment allocations, don't parse the report
The `*-memory-usage-report.txt` lists memory SLOTS with **time-multiplexed shape lists**
(`used_by_n_values=56; shapeA, 52×shapeB, ...`). Summing those shapes overcounts 2–3× (I got
295/256 GB vs a 163 GiB peak, three times). **Instead:**
- **Top-level split:** sum `*-buffer-assignment.txt` allocations once each by size —
  `^allocation N: size (\d+), (parameter|output shape is|constant|...)`. Sum = the real peak.
  Classify: `parameter` → input (weights/optstate/masters), else → the one big
  `preallocated-temp` arena (all intermediates, packed with reuse). This sum MATCHES the
  report's `Total bytes` — that's your correctness check.
- **Authoritative numbers without parsing at all:** XLA's own `memory_analysis()` (static:
  argument/output/temp GiB) and `device.memory_stats()` (runtime `peak_bytes_in_use`). The
  trainer prints these under `PD_MEM_PROFILE=1` (`run.py`). Trust these over any text parse.
- **Per-tensor attribution of the temp arena:** `jax.profiler.save_device_memory_profile`
  writes a pprof (`device_memory_*.prof`) keyed by allocation site — the right tool for
  "which intermediate is the 122 GB", vs guessing from HLO text.
- Useful identity: for this trainer, peak ≈ params (~53 GB: weights+optstate+masters+frozen
  target) + temp arena (~122 GB: intermediates, scales with batch → the OOM lever). The temp
  is dominated by `[B,S,C]` CI/activation tensors (per token, ΣC≈1.25M ≫ vocab≈128k).

## Lesson 2 — Trust the runtime decision-log, not the static symbol
- **NCCL protocol:** the xplane kernel symbol (`ncclDevKernel_AllGather_RING_LL`) is a
  MISLEADING name — it does NOT reflect the runtime protocol. Trust `NCCL_DEBUG=INFO
  NCCL_DEBUG_SUBSYS=TUNING` ("AllGather: N Bytes -> Algo RING proto SIMPLE"). We chased a
  phantom "stuck on LL" for hours; the gathers were always Simple.
- **HLO op→source metadata** (op_name/stack_frame) is unreliable inside scan/jvp/remat/cond.
  Classify ops by tensor SHAPES, not source frame. Use `jax.named_scope` for honest phase tags
  (and read the INNERMOST `pd_*` scope to separate nested phases — see `exposed_by_phase.py`).

## Lesson 3 — The persistent compile cache hides env-only A/Bs
Cache key = HLO + XLA_FLAGS + topology + jax/xla version. **Env vars (NCCL_*) are NOT in it.**
A pure-env A/B silently reloads the stale executable → byte-identical wall AND gather time is
the tell ("change never ran", not "no effect"). For an env-only A/B, force a fresh compile
(point `jax_compilation_cache_dir` at an empty dir, or change an XLA flag). HLO-changing A/Bs
(TP, fp8, batch, unroll) recompile naturally — those are valid.

## Lesson 4 — Roofline: arithmetic intensity of an FSDP weight-gather = per-rank tokens
Every gathered weight (V, U, frozen W) is matmul'd over BT tokens → `FLOPs/bytes = BT` exactly.
Ridge = `peak_flops / gather_BW` (B200 bf16 ≈ 2.25e15; NVLink5 peak ≈ 900 GB/s → ridge ≈ 2500).
`AI(=BT) ≥ ridge` ⇒ compute-bound (overlap can hide comms); `<` ⇒ comms-bound. At b128/dp32,
BT=2048 < 2500 → comms-bound; crossing needs ~+22% batch at peak BW (but batch is often
CI-memory-blocked — see Lesson 1). Effective BW well below peak (contention) makes it worse.

## Lesson 5 — Collective overlap: measure concurrency per-stream, mind the overlap LIMIT
- Overlap is real only if a gather kernel runs while a compute kernel runs on a DIFFERENT
  stream. Measure with `tools/async_fraction.py` (gather time on the compute-stream = serialized
  = unhideable) and `tools/exposed_by_phase.py` (exposed gather by phase). 13% concurrency = bad.
- `xla_gpu_experimental_parallel_collective_overlap_limit` **defaults to 1** (only 1 in-flight
  collective) → excess gathers spill onto the compute stream. Raising it moves them to async
  streams — but that ALONE didn't improve overlap (the binding issue was prefetch *timing*, not
  stream placement). `xla_gpu_executable_num_communication_streams` errored on our XLA.
- Within-iteration (unroll K) vs between-iteration (double-buffer + pipelined-all-gather) are
  ALTERNATIVE ways to get `gather(L+1) ‖ compute(L)`, not complementary. Unroll costs K× the
  per-layer intermediates (OOMs); double-buffer needs only 2 cheap weight buffers (weights are
  transient — confirmed via Lesson 1 — so it's NOT memory-blocked; it's scheduling-blocked).

## Lesson 6 — Wins that landed vs levers that are gated (this trainer, 2026-06-30)
- **Landed, shippable:** `all_gather_combine_threshold=1GB` (−5.7%; knee at 1GB, 2G/4G worse);
  CI-fn computed once via `eqx.filter_vjp` (−3.4%, bit-identical); disable-XLA-remat (1.6×, prior).
- **Gated:** overlap (~1.5× potential, locked behind prefetch-timing); batch→compute-bound
  (CI-memory-blocked). Both need the 122 GB temp-arena reduced first.

## Tools (`tools/`, run from this dir so `xplane_pb2` imports)
- `xplane_overlap.py <xplane.pb>` — gather/compute union + both-active overlap + top kernels.
- `async_fraction.py <xplane.pb>` — % gather transfer ON the compute stream (serialized) vs
  async stream + genuine concurrency.
- `exposed_by_phase.py <xplane.pb>` — exposed (non-overlapped) gather attributed by `pd_*` phase.
- Memory: prefer `PD_MEM_PROFILE=1` (prints `memory_analysis` + `memory_stats`) or the
  buffer-assignment allocation-sum (Lesson 1) over the memory-usage-report.

xplane lives at `runs/<id>/profile/plugins/profile/*/*.xplane.pb`; HLO dumps at `runs/<id>/hlo/`.
