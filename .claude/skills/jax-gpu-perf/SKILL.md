---
name: jax-gpu-perf
description: Debugging/optimizing JAX+XLA GPU training perf (FSDP weight-gather overlap, memory/peak attribution, collective scheduling, roofline). Use when investigating step-time, comms overlap, OOM, or MFU on the param-decomp JAX trainer (or any JAX/GSPMD GPU workload). Leads with native tooling; carries hard-won lessons.
---

# JAX/XLA GPU performance debugging

**Meta-rule: reach for the authoritative native tool first. Do NOT hand-roll xplane/HLO
parsers** — they're easy to get subtly wrong (this skill's original hand-rolled scripts had
real bugs: a "gather" regex that silently matched all collectives, a fragile compute-stream
heuristic, and double-counted exposed time; and the memory-report text was mis-summed 3×).
When you must read a raw artifact, cross-check against a native number.

## The right tool for each question

| Question | Native tool | How |
|---|---|---|
| **Peak memory** (runtime) | `device.memory_stats()` | `jax.devices()[0].memory_stats()["peak_bytes_in_use"]` |
| Peak memory (static, per executable) | `compiled.memory_analysis()` | `jit(fn).lower(*a).compile().memory_analysis()` → argument/output/temp GiB. This trainer prints both under `PD_MEM_PROFILE=1` (`run.py`). |
| **Memory breakdown — top-level** | buffer-assignment allocation-sum | sum `*-buffer-assignment.txt` `^allocation N: size (\d+),` once each; `parameter`→input, else the one `preallocated-temp` arena. Sum MUST equal the report's `Total bytes` (correctness check). This is the static executable heap; runtime peak comes from `memory_stats()`. |
| Memory — live-alloc by Python stack | `jax.profiler.save_device_memory_profile` + `pprof` | `pprof --web mem.prof`. NOTE: attributes by **Python allocation stack**; a jitted step is largely **opaque**, so this is NOT reliable per-HLO-intermediate attribution. Use the buffer-assignment + A/B instead for "which intermediate." |
| **Comms/compute overlap, stream concurrency** | **Nsight Systems** (exact) or XProf trace viewer (visual) | `nsys profile --trace=cuda,nvtx,nccl -o nsys_%q{SLURM_PROCID}_%p python -m ...` then `nsys stats --report cuda_gpu_kern_sum,cuda_gpu_trace`. Or `jax.profiler.trace(dir)` → `xprof <dir>` → Trace Viewer / GPU kernel stats. Don't hand-roll interval sweeps over xplane. |
| **NCCL protocol/algo actually used** | NCCL TUNING log | `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING` → "AllGather: N Bytes -> Algo RING proto SIMPLE". |
| Exposed collective time **by our `pd_*` named scope** | Nsight + NVTX, or XProf | `jax.named_scope` surfaces as NVTX ranges under `nsys profile --trace=...,nvtx`; export `nsys stats` to SQLite and join kernel intervals against the NVTX range, or filter the XProf trace viewer by scope. (No turnkey native metric for "exposed time per scope" — build it from the SQLite export only if needed, and validate the total against `nsys`'s own collective totals.) |

## Lessons (corrected per review)

1. **Memory: don't sum the `*-memory-usage-report.txt` shape lists** — slots are
   time-multiplexed (`used_by_n_values=N; shapeA, 52×shapeB`); summing overcounts 2–3×. Use
   `memory_analysis()`/`memory_stats()`, or the buffer-assignment allocation-sum (above).
2. **Don't infer NCCL protocol SOLELY from the xplane CUDA kernel symbol** (`..._RING_LL` can
   be a misleading name); confirm with the NCCL TUNING log or Nsight's NCCL trace.
3. **HLO op→source metadata lies inside scan/jvp/remat/cond** — classify by tensor SHAPES,
   not source frame; tag phases with `jax.named_scope` and read the INNERMOST scope.
4. **Persistent compile-cache key includes HLO + `XLA_FLAGS` + compile options + topology +
   version, but NOT arbitrary runtime env vars** (`NCCL_*`). A pure-`NCCL_*` A/B silently
   reloads the stale executable → byte-identical wall AND collective time = "never ran". Force
   a fresh compile for env-only A/Bs. HLO/XLA_FLAG-changing A/Bs recompile naturally.
5. **Roofline: AI of an FSDP weight-gather ≈ per-rank tokens** (`FLOPs/byte ≈ BT`, under bf16
   full-gathered-weight-byte accounting; the real wire bytes scale `BT·dp/(dp−1)` and shift
   with dtype/FLOP convention/reuse). Ridge = `peak_flops / gather_BW` (B200 bf16 ≈ 2.25e15;
   NVLink5 peak ≈ 900 GB/s → ridge ≈ 2500). `AI ≳ ridge` ⇒ compute-bound.
6. **Collective overlap knob:** `xla_gpu_experimental_parallel_collective_overlap_limit`
   defaults to **1** on current OpenXLA (experimental/version-specific — verify) → excess
   collectives spill onto the compute stream. Raising it moves them to async streams, but that
   ALONE may not improve overlap (binding issue can be prefetch *timing*). Within-iteration
   (unroll K) and between-iteration (double-buffer + pipelined-all-gather) are ALTERNATIVE ways
   to get `gather(L+1) ‖ compute(L)`, not complementary (unroll costs K× per-layer
   intermediates → OOM; double-buffer needs only 2 cheap weight buffers).
7. **This trainer (2026-06-30):** landed = `all_gather_combine_threshold=1GB` (−5.7%),
   CI-fn-once via `eqx.filter_vjp` (−3.4%, bit-identical), disable-XLA-remat (1.6×). Gated =
   overlap (prefetch-timing) and batch→compute-bound (memory). Peak ≈ params ~53 GB + temp
   arena ~122 GB; temp scales with batch and is the OOM lever.

## No bundled scripts — by design
This skill ships **no hand-rolled xplane/HLO parsers**. Earlier versions did; a review found
real bugs in every one (a collective regex that silently matched all-collectives, a fragile
compute-stream heuristic, double-counted exposed time), so they were deleted rather than
shipped. Use the native tools in the table above. If a genuinely-missing metric needs a custom
script, write it against the SQLite/`memory_analysis` outputs and **validate its totals against
a native number before trusting it** — then it can earn a place here.

Artifacts: xplane at `runs/<id>/profile/plugins/profile/*/*.xplane.pb`; HLO at `runs/<id>/hlo/`.
