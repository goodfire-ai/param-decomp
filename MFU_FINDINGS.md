# Full-32L Llama-8B VPD — MFU / step-time findings

Canonical state doc for the MFU push on the full-32-layer Llama-3.1-8B VPD training step
(32 B200, 4 nodes × 8, JAX/GSPMD, HSDP mesh `replicate=4 (IB) × fsdp=8 (NVLink)`, tp=1).
Supersedes the ephemeral scratchpad `OVERLAP_MFU_PLAN.md`.

## TP2 implementation progress (2026-06-29, in-flight on main checkout, uncommitted)

Decision: TP2 = 3-D mesh `(replicate, fsdp, tp)`, tp carved in-node, C-on-tp (Megatron-C)
to halve the per-layer weight comm; v1 = full-residual, target replicate-over-tp, attn
left FSDP. Done in main checkout (needs the disable-remat/flash wins present for the bench).

**DONE + validated (correctness):**
- Foundation: `hsdp_mesh(tp)` → 3-D mesh (sharding.py); `runtime.tp` un-dropped + real
  `RuntimeConfig.tp` field (ge1 le8); threaded `hsdp_mesh(built.runtime.tp)`. tp=1 degenerate
  = behaviour-preserving (4/4 multidevice sharding + 12/12 llama8b, type green).
- V/U C-on-tp: `DecompVU.shardings` (master: `P(data,"tp")` / `P("tp",data)` → ÷N regained),
  `_reconstruct_compute_weights` (compute: `P(None,"fsdp","tp")` / `P(None,"tp","fsdp")`),
  `site_out` (xV pinned C-on-tp + output pinned d-full for symmetric tp-reduce + fsdp-gather).
  Updated test_sharding's old "C never sharded" spec assertion. CPU-HLO: **involuntary_remat=0**
  (the 2026-06-26 trap avoided) for both mask-replicated and mask-C-on-tp.

**OPEN / gotchas:**
- **Perf mechanism unconfirmed on CPU.** At realistic dims GSPMD compiled `x@V` as an OUTPUT
  ALL-REDUCE (over fsdp), NOT a weight gather — i.e. it may win via output-reduce rather than
  the halved gather we predicted. tiny/CPU GSPMD ≠ real-scale (lore). **Needs real GPU HLO** to
  see the actual compiled strategy (gather halved? output all-reduce? batch behaviour?).
- **CI fn docstrings MISLEAD** (verify against the forward, not the docs): CIBlock `.shardings`
  says "every weight shards d_model ÷N, head/mlp_hidden replicated" — but `w1 [nc,d_model,
  mlp_hidden]` spec `P(None,None,full)` shards **mlp_hidden** (axis2), so the MLP is ALREADY
  Megatron-on-mlp_hidden (sharded+reduced) on the data axes; only attention is FSDP-on-d_model.
  So CI fn tp wiring must be grounded in the real forward+shapes.
- CI fn tp NOT done yet: need `out_ws` C-on-tp (keystone — masks come C-sharded to match V/U),
  MLP mlp_hidden→tp, the compute re-pin (`_reconstruct_ci_compute_weights`), attn left as-is.
  Until done, a tp=2 run regresses the CI fn (replicated over tp → ~2× mem → likely OOM at b128),
  so the GPU "see what happens" check is gated on finishing the CI fn.
- Repro: `scratchpad/tp_siteout_hlo.py` (real site_out, 3-D mesh, involuntary-remat check).

## Current state (2026-06-29)

- **Step wall: 8.3s** (down from 13.2s — see disable-remat win below). Device-bound (~96% busy).
- The step is now **GATHER-BOUND**, uncapped-xplane attribution at b128:
  - gather union **~5.5s** (dominant: `ncclDevKernel_AllGather` ~3.9s = the ÷fsdp→full NVLink
    weight gather, paid fwd + bwd-recompute; plus IB AllReduce ~1.1s + collective-permute ~1s).
  - compute union **~2.8s**.
  - overlap (both-active) only **~16%** — the gather is essentially exposed/serial.
- Roofline ceiling from here: perfect overlap → `max(gather,compute)=5.5s`. Since gather > compute,
  `5.5-2.8=2.7s` of gather is **unhideable** (not enough compute to tuck it under). So overlap
  ALONE caps at ~5.5s. To beat that you must **cut the gather**.

## THE WIN: disable XLA's rematerialization pass (1.6×)

`--xla_disable_hlo_passes=rematerialization` (JAX-native equivalent:
`jax.config.update('jax_compiler_enable_remat_pass', False)`) → **step 13.2s → 8.3s**.

- Mechanism (confirmed): XLA's own HLO rematerialization pass was **double-rematerializing on
  top of our JAX `nothing_saveable` remat** — we were paying ~2.7× the necessary backward
  recompute. Disabling it collapsed **compute union 7.6s → 2.8s**. NOT an overlap win (`is_sync`
  barely changed). JAX docs confirm: when you remat manually, the compiler's basic auto-remat
  layers extra recompute on top; disable it so your policy alone governs.
- **It's the standard MaxText GPU default** (every 7B/405B GPU config ships it). Our whole
  session of overlap no-movers was because the wall was dominated by *redundant recompute*, not
  exposed comms — we were optimizing the wrong half.
- **Caveat:** it removes XLA's OOM safety net. Our manual `nothing_saveable` must fit on its own.
  It fit at b128 in a throwaway trace; **STILL TO VALIDATE: a real save/resume training smoke**
  (checkpoint-save is the memory-peak moment) before shipping to production configs.
- Action: add to production launch_env (prefer the JAX-native config toggle). Also adopt the rest
  of MaxText's canonical GPU bundle (triton_gemm=false, command buffers off, combine thresholds,
  highest_priority_async_stream, pipelined AG/RS/AR, double_buffering).

## Other changes landed this session (working tree, uncommitted)

- **CI fn → cuDNN flash** (`ci_fn.py`): was `implementation="xla"` (materialized `[B,H,T,T]`
  scores, GBs/chunk) because heads were Megatron-sharded. tp=1 now → switched to
  `implementation = "cudnn" if jax.default_backend()=="gpu" else "xla"` (XLA fallback for CPU
  tests). Frees the score-slab off the peak. **Speed-neutral** (compute is matmul-bound, not
  attention) but a memory enabler.
- **CI fn remat-policy bug FIX** (`ci_fn.py`): `remat_ci_fn=False` previously meant *no
  checkpoint at all* (→ stacks full gathered 31B weights as residuals → OOM), inconsistent with
  llama8b.py. Now `nothing_saveable if remat else dots_saveable`, matching the recon forward.
- **`PD_UNROLL_K` chunk-unroll** (`llama8b.py`, env-gated, default 1 = today's per-layer scan):
  scans groups of K layers unrolled. **NEGATIVE RESULT — do not use.** K=4 made overlap *worse*
  (15%→11.8%, compute rose, wall 12.6→12.8 pre-disable-remat) and produced more sync gathers
  (1495 vs 359). Kept in code (off by default) but it's a dead direction.

## Dead ends (all measured, all confirmed non-movers on the wall)

- XLA overlap flags (LHS, pipelined_all_gather, double_buffering, combine threshold) — no-movers
  at tp=1; XLA's `ConvertAsyncCollectivesToSync` actively flips our gathers back to sync.
- IB-reconstruction gradient dedup (jax.vjp) — gradient-correct, ~no wall change.
- cast-order optimization_barrier fix — no-mover.
- `dots_saveable` (save activations) — **OOM (263 GiB)**: on a scan, saved activations stack
  `[n_layer,…]`; full activation saving is infeasible at b128. (This is *why* remat is mandatory.)
- b256 (even with flash + bf16 src) — **OOM** (~201 GiB needed, 151 GiB temp; cap ~180). Batch↑
  to 256 is out; b192 declined by Oli. Batch stays 128.
- chunk-unroll (above).
- flash for *speed* — neutral (compute is matmul-bound).

NB: no remat policy avoids the backward re-gather (the 2× FSDP gather is inherent; only the
36.6 GB DDP-stack avoids it, which OOMs). `nothing_saveable` is correct: min-memory, and the
extra recompute it causes is no longer the bottleneck.

## MaxText prior art (mined from ../maxtext)

- **Same structure as us**: `scan_layers=True` + `remat_policy=full` (≈ nothing_saveable) + FSDP
  on GPU. So our architecture isn't the problem.
- **fp8 Quantized All-Gather (QAG)** — the blueprint for cutting the gather: quantize weight to
  fp8 FIRST, then `all_gather` the `.qvalue` (fp8 bytes) → **comm halved**
  (`kernels/megablox/ops.py:180-187`). Gated on **static scaling** (`weight_calibration_method=
  "fixed,…"`) so the per-axis scale survives the gather. `quantization=fp8_gpu` +
  `use_qwix_quantization=true` for NVIDIA; e4m3 fwd / e5m2 grad. Their shipped QAG is in the MoE
  GMM kernel, not dense-Linear FSDP — we'd replicate the *pattern* on `site_out`.
- **Windowed einsum (collective matmul)** = `xla_gpu_multi_streamed_windowed_einsum=true` +
  `xla_gpu_threshold_for_windowed_einsum_mib=0`. **Requires TP** (their test runs `tp=8, fsdp=1`)
  — it overlaps the TP *activation* collective with the matmul. **This is the only GPU overlap
  that actually works, and it needs TP.** Not available at tp=1.
- FSDP gather arithmetic intensity = **local_batch** (flops/byte). To make the gather hideable:
  raise per-device batch (OOM'd) or halve bytes (fp8) — or shard the weight more (TP).
- remat policy menu exists (minimal, save_qkv_proj, …) + an AOT `estimator` to pick the
  fastest-fitting. A *light* selective policy is a fallback to reclaim the exposed ~1.9s compute
  if overlap proves unattainable — NOT the next move.

## DECISION: how to attack the comm/gather wall

Three composing levers (multiplicative on the gather):
1. **Overlap** — capped at 5.5s alone; and the only working GPU mechanism (windowed einsum)
   **requires TP**. So overlap ≈ requires lever 3. It's the *finisher*, not primary.
2. **fp8 QAG** — halves gather bytes, but **lossy on the core V/U matmul** (VPD is a
   precision-sensitive decomposition → real quality risk; must validate).
3. **TP2** — shard C on tp=2 (Megatron column/row-parallel: `x@V` col-parallel, mask on `C/tp`,
   `@U` row-parallel + output all-reduce). Makes V/U half-size on C → **the FSDP all-gather
   halves**. The new TP output all-reduce is smaller (activations ≪ V/U) AND **overlappable via
   windowed einsum**. **EXACT** (only fp32 reassociation, no precision loss) — decisive for VPD.

**Chosen order:**
1. **TP2** — exact halving (5.5→~2.8s) + unlocks windowed-einsum overlap; tp mesh infra already
   exists in repo (tp2/tp4/tp8 configs) → closer to a config experiment than a rewrite.
2. **Overlap (windowed einsum)** — now bites, because TP gives it the collective it can overlap.
3. **fp8 QAG** — optional second halving to reach ~3s, gated on a decomposition-quality check.

Endgame to ~3s: **TP2 + windowed-einsum overlap**, fp8 as the final squeeze.

## TP2 — implementation scope (the real work / risk)

Make the masked low-rank forward shard cleanly on C at tp=2:
- `components.py:site_out` — V column-parallel (C on tp), U row-parallel (C on tp), output
  all-reduce over tp. Mask `[tok, C]` → `[tok, C/tp]`.
- `targets/llama8b.py` `_reconstruct_compute_weights` / `_attach_per_kind_masks` — C-on-tp specs.
- The CI fn must emit masks C-sharded (the masks live on the C axis).
- **Verify matmul AI stays high** (AI ∝ M/tp) so TP2 doesn't trade a gather wall for a
  small-matmul wall. Measure, don't assume.

## Key artifacts / tooling

- Uncapped overlap attribution: `scratchpad/xplane_overlap.py <xplane.pb>` (the 1M-event JSON cap
  never touches the raw `*.xplane.pb`; monkeypatches the protobuf version check).
- `is_sync` scorecard: grep `"is_sync":true` in `runs/<id>/hlo/*after_optimizations.txt` among
  `all-gather-start` lines.
- Reference runs (b128, full32L, trace step 2): baseline+flash `p-c9b9711f` (12.6s);
  **MaxText-flags `p-c49f8c21` (8.3s)**; unroll `p-c68d9031` (worse); b256 OOM probe `p-ac…`/131842.
- MaxText clone: `../maxtext` (unshallowed). Canonical GPU FSDP flag bundle:
  `src/maxtext/configs/gpu/a3/llama_3.1_405b/128vm.sh`.
