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

## Overnight 2026-06-29 (late): TP result, harness bug, fp8 QAG

**TP shelved (Oli's call) — and the b64 A/B says it doesn't help as-built.** At fixed global
batch 64 + 32 GPUs: tp=1 b64 = **7.41s** (gather 5.44, AllGather 4.0) vs tp=2 b64 = **9.68s**
(gather 7.59, AllGather 4.65). The C-on-tp gather-halving did NOT materialize — the AllGather
*grew*, plus the tp output-reduces + per-rank-batch doubling (2→4/rank) made it net worse. So
tp=2 loses at fixed global batch. b128 tp=2 OOM'd (per-rank doubling + target-replicate + disable-
remat). TP is SHELVED, not removed: tp=1 is the default (behaviour-preserving), all C-on-tp code
+ configs stay; it stacks with fp8 (TP halves gather, fp8 halves bytes → ÷4) if revisited. Open:
*why* GSPMD didn't shrink the gather despite C-on-tp (needs HLO dig; deferred).

**CPU-sharding harness bug (important, affects any CPU sharding validation).** `with mesh:`
leaves `get_abstract_mesh().empty=True`, so the sharded-compute guards (`_reconstruct_compute_
weights`, site_out constraints) ALL no-op — the smoke ran the *unsharded* path. The engine uses
`jax.set_mesh(mesh)` (run.py:457) which sets the abstract mesh. Fix: smokes must `jax.set_mesh`
(assert `not get_abstract_mesh().empty`). After the fix, TP tp=1≈tp=2 on the REAL sharded path
(reassociation), and fp8 actually fires. `scratchpad/tp2_step_smoke.py` fixed.

**fp8 Quantized All-Gather (QAG) — the supplementary lever, V/U done + CPU-validated.**
Helpers `quantize_fp8`/`dequantize_fp8` (components.py): per-LEADING-ROW e4m3 scale (reduces all
but axis0 — the scan axis; a scalar scale has no axis to scan → IndexError). Env-gated
`PD_GATHER_FP8` (default off, behaviour-preserving). `_reconstruct_compute_weights` quantizes V/U
to fp8 ÷fsdp + per-layer scale; `masked_site` gathers the fp8 to full d (½ bytes) then dequants
to bf16 (optimization_barrier keeps the convert AFTER the gather). CPU-validated: **fp8 on the
wire (36 f8e4m3 all-gathers, 0 bf16 in the HLO)**, lossy-small (recon 0.0006205→0.0006208,
~0.04%), finite, default path unbroken (12/12 test_llama8b). GPU trace `133487` in flight.

**Next:** (1) GPU fp8 result (does the V/U gather shrink + step time vs 8.3s). (2) **CI fn fp8**
— the bigger gather (CI fn ~54% vs V/U 32%): same QAG pattern in `_reconstruct_ci_compute_weights`
+ dequant at each weight use in the CI forward (more dequant points than V/U's single site_out).
Faith loss is fp8-invariant (uses fp32 masters), so quality risk is only on recon/PPGD — validate.

## fp8 GPU result (V/U) — NET NEGATIVE, and it pinpoints the comms-BW root cause

fp8 V/U trace (`p-c7d98c11`, tp=1 + disable-remat): **wall 9.39s vs 8.3s baseline** — SLOWER.
gather union 5.5→6.36s, AllGather 3.9→**4.45s** (↑16%), compute 2.8→3.0s. fp8 IS on the wire
(514 `f8e4m3` all-gather-start vs 2046 bf16 for the still-bf16 CI fn/target). So halving the
V/U bytes made the gather *grow*.

**Why — the key insight:** the in-step comms run at ~27 GB/s vs ~600 GB/s isolated, i.e. deep in
the **small-message / low-BW regime** (BW curve: 1MB=23, 64MB=582, 1GB=646 GB/s). Halving the
bytes (fp8) makes each message *smaller* → *lower* BW → the transfer time goes UP, not down. The
gather time is **BW/latency-bound, not byte-bound**, so byte-reduction levers (fp8, TP) CANNOT
help until the effective bandwidth is fixed. (Plus the explicit per-site gather+dequant in
masked_site fragments further + adds convert overhead.)

**Consequence / reprioritization:** the **low effective comms bandwidth is the ROOT lever**;
fp8 and TP are downstream of it (they only pay once bytes→time is linear). fp8 left dormant
(`PD_GATHER_FP8` off by default — code works, just net-negative until BW is fixed). The comms-BW
investigation (RING_LL-on-large-messages? NCCL channel/proto/buffsize tuning? message coalescing
vs the scan; all vs MaxText) is now THE priority — fixing it should help the baseline gather
directly AND unlock fp8/TP. Prime suspect: `ncclDevKernel_*_RING_LL` (NCCL low-latency protocol,
low peak BW) being used on large gathers.

## ★ ROOT CAUSE of the low comms BW: NCCL stuck on the LL protocol (the #1 lever)

Diagnosis (subagent, trace `p-c49f8c21`, tooling `scratchpad/{hlo_collectives,trace_nccl_bw,join_gather_bw}.py`):

**Every NCCL collective runs on `_LL` (low-latency) protocol** — the trace has ONLY
`ncclDevKernel_*_RING_LL`, zero `Simple`/`LL128`/NVLS. LL is for tiny latency-bound messages and
wastes ~½ the link on per-flit flag sync; our gathers are **large (132 MB modal = the 128 MB
combine cap)** but stuck on LL. The XLA-flag layer is already MaxText-equivalent (MAXTEXTFLAGS) —
**the gap is purely the NCCL env, which we leave at defaults.**

Quantified (per-rank, joined trace↔HLO):
| link | kernels | time | moved | busBW |
|---|---|---|---|---|
| NVLink in-node (group=8) | 4097 | 1567 ms | 420 GB | **268 GB/s** (Simple ~582 → ~1.66× left on table) |
| **IB cross-node reconstruct (group=4)** | 44 | **500 ms** | **9 GB** | **19 GB/s** ← the "27 GB/s"; only grid:4 channels |
| full-mesh grad (group=32) | 64 | 171 ms | 33 GB | 195 GB/s |

Step is **AllGather-bound** (2237/2266 ms collective; AllReduce/ReduceScatter are hidden/overlapped
— grad-sync is NOT the problem). Message size is NOT the problem (all gathers large, coalescing
works). The IB reconstruct gather moves only 9 GB but takes 500 ms at 19 GB/s on 4 channels — a
real waste, not unavoidable cross-node work.

**This also explains why fp8/TP backfired:** on LL, smaller messages = even lower BW, so
byte-cutting can't help until the protocol is fixed.

### Fix (NCCL env in `runtime.launch_env.env`; we currently set NONE of these), ranked:
1. **`NCCL_PROTO=Simple`** (try `Simple,LL128`) — removes the ~1.66× LL penalty on 1567 ms of
   NVLink gathers + the 19 GB/s IB gathers. **Highest-confidence single lever.** [trace 133605]
2. **`NCCL_NVLS_ENABLE=1`** — NVLink-SHARP in-network reduction on B200/NVSwitch, can ~2× in-node
   AG/RS (currently unused). [in 133611]
3. **`NCCL_MIN_NCHANNELS=16`** — the IB gathers use only 4 channels. [in 133611]
4. **`CUDA_DEVICE_MAX_CONNECTIONS=1`** — overlap scheduling (MaxText sets it). [in 133611]
5. (algorithmic, bigger ceiling) cut the 4097 gather count — hoist the per-layer fsdp gather out
   of the PPGD ascend inner loop (gather once, reuse across ascend steps; keep per-layer transient,
   NOT the full-model stack which OOM'd 2026-06-26).

Caveat: MaxText sets these only in its GCP TCPX/Ethernet branch; on IB it rides defaults too — so
it's not "MaxText does X on IB," it's "Simple is exactly the lever our LL-stuck IB run needs."

**Traces in flight:** `133605` (Simple alone), `133611` (full bundle) vs the 8.3s baseline.
Validate: `scratchpad/join_gather_bw.py <trace> <hlo>` — expect NVLink busBW 268 → ~450+.
**REPRIORITIZATION: NCCL_PROTO=Simple is now the #1 MFU lever** (root cause, one config line);
fp8/TP are downstream of it.

## NCCL_PROTO=Simple — set correctly but NOT honored (the twist; needs diagnosis)

Launched `NCCL_PROTO=Simple` (133605) + full bundle (Simple+NVLS+16ch+CUDA_DEVICE_MAX_CONNECTIONS,
133611). **Result: no change** — wall 8.32 / 8.25 vs 8.3s baseline; the trace kernels are STILL
`ncclDevKernel_*_RING_LL` (LL, 0 Simple); gather union ~5.5s unchanged. Verified the env DOES
reach the rank: source config has `launch_env.env.NCCL_PROTO=Simple` and `LaunchEnv.as_env()`
emits it (configs.py:718 `rendered |= self.env`; launch.py:73 exports each). So **NCCL is not
honoring `NCCL_PROTO` under XLA on this stack** (NCCL 2.30.7+cuda12.9, B200) — NCCL_DEBUG was at
WARN (4 lines) so we couldn't see why. Launched `NCCLDIAG` (133688): `NCCL_PROTO=Simple` +
`NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT,TUNING,ENV` to confirm whether NCCL reads the env and
what proto it selects. Hypotheses to check next: (a) XLA's GPU NCCL integration overrides the
protocol (look for an `xla_gpu_*` NCCL-proto/algo flag rather than the env), (b) `NCCL_PROTO=Simple`
syntax/version issue → try `Simple,LL128`, (c) the message size still routes NCCL to LL despite
the env. The LL→Simple ~1.66× remains the diagnosed prize; we just haven't found the working knob.

### NCCL DOES read NCCL_PROTO=Simple — diag (133688)
`NCCL INFO NCCL_PROTO set by environment to Simple` printed on all 32 ranks — so the env reaches
NCCL (the rendering/propagation is fine). NCCL then builds its "Enabled Func/Proto/Algo Matrix".
NCCL comms seen: `nRanks 4 nNodes 4` (the cross-node replicate/IB axis; the in-node fsdp=8 comm
should exist too). The contradiction (NCCL reads Simple, but the earlier 133605/133611 traces
showed RING_LL kernels + no wall change) is being settled by the diag run's OWN trace — pending
(poll bmd12b7rp). Two outcomes: (a) diag kernels = Simple → the env works and the earlier "no
effect" was a stale/wrong-trace attribution or the gather-BW gain not translating to wall (gather
not the binding path as believed); (b) diag kernels still LL → NCCL reads Simple but emits LL
(proto matrix keeps LL enabled / XLA-launch path bypasses the restriction) → need a different knob.
Either way: NCCL_PROTO reaches NCCL; whether it changes the emitted kernel is the open question.

### NCCL_PROTO=Simple — ACTIVE but NO EFFECT; the gather is not BW/protocol-bound
Diag (133688, rid p-d98d4ed5): NCCL read `NCCL_PROTO=Simple` (×32) AND its Func/Proto/Algo
matrix shows **LL=0, LL128=0, Simple=1** for every collective — Simple is genuinely active. YET
the dominant kernel is **still `ncclDevKernel_AllGather_RING_LL` (3877 ms)**, gather union **5.31s**,
wall **8.48s** — unchanged vs 5.5s/8.3s baseline. (Why the kernel stays RING_LL despite matrix
LL=0 is unresolved — likely a per-comm matrix for the in-node fsdp comm, or NCCL keeping LL for the
intra-node NVLink path; minor vs the bottom line.)

**BOTTOM LINE — every byte/protocol lever has failed to move the ~5.5s gather:**
- TP (C-on-tp): gather 5.5→7.6s (worse)
- fp8 QAG (V/U): gather 5.5→6.36s (worse)
- NCCL_PROTO=Simple: gather 5.5→5.31s (unchanged)

So the in-step gather (~300 GB/s effective vs ~582 isolated) is **NOT** protocol/byte/BW-bound in
any accessible way. The remaining hypotheses (the real morning frontier):
1. **COUNT-bound** — the subagent found **4205 gather kernels**, re-gathering the SAME per-layer
   weights every PPGD-ascend iter (1664), every value_and_grad (1699), every ci_fwd (511). Lever:
   **hoist the per-layer ÷fsdp gather out of the PPGD-ascend inner loop** (gather once/step, reuse
   across ascend steps; keep per-layer transient, NOT the full-model stack — that OOM'd 2026-06-26).
   This is the one comms lever NOT yet tried, and the only one consistent with "bytes/protocol
   don't help."
2. **CONTENTION-bound** — per-kernel ~300 vs 582 isolated GB/s suggests gathers contend with
   compute / each other on the link; a count reduction would also relieve this.
3. Re-examine whether the gather is even the true wall-binding path (union math says yes, but
   nothing that targets bytes/protocol moves it — suspicious).

**The protocol/bytes track is dead; the count/contention track (lever #5) is the live one.**
fp8/TP stay shelved (they make a count/contention-bound gather *worse*, not better).

### ★ THE CONFOUND: the persistent compile cache invalidated the NCCL_PROTO test
The diag's three NCCL comms (global nRanks32, cross-node nRanks4, in-node-fsdp nRanks8) ALL show
AllGather matrix `LL=0 LL128=0 Simple=1` — NCCL genuinely disabled LL — yet the kernel is still
`ncclDevKernel_AllGather_RING_LL`. Both can only be true if the run **ran a STALE cached executable**
with `RING_LL` baked into the collective thunk. The persistent XLA compile cache
(`xla_compilation_cache/`, keyed HLO+XLA_FLAGS+topology+version) does **NOT** include `NCCL_PROTO`
(a pure NCCL env var). So all three NCCL_PROTO runs (133605/133611/133688) loaded the IDENTICAL
pre-NCCL_PROTO executable → byte-identical gather (5.3-5.5s) and wall (8.3-8.48s). **The "NCCL_PROTO
doesn't help" result is invalid — the protocol was never actually exercised at execution.** This is
the long-sought explanation for "nothing moves the needle" on env-only A/Bs (TP/fp8 DID change the
HLO → fresh compile → those tests were valid). Coherent model: **XLA picks the collective kernel at
COMPILE time** (querying NCCL tuning, which reads NCCL_PROTO) and bakes it; a runtime env can't
override a baked kernel.

**Re-test in flight (133843, NCCLFRESH):** added `PD_XLA_CACHE_DIR` override to
`_enable_persistent_compilation_cache` (honors an env cache-dir; backward-compat) → pointed at an
EMPTY dir → forces a full fresh compile with `NCCL_PROTO=Simple` in scope. If the AllGather kernel
becomes `RING_Simple` (and gather/wall drop) → NCCL_PROTO works, the cache was the confound, and the
real fix is busting/keying the cache on NCCL_PROTO. If still `RING_LL` → XLA bakes LL regardless →
the lever is an XLA collective flag, not the env. Either way the ~1.66× LL→Simple prize is back in
play — it was never validly tested.

### ★★ CORRECTION: there was NEVER an LL-protocol problem — kernel-name red herring
The fresh compile (133843, p-0b87b5bc, empty cache, NCCL_PROTO=Simple) settles the whole comms-BW
thread. NCCL's per-op TUNING log is ground truth, and it shows **every AllGather at every size uses
`Algo RING proto SIMPLE`** — 1MB/4MB/14MB/134MB/1GB, ALL proto SIMPLE, ZERO proto LL. Yet the xplane
trace labels the dominant kernel `ncclDevKernel_AllGather_RING_LL`. **The xplane NCCL kernel-symbol
name is UNRELIABLE** (NCCL 2.30 / CUPTI symbol resolution) — it does NOT reflect the runtime protocol.
Trust NCCL's `SUBSYS=TUNING` log, not the kernel symbol (cf. "HLO stack-frame attribution lies" —
same lesson, different layer).

Consequences:
- **The subagent's "gathers stuck on LL, ~1.66× penalty" diagnosis was WRONG** — it misread the
  kernel symbol. The gathers were on SIMPLE all along.
- **NCCL_PROTO=Simple is a no-op because the gathers were already Simple.** The earlier "no effect"
  (cached) AND the fresh compile (8.53s, gather 5.47s) agree: protocol changes nothing. The
  cache-confound theory is ALSO moot — fresh and cached both run proto SIMPLE, same wall.
- **The ~300 vs ~582 GB/s gap (in-step vs isolated microbench) is NOT protocol.** Both were Simple.
  The gap is in-step CONTENTION + COUNT: ~4205 per-layer gathers (re-gathered every PPGD-ascend iter
  / value_and_grad / ci_fwd), small and contended, vs one big clean isolated gather.

**THE REAL (and only remaining) comms lever — #5, reduce the re-gather COUNT/contention:**
hoist the per-layer ÷fsdp→full gather out of the PPGD-ascend inner loop (gather once/step, reuse
across ascend steps), keeping per-layer transient (NOT the full-model stack — OOM'd 2026-06-26).
This is the one lever consistent with ALL the negatives: protocol (no-op), bytes (fp8 worse), TP
(worse) — because they all touch per-gather size/proto, not the gather COUNT. fp8/TP make a
count/contention-bound gather WORSE (more, smaller gathers). Stay shelved.

## ★ Lever #5 SHIPPED-PENDING: PD_ASCEND_REPLICATE — the first comms lever that WORKS (+4%)

Implemented (`param_decomp/train.py`, flag `PD_ASCEND_REPLICATE`, off by default, like
`PD_REPLICATE_WEIGHTS`/`PD_GATHER_FP8`): in the PPGD warmup ascend the compute weights V/U are
DETACHED + constant across all `n_warmup` iterations, yet each iteration's forward re-gathers
every layer's ÷fsdp→full slice. `replicate_for_ascend` gathers V/U to FULL/replicated ONCE
before the ascend scan (`jax.tree.map` + `with_sharding_constraint(P())` over the opaque
`prepared` pytree — generic, no protocol change), so the `n_warmup` ascend forwards run plain
matmuls with NO per-layer gather. Pure data movement → **numerics bit-identical** (CPU smoke
worst Δ = 0.0; equivalence goldens + sharding/llama8b/ascent tests green).

**A/B trace at b128/dp32 (n_warmup=2), disable-remat baseline:**
| | baseline (p-6733c236) | ASCENDREP (p-8de78b2b) | Δ |
|---|---|---|---|
| step wall | 8.362 s | **8.007 s** | **−4.2%** |
| gather union | 5551.7 ms | 5023.1 ms | −528 ms (−9.5%) |
| span | 8364 ms | 7425 ms | −11.2% |

- **No OOM at b128** (the 2026-06-26 full-stack OOM was the WHOLE-step replicate; here it's
  ascend-only, and the ascend phase is param-detached / low-activation so it has the headroom).
  The 8 Tracebacks in the log are the known cosmetic `_write_perfetto_trace_file` crash
  (post-steps), NOT an OOM.
- Modest (4%) because the ascend is only `n_warmup=2` forwards; the trick canNOT extend to the
  main value_and_grad (live weights need grads; full-replicate there = the whole-step OOM). This
  is the gather-COUNT lever working as predicted (vs protocol/bytes which don't), bounded by the
  ascend's share. Stacks on disable-remat (8.3→8.0s).
- **Ship gate: peak-HBM A/B in flight** (MEMBASE 138595 vs MEMASCREP 138602) — confirms whether
  the +full-V/U-during-ascend leaves enough headroom for production save-time fragmentation.

### Peak-HBM A/B: lever #5 is FREE (zero memory cost) — SHIP IT
| | peak HBM | temp | headroom (limit 164.08) |
|---|---|---|---|
| baseline (MEMBASE 138595) | 148.76 GiB | 99.25 GiB | 15.3 GiB |
| ASCEND_REPLICATE (MEMASCREP 138602) | **148.48 GiB** | 98.97 GiB | 15.6 GiB |

The full-V/U-during-ascend adds ZERO to the peak (marginally LESS) — the global peak lives in the
main value_and_grad (~148 GiB) and the ascend phase sits well under it, so the extra full stack
(freed before the main backward) never raises the ceiling; the per-ascend-iteration transient
gathers it replaced were slightly costlier. **Verdict: +4% wall, zero memory cost, bit-identical
numerics, 15 GiB headroom → strictly better, safe to ship.**

**Delivery:** the code is `param_decomp/train.py` (`PD_ASCEND_REPLICATE`, off by default). To
enable for a run, add `runtime.launch_env.env.PD_ASCEND_REPLICATE: '1'`. NOTE: the base training
config `llama8b_full32L_HSDP_b128_dp32.yaml` currently has NO `launch_env` — it lacks even the
disable-remat 1.6× win (which lives only in the MAXTEXTFLAGS trace config). Promoting wins into
the canonical training config is a separate deliberate step (Oli's call), so the production config
was NOT edited — the lever is implemented + proven, ready to wire in.

## ★ STRATEGIC CORRECTION (Oli, 2026-06-30): coalesce + overlap, NOT frontload-full-gather
Lever #5 (`PD_ASCEND_REPLICATE`) is mechanically **coalesce + HOLD** — it collapses the ascend's
per-layer gathers into one, but then HOLDS the full-replicated V/U for the 2 ascend forwards.
That "hold" is **lazy DDP** — it defeats (H/F)SDP and does not scale (it's free only because the
ascend phase has HBM slack under the main peak at THIS size; U/V are 4× the model, CI fn 10× —
you can never hold them full in general). So #5 is at most a narrow parked opt-in, NOT the
strategy and NOT to be shipped to the production config as the headline.

**The principled lever = coalesce + DON'T hold (keeps ÷fsdp):** per transformer block there are
~**21 separate weight all-gathers** (7 frozen W + 7 V + 7 U). Coalesce them into ONE collective
per block (gather → slice back → free per block, stays ÷fsdp), and overlap `gather(L+1)` under
`compute(L)`. This keeps FSDP intact, applies to the MAIN value_and_grad (the big gather chunk,
not just the 2 ascend forwards), and scales. It attacks the count/contention diagnosis on both
axes (count 21→1, contention via overlap). XLA's all-gather combiner is configured
(`gpu_all_gather_combine_threshold_bytes=128MB`) but UNVERIFIED — next step is to inspect the
scan-body HLO and count the actual gathers/block before deciding manual concat vs combiner.
Param scale (target 1× / U/V 4× / CI fn 10×) is why holding-full is off the table.

## ★★ COALESCE LEVER (the real one): all_gather_combine_threshold 128MB → 1GB
Per-block there are ~21 weight gathers; XLA's combiner was capped at 128MB. The HLO showed
886/1156 gathers were arity-1 (uncoalesced), with 363 parked at 96–128MB (throttled by the cap)
and split into two pools by gather DIMENSION (V/Wd/wo shard axis-0 → dim-0; U/Wg/Wu/wqkv shard
axis-1 → dim-1; combiner can't merge across dims). Our config also capped all_gather at 128MB
while all_reduce was already 1GB — an unjustified 8× asymmetry.

**Raising `gpu_all_gather_combine_threshold_bytes` 128MB→1GB (trace p-4a1e5a25):**
| | all-gather-start | arity-1 singletons | wall |
|---|---|---|---|
| baseline 128MB | 1156 | 886 | 8.362 s |
| **1GB** | **475 (−59%)** | **215 (−76%)** | **7.883 s (−5.7%)** |
Combiner now fuses up to 27 tensors/gather. **One XLA flag, no code change, no held stack, no
DDP concern** — strictly better than the ascend hack (8.007s) and the right kind of lever
(count reduction, FSDP intact). Peak-HBM probe in flight (bigger combined gathers → more
transient HBM; ran fine at b128, confirming no OOM, peak TBD).

**Remaining 215 singletons** = the individually-over-1GB matrices (down/gate-up, C huge) + the
dim-split residue. Next structural step = Idea 1 (`[d_in+d_out, C]` per site → uniform dim-0,
merges the two pools + halves target site gathers) and the CI-fn `out_ws` re-glue (CI fn is 10×
→ dominates the gather count). Staging: threshold (done) → Idea 1 → CI-fn re-glue → cross-site/QKV.
