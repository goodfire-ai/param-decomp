# Scaling-book notes — applied to full-32L VPD on B200

Source: DeepMind, *How to Scale Your Model* (jax-ml.github.io/scaling-book). Read in
full 2026-06-26 (roofline + training directly; gpus/sharding/transformers/jax/profiling/
applied-training via fan-out). This distils only what bears on **our** decision: which
(dp, tp) topology and batch for the full Llama-3.1-8B decomposition on CoreWeave B200.

Companion to `FULL32L_MEMORY_MODEL.md` (resident-state memory accounting).

---

## 0. The one-line answer to "are we flailing?"

**No — we're in a structurally hard corner of the design space, and we've been pushing the
wrong lever.** The book gives a clean procedure (below). Run through it, our situation is:

- Our batch-per-chip is ~10–100× **below** the compute-bound floor → we are firmly
  **comms-bound**, and that is expected, not a bug.
- We got there by **scaling node count up** (to shard resident memory / chase batch). On a
  GPU cluster that is exactly the move that deepens comms-boundedness: every added node
  pushes the heavy weight-gather traffic onto the slow inter-node link and lowers tokens/chip.
- The book's prescription for our regime (fixed-ish small batch, memory-heavy, many chips)
  is the **opposite of our instinct**: use the *fewest* nodes memory allows, fill the freed
  HBM with batch, keep all heavy collectives on NVLink (within a node), cross nodes only
  with the cheap async gradient all-reduce.

So the cure for the "flailing" feeling: stop hunting for a compute-bound config (it may not
exist at our batch×memory) and instead use the formulas to find the **least-comms-bound**
point and commit.

---

## 1. B200 hardware numbers (the roofline inputs)

| quantity | B200 | H100 (contrast) |
|---|---|---|
| bf16 dense FLOP/s per GPU | **2.25e15** (measured peak ~1.7e15) | 0.99e15 |
| HBM capacity | **192 GB** | 80 GB |
| HBM bandwidth | 8.0 TB/s | 3.4 TB/s |
| NVLink egress / GPU (intra-node) | **~900 GB/s** | ~450 GB/s |
| InfiniBand egress / **node** (inter-node) | ~400 GB/s (**~50 GB/s per GPU**) | ~400 GB/s |
| NVLink domain (= TP ceiling) | **8 GPUs** (HGX) / 72 (GB200 NVL72) | 8 |

The cliff that governs everything: **NVLink is ~18× the per-GPU bandwidth of IB**
(900 vs 50 GB/s). The node boundary is a hard bandwidth cliff, not a gentle slope.

---

## 2. The procedure (book's decision recipe)

For a fixed model, cluster, and target batch B (tokens):

1. **Pure DP?** Only if params+optimizer fit on one chip. They don't for us → no.
2. **Pure FSDP?** Compute-bound iff per-chip batch `B/N > C/W ≈ 2550/M` (TPU number; on
   GPU use C/W below). If below → comms-bound → no.
3. **Mixed FSDP+TP?** Lowers the floor ~8×: compute-bound iff `B/N > α²/(M_X·M_Y·F)`.
   Optimal split: **`X_opt(FSDP) = sqrt((B/F)·(M_X/M_Y)·N)`**, `Y_opt(TP) = N/X_opt`.
4. **TP ceiling:** comms-bound when `Y > F/(C/W)`. For us F=14336, C/W_nvlink≈2500
   → **Y ≲ 5–8**. We cap TP=8 (one node). ✓ Already correct.
5. **Across nodes:** do model-parallel (TP/PP) *within* a node, pure DP *across* nodes.

### Operational intensities for us (C/W, "tokens per chip to be compute-bound")
- Single matmul on one B200: **~300 tokens** (the `B>240` GPU analogue).
- **FSDP/DP over NVLink** (intra-node): `2.25e15 / 9e11 ≈ 2500 tokens/GPU`.
- **FSDP/DP over IB** (cross-node): `2.25e15 / 5e10 ≈ 45,000 tokens/GPU` — the killer.
  (cf. the book's TPU cross-pod DCN figure of ~73k tokens/chip — same order; crossing the
  slow link always demands a huge per-slice batch.)
- **Mixed FSDP+TP** floor: `α²/(M_X M_Y F)` — roughly **~100 tokens/chip on TPU**;
  on GPU, dominated by the IB α, materially higher.

---

## 3. Where WE sit (plug in our numbers)

Current runs: per-DP batch 1 sequence = **512 tokens/GPU**, ~**4096 tokens/node**.

- vs single-matmul floor (~300 tok): fine, the matmuls themselves are compute-bound.
- vs **cross-node FSDP floor (~45k tokens/GPU)**: we are **~90× under**. Any step that
  gathers/scatters weights across IB at this batch is overwhelmingly comms-bound.
- The 256-GPU config (tp8/dp32, 32 nodes) at 2.23 s/step is consistent with being
  dominated by **cross-node weight movement**, not compute.

**Why scaling out hurt us:** FSDP moves *weights* (700 GB+ resident, see §4), TP moves
*activations*. Spreading resident state across 32 nodes means the weight all-gather/
reduce-scatter rides the **50 GB/s IB** link every step. More nodes = thinner shards but
the *same total weight bytes over a slower aggregate path per token* = deeper comms-bound.

---

## 4. The memory ⟷ batch ⟷ node-count squeeze (our actual constraint)

The book assumes activation memory dominates and rematerialization makes capacity a
non-issue. **For us it's inverted: resident state dominates.**

- V/U + Adam ≈ **~700 GB**. CI function ~31B params × 14 B ≈ **~430 GB**. Total resident
  **~1.1 TB**.
- One B200 node = 8 × 192 = **1536 GB HBM**. Resident ~1.1 TB leaves ~430 GB for
  activations+overhead across 8 GPUs (~50 GB/GPU). **Plausibly fits in 1–2 nodes.**
- This is *why the second-half model fit on one node* and surprised us: halving resident
  (~550 GB) drops comfortably onto 8×192. The full model should need ~2 nodes for memory.

### Empirical verdict (2026-06-26): tp is a *memory* requirement, not just a comms knob

- **B32 tp1/dp32 (4 nodes, job 130424): OOM** — single 94.86 GiB allocation failed before
  step 1. With tp=1 the ~31B CI function is **not** Megatron-sharded, so the per-chunk CI
  gather is `param/tp = param/1` (full) → blows up. This is the "hoisted CI gather ∝ param/tp
  kills low-tp" lever, now confirmed.
- **B64 tp2/dp64 (8 nodes, job 130423): healthy** — cleared faith warmup (final faith
  3.56e-4), compiling main step. tp=2 shards the CI fn enough to fit.

So **tp ≥ 2 is a hard memory floor** (independent of comms): §5's "could we drop tp?" is
bounded below by the CI-fn size. The real question is the *minimum tp that fits*, then
spend the rest on dp/batch. The §2 TP comms ceiling (≲8) is the upper bound; the CI-fn
gather is the lower bound. Our usable TP window is roughly **2 ≤ tp ≤ 8**.

**Consequence — the lever we've been ignoring:** the memory floor is ~2 nodes, not 32.
Running on 32 nodes does **not** buy correctness; it spreads resident thin (memory
headroom we don't need) at the cost of forcing every weight collective across IB. The
right move is to run on the **fewest nodes memory allows** and spend the recovered HBM on
**batch** — which simultaneously raises tokens/chip (toward compute-bound) and keeps the
heavy FSDP traffic on NVLink.

A 2-node config (16 GPUs) at our 16k-token batch = **1024 tokens/GPU** — within striking
distance of the *intra-node* mixed floor, versus ~90× under the *cross-node* floor at 256
GPUs. Same batch, ~10× better arithmetic intensity, just by not scaling out.

---

## 5. Levers, ranked

1. **Shrink node count to the memory floor (~2 nodes).** Biggest win; keeps weight
   collectives on NVLink. Inverts our "add GPUs" instinct. Verify the true memory floor
   with an AOT peak probe (compile-only) at 1 / 2 / 4 nodes.
2. **Raise batch to fill freed HBM.** Both raises tokens/chip and amortizes comms. Capped
   above by activation memory and by the science (critical-batch diminishing returns).
3. **Keep TP=8, within node.** Already correct. Note our TP is heavier than the book's
   vanilla TP — it also carries the CI-function Megatron activations — so don't push past 8.
4. **Cross nodes only with async gradient all-reduce (pure DP), not FSDP weight-gather.**
   Viable *iff* resident fits FSDP-sharded *within* a node. This is the structural prize.
5. **Pipeline parallelism across nodes** (LLaMA-3 did TP8 × PP16 × DP128). GPU-native, cheap
   cross-node. But conflicts with ZeRO-3 (need ZeRO-1 → more weight memory) and sits awkwardly
   with our scan-over-layers + frozen-target + recon structure. Flag, don't adopt yet.
6. **Overlap comms with compute** (collective-matmul / `shard_map`). Only after a profile
   says comms isn't already overlapped.

---

## 6. What the book does NOT model for us (measure, don't assume)

Our step is 2.23 s, not microseconds, because of structure absent from the book's plain
MLP roofline:
- The **CI function** (its own Megatron-TP activation comms on the tp axis).
- The **PPGD adversary** (N ascent steps → N× forward).
- **Chunkwise recon** (extra suffix forwards, KL on final logits).

So the roofline is necessary-not-sufficient. **Action: take one real profile** (book's
profiling chapter + our HLO-dump habit): `jax.jit(step).lower(...).compile()` →
`.as_text()` + `.cost_analysis()`; capture an xprof trace; attribute time to
compute / NVLink-comms / IB-comms / remat. Confirm the weight-gather hypothesis before
re-architecting. Look in HLO for cross-node `all-gather`/`reduce-scatter` (`replica_groups`
spanning nodes) and for involuntary remat.

---

## 7. Concrete next actions

- [ ] AOT peak-memory probe (compile-only) at **1, 2, 4 nodes** → find the true memory floor.
- [ ] One **profile** of a real step at the current topology → compute/comms/remat split.
- [ ] Re-run the **fixed-global-batch tp sweep** with §2 as the prior (expect: fewest nodes
      that fit + TP=8 + batch maxed wins; cross-node FSDP loses).
- [ ] If resident fits FSDP-within-a-node: prototype **DP-across-nodes / FSDP+TP-within-node**.
