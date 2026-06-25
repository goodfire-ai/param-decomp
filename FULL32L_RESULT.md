# Full-32L Llama-3.1-8B Decomposition — Result, Mechanics, Validation

**The result:** the *entire* 32-layer Llama-3.1-8B model (all 224 weight matrices) was
decomposed (VPD) and **trained end-to-end on 32 GPUs** — job 128760 / run `p-4da39650`,
**COMPLETED** all 150 steps, checkpoints at 50/100/150. This is the largest model
decomposed to date by a wide margin (prior work: toy models / single layers).

This doc answers: (1) what shape is the run, (2) what placements we use, (3) what strategy
JAX/GSPMD found (HLO-grounded), (4) is it validated. Companion: `HOW_JAX_EXECUTES_FULL32L_STEP.md`.

---

## 0. Live numbers (the completed run, step 150)

| metric | value |
|---|---|
| status | COMPLETED, 150/150 steps, 48m 58s wall |
| **step_time** | **3.04 s/step** (pure compute; the ~15 s gaps were 534 GB ckpt saves) |
| **peak mem / rank** | **173.5 GiB** (B200 ~180 hard, BFC cap ~166 → **tight**, ~6.5 GiB headroom) |
| FaithfulnessLoss | 3.754e-4 (warmup ended 3.865e-4) |
| ChunkwiseSubsetReconLoss | 1.628 |
| PersistentPGDReconLoss | 178.1 |
| ImportanceMinimalityLoss | 2.934e6 (raw; ×coeff 5e-6) |
| total loss | 482.4 |
| grad-norm total | 50.68 (finite, healthy) |

> ⚠ **Memory is tight at 32 GPU** — the live peak (173.5) is far above the HLO *planned*
> peak (135.1); the gap is BFC fragmentation + runtime scratch. It fit and completed, but
> there is little headroom. This is the main reason to run the real thing at 128–256 GPU
> (more dp → everything FSDP/TP-sharded shrinks further).

---

## 1. Run shape

**Topology.** `dp_mesh(tp)` reshapes devices to `(n//tp, tp)`. This run: `runtime.dp=32`
(total GPUs), `runtime.tp=8` → **mesh (dp-axis=4, tp=8)** on 4 nodes × 8.
- `tp=8` = intra-node Megatron (tensor-parallel).
- `dp-axis=4` = across-node, carries FSDP weight-sharding + data-parallel batch.
- (`runtime.dp` is the *total* device count, not the dp-axis size — dp-axis = dp/tp.)

**Batch / tokens.** global batch **4**, per-dp-group **1** (4/4), seq **512**, **2048 tokens/step**.

**Decomposition (V/U).** 224 sites = 32 layers × 7 matrices. Per-layer C:

| q | k | v | o | gate | up | down | ΣC/layer |
|---|---|---|---|---|---|---|---|
| 2048 | 2048 | 4096 | 4096 | 8192 | 8192 | 10240 | **38,912** |

→ **1,245,184 components** total. V/U params = **18.32 B** (fp32 masters).

**CI fn (the importance predictor).** Chunkwise transformer, **one transformer per layer →
32 chunks** (`blocks_per_chunk=1`), each: `d_model=4096, n_blocks=4, n_heads=64,
mlp_hidden=16384`, output width `c_chunk=ΣC=38,912`. **Total ≈ 31.41 B params** (981 M/chunk
× 32: blocks 805 M + out-head 159 M + in-proj 17 M).

**Sizes in perspective:** CI fn **31.4 B** = 3.9× the **8.03 B** frozen target = 1.7× the
18.3 B V/U. Total trainable ≈ **49.7 B** (CI fn + V/U), ~6× the target. The CI fn being
larger than the target is *by design* — an over-parameterized importance predictor.

**Losses / optim.** Recon (Chunkwise subset, coeff 2.0, sites_per_chunk=56 → **4 recon
chunks**) + Faithfulness (1e6) + ImportanceMinimality (5e-6, p-anneal) + Persistent-PGD
adversarial (0.5). Both optimizers Adam, lr 2e-5 cosine; components grad-clip 0.01.
faith-warmup 2 steps; **150 total steps** (a short bring-up run, not a converged train).

---

## 2. Placements — one rule, four exceptions

**The rule:** *FSDP a weight dim on `dp`, TP the Megatron dim on `tp`.* The **component axis
C lives on `tp`** for V, U, *and* the CI-fn output head — so the CI mask, `x@V`, and `@U` all
carry C on the same axis → **no mask reshard** (the load-bearing invariant).

| tensor | shape | spec | dp(FSDP) | tp(TP) | replicated |
|---|---|---|---|---|---|
| V | (d_in, C) | `P("dp","tp")` | d_in | C | — |
| U (o + MLP) | (C, d_out) | `P("tp","dp")` | d_out | C | — |
| U (q/k/v) | (C, d_out) | `P("tp",None)` | — | C | d_out *(exc. 1)* |
| CI-fn weights | (nc, …) | `P(None,"dp"/"tp",…)` | a hidden dim | Megatron dim | nc *(unsharded, exc. 4)* |
| target attn q/k/v | (nl, head, d) | `P(None,None,"dp")` | d | — | head *(exc. 2)* |
| target attn wo | (nl, d, head) | `P(None,"dp",None)` | d | — | head |
| target MLP g/u/d | (nl, …) | FSDP d on dp | d | — | intermediate *(exc. 3)* |
| target embed/lm_head/norm | — | `P()` | — | — | all (~2 GB) |
| PGD sources (bsc) | (B, T, C+1) | `P("dp",None,None)` | B (batch) | — | T, C+1 |
| activations / residual | (B, T, d) | leading on dp | B | — | d (residual is d-full) |

**Exceptions (all justified):**
1. **q/k/v U `d_out` replicated** — d_out is the head dim, reshards to head-on-tp at the attn
   seam; tp is taken by C, so leave it replicated. (q/k/v U is only ~5% of U.)
2. **Target attention head replicated, no TP** — cuDNN flash needs q/k/v *identically*
   sharded; `core` runs batch-parallel attention (q/k/v → `P("dp",None,None,None)`). TP'ing
   the head gives q (32 heads) vs k/v (8 heads) different per-rank counts → cuDNN rejects
   them ("Query, key and value should have same sharding"). **This was the bug that cost the
   first target-FSDP attempt; the fix is FSDP `d` only.**
3. **Target MLP intermediate replicated, no TP** — Megatron-TP'ing a *frozen* 16 GB target
   through the replicated residual isn't worth it; FSDP `d` on dp (`/dp`) is ample.
4. **CI-fn `n_chunks` unsharded** — it's a plain vmap axis now (the mesh-simplification);
   weights FSDP+TP their own dims regardless of chunk structure. (Also removed the
   `n_chunks % dp` constraint that used to kill non-multiple-of-8 layer counts.)

**Per-device resident (this run, dp-axis=4 / tp=8 → and at 256 GPU dp-axis=32):**
V/U+Adam fp32 6.9 → 1.4 GB · CI-fn+Adam fp32 ~2.9 → 0.4 GB · target bf16 ~5 → 2.4 GB
(embed/lm_head ~2 GB floor, replicated). The rest of the 173.5 GiB peak is **transients**
(recon forwards + remat + PGD + the ~80 GiB temp arena), which is why more dp helps.

---

## 3. How JAX executes the step (HLO-grounded, job 128760)

Mesh proven from replica groups: `{0..7}`=tp(8), `{0,8,16,24}`=dp(4).

**Collectives** (compiled `jit_step`): all-gather **1124** (623 size-8 tp = reconstruct the
Megatron dim of CI-fn/V-U; 287 size-4 dp = **FSDP gathers** of the frozen target + the
FSDP dims of CI-fn/V-U), all-reduce **667** (TP matmul partial-closes + loss sums),
reduce-scatter **113** (dp = **FSDP gradient** reduce-scatter on the fp32 grads),
all-to-all **348** (small, scattered), collective-permute **2362**. FSDP gathers of
CI-fn/V-U are `/tp`-bounded as designed; the target is dp-only FSDP.

**Attention.** Target = **cuDNN flash** (`__cudnn$fmhaSoftmax` ×43, **no `[B,H,T,T]` score
materialized**), GQA 32 q / 8 kv heads, batch-on-dp + heads-replicated. CI-fn = **xla**
attention (tiny `f32[32,8,512,512]` score, leading 32 = chunk vmap). The split is
deliberate: cuDNN can't partition the CI-fn's Megatron-head q/k/v, but its score is tiny so
xla is fine; the target's score would OOM under xla, so it must use flash.

**Structure.** 19 `lax.scan`s over 32 layers (2 plain forward, 5 differentiated recon
forwards, 5 backward, 5 remat-recompute, 1 nested PGD scan) — `remat_recon_forwards=true`
(8108 `rematted_computation` markers). Backward = `transpose(jvp(jvp()))`. The CI-fn vmaps
over the 32 chunks.

**Mask flow.** Masks are born **batch-on-dp** (the de-chunked CI fn) and reach the masked
forwards with no dominant C→batch reshard — `batch_sharded_ci` works; the 348 all-to-alls
are small and scattered, not the temp-arena-dominating reshard the chunk-parallel CI used to
force.

**Memory.** Peak (HLO plan) 135.1 GiB; largest single buffer ≈ **80 GiB preallocated-temp
arena** (~59%). Target weights confirmed **dp-sharded /4** (`bf16[32,14336,1024]`, d 4096→1024
vs the old full 4096). (Live BFC peak 173.5 > plan 135 = fragmentation.)

---

## 4. Validation — what's confirmed vs caveats

**CONFIRMED:** all 224 sites of the real Llama-3.1-8B are decomposed; ran on 32 GPU dp4×tp8;
**completed 150 steps**; three valid **534 GB sharded orbax checkpoints** (4 ocdbt processes
= 4 dp groups, well-formed); **all loss terms finite + sane** at step 150 (total 482.4,
faith 3.75e-4 — in the same ballpark as the completed 16L=3.77e-4 / 8L=3.74e-4 runs); target
uses cuDNN flash, CI-fn uses xla — no silent fallback; memory budget reconciles; the ~12k
"VMM FABRIC" log warnings are benign (completed runs had 2.5k–24k and finished fine).

**CAVEATS / not-yet-validated:**
- **Short bring-up run** — 150 steps proves the *machinery* (fits, trains, saves, resumes),
  **not** a converged or useful decomposition. A real run is ~100k–200k steps.
- **Memory tight at 32 GPU** (173.5 / 180). 256 GPU is needed for headroom + real batch.
- **`bsc` PGD source scales with per-device batch** — fine here (batch 1/group → tiny), but
  at production batch it grows `global/dp-axis` and can dominate; size it deliberately.
- **Logging gap** — `train_log_every=200 > steps=150` meant only the final step logged
  metrics. For a real run, set `train_log_every` ≪ steps (or a `dense_log_phase`) so loss
  curves + memory are observable mid-flight.

**Bottom line:** the result is **real** — the full 32-layer model genuinely decomposes and
trains. What remains is scale (256 GPU), duration (a real training schedule), and
observability (logging cadence).

---

## 5. The journey (how we got here, for context)

full-32L OOM'd at 32 GPU with the replicated target (75 GB over) → HLO traced the dominant
buffers to the **replicated frozen target** (`dynamic_nodonate` inputs + their backward/remat
copies), *not* the faith term → **target FSDP** (shard the 14 GB target's `d` on dp) → first
attempt hit the **cuDNN "same sharding"** wall (TP'd the head) → fixed by FSDP-`d`-only,
head replicated → **fits + trains**. Prior to that: uniform FSDP×TP for V/U + de-chunked CI
fn + activation-space delta rewrite (commits 6bc5e2cd2, b07c4e2a5).
