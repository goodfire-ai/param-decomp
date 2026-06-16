# Investigation: JAX run p-7847c8aa diverges after ~310k (baseline p-20f9fc15 does not)

## What we already know (grounded in the run data)

JAX run **p-7847c8aa** replicates torch **p-20f9fc15**. It tracks the baseline almost
exactly until ~step 310k, then diverges; the baseline stays flat to 400k.

**It is not a resume artifact.** The job ran straight through in one allocation: exactly
one training banner, one faith-warmup, no requeue/restart. The GRPC "connection refused"
lines are ranks exiting at teardown after step 400k.

**The signature is a sparsity-driven collapse, led by an exploding CI-fn gradient:**

| step | total | recon | ppgd | impmin(no_beta) | grad ci_fns | grad components |
|---|---|---|---|---|---|---|
| 300k | 24.6 | 0.23 | 0.53 | 162 | 0.16 | 0.21 |
| 320k | 24.6 | 0.25 | 0.67 | 149 | 0.25 | 0.22 |
| 340k | 24.7 | 0.31 | 0.77 | 114 | **0.87** | 0.26 |
| 360k | 25.4 | 0.63 | 2.08 | 70 | **4.8** | 0.45 |
| 380k | 28.2 | 1.93 | 6.41 | 34 | **16.5** | 1.14 |
| 400k | 30.8 | 3.15 | 10.4 | 15 | **57.3** | 2.11 |

(torch baseline at every one of these steps: total ≈24.6, recon ≈0.23, ppgd ≈0.57,
impmin ≈1100 — flat.)

Reading: the **CI-fn gradient norm explodes ~300×** while component grads grow ~10×. The
importance-minimality term collapses (CI → 0, components dying); reconstruction and PPGD
losses blow up as a *consequence* of components dying. `total` is dominated by the
faithfulness term (1e7 × 2.4e-6 ≈ 24, flat), which is why the collapse is invisible in
`total` until ~340k.

**Timing correlates with the pnorm anneal.** `p` anneals linearly 2.0 → 0.4 over the full
run (`annealed_pnorm`, `losses.py:52`). Onset (~310–340k) is where `p` drops below ~0.8.
The Lp penalty gradient w.r.t. CI is `p·(ci+eps)^(p-1)`; for `p<1` and `ci→0` this
*explodes* (exponent negative). The **CI-fn optimizer has no gradient clipping**
(`ci_fn_optimizer.grad_clip_norm: null` → `build_optimizers`, `run_state.py:37` is a bare
`adamw`), while components are clipped at 0.01 (`run_state.py:33-36`). So nothing contains
an exploding CI-fn gradient.

**Importance-min is already fp32** (`losses.py:43`), so a naive bf16-in-the-loss
explanation is out. But CI *values* come from the CI-fn forward, which runs in bf16
(COMPUTE_DT), then cast to fp32 — bf16 rounding of near-zero CI is amplified by the low-p
gradient.

**Crucial caveat — same config, both use bsc.** torch p-20f9fc15 *also* used
`per_batch_per_position` (bsc) and the *same* schedule and the *same* no-CI-clip config,
yet stayed stable. So "no CI-fn clip" is necessary-but-not-sufficient: the system is
**metastable at low p**, and something JAX-specific (bf16/float-reassociation numerics, or
a bug in the new bsc path) tips it over the edge torch sits behind.

**Asset constraints.**
- p-7847c8aa kept only `ckpts/395000` and `ckpts/400000` (`keep_last_n=2`) — both
  *post-collapse*. No pre-onset checkpoint exists.
- p-7847c8aa is the **only JAX run that ever reached >310k** (all others stopped ≤56k), so
  history can't tell us whether non-bsc JAX also collapses. A controlled run is required.

## Hypotheses (ranked)

- **H1 — Low-p Lp instability on an unclipped CI-fn, tipped by JAX bf16 CI numerics.**
  The leading explanation given the grad-norm signature and the p correlation. Predicts a
  fix via CI-fn grad clipping and/or an fp32 CI path and/or eps in the Lp term.
- **H2 — bsc persistent-adversary feedback (the new code).** The bsc batch-sharded
  persistent sources interact with low-p sparsity to drive the runaway. Must be tested for
  bsc-specificity since this is the freshly added path.
- **H3 — importance-min formula / eps / p-anneal mismatch vs torch at low p.** Equivalence
  tests exist but may not cover `p<1`, the entropy term at scale, or the gradient. A
  subtle divergence would only bite late.
- **H4 — CI-fn forward / `leaky_hard` squashing bf16 pathology** producing near-zero CI
  that the Lp gradient amplifies (a specific mechanism under H1).
- **H5 — accumulated bf16 / optimizer drift** unrelated to p (least likely — onset tracks
  p, not wall-time).

## Investigation plan

### Phase A — Post-mortem on the surviving checkpoints (fast, read-only)
Use `ckpts/395000` + `ckpts/400000` (load via `run_state.init_train_state` +
`checkpoint` restore; `export.py`/`tests/test_checkpoint.py` show the restore idiom).
- Per-site CI distributions: histogram of `ci_upper`, fraction of components with
  CI≈0 (dead), which sites/layers died first. Confirms the "components dying" reading.
- Inspect the **bsc persistent sources** (`TrainState.sources`) and their Adam state
  (`sources_opt_state`): magnitudes, saturation at the [0,1] clamp, NaN/Inf, per-element
  drift. Tests H2 directly.
- Component (V/U) norms and CI-fn weight norms vs a healthy early checkpoint if any can be
  recovered (else vs init scale).
- Recompute the Lp gradient w.r.t. CI at p(395k)≈0.42 on the 395k CI tensors; confirm the
  magnitude matches the exploded grad-norm and localize it to lp vs entropy.

### Phase B — Build a fast, faithful repro (the testbed for all fixes)
No pre-onset checkpoint exists, so:
- **B1 (ground truth):** re-run the exact config but with `keep_last_n` raised and
  `save_every` denser around 280–340k, and stop at ~360k (~9h, or fewer GPUs). Gives real
  checkpoints straddling the onset and a definitive repro.
- **B2 (fast iteration):** a short "accelerated-anneal" harness — same model/losses but
  `steps` ~3–5k with `p_anneal` 2.0→0.4 compressed, from a warm checkpoint (395k is wrong
  side; better: a mid-training checkpoint produced by B1). Use to iterate on fixes in
  minutes. Validate every B2 conclusion against the B1 signature before trusting it.

### Phase C — Discriminate hypotheses (short runs on the Phase-B testbed)
- **C1 (H1):** add CI-fn gradient clipping (start at the components' 0.01; also try a
  looser cap) in `build_optimizers`. Does the runaway stop and the trajectory rejoin
  baseline? Cheap, high-information.
- **C2 (H1/H4):** force the CI-fn forward / CI path to fp32 (cast in `ci_fn.py` /
  `train.py`); re-run. Isolates bf16-CI as the tipping factor.
- **C3 (H3):** extend `tests/equivalence` to compare JAX vs torch **importance-min loss
  *and its gradient*** on identical CI tensors containing near-zero entries, at p ∈
  {0.4,0.6,0.8}. Confirms/denies formula+eps+anneal parity in the regime that matters.
- **C4 (H2 — bsc-specificity):** run a non-bsc leg to the low-p regime — easiest is the
  existing fresh-PGD config `pile_llama_simple_mlp_4l_pgd1.yaml`, and/or an `sc`-persistent
  variant — under the Phase-B accelerated anneal. If it *also* collapses, the cause is
  general-JAX, not bsc; if only bsc collapses, focus on the bsc adversary.
- **C5 (H2):** instrument the bsc source ascent near onset (source norms, grad norms,
  fraction clamped) to see whether the adversary is the driver or a passenger.

### Phase D — Confirm root cause & land a fix
- Pick the minimal change that makes the Phase-B repro track baseline (most likely
  CI-fn grad clipping and/or fp32 CI; or a genuine bsc fix if C4/C5 implicate it).
- Validate with a from-checkpoint continuation (from a B1 pre-onset checkpoint) through
  400k matching the torch baseline (impmin stays ~1100, recon ~0.23).
- If the fix is "add CI-fn clipping," note it changes the algorithm vs the stored config —
  decide with the team whether to (a) treat it as a JAX-trainer robustness fix, or
  (b) chase exact-parity numerics so the unclipped config is stable as in torch.

## Key files
- `param_decomp_jax/jax_single_pool/losses.py` — `importance_minimality_terms`,
  `annealed_pnorm` (the Lp penalty + schedule).
- `param_decomp_jax/jax_single_pool/run_state.py:28-38` — `build_optimizers`
  (CI-fn = unclipped adamw; components = clip 0.01).
- `param_decomp_jax/jax_single_pool/train.py` — `loss_fn`, grad flow, dtype of CI path.
- `param_decomp_jax/jax_single_pool/ci_fn.py` — CI-fn forward + `leaky_hard` squash dtype.
- `param_decomp/metrics/` (torch `ImportanceMinimalityLoss`) — formula parity reference.
- `param_decomp_jax/jax_single_pool/adversary.py` — bsc persistent sources (H2).

## Precision audit: JAX vs torch-`main` on the training step

Traced every dtype on the step path. **CI values are bf16 in both** (CI-fn forward +
`leaky_hard` squash run in bf16 — JAX `cast_floating(ci_fn, COMPUTE_DT)` `train.py:368`;
torch autocast). The differences are in the **loss computations** and the **frozen
target**:

| Quantity | torch-`main` | JAX | same? |
|---|---|---|---|
| CI-fn forward + leaky_hard squash | bf16 (autocast) | bf16 (`COMPUTE_DT`) | ✅ |
| ci.upper / ci.lower values | bf16 | bf16 | ✅ |
| **Importance-min `(ci+eps)^p` + Σ over positions** | **bf16** (no `.float()`; `importance_minimality.py:64-65`) | **fp32** (`ci.astype(f32)`, `losses.py:43-47`) | ❌ |
| **Importance-min entropy term** | bf16 | fp32 | ❌ |
| **Recon / KL loss (logits → KL)** | **bf16** | **fp32** (logits cast, `losses.py:12-13`) | ❌ |
| Faithfulness (Σ delta²) | fp32 (deltas built outside autocast) | fp32 (built from fp32 `components`) | ✅ |
| Component V@U / masked forward matmuls | bf16 (autocast) | bf16 (`components_bf16`) | ✅ |
| **Frozen-target / clean-logits (recon target)** | **fp32** | **bf16** | ❌ |
| Optimizer masters + Adam state | fp32 | fp32 | ✅ |
| Grad clip — components / ci-fn | 0.01 / **none** | 0.01 / **none** | ✅ |

(One non-dtype note: `main`'s entropy uses `log2(1 + layer_sums·world_size)` on per-rank
sums; JAX's `jnp.sum` is already the global sum under GSPMD, so JAX matches `main` there.
The *worktree* torch copy dropped the `world_size` factor, but we don't run torch.)

**Prime suspect — importance-min is fp32 in JAX, bf16 in torch.** This is the dtype
difference most directly tied to the observed failure (exploding **CI-fn** gradient, which
the importance-min term drives). Mechanism: at low `p` (<1) the Lp gradient
`p·(ci+eps)^(p-1)` is huge as `ci→0`. Torch sums `(ci+eps)^p` over the 32,768
batch×seq positions **in bf16** — bf16's 8-bit mantissa drops most small contributions
once the partial sum grows (granularity at sum≈16k is ~128), so torch's sparsity
loss/gradient is systematically **undercounted and blurred**. JAX computes the Σ in fp32:
the **true, sharp** low-p gradient. With **no CI-fn gradient clip**, that accurate-but-
explosive gradient drives the CI-fn runaway → components die → reconstruction collapses.
So, counter-intuitively, JAX being *more numerically faithful* is what tips a system that
torch's lossy bf16 reduction was accidentally damping. This predicts the onset timing
exactly (it begins when `p` crosses below ~0.8).

**Decisive experiment (top of Phase C):** make JAX's `importance_minimality_terms` compute
in bf16 (don't cast `ci` to fp32; do `(ci+eps)^p` and the Σ in bf16, matching torch). If
the 40k repro then stays stable, precision is confirmed as root cause. Cross-check by
adding CI-fn grad clipping (the other thing that would contain it). Secondary: the
recon-KL fp32-vs-bf16 and frozen-target bf16-vs-fp32 differences (less likely to drive a
*CI-fn*-localized blowup, but worth toggling).

## Running experiments
- **40k accelerated-anneal repro** launched (`p-4e59a901`, opportunistic, 16 GPU): same
  config, `steps=40000` so `p`<1 near step ~31k. Watching whether the same
  CI-fn-grad/impmin/recon signature appears at the compressed fraction.
- User also reports `jax-l18-C49k-200k` (param-decomp-llama, a *different* model/config)
  shows the same pattern → supports a general-JAX precision cause over a bsc-specific one.

## RESULT (2026-06-16): decisive precision test — HYPOTHESIS REFUTED

Ran a 40k accelerated-anneal A/B (dp=16), identical except importance-min precision:
- **Control `p-4e59a901`** (fp32 importance-min = current JAX): **reproduced the divergence
  exactly** — at p<0.6 the CI-fn grad norm explodes (0.36→0.78→4.2→23→**149**), no_beta
  collapses (375→81), recon/ppgd blow up (0.74→2.84, 2.2→23). Validates the 40k repro and
  shows the collapse does NOT need a fully-converged model — it's driven by low `p`.
- **Test `p-3899ad8e`** (bf16 importance-min = matches torch): **diverges identically.**
  At matched steps the curves overlap — 34k: CI-grad 0.78 vs 0.83; 36k: 4.2 vs 4.51;
  no_beta 244.5 vs 244.3.

**Conclusion: the fp32-vs-bf16 importance-min difference is NOT the root cause.** Making
JAX match torch's bf16 Lp computation does nothing. The runaway is a low-`p` CI-fn
instability independent of importance-min precision. The `losses.py` bf16 diagnostic edit
should be reverted (it changes nothing).

Open: the JAX-vs-torch difference that actually matters is still unidentified. Remaining
precision suspects from the audit: **recon-KL fp32 (JAX) vs bf16 (torch)** and
**frozen-target/clean-logits bf16 (JAX) vs fp32 (torch)** — the latter is interesting: a
bf16 recon *target* is a blurrier signal to keep components alive, which could let the
low-`p` sparsity win. Non-precision suspects: CI-fn has no grad clip in either, yet torch
doesn't explode — so torch's all-bf16 forward (or fp32 recon target) must keep the CI-fn
gradient bounded in a way JAX's path doesn't. Next decisive tests: (a) **CI-fn grad
clipping** (likely the practical fix + a strong diagnostic), (b) **fp32 frozen
target/recon target** to match torch, (c) toggle recon-KL to bf16.

## DEEP ANALYSIS (2026-06-16, no new training): squash parity, forward diffs, mechanism

**Squashes are byte-identical to torch-main** (verified): lower = custom-VJP, asymmetric
0.01 leak below 0 only on negative grads, zero grad above 1; upper = clip with 0.01 leak
above 1, hard-zero (zero grad) below 0. upper→impmin, lower→recon wiring identical.
Clean/recon-target logits are bf16 in BOTH (torch computes them inside autocast) — the
earlier "fp32 frozen target" was wrong; not a divergence.

**Confirmed JAX-vs-torch differences are all small, in the shared CI-fn forward** (flagged
"un-unified" in jax CLAUDE.md): GELU tanh-approx (`jax.nn.gelu` default) vs torch exact
erf; RMSNorm eps 1e-5 vs ~1e-6; attention-softmax dtype (jax dpa vs torch SDPA fp32). The
GELU diff measured tiny: max 4.7e-4 raw; propagated through 8 blocks in bf16 → ~0.009 rms
/ 0.14 max on ci.upper, ~0.27% alive/dead boundary flips. No single op is a smoking gun.

**Grad-clip(1.0) did NOT fix it** (20k arm p-6882f7ad still collapsed: no_beta 50.6, recon
4.64) — so it's a DIRECTIONAL attractor, not a magnitude spike; clipping bounds step size
but not the kill-components direction.

**Mechanism (from logged L0 + grad-norms):** torch holds importance-min ≈1080 FLAT as p
anneals 0.8→0.4 (stable sparsification equilibrium). JAX over-sparsifies: L0 active comps
549(p1.04)→482(.80)→425(.64)→349(.56)→274(.48)→125(.40); recon-KL holds ~0.8 until
p≈0.6 then runs away (0.8→3.8). The grad explosion is localized to out_w + in_proj_w +
late-block MLP w2 (the logit/impmin interface, backprop through later blocks). Barely-alive
components (small positive logit) sit in the upper-squash identity region (grad=1) where
the low-p impmin gradient `p(ci+eps)^(p-1)` explodes; routed through the SHARED transformer
this perturbs all components → collective runaway.

**Leading conclusion: H1 metastability.** The low-p landscape is ill-conditioned (impmin
grad ~1e6 at ci→0 due to eps=1e-12 with no floor on the (ci)^(p-1) term). torch sits on
the stable side of the bifurcation; JAX's accumulated ~0.01-scale numerical differences
(gelu/rms/attn/bf16 reassociation) tip it off. Implication: the fix is LANDSCAPE
CONDITIONING (larger eps / floor the impmin gradient / higher final p / softer squash near
0), not chasing exact torch numeric parity. NOT yet done: checkpoint-based logit-
distribution comparison (JAX-numerics vs torch-numerics on identical trained CI inputs) to
distinguish pure metastability from a systematic over-sparsification bias.

## Success criterion
Identify the mechanism that makes the JAX run diverge where torch does not, and a change
under which a JAX run with this exact config tracks p-20f9fc15 through 400k
(importance-min ~1100, recon ~0.23, no CI-fn grad-norm blowup).
