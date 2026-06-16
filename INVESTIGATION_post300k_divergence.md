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

## Success criterion
Identify the mechanism that makes the JAX run diverge where torch does not, and a change
under which a JAX run with this exact config tracks p-20f9fc15 through 400k
(importance-min ~1100, recon ~0.23, no CI-fn grad-norm blowup).
