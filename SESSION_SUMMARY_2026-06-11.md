# Session summary: accelerating adversarial training for parameter decomposition

_Written 2026-06-11. Branch `feature/adv-lee` (commit `5e6e26235`). Companion to
`ADVERSARIAL_LIT_REVIEW.md` (the literature analysis) and `CLAUDE_RUN_SUMMARY.md` (the
first-day learned-head results)._

## Goal

Speed up the adversarial-mask training used in our parameter-decomposition method without
losing robustness. The success metric is **`eval/loss/PGDReconLoss`** — recon (KL) under a
20-step random-init sign-PGD attack over the mask box, `mask = ci + (1-ci)*source`,
`mask_scope=shared_across_batch`. Lower = more robust. The bar is the **Jose / PPGD control**
(`goodfire/spd/s-55ea3f9b`, replicated as `pd-adv-control`): β₁=0.5, 2 warmup steps,
per-position PersistentPGD, eval PGDReconLoss ≈ 0.66–0.9, stable.

## Headline result

A **cheaper PersistentPGD recipe matches Jose/control quality at ~33% less compute**:

> **per-position PersistentPGD · β₁≈0 · β₂=0.99 · n_warmup_steps=1 · lr=0.01 · default grad clip**

It reaches eval PGDReconLoss ≈ 0.6–1.0 (Jose/control territory). The saving is dropping the
inner warmup steps from 2 → 1 (3 → 2 model forward/backwards per step). The originally-pursued
*learned/amortized* adversary did **not** work; the win is "the existing persistent-PGD recipe,
made cheaper," not "a successful learned adversary."

## What we tried, in order

### 1. Learned-head adversary arms (first day; `CLAUDE_RUN_SUMMARY.md`)
Seven config-level arms around the `HeadInitPGDReconLoss` design (an amortized head predicts a
PGD init; defender trains on the refined endpoint): `defender_target` ∈ {winner_take_all,
head_and_random, **random_only** (new)}, PGD depth ∈ {var[1,8], 4, 12}, head LR ∈ {1e-4…1e-2}.
**All failed** — eval PGDReconLoss oscillated ~2↔50, never stable, never matched control. A
single head pass + a few PGD steps is too weak/narrow an attacker; the defender overfits to it.

### 2. Literature review (`ADVERSARIAL_LIT_REVIEW.md`; 6 papers in `~/paper-corpus`, tag `adv-training`)
FGSM-SDI, GradAlign, N-FGSM, ADT, FGSM-PGI/MEP, FGSM-CKPT. Initial hypothesis: our oscillation
is **catastrophic overfitting** (defender warps the loss surface so the cheap attack fails while
the 20-step eval PGD succeeds). Mapped the control's success to FGSM-MEP (persistent buffer =
accumulated search the defender can't outrun).

### 3. Diagnostics — and a refuted hypothesis
Built two eval metrics: **`SourceGradAlignment`** (GradAlign Eq. 5 in mask space) and
**`PGDInteriorScaleCurve`** (Kim et al. distortion check). **They refuted the CO hypothesis**:
gradient alignment did *not* collapse when PGDReconLoss spiked (spikes at alignment 0.24–0.90,
troughs at 0.37–0.89; no anticorrelation), and interior-scale distortion was sporadic and
untracking. So the defender is **not** silencing the attack by warping the local surface.

### 4. Real mechanism: adversary coverage + detection lag (not surface warping)
PersistentPGD **scope sweep** (`single_source` vs `per_batch_per_position`, warmup 0/2):
- `single_source` (PB1/PB2) — the *exact eval-threat shape* (one shared `[C]` mask) — **fails**
  (stuck 43–84) at *high* gradient alignment. Not geometry: one persistent source covers one
  mode; the 20-step random-restart eval attack explores a multimodal universal-mask space.
- `per_batch_per_position` (32,768 distinct masks/step) is the load-bearing ingredient — broad
  per-step coverage. PB4 (per-position, warmup 0) oscillated 3–34; control (warmup 2) is stable.

So the lever is **per-step adversary coverage + freshness**, and the oscillation is a
relaxation oscillation: when ImpMin re-opens a vulnerable direction, a low-coverage/lagging
adversary takes thousands of steps to re-find it. The control's per-position persistent buffer
has ~zero detection lag.

### 5. Acceleration frontier: warmup steps
- **warmup 1** (PB6, and β₁=0 / synth variants): reaches ~0.6–1.0, ≈ control, ~33% cheaper. ✅
- **warmup 0** (Free-AT pricing, one free coupled step/iter): **fails** at *every* adversary lr
  tried (0.01, 0.03, 0.06, 0.1, 0.3) — stuck 30–80 even at 120k+ steps. The "one big step per
  iteration" idea is refuted; ≥1 dedicated warmup step is necessary.

### 6. Adversary-optimizer sweeps (on the warmup-1 recipe)
- **β₁ (momentum / first-moment memory):** shorter is better. β₁=0.0/0.01 ≈ control;
  β₁=0.8 worse (oscillates 3–11). A reactive adversary re-aims faster at the moving defender.
- **β₂ (second-moment / per-coordinate scale memory):** **high is required.** β₂=0.99 and
  0.999999999 both reach ~1; β₂ ∈ {0.0, 0.01, 0.05} all **fail** (stuck 40–65) — low β₂ gives a
  noisy scale estimate → erratic attack. No benefit beyond 0.99.
- **Components grad clip:** **irrelevant to the endpoint.** clip ∈ {0.01, 0.1, 1.0} give
  identical PGDReconLoss (~1.0), KL (~0.47), CIHiddenActs (~0.77), L0 (~2450) at matched step.
  The pre-clip components grad norm sits at ~0.44 regardless, so clip=0.01 throttles it ~44×
  while clip=1.0 doesn't clip — yet same decomposition. The clip shapes convergence dynamics,
  not the endpoint. (Corrects an earlier guess that the tight clip was load-bearing.)
- **Muon optimizer (adversary):** **failed.** Newton-Schulz orthogonalization equalizes the
  update's singular values across the per-position×component matrix, destroying the
  gradient-magnitude concentration an attacker needs → diffuse weak attack. Own-adversary recon
  0.78 (looks fine) but eval PGD 38–47 (defender overfit a weak adversary). Orthogonalization is
  good for an *optimizer*, anti-helpful for an *adversary*.
- **Per-warmup-step lr taper (0.01→0.003):** no edge over flat warmup.

## Honest caveats
- "Beats control" was **overstated** mid-session — it came from a stale 1.4 control figure.
  Fair comparison: short-memory/warmup-1 recipes **match** control (~0.6–1.0), with comparable
  KL/faithfulness; control has a slightly better median, the cheaper recipes a tighter band.
- Most comparisons are at 45k–310k steps, **not** a completed 400k, and **single seed**. KL and
  CI L0 fall over training via p-annealing, so younger runs look denser/less-faithful purely
  from training fraction — comparisons must be **matched-step**.
- Jose vs our runs: same recipe (C sizes, β₁=0.5/β₂=0.99, lr 0.01, dataset, seq len, steps,
  delta, sigmoid all identical) but **different codebase** (`goodfire/spd` vs `param-decomp`,
  reorganized config schema) + different eval-metric set — a faithful baseline, not "identical
  except N knobs."

## Code changes on `feature/adv-lee` (committed `5e6e26235`)
- `param_decomp/metrics/head_init_pgd_recon.py` — `defender_target: random_only`.
- `param_decomp/metrics/persistent_pgd_state.py` — **MuonPGD** optimizer (+ Newton-Schulz) and
  per-warmup-step lr (`warmup_step_lrs`) threading.
- `param_decomp/metrics/persistent_pgd_recon.py` — `warmup_step_lrs` config field + validator.
- `param_decomp_lab/eval_metrics/{source_grad_alignment,pgd_interior_scale_curve}.py` — the two
  diagnostics; registered in `eval_metrics/__init__.py`.
- ~25 sweep configs under `param_decomp_lab/experiments/lm/pile_llama_simple_mlp-4L_adv-*.yaml`.

## Infra notes
- Started on **andromeda** (chronic NODE_FAILs — killed ~8 jobs; runs still reached 250–310k).
- Moved to **CoreWeave reno** (committed branch → fresh checkout + `uv sync` + copied `~/.netrc`
  for wandb). Reno is faster to schedule but flaky in two ways: **HF-CDN parquet timeouts**
  (408/504 on the streamed Pile dataset) and **CUDA device-side asserts** (≥4 — at least one hit
  two healthy jobs simultaneously, i.e. a correlated cluster event). Mitigation (HF cache /
  longer `HF_HUB_DOWNLOAD_TIMEOUT`) not yet applied.

## WandB
- Runs are display-named by config: `beta1=… [beta2=…] n_warmup_steps=… lr=… [comp_grad_clip=…]`.
- Full report (22 runs + Jose baseline, 9 metric panels):
  https://wandb.ai/goodfire/param-decomp/reports/Adversary-investigation-FULL-v3-2026-06-11--VmlldzoxNzIzMjg0OQ==
- Group `adv-lee`. Jose baseline: `goodfire/spd/s-55ea3f9b` (`jose-s-55ea3f9b`).

## Recommended next steps
1. Resume one **winner** (β₁=0, warmup 1, lr 0.01 — or β₁=0.01/synth) to a clean **400k** as the
   headline curve, ideally with a 2nd seed; apply the reno HF/timeout mitigation first.
2. Kill the confirm-only runs still burning reno GPUs (β₂-low especially — settled negative).
3. The amortized-head direction is parked: if revisited, the lesson is it must reproduce the
   control's *broad per-step coverage* (e.g. a head emitting a population/distribution of
   sources), not a single mode — and be gradient-conditioned (FGSM-SDI) rather than distilled.
