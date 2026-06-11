# Literature review: stabilizing learned/amortized adversaries for PGD-robust parameter decomposition

2026-06-10. Follow-up to CLAUDE_RUN_SUMMARY.md (6 failed config-level arms). All papers below
are saved in `~/paper-corpus` (tag `adv-training`) on Mac + both cluster login nodes (pc-sync'd).

## Papers read in detail

| Paper | Venue | The one-line lesson for us |
|---|---|---|
| FGSM-SDI — "Boosting Fast AT with Learnable Adversarial Initialization" (Jia et al., [2110.05007](https://arxiv.org/abs/2110.05007)) | TIP 2022 | Our head-init design, in image space — but trained as a *game through the attack*, conditioned on *gradient information*, updated *every k=20 defender steps*. Overcomes catastrophic overfitting. |
| GradAlign — "Understanding and Improving Fast AT" (Andriushchenko & Flammarion, [2007.02617](https://arxiv.org/abs/2007.02617)) | NeurIPS 2020 | CO = defender learns local *nonlinearity* inside the threat set so weak attacks fail while strong eval PGD succeeds; detectable as collapse of gradient alignment between nearby points in the ball; preventable by regularizing that alignment. |
| N-FGSM — "Make Some Noise" (de Jorge et al., [2202.01181](https://arxiv.org/abs/2202.01181)) | NeurIPS 2022 | Strong noise augmentation *beyond the threat radius*, no clipping around the clean point, implicitly regularizes the loss surface — matches GradAlign at 1/3 cost. |
| ADT — "Adversarial Distributional Training" (Dong et al., [2002.05999](https://arxiv.org/abs/2002.05999)) | NeurIPS 2020 | Without an entropy regularizer the inner-max distribution *provably collapses to a Dirac*. λ·H(p) (λ≈0.01) keeps the adversary diverse; amortized generator takes a noise input z for multimodality. |
| FGSM-PGI/MEP — "Prior-Guided Adversarial Initialization" (Jia et al., [2207.08859](https://arxiv.org/abs/2207.08859)) | ECCV 2022 | Historical-perturbation priors (momentum over epochs) prevent CO at zero extra attack cost — this is what our PersistentPGD control already is. Plus: a consistency regularizer between current-attack and prior-attack predictions. |
| FGSM-CKPT — "Understanding CO in Single-step AT" (Kim et al., [2010.01799](https://arxiv.org/abs/2010.01799)) | AAAI 2021 | CO = "decision boundary distortion": model robust at the attack's *endpoint magnitude* but vulnerable at *interior* points along the same direction. Fix: also train at interior scales of the perturbation. |

## Diagnosis: our oscillation is recurring catastrophic overfitting

The fast-AT literature describes exactly our failure, with one twist.

**The phenomenon there:** train with a cheap attack (FGSM, 1 forward/backward) → robust accuracy
against the cheap attack goes to ~100% while robustness against multi-step PGD *suddenly drops to
0* within a fraction of an epoch. The defender hasn't become robust; it has learned to *defeat the
attack's search procedure*.

**The mechanism (GradAlign + Kim et al.):** the defender warps the loss surface inside the threat
set — gradients at nearby points become orthogonal (alignment 0.95 → 0.05), the surface becomes
locally highly nonlinear/distorted, so any attack that relies on few local gradient evaluations
lands at a bad point. A 20-step PGD with enough budget still navigates the warped surface and
finds the true worst case. Kim et al. sharpen this: post-CO models are often robust at the *full*
perturbation magnitude (where the cheap attack always lands, e.g. the sign-step corner) but
vulnerable at *interior* magnitudes along the same direction.

**Mapping to us:** "input" = source vector s in the box [0,1]^C per module; "model" = components +
CI fn (the defender controls the entire loss landscape over masks, far more directly than an image
classifier controls its input loss surface — so warping is *easier* here). Cheap attack = head
prediction + 1–8 sign-PGD steps at step 0.1. Eval attack = 20-step sign-PGD, random init, mask
shared across the batch. Evidence from our runs that fits:

- Defender's own-adversary recon is low (~0.1–11) while eval PGDRecon is 45–104: robust to its
  attack, broken under the real one. Textbook CO signature.
- The head always wins against the single random restart (`random_restart_win_frac = 0.0`) yet
  eval still finds 10–30× worse points — the warped surface defeats *fresh short searches*, and our
  training restart is refined by only 1–8 steps vs eval's 20.
- Our distribution-head arms collapsed to near-binary masks (87–99% saturated) — the Dirac
  collapse ADT proves happens without entropy regularization, plus Kim et al.'s endpoint-only
  training (saturated = box corner = "full magnitude").
- The oscillation (5 ↔ 40, period of a few thousand steps) is the one twist: in fast AT, CO is
  one-way because FGSM never adapts. Our adversary *partially* adapts (head retrains, PGD refines
  from its init), so each collapse gets punished after a delay and partially reverts → limit cycle
  instead of one-way collapse. Same disease, different presentation because the adversary has a
  pulse.

This also explains why all six config arms failed identically: defender_target pooling, PGD depth
(4 vs 12), and head LR (1e-4…1e-2) all change *which* cheap attack the defender sees, not the
defender's incentive to warp the surface against whatever cheap attack it sees.

And it explains why the PersistentPGD control works: a persistent buffer with fresh-ish gradient
steps every iteration is FGSM-MEP (the best prior-guided variant in Jia et al. 2022) — the attack
inherits the *accumulated* search of all previous steps, so the defender can't outrun it by local
warping; warping the surface invalidates the buffer only slowly, and the buffer re-adapts every step.

## Lessons → concrete interventions, ranked

### Tier 1 — verify the mechanism first (cheap eval metrics, no training change)

1. **Mask-space gradient alignment metric.** GradAlign's Eq. 5 transplanted: at eval, compute
   `cos(∇_s L(s₁), ∇_s L(s₂))` for two random sources in the box (per module and pooled). The CO
   story predicts this *collapses exactly when eval PGDRecon spikes* and recovers in the troughs.
   One eval metric, ~2 extra backward passes per eval. If it doesn't track the spikes, the CO frame
   is wrong and tiers 2–3 need rethinking.
2. **Interior-scale robustness curve** (Kim et al.'s distortion check): evaluate recon at
   `s = k·s_pgd` for k ∈ {0.1,…,1.0} along the eval-PGD endpoint direction. Distortion =
   non-monotonicity (vulnerable at k=0.4, robust at k=1.0). Plot per eval like PGDSourceHistogram.

### Tier 2 — defender-side regularization (the literature's actual fixes; need code, small)

3. **N-FGSM analogue (cheapest, try first).** Before the training attack, jitter the sources with
   strong noise and *don't clamp the attack step*: `s₀ ~ U(-k_extra, 1+k_extra)` (k_extra ≈ 0.5,
   i.e. noise beyond the threat box, the k=2ε analogue) and let the 1–8 PGD steps run unclamped,
   clamping only the final mask product if needed for semantics. Training against a slightly
   over-threat box means eval's in-box attack is interior to the training distribution. Cost: zero
   extra forwards. Caveat: masks outside [0,1] mean amplified/sign-flipped component contributions —
   semantically aggressive but exactly the "train beyond the threat" trick that made N-FGSM work.
4. **GradAlign analogue.** Add regularizer `λ·(1 − cos(∇_s L(s₁), ∇_s L(s₂)))` with s₁, s₂ random
   in the box (or s₁ = head init, s₂ random). Directly removes the defender's incentive to warp.
   Cost: double backprop, ~2× the recon-loss cost — still cheaper than the PPGD control's inner
   loop. GradAlign is the reliable-but-slow option in every comparison; N-FGSM matches it at 1/3
   cost, so try 3 first.
5. **Interior-point training** (Kim et al.): with probability ½ train the defender at `u·s_attack`
   for u ~ U(0,1) instead of the endpoint. One-line change where the defender recon is computed.
   Directly prevents endpoint-only robustness / saturated-corner overfitting.

### Tier 3 — adversary-side redesign (FGSM-SDI's recipe for the head)

If we keep the amortized head, FGSM-SDI says our current head differs from the working design in
four specific ways, each independently testable:

6. **Condition the head on gradient information.** FGSM-SDI's ablation is unambiguous: generator
   conditioned only on the clean sample *still catastrophically overfits*; adding the signed
   gradient of the current loss fixes it, and gradient-only beats sample-only by a large margin.
   Our head sees only trunk features (the "sample"). Add `sign(∇_s L)` evaluated at s=0 (or at the
   ci point) as a per-component input feature — one extra forward+backward, same cost class as the
   PGD steps we already pay for. This is the single most evidence-backed change to the head.
7. **Train the head adversarially through the attack, not by distillation.** FGSM-SDI's generator
   maximizes the defender's loss *through* the FGSM step (differentiate the endpoint loss w.r.t.
   generator params). Our distillation (`MSE(s₀, s_k.detach())`) makes the head chase the PGD
   endpoint — but the PGD endpoint is computed *from the head's own init*, a self-referential
   target that can drift into a narrow mode (and a stale one, since the defender moves). Game
   training keeps the head pointed at "what currently hurts the defender most."
8. **Two-timescale: update the head every k ≈ 10–20 defender steps.** FGSM-SDI's robustness
   *improves* monotonically with k up to 20 (their chosen value), at lower cost. We update every
   step. This is TTUR for our saddle problem and it's free (saves compute).
9. **Multimodality + entropy (ADT).** Give the head a noise input z ~ U(-1,1) so it defines a
   conditional *distribution* over attacks, and add λ·H ≈ 0.01·entropy (variational bound, as in
   ADT-IMP-AM) to stop Dirac collapse. Without this a deterministic head can only ever encode one
   attack mode per datapoint — while eval explores with random restarts.

### Tier 4 — threat-model alignment + optimizer hygiene (orthogonal, cheap)

10. **Match `shared_across_batch` in the training attack.** Training attacks are per-position;
    eval optimizes ONE mask for the whole batch (a universal-perturbation threat). Per-position
    robustness in expectation does not imply shared-mask robustness when the inner max is solved
    approximately. Make some fraction of training attack steps aggregate gradients across the
    batch (shared source), exactly like eval. Cost-neutral.
11. **EMA / weight averaging of the defender** for eval (and possibly as the training target à la
    robust-overfitting literature, Rice et al. 2020 + Yazıcı et al. 2019). Smooths the limit cycle's
    user-facing metric and typically improves the underlying robustness too. Cheap, orthogonal.
12. **Persistent shared-mask adversary (FGSM-MEP for the eval threat).** Since the eval threat is
    one shared mask, keep a *single persistent source vector per module* (no batch dim — tiny),
    updated by momentum sign-steps once per training step, with occasional random restarts; train
    the defender against it alongside the per-position head. This is the control's trick aimed
    exactly at the eval threat, at ~zero memory cost, and it composes with the head (head proposes
    per-position, buffer covers the universal mode).

## Suggested experimental sequence

1. Land Tier 1 metrics (gradient alignment + interior-scale curve) — they confirm/refute the
   mechanism and make every later arm interpretable.
2. Arm H: **N-FGSM-style noise + no-clamp** (3) + **interior-point training** (5) — pure config/
   small-code defender-side fixes, no new adversary machinery, cost-neutral. If CO is the story,
   this alone should kill the spikes (it did for them, at radii where RS-FGSM always collapsed).
3. Arm I: **FGSM-SDI-faithful head**: gradient-conditioned (6), game-trained (7), k=20 (8),
   z-input + entropy (9), against the same defender recipe as H.
4. Arm J: H + **shared-mask training attack** (10) + EMA eval (11).
5. If H–J still oscillate: add GradAlign (4) — the expensive-but-reliable hammer.

Each lesson is from a setting (image classification, ℓ∞ ball) that differs from ours in one
important way: our defender *owns* the loss landscape over masks (the components and CI fn are the
parameters being trained), so surface-warping is easier and the regularization pressure likely
needs to be stronger than the image-domain defaults (their λ's are a starting point, not gospel).

The deeper question from the handover — "can a learned adversary ever beat a free 20-step PGD at
its own metric" — gets a nuanced answer from this literature: no single-shot adversary matches
multi-step PGD *as an attack*, but that's not required. What's required is that the training attack
be reliable enough that the defender can't cheat it by warping the surface. FGSM-SDI/N-FGSM/MEP all
achieve PGD-AT-level *robustness* with ~1-step attacks + the right regularization/priors. The fix
is less about making the adversary stronger and more about removing the defender's incentive to
exploit its weakness.

---

# Part 2: detailed mapping to our setting (added after deeper analysis)

## The game, precisely

Adversary: s ∈ [0,1]^C per module; mask m = ci + (1−ci)·s ∈ [ci, 1]. Can push masks UP from
the CI floor, never below. Defender: components + CI fn — controls the entire function
s → recon AND the box geometry (ci). Eval judge: 20-step sign-PGD (step 0.1), random init,
ONE mask shared across the eval batch (a universal-perturbation threat).

Structural differences from image AT:
1. **Defender owns the loss surface** — warping is strictly easier than in image AT, where the
   classifier shapes ℓ(x+δ) only indirectly.
2. **Threat set = whole unit box, not a small ε-ball.** GradAlign's local-linearity story doesn't
   transplant globally: recon is legitimately nonlinear in masks across the box. The right
   alignment notion is LOCAL at the PGD step scale: cos(∇L(s), ∇L(s+η)), ‖η‖∞ ≈ 0.1–0.2.
3. **Streaming data** (no per-sample identity → FGSM-MEP per-sample priors impossible) and a
   **universal eval threat** (shared mask) — these two constraints jointly suggest the
   persistent-universal-buffer arm below.

## Refined mechanism: relaxation oscillation from ImpMin vs. adversary detection lag

The dips to eval ≈ 2 prove the defender sometimes genuinely approaches robustness — a purely
surface-warping defender would not produce low eval PGD (eval has budget to navigate warps).
The cycle:

1. Near-robust state: training attack (head + 1–8 shallow steps) finds nothing → adversarial
   gradient pressure vanishes.
2. ImpMin + p-annealing keep pushing ci down → vulnerable directions re-open; nothing pushes back.
3. Eval's 20-step PGD spots the regression immediately (spike). The training adversary needs
   thousands of steps to re-discover the hole (shallow, i.i.d. searches; stale head prior).
4. Adversary catches up → defender re-hardens (trough) → goto 1.

The oscillation period = the detection lag. The control (PersistentPGD) works because its buffer
is FGSM-MEP: standing pressure at the worst points; when ImpMin re-opens a hole, the buffer's
2 warmup steps catch it within a step. Zero lag. None of the six config arms touched the lag.

Surface-warping (CO proper) is the second-order effect: during the vulnerable phase, warping is
the cheapest way to silence the shallow attack, which lengthens the lag. The mechanisms compound.

Falsifiable predictions: (a) eval PGDRecon spikes correlate with CI_L0 decreases;
(b) local mask-space gradient alignment collapses during vulnerable phases; (c) eval PGD endpoints
are interior (not corner-saturated) when training attacks are corner-saturated.

## Asymmetries that change the transplants

- **N-FGSM one-sided only.** s > 1 (mask > 1) = amplification — coherent over-threat training.
  s < 0 (mask < ci) = ablating *important* components and demanding output invariance — wrong
  objective (forces ci up or forces redundancy). Transplant: unclamped steps on the high side
  only; Kim-style interior scaling (train at u·s_attack, u~U(0,1)) covers the low side in-box.
- **Random-mask noise alone cannot work** (we already have StochasticReconSubset): random points
  in a 10k-dim box concentrate on typical low-loss masks; the adversarial needle has vanishing
  measure. N-FGSM keeps the full attack step after noising — both halves needed.
- **Game-training the head ≠ the diverged deterministic-reverse arm.** That blow-up was caused by
  adversarial gradient flowing INTO the shared CI trunk through double_leaky_hard's non-vanishing
  rail gradient. The head is trunk-detached and sigmoid-bounded; ascending endpoint recon w.r.t.
  φ only (straight-through the detached PGD deltas, or GRL on the s₀ path sharing the defender's
  backward) touches none of that machinery.

## Arm PB: persistent universal source buffer (new top pick)

The eval threat is one shared mask → a universal perturbation → needs NO sample identity:
- K ≈ 4–8 persistent source vectors per module (shape C each — a few thousand floats, no batch dim).
- Each training step: one momentum sign-step on one buffer (round-robin) using the current batch
  (1 extra forward+backward); occasional random restarts for mode coverage.
- Defender trains recon at the active buffer's mask (+ existing losses).
- = the control's mechanism aimed exactly at the eval threat, at ~control throughput;
  composable with a per-position head later.

Why per-position training attacks under-serve the shared-mask metric: defender gradients spread
over idiosyncratic per-token directions; per-position robustness implies shared-mask robustness
only when the inner max is solved exactly — which a shallow attack never does.

## Cost accounting (model forwards f / backwards b per training step, beyond shared losses)

| Arm | attack cost | defender cost | est. throughput |
|---|---|---|---|
| Control PPGD (n_warmup=2) | 2(f+b) | f+b | 3.1 it/s (measured) |
| Current head arms (v1/v2) | head f + ~4.5(f+b) PGD + restart ~4.5(f+b)+f | f+b (×2 endpoints in v2) | 1.2–1.7 it/s (measured) |
| **PB (K buffers, round-robin)** | 1(f+b) | f+b | ~3 it/s (est.) |
| **SDI-rebuilt head** | head f + grad-cond f+b + 2–4(f+b) PGD | shared with head ascent via GRL | ~2 it/s (est.) |
| + interior scaling | +1–3 f (no b) | — | small hit |
| GradAlign-local | +2b + double-backprop | — | ~2–3× recon cost |

## Revised experiment ranking

1. **Diagnostics first** (eval metrics, no training change): local gradient alignment at step
   scale; interior-scale loss curve along eval endpoint; spike↔CI_L0 cross-correlation.
2. **Arm PB** — persistent universal buffers. Directly kills the lag AND the threat mismatch.
3. **Arm SDI** — gradient-conditioned, game-trained, k≈20-interval head (+optional z+entropy).
4. **Defender-side**: interior-scale training + one-sided unclamped steps; EMA-of-defender at
   eval regardless (averaged iterates of a cycling saddle trajectory ≈ equilibrium).
5. **GradAlign-local** as last resort.
