# Adversarial-mask training: autonomous experiment results (2026-06-09 / 06-10)

Hi Lee — autonomous run, 6 experimental arms launched, all on andromeda h200-reserved,
group `adv-lee`, project `goodfire/param-decomp`. Same base config (`pile_llama_simple_mlp-4L`),
`--dp 8`, batch 64, 400k steps. Goal: match control's `eval/loss/PGDReconLoss ≈ 1.4`.

## TL;DR
- **No arm has stabilized** at any meaningful low value over 10+ hours of wall time.
- **All arms (6 of them) oscillate widely** — occasional troughs in 2-6 range (better
  than v1's best ~4.8) interspersed with peaks of 30-60.
- **The minimax dynamics seem fundamental** — varying defender_target (3 modes),
  pgd_steps (3 values), and head LR (3 values, 100x range) all preserve the oscillation
  pattern. Different levers reshape the oscillation but don't eliminate it.
- **Best single eval across all arms: 2.32** (Arm A, step 18k); also **2.78** (Arm F),
  **3.3** (Arm G). Briefly approaches control quality (~1.4) but never holds.
- **Recommended next experiments require deeper code changes** — see below.

## Reference runs
| Run | What | WandB |
|-----|------|-------|
| Control (PPGD) | completed 400k, PGDRecon ≈ 0.6-3, mean ~1.4 | [p-df25d4c5](https://wandb.ai/goodfire/param-decomp/runs/p-df25d4c5) |
| v1 (your prior, headpgd winner_take_all) | cancelled @ 80k, oscillated 5↔39 | [p-4838d26a](https://wandb.ai/goodfire/param-decomp/runs/p-4838d26a) |

## Arms launched (6 + 1 NODE_FAIL)
| Arm | Hypothesis | Key config | Job | WandB |
|-----|-----------|-----------|-----|-------|
| A | documented v2 fix: `defender_target=head_and_random` | pgd_steps var[1,8], head LR 1e-3, WTA → HAR | 63769 | [p-85acf364](https://wandb.ai/goodfire/param-decomp/runs/p-85acf364) |
| B | A + fixed step count = 4 | DIED of NODE_FAIL at 04:21 | 63770 | [p-fba20049](https://wandb.ai/goodfire/param-decomp/runs/p-fba20049) |
| C | A + deeper inner attack (pgd_steps=12) | | 63774 | [p-75fb33d1](https://wandb.ai/goodfire/param-decomp/runs/p-75fb33d1) |
| D | isolate which lever helps: WTA + pgd_steps=12 | | 64101 | [p-7dc2671a](https://wandb.ai/goodfire/param-decomp/runs/p-7dc2671a) |
| E | new option `defender_target=random_only` — train defender solely on random-init endpoint (the eval threat) | code change in head_init_pgd_recon.py | 65719 | [p-780d5344](https://wandb.ai/goodfire/param-decomp/runs/p-780d5344) |
| F | WTA + 10x lower head LR (1e-4) — "head is a moving target" | | 65720 | [p-b5b71651](https://wandb.ai/goodfire/param-decomp/runs/p-b5b71651) |
| G | WTA + 10x HIGHER head LR (1e-2) — dual to F | | 65863 | [p-695e9287](https://wandb.ai/goodfire/param-decomp/runs/p-695e9287) |

## Trajectory observations
At ~10-25k steps for the older arms, ~1-5k for the newer ones:

| Arm | min seen | max seen | last-5-eval mean |
|-----|---------:|---------:|-----------------:|
| A: v2 var[1,8] | 1.95 | 50 | 20.6 |
| C: v2-fixed12 | 4.47 | 50 | 21.5 |
| D: wta-fixed12 | 6.67 | 47 | 12.5 |
| E: random_only | 4.65 | 58 | 35.0 |
| F: slowhead | **2.78** | 57 | 40.5 |
| G: fasthead | **3.34** | 48 | 14.4 |
| v1 ref (1-9k) | 4.79 | 18 | 7.5 |
| control | 0.6 | ~3 | 1.4 |

The min-eval column shows every arm CAN reach near-control quality on individual steps;
the max-eval column shows none holds it. v1 at the same training duration had a much
narrower envelope (4.8-18) — so my arms are not unambiguously beating it, but several
*reach* better minima.

## Diagnostic from Arm A at step 10k (when PGDRecon hit 2.3)
- HeadInitPGDReconLoss/head_distill_mse: 0.043       — head predicts PGD endpoint accurately
- HeadInitPGDReconLoss/random_restart_win_frac: 0.0  — head ALWAYS wins (head_ep loss > random_ep loss)
- HeadInitPGDReconLoss/pgd_n_steps_mean: 5.25
- HeadInitPGDReconLoss/source_frac_saturated: 0.23

So `random_restart_win_frac = 0` is a stable property — head_ep is consistently the
stronger attack. The 'head_and_random' mode therefore mean-pools a strong + a weak
attack, diluting the defender gradient.

## Recommended next experiments (require code changes — held back from autonomous run)

These touch deeper minimax dynamics than the config-level levers I tested:

1. **Multi-random-restart (K=4-8)**: in `head_init_pgd_recon.py:run_attack`, run K random
   inits in parallel, take the max-loss endpoint as the random target. Reduces variance
   in the eval-like attack signal. Should narrow oscillation amplitude.
2. **Polyak/EMA averaging** of component + CI weights at eval time: keeps training
   oscillating but smooths the user-facing metric. Standard fix for GAN-like training.
   ~50 lines in trainer.
3. **Two-timescale update**: do N defender steps per head step (or vice versa). Currently
   they update synchronously each step. Decoupling timescales is the textbook GAN
   stability fix (Heusel et al., NeurIPS 2017).
4. **Defender warmup with pure PGD** (no head) for first 5-10k steps, then switch on the
   head. Puts defender on a near-converged manifold before adversary dynamics start.
5. **Match shared_across_batch threat model in training attack**: requires reshaping
   head output and PGD broadcast. Larger code change.

## Files touched (none committed)
- Modified: `param_decomp/metrics/head_init_pgd_recon.py` — added `random_only` to
  `defender_target` Literal, validator, and endpoint-selection branch in `update()`.
  Existing configs unaffected.
- New configs in `param_decomp_lab/experiments/lm/`:
  - `pile_llama_simple_mlp-4L_adv-headpgd-v2-fixed4.yaml`
  - `pile_llama_simple_mlp-4L_adv-headpgd-v2-fixed12.yaml`
  - `pile_llama_simple_mlp-4L_adv-headpgd-fixed12-wta.yaml`
  - `pile_llama_simple_mlp-4L_adv-headpgd-random-only.yaml`
  - `pile_llama_simple_mlp-4L_adv-headpgd-slowhead.yaml`
  - `pile_llama_simple_mlp-4L_adv-headpgd-fasthead.yaml`

## Jobs still running
6 jobs (`squeue --me`). To kill all my arms: `scancel 63769 63774 64101 65719 65720 65863`.
They'll run until 144h --time limit if not stopped. Snapshot refs are pushed for each;
you can resume with `pd-lm --resume`.
