# Adversarial-mask training: autonomous experiment results (2026-06-09 / 06-10)

Hi Lee — I ran 4 parallel experimental arms on andromeda (h200-reserved) targeting the
`eval/loss/PGDReconLoss` instability you flagged in feature/adv-lee. All 4 are still
running (jobs 63769, 63770, 63774, 64101). Control completed (61155). WandB group
`adv-lee`.

## TL;DR
- **Documented v2 fix (head_and_random) does not cleanly stabilize training.** Lower
  troughs than v1, but much higher peaks. Net effect: wider oscillation, worse mean.
- **None of my 4 arms clearly beats v1.** v1's mean PGDRecon at steps 1-9k was ~9.9;
  my best (Arm A) at steps 10-24k has mean ~13.6.
- **Best single point seen across all arms: 1.8** (Arm B, step 26k) — better than v1's
  best (~4.8) but not stable. Suggests the defender CAN reach near-control quality
  occasionally but is dislodged by the adversary on the next step.
- **Disproven hypotheses:**
  - Fixed PGD step count doesn't reduce oscillation (Arms B, C, D all worse than v1)
  - Deeper PGD (n=12) alone hurts (Arms C, D)
- **Probable real lever (not yet tested):** the v2 'pooling' is currently a SUM of two
  recon losses. With `random_restart_win_frac = 0.0` (confirmed in eval logs), the
  random_ep attack is always WEAKER than head_ep. The current loss = mean(strong, weak)
  dilutes the gradient and creates the wider oscillation. True `max` over endpoints,
  or weighting the harder endpoint more, would be the principled fix.

## Arms launched
| Arm | Config | defender_target | pgd_steps | WandB | Job |
|-----|--------|-----------------|-----------|-------|-----|
| A | pile_llama_simple_mlp-4L_adv-headpgd-v2.yaml          | head_and_random  | uniform[1,8] | p-85acf364 | 63769 |
| B | pile_llama_simple_mlp-4L_adv-headpgd-v2-fixed4.yaml   | head_and_random  | fixed 4      | p-fba20049 | 63770 |
| C | pile_llama_simple_mlp-4L_adv-headpgd-v2-fixed12.yaml  | head_and_random  | fixed 12     | p-75fb33d1 | 63774 |
| D | pile_llama_simple_mlp-4L_adv-headpgd-fixed12-wta.yaml | winner_take_all  | fixed 12     | p-7dc2671a | 64101 |

Arm A is the documented v2 (head_and_random pooling) from your handover.
Arm B and C vary pgd_steps to fixed values (motivated by 'reduce step-count variance'
lever in HANDOVER_adv-lee.md).
Arm D was a late addition to isolate which change matters: it's v1's defender_target
(winner_take_all) paired with Arm C's deeper PGD. Result: it's WORSE than v1, which
means deeper PGD alone hurts — the head_and_random pooling does provide *some* benefit,
just not stability.

## Observed trajectory ranges (eval/loss/PGDReconLoss)
| Arm              | steps seen | min  | max  | mean (last 10 evals) | vs v1's ~9.9 mean (1-9k) |
|------------------|-----------:|-----:|-----:|---------------------:|--------------------------|
| A: v2 var[1,8]   | 24k        | 1.95 | 50   | 13.6                 | worse                    |
| B: v2-fixed4     | 30k        | 1.84 | 59   | 18.4                 | worse                    |
| C: v2-fixed12    | 18k        | 5.6  | 60   | 27.7                 | much worse               |
| D: wta-fixed12   | 12k        | 7.2  | 47   | 23.4                 | much worse               |
| control (PPGD)   | 400k       | ~0.6 | ~3   | ~1.4                 | the bar                  |

## Diagnostic from Arm A at step 10k (when PGDRecon hit 2.3)
- HeadInitPGDReconLoss/head_distill_mse: 0.043       — head is predicting PGD endpoint well
- HeadInitPGDReconLoss/random_restart_win_frac: 0.0  — head ALWAYS wins (head attack > random attack)
- HeadInitPGDReconLoss/pgd_n_steps_mean: 5.25
- HeadInitPGDReconLoss/source_frac_saturated: 0.23

So even in v2, the head's PGD endpoint is consistently the stronger attack. The 'head_and_random'
defender_target therefore puts equal weight on a strong and a weak attack in the loss — that
asymmetry, IMO, is what's creating the wider amplitude (defender keeps switching which attack
to harden against).

## Suggested next experiments (require code changes — not run autonomously)
1. **`defender_target: max`** — train defender on max(recon(head_ep), recon(random_ep))
   instead of the average. With `random_restart_win_frac = 0` this would equal v1 in
   expectation; under shifts (when random_ep wins), it adapts. Roughly two-line change in
   `head_init_pgd_recon.py` around the `endpoints = [head_ep, random_ep]` block.
2. **Multi-random-restart**: run 4-8 random inits in parallel, take max — much better
   estimate of the eval threat model on each step. Replaces the single random restart.
3. **Lower head LR** (currently 1e-3 constant) — handover lists as untested. The head is
   a moving target; slower head means the defender doesn't constantly re-target.
4. **Defender warmup with pure PGD (no head)** for first 5-10k steps, then introduce
   the head. This would put the defender on a near-converged manifold before adversary
   dynamics kick in.
5. **Match shared_across_batch in training** (handover-listed) — requires reshape in
   the head's source output. Test: does threat-model parity help?

## Jobs still running
`squeue --me` will show them. To kill: `scancel <jobid>`. They'll keep eating GPUs
until you stop them or reach the 144h --time limit. Config files for arms B/C/D are in
`param_decomp_lab/experiments/lm/pile_llama_simple_mlp-4L_adv-*.yaml` (new today).
