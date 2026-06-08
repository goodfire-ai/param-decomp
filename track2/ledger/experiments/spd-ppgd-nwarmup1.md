# spd-ppgd-nwarmup1 — PPGD n_warmup_steps 2 → 1

- **Claim type:** speedup
- **Stage:** killed (PPGD primary regressed at 20k; gate held)
- **Branch / worktree:** `feature/spd-ppgd-nwarmup0-t1` (off `feature/track2-t1`)
- **Owner agent:** claude-opus (2026-06-08 session)

## Hypothesis
Milder version of nwarmup0: drop one of the three PPGD inner forwards (2→1) rather than
two. Less speedup than 2→0 but a gentler perturbation to the adversarial inner loop — a
fallback sweet-spot if 2→0 hurts faithfulness.

## Diff vs baseline
Config-only: `PersistentPGDReconLoss.n_warmup_steps: 2 → 1`. Quality config keeps
`pd.steps: 400000`. `pile_llama_simple_mlp-4L-nwarmup1.yaml`.

## Success / kill thresholds
- ≥5% speedup vs baseline config (`pd-speedup-bench`, b16).
- Quality bundle within band at both 20k and 50k; faithfulness gate hard.

## Results (artifacts required)
- **benchmark:** baseline 649.99 ms/step → **557.71 ms/step** = **14.2% faster**
  (b16/s512, 1×H100, eval excluded). `nwarmup1_b16.md`.
- **quality @20k** (run `p-ebc2de5b` vs baseline `p-20f9fc15`,
  https://wandb.ai/goodfire/param-decomp/runs/p-ebc2de5b) — **FAIL, primary (PPGD)
  regressed** (gate held):

  | tier | metric | baseline | variant | Δ% | verdict |
  |---|---|---|---|---|---|
  | primary | PGD recon (PPGD) | 1.4774 | 4.8486 | +228.2% | REGRESSED |
  | primary | no_beta | 275.33 | 232.25 | −15.7% | improved |
  | secondary | Stochastic recon | 0.53068 | 0.45082 | −15.0% | improved |
  | secondary | CI-mask recon | 0.90267 | 0.76133 | −15.7% | improved |
  | secondary | L0 (info) | 1879.5 | 1925.4 | +2.4% | REGRESSED |
  | gate | CE diff | 0.30395 | 0.29455 | −3.1% | improved |
  | gate | KL | 0.64088 | 0.60447 | −5.7% | improved |
  | gate | CE unrec | 0.0045113 | 0.0043713 | −3.1% | improved |

- **quality @50k:** not run — killed early on the 20k primary regression. Job 644568
  cancelled (was at step 34k).

## Verdict
**killed** — the faithfulness **gate held** (all within-or-better), but the eval PPGD-recon
attack **regressed +228%** (1.48→4.85), far outside band. Milder than nwarmup0's +3358% but
the same failure mode: fewer PPGD warmup steps → a decomposition less robust to the eval PPGD
attack. PPGD is same-step-meaningful and +228% won't recover by 50k, so cancelled early. (If
Dan wants a 50k confirmation, re-run — the gate holding is the one mildly interesting note.)
