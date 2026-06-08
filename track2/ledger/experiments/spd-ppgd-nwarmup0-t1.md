# spd-ppgd-nwarmup0-t1 — PPGD n_warmup_steps 2 → 0

- **Claim type:** speedup
- **Stage:** killed (faithfulness gate + PPGD regressed at 20k)
- **Branch / worktree:** `feature/spd-ppgd-nwarmup0-t1` (off `feature/track2-t1`)
- **Owner agent:** claude-opus (2026-06-08 session)

## Hypothesis
The PPGD inner loop runs `n_warmup_steps` extra masked-component forward/backward passes
per batch before the loss. Dropping 2→0 removes 2 of the 3 inner forwards — the dominant
per-step cost — for (hopefully) similar decomposition quality.

## Diff vs baseline
Config-only: `PersistentPGDReconLoss.n_warmup_steps: 2 → 0`. Quality config keeps
`pd.steps: 400000`. `pile_llama_simple_mlp-4L-nwarmup0.yaml`.

## Success / kill thresholds
- ≥5% speedup vs baseline config (`pd-speedup-bench`, b16).
- Quality bundle within band at both 20k and 50k; faithfulness gate hard. A primary
  (PPGD / no_beta) WIN is a 400k-confirm candidate handed to Dan.

## Results (artifacts required)
- **benchmark:** baseline 649.99 ms/step → **467.76 ms/step** = **28.0% faster**
  (b16/s512, 1×H100, eval excluded; +39% tok/s, −1.3 GB). `nwarmup0_b16.md`.
- **quality @20k** (run `p-3105a340` vs baseline `p-20f9fc15`,
  https://wandb.ai/goodfire/param-decomp/runs/p-3105a340) — **FAIL, gate regressed**:

  | tier | metric | baseline | variant | Δ% | verdict |
  |---|---|---|---|---|---|
  | primary | PGD recon (PPGD) | 1.4774 | 51.085 | +3357.7% | REGRESSED |
  | primary | no_beta | 275.33 | 202.05 | −26.6% | improved |
  | secondary | Stochastic recon | 0.53068 | 0.36084 | −32.0% | improved |
  | secondary | CI-mask recon | 0.90267 | 0.64863 | −28.1% | improved |
  | secondary | L0 (info) | 1879.5 | 2599 | +38.3% | REGRESSED |
  | gate | CE diff | 0.30395 | 0.36025 | +18.5% | REGRESSED |
  | gate | KL | 0.64088 | 0.62066 | −3.2% | improved |
  | gate | CE unrec | 0.0045113 | 0.0053461 | +18.5% | REGRESSED |

- **quality @50k:** not run — killed on the 20k gate failure. Job 644454 cancelled.

## Verdict
**killed** — faithfulness **gate regressed** at 20k (CE-diff/CE-unrec +18.5%) and the eval
PPGD-recon attack **blew up +3358%** (1.48→51.1). Dropping both PPGD warmup steps produces a
decomposition that is far less robust to the eval PPGD attack — the 28% speedup isn't worth
it. (no_beta + the masked-recon secondaries improved, but that's irrelevant once the gate
fails / PPGD blows up.)
