# spd-ppgd-nwarmup1 — PPGD n_warmup_steps 2 → 1

- **Claim type:** speedup
- **Stage:** running (quality screening at 20k/50k)
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
- **quality @20k / @50k:** _pending_ — screening run `p-ebc2de5b`
  (https://wandb.ai/goodfire/param-decomp/runs/p-ebc2de5b), compare vs baseline
  `p-20f9fc15`.

## Verdict
_pending the 20k/50k quality screen._
