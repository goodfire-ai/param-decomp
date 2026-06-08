# spd-ppgd-nwarmup0-t1 — PPGD n_warmup_steps 2 → 0

- **Claim type:** speedup
- **Stage:** running (quality screening at 20k/50k)
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
- **quality @20k / @50k:** _pending_ — screening run `p-3105a340`
  (https://wandb.ai/goodfire/param-decomp/runs/p-3105a340), compare vs baseline
  `p-20f9fc15`.

## Verdict
_pending the 20k/50k quality screen._
