# spd-ppgd-sign — PPGD source optimizer adam → sign (signSGD)

- **Claim type:** speedup
- **Stage:** killed
- **Branch / worktree:** `feature/spd-ppgd-nwarmup0-t1` (off `feature/track2-t1`)
- **Owner agent:** claude-opus (2026-06-08 session)

## Hypothesis
The PPGD inner loop optimizes adversarial sources with Adam. Switching to sign-PGD
(signSGD) drops the second-moment state/arithmetic, so each inner source step is cheaper.

## Diff vs baseline
Config-only, one block: `PersistentPGDReconLoss.optimizer.type: adam → sign` (drops
beta1/beta2/eps; same `lr_schedule`). Config
`pile_llama_simple_mlp-4L-ppgd-sign.yaml`.

## Success / kill thresholds (quoted from track2/README.md)
- ≥5% speedup (`pd-speedup-bench`, batch 16, vs the baseline config).
- Quality bundle within band at both 20k and 50k; faithfulness gate hard.

## Results (artifacts required)
- **benchmark:** baseline 649.99 ms/step → sign **627.34 ms/step** = **3.5% faster**
  (b16/s512, 1×H100, eval excluded). Peak mem unchanged. `ppgd-sign_b16.md`.
- **quality:** not run — failed the speed gate first.

## Verdict
**killed** — only 3.5% faster, below the 5% floor. The optimizer arithmetic is a tiny
slice of the PPGD step (the masked component forward/backward dominates), so swapping
Adam→sign barely moves wall-clock. The result is lr-independent (sign is ~3.5% faster
regardless of `lr`), so retuning the source lr wouldn't recover a speed win. No quality
run launched.
