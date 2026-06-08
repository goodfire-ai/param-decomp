# spd-ci-blocks2 — halve CI-fn transformer depth (n_blocks 4 → 2)

- **Claim type:** speedup
- **Stage:** proposed → killed (bench gate)
- **Branch / worktree:** none (config-only, on `feature/track2-setup`)
- **Owner agent:** /loop T0 iteration

## Hypothesis
The CI function (`global_shared_transformer`) runs every train step to produce causal
importances. The baseline uses a 4-block transformer (d_model 512, mlp 2048). Halving its
depth to 2 blocks should cut per-step time if the CI fn is a meaningful compute center —
orthogonal to the in-flight `spd-ppgd-nwarmup0` (which targets the PPGD inner loop).

## Diff vs baseline
Config only: `ss_llama_simple_mlp-2L-ci-blocks2.yaml` = `…-2L-baseline.yaml` with
`pd.ci_config.simple_transformer_ci_cfg.n_blocks: 4 → 2` (one line). Eval block unchanged.

## Success / kill thresholds
- **≥10% speedup** (step time, `pd-speedup-bench`) — else not worth it.
- Faithfulness gate + primary within the 3-seed T0 band.

## Results (artifacts required)

### T0 (`ss_llama_simple_mlp-2L`, 50k/b32)
- **benchmark: 156.02 → 148.55 ms/step = ~4.8% faster** (1×H100 slurm-dev-192-129, b32/s512,
  `pd-speedup-bench`, warmup5/measure20). Peak mem 14.23 → 13.33 GB. **Below the 10% floor.**
- smoke: `/tmp/ci-blocks2-smoke.yaml` (n_blocks 2, 20 steps) ran clean, no NaNs.
- No 50k run submitted (killed at bench).

## Verdict
**killed** — only ~4.8% faster, below the 10% gate. Finding: the CI transformer is only ~5%
of step time at this scale; per-step cost is dominated by the masked component forwards in
the PPGD inner loop (corroborated by `spd-ppgd-nwarmup0`'s ~33% from cutting PPGD forwards
3→1). Future speed ideas should target the component forward / PPGD, not the CI fn.
