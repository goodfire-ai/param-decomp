# Smooth-L0 importance-minimality — configs to run

New CI-sparsity penalty `φ(c) = c²/(c²+γ²)` (Geman–McClure), an alternative to the `L_p`
`ImportanceMinimalityLoss`. Flat gradient at 0, bounded `~0.65/γ` near `c≈γ` — no cliff as
the threshold tightens (`L_p`'s `p·c^(p-1)` blows up as `c→0` for `p<1`). `SmoothL0ImportanceMinimalityLoss`
in `param_decomp/metrics/smooth_l0_importance_minimality.py`.

## Configs

Both decompose `pile_llama_simple_mlp-4L` (target `t-9d2b8f02`), 400k-step schedule, and
cross-log the *other* penalty's sparsity proxy at eval (`eval/loss/{ImportanceMinimalityLoss,
SmoothL0ImportanceMinimalityLoss}_no_beta`) so the two are directly comparable on one run.

- `param_decomp_lab/experiments/lm/pile_llama_simple_mlp-4L_impmin_lp.yaml` — L_p control
  (`pnorm 2.0→0.4`) + smooth-L0 eval probe.
- `param_decomp_lab/experiments/lm/pile_llama_simple_mlp-4L_impmin_smoothl0.yaml` — smooth-L0
  driver (`γ 1.0→0.1`) + L_p eval probe.

Both use `coeff: 2e-4`, `beta: 0.5`. The smooth-L0 loss scale (≈ active-count) differs from
`L_p`'s, so sweep `coeff` (e.g. ×{0.3, 1, 3}) on each to trace the sparsity/faithfulness curve.

## Run

```bash
source .venv/bin/activate
pd-lm param_decomp_lab/experiments/lm/pile_llama_simple_mlp-4L_impmin_smoothl0.yaml \
  --dp 8 --group smoothl0-vs-lp --tags smoothl0,c2e-4
pd-lm param_decomp_lab/experiments/lm/pile_llama_simple_mlp-4L_impmin_lp.yaml \
  --dp 8 --group smoothl0-vs-lp --tags lp,c2e-4
```

`batch_size` is global (split across ranks), so `--dp 8` and `--dp 16` give the same
trajectory. Compare at 5k / 10k: `eval/.../CI_L0` total (sparsity, lower=sparser) vs
`eval/ce_kl/kl_ci_masked` (faithfulness, lower=better) and `eval/loss/PGDReconLoss` (20-step
adversarial). Parity = smooth-L0 matches/beats L_p's CI_L0-vs-KL frontier.

## Another cluster / data

Configs stream `danbraunai/pile-uncopyrighted-tok-shuffled` from HF (portable). If HF
streaming is unavailable, point at a local snapshot instead:

```yaml
data:
  dataset_name: parquet
  data_files: {train: "<snapshot>/data/train-*.parquet", val: "<snapshot>/data/val-*.parquet"}
```

## Prior result

A 10-run sweep on the sibling `feature/jax` branch (same loss math) found smooth-L0 **dominates
the L_p trade-off** at 5k/10k — ~0.1–0.15 lower KL at matched CI_L0 and lower 20-step PGD
recon, including a fast-anneal (γ→0.05 / p→0.4 by step 8k) cliff-regime pair. Re-running on
`main` is the point of this PR.
