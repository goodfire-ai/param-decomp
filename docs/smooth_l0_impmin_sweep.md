# Smooth-L0 importance-minimality — the change & what's being tested

Companion to [`smooth_l0_importance_minimality.md`](smooth_l0_importance_minimality.md)
(which lists the comparison configs + run commands). This note explains *why* the change
exists and what the currently-running sweep is measuring.

## The change

A new CI-sparsity penalty, `SmoothL0ImportanceMinimalityLoss`
(`param_decomp/metrics/smooth_l0_importance_minimality.py`), as a drop-in alternative to
the existing `L_p` `ImportanceMinimalityLoss`.

Per-CI-value penalty (Geman–McClure):

```
φ(c) = c² / (c² + γ²)        instead of L_p's   (c + eps)^p
```

Summed over components, `φ` saturates to ≈ the active-component count — a smooth surrogate
for L0. The motivation is the **gradient shape near zero**, where almost all components sit:

- `L_p` gradient is `p·c^(p-1)`, which **blows up as `c→0`** for `p<1` — a gradient cliff
  at the accumulation point that gets worse as `p` anneals down.
- `φ` is **flat at 0** (`φ'(0)=0`) and has a **bounded** gradient (`~0.65/γ` near `c≈γ`),
  so tightening the threshold never produces a cliff.

`γ` is the threshold; it linearly anneals from `gamma` → `gamma_final` over
`[gamma_anneal_start_frac, gamma_anneal_end_frac]` of training, mirroring how `p` anneals
in the `L_p` variant. The loss is self-contained (entropy term + DDP `world_size` handling
duplicated, not shared) so it adds no coupling to `importance_minimality.py`. Registered
loss-side (`configs.py`, `dispatch.py`) and eval-side (`eval_metrics/__init__.py`) so a run
driven by one penalty logs the other's `_no_beta` sparsity proxy at eval — the two are
directly comparable on a single run.

## What's being tested

Whether smooth-L0 traces a **better sparsity ↔ faithfulness frontier than `L_p`** when
decomposing `pile_llama_simple_mlp-4L` (target `t-9d2b8f02`), 400k steps. A prior 10-run
sweep on the sibling `feature/jax` branch found smooth-L0 *dominates* the `L_p` trade-off
(~0.1–0.15 lower KL at matched CI_L0); this re-runs the comparison on `main` (PR852).

Current run: a **3-point coefficient sweep** of the smooth-L0 driver, all with `γ 1.0→0.1`
annealed over the full run, `beta 0.5`. The coefficient sets the sparsity pressure:

| run | job | `coeff` | wandb |
|---|---|---|---|
| c6e-5 | 733382 | 6e-5 (weakest) | — |
| c2e-4 | 733381 | 2e-4 | — |
| c6e-4 | 733376 | 6e-4 (strongest) | p-74f9e274 |

Each cross-logs the `L_p` proxy. Success = the smooth-L0 CI_L0-vs-KL points sit on or below
the `L_p` control's frontier (lower CI_L0 = sparser, lower `kl_ci_masked` = more faithful).

**Key eval metrics:** `eval/l0/0.0_total` (sparsity ↓), `eval/ce_kl/kl_ci_masked` and
`eval/ce_kl/ce_difference_ci_masked` (faithfulness ↓), `eval/loss/PGDReconLoss` (20-step
adversarial recon).

## Live status (≈ step 12–13k / 400k, ~2.9 it/s, ETA ~37h, 3-day limit)

All three healthy on H100, no errors, and already tracing a clean monotonic frontier:

| `coeff` | CI_L0 total | kl_ci_masked | ce_diff_ci_masked |
|---|---|---|---|
| 6e-5 | 3481 | 0.445 | 0.052 |
| 2e-4 | 1510 | 0.660 | 0.218 |
| 6e-4 | 660 | 0.901 | 0.391 |

Higher coeff → sparser (lower L0) but less faithful (higher KL/CE), as expected. The `L_p`
control's frontier (from the cross-logged proxy / the `_impmin_lp.yaml` companion run) is
what these get compared against to judge dominance.
