# Smooth-L0 vs L_p importance-minimality — comparison plan

## What this tests

Whether the bounded **smooth-L0** (Geman–McClure) importance-minimality penalty
`φ(c) = c² / (c² + γ²)` reaches **performance parity (or improvement)** with the
incumbent **`L_p`** penalty `(c + eps)^p` (`p` annealed to 0) on the
`pile_llama_simple_mlp-4L` decomposition.

Motivation: the `L_p` gradient `p·c^(p-1)` blows up as `c→0` for `p<1` — an infinite
cliff at the accumulation point where most components sit, which makes low-`p` training
metastable. Smooth-L0 is flat at 0 (`φ'(0)=0`), saturates to 1 (its per-component sum ≈
the active-component count), and has a bounded gradient (`~0.65/γ` at the finite, meaningful
threshold `c≈γ`).

## What was implemented (this branch)

- `SmoothL0ImportanceMinimalityLoss` (`param_decomp/metrics/smooth_l0_importance_minimality.py`)
  + `SmoothL0ImportanceMinimalityLossConfig` (`param_decomp_config/losses.py`). New loss;
  the `L_p` `ImportanceMinimalityLoss` is untouched (so the app's `pnorm` DB/API are
  unaffected).
- Both penalties registered **eval-side** (`EVAL_METRIC_CLASSES` + `AnyEvalMetricConfig`)
  so a run driven by one logs the other as an eval-only sparsity proxy.
- Shares the `L_p` variant's `beta`-entropy term and exact-DDP all-reduce
  (`finalize_imp_min` / `lp_and_entropy_terms` are penalty-agnostic).
- Tests: `param_decomp/tests/metrics/test_smooth_l0_importance_minimality_loss.py`
  (15, incl. the defining flat-finite-gradient-at-0 and bounded-gradient-peaks-at-γ checks).

## Runs

Baseline reference: **`p-20f9fc15`** (`jose-refactor-p-20f9fc15`, torch, 400k steps,
`L_p` `pnorm 2.0→0.4`, `beta 0.5`, `coeff 2e-4`, dp=16 on h100).

Two fresh runs (identical code + b200 hardware + seed=0), both cross-logging the *other*
penalty as an eval-only proxy:

| Leg | Config | Imp-min driver | Eval-only proxy |
|---|---|---|---|
| **L_p control** | `pile_llama_simple_mlp-4L_impmin_lp.yaml` | `ImportanceMinimalityLoss` (`pnorm 2.0→0.4`) | `SmoothL0ImportanceMinimalityLoss` (`γ 1.0→0.1`) |
| **smooth-L0** | `pile_llama_simple_mlp-4L_impmin_smoothl0.yaml` | `SmoothL0ImportanceMinimalityLoss` (`γ 1.0→0.1`) | `ImportanceMinimalityLoss` (`pnorm 2.0→0.4`) |

Both: `steps: 400000`, `coeff: 2e-4`, `beta: 0.5`, anneal over the full run, dp=16, normal QoS.

The L_p control reproduces `p-20f9fc15` exactly except for the added smooth-L0 eval probe,
so the primary comparison (smooth-L0 vs L_p control) is apples-to-apples on identical
code+hardware; `p-20f9fc15` is the secondary sanity reference.

## Measurement points

**Step 5k and 10k** (per the request). Note: at 5k/10k of 400k the anneal has barely
moved (`γ≈0.98`, `p≈1.96`), so this measures the *early-training establishment* phase, not
the low-`γ`/low-`p` cliff regime (that is a >300k phenomenon). Eval cadence is every 1k
steps (slow eval at 0/10k), so both points have logged evals.

## Metrics to compare (at 5k and 10k)

Primary — directly comparable across both penalties:
- `eval/.../CI_L0` **total** (and per-layer) — hard active-component count, the
  penalty-agnostic sparsity number.
- `eval/.../CEandKLLosses` — recon faithfulness (KL to clean logits, CE). The
  faithfulness side of the sparsity/faithfulness trade-off.
- `eval/loss/PGDReconLoss` (20-step) and `eval/loss/StochasticHiddenActsReconLoss` /
  `CIHiddenActsReconLoss` — reconstruction under ablation.

Secondary:
- Each penalty's own `eval/sparsity/<Name>_no_beta` proxy (on its own scale — read
  trajectory shape, not absolute cross-penalty value).
- `ComponentActivationDensity`, `CIMeanPerComponent`, `CIHistograms` — CI distribution
  shape (is smooth-L0 producing the cleaner bimodal split it's designed for?).

Parity verdict: smooth-L0 reaches **≤** the L_p control's CI_L0 at **≈ equal** recon
KL/CE (or a better point on that trade-off curve), with finite, non-spiking CI gradients.

## How to launch

```bash
source .venv/bin/activate
pd-lm param_decomp_lab/experiments/lm/pile_llama_simple_mlp-4L_impmin_lp.yaml \
  --dp 16 --time 04:00:00 --job_name ai-pd-lm-lp \
  --group smoothl0-vs-lp --tags smoothl0-compare,lp-control
pd-lm param_decomp_lab/experiments/lm/pile_llama_simple_mlp-4L_impmin_smoothl0.yaml \
  --dp 16 --time 04:00:00 --job_name ai-pd-lm-sl0 \
  --group smoothl0-vs-lp --tags smoothl0-compare,smoothl0
```

10k steps ≈ 50 min at the baseline's ~307 ms/step. Cancel each job once its 10k eval has
logged (we don't need the full 400k for this comparison; the 400k `steps` only fixes the
anneal/LR schedule shape).

## Results (2026-06-16, dp=8, b200, local-data subset)

Verdict: **smooth-L0 beats parity** — it dominates the L_p sparsity/faithfulness trade-off
at both 5k and 10k. At matched CI_L0, smooth-L0 gives lower KL-to-clean-logits and lower
20-step adversarial PGD recon.

Step 10k, std-schedule coeff sweep (CI_L0 total / KL_ci / PGD20):

| coeff | smooth-L0 | L_p |
|---|---|---|
| 1e-4 | 2495 / 0.556 / 1.46 | 2279 / 0.676 / 2.77 |
| 2e-4 | 1530 / 0.696 / 2.89 | 1256 / 0.840 / 3.95 |
| 5e-4 | 622 / 0.904 / 3.73 | 444 / 1.030 / 3.87 |
| 1e-3 | 324 / 1.047 / 5.46 | 262 / 1.275 / 5.52 |

Interpolating L_p onto matched CI_L0: smooth-L0's KL is ~0.1–0.15 lower across the whole
frontier (e.g. L0≈1530 → 0.696 vs ~0.80; L0≈324 → 1.047 vs ~1.19). Same shape at 5k.

Cliff pair (fast anneal → γ=0.05 / p=0.4 by step 8k), step 10k:

| | L0 | KL_ci | PGD20 |
|---|---|---|---|
| smooth-L0 fast | 340 | 0.886 | 3.05 |
| L_p fast | 311 | 0.917 | 4.09 |

Even in the aggressive low-γ/low-p regime smooth-L0 is more faithful and much better on
adversarial recon, and trained stably (no collapse — the original motivation).

Runs (wandb group `smoothl0-vs-lp`; wandb sync was failing during the network incident, so
the analyzed data is each run's local `metrics.jsonl`): smooth-L0 p-230213c2/p-4bc9946d/
p-bd7016b5/p-3a9545be/p-5c75cc70; L_p p-7d51ec1e/p-28af0784/p-7de2743e/p-2fad6768/p-1ca20256.
Raw 5k/10k table: `/tmp/sl0_results.tsv` (regenerate with `/tmp/extract_cmp.py`).

Caveat: 5k/10k is early training — std runs barely move the anneal (γ≈0.98, p≈1.96), so they
probe the soft regime; the cliff pair probes the sharp regime. The full-trajectory collapse
story (>300k) is untested here.

### Data source note

HF `streaming` of `danbraunai/pile-uncopyrighted-tok-shuffled` was failing (xet-CDN 408s on
every shard). These runs read the complete local snapshot instead, via the `parquet` builder:
`dataset_name: parquet` + `data_files: {train: <snap>/train-00[0-1]*.parquet, val:
<snap>/val-*.parquet}` (200-shard subset, ~76M sequences). Snapshot at
`/mnt/data/artifacts/hf_cache/hub/datasets--danbraunai--pile-uncopyrighted-tok-shuffled`.
`LMDataConfig.data_files` was widened to accept a per-split `dict[str,str]` for this.

## Contingency: hyperparameter sweep

Smooth-L0's loss scale (≈ active count) differs from `L_p`'s, so `coeff 2e-4` is a first
guess at matched sparsity pressure. If at 5k/10k the smooth-L0 leg is off the L_p
trade-off curve (too sparse / not sparse enough), the next step is a short sweep over
`coeff` (e.g. ×{0.3, 1, 3}) and/or `gamma_final` (e.g. {0.05, 0.1, 0.2}) at the same
5k/10k measurement points.
