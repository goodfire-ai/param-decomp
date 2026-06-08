# Locked baseline — READ-ONLY for experiments

The canonical run every experiment compares against. Agents compare against **this exact artifact**,
never a re-derived baseline. Re-run it only when a human deliberately rebases it (keep the old one).

**Band policy:** single baseline (1 seed) → no spread, so the band is just **±`tol_pct`** (default
±2%); treat deltas near that floor as noise. **20k/50k are an early screen, not a converged
judgment:** faithfulness is ~settled by 50k (the gate is meaningful), but the primary metrics are not
— PGD-recon and especially L0 are far from their 400k values, and L0 is non-monotonic. So a **primary
WIN needs a longer (400k) confirmation run** (see `plan_t1.md` §Why early-step screening).

---

## `pile_llama_simple_mlp-4L` (post-refactor) — ✅ LOCKED

The post-large-refactor replica of Jose's canonical run. We do **not** re-run it; the harness reads
its stored W&B metrics. Judged **early** (20k & 50k). (Pre-refactor predecessor: `s-55ea3f9b`, whose
final 400k metrics this run matches — PGD recon 0.660 vs 0.652, L0 201 vs 201, CE diff 0.273 vs 0.286.)

| field | value |
|---|---|
| run short code | `p-5b17949e` |
| W&B | https://wandb.ai/goodfire/param-decomp/runs/p-5b17949e |
| steps | 400000 (batch 64, dp 16) |
| wall-clock | ~79,308 s (~22.0 h on 16 GPUs) → ~0.198 s/step (≈1.1 h to 20k, ≈2.75 h to 50k) |
| local artifact | `runs/p-5b17949e/metrics.jsonl` (distilled from W&B history; eval points every 10k) |
| eval-ruler | `runs/p-5b17949e/experiment_config.yaml` = the `pile_llama_simple_mlp-4L` config (eval block verified identical to the run's W&B eval, incl. PGD attack `n_steps 20`/`step_size 0.1`) |

Compare with: `pd-speedup-compare p-5b17949e <variant> --at_step 20000` and `--at_step 50000`.

Quality-bundle **bar at the early checkpoints** (20k/50k are both `slow_every`=10k multiples, so all
bundle metrics are present); 400k shown for reference / convergence context:

| tier | metric | @20k | @50k | @400k | early-step note |
|---|---|---|---|---|---|
| primary | PGD recon (PPGD) | 3.0936 | 0.96510 | 0.65964 | monotone↓ — ranks variants |
| primary | L0 total (sparsity) | 1818.99 | 2465.32 | 200.78 | **non-monotone** — within-band gate only |
| secondary | Stochastic hidden-acts recon | 0.53357 | 0.44733 | 0.40159 | monotone↓ |
| secondary | CI-masked hidden-acts recon | 0.90560 | 0.80671 | 0.82510 | ~flat |
| gate | CE diff (CI-masked) | 0.28642 | 0.27975 | 0.27273 | settled early |
| gate | KL (CI-masked) | 0.62617 | 0.45567 | 0.33454 | drifting↓ — same-step only |
| gate | CE unrecovered (CI-masked) | 0.0042498 | 0.0041561 | 0.0040755 | settled early |
