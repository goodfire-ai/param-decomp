# Locked baselines — READ-ONLY for experiments

The canonical configs run on each tier, used as the fixed comparison point for every
experiment. Agents compare against **these exact artifacts**, never a re-derived baseline.
Re-run a baseline only when a human deliberately rebases it (keep the old one).

**Band policy (2026-06-08):**
- **T0 = 3 seeds** → the per-metric band is their **observed spread** (floored at `--tol_pct`,
  default ±2%). Pass the 3 seed ids comma-separated to `pd-speedup-compare`.
- **T1 = 1 seed** (`s-55ea3f9b`, can't re-run) → no spread, so the band is just **±`tol_pct`**;
  treat T1 deltas near that floor as noise and lean on the T0 band.
- **50k is a screen, not a converged judgment.** On `s-55ea3f9b`, faithfulness is ~settled by 50k
  (CE/KL within ~5%) but the *primary* metrics are not (PGD-recon ~53% and L0 ~12× off their 400k
  values). So at 50k the **faithfulness gate is meaningful**, but a **primary WIN needs a longer
  confirmation run** (§3.3/§7). The 3-seed T0 band already absorbs the large 50k L0 variance.

---

## T1 — `pile_llama_simple_mlp-4L` — ✅ LOCKED

Jose's existing run (we do **not** re-run it; the harness reads its stored W&B metrics).

| field | value |
|---|---|
| run short code | `s-55ea3f9b` |
| W&B | https://wandb.ai/goodfire/spd/runs/s-55ea3f9b |
| steps | 400000 |
| wall-clock | ~92,711 s (~25.8 h) |
| local artifact | `runs/s-55ea3f9b/metrics.jsonl` (full trajectory, distilled from W&B history) |
| eval-ruler | `runs/s-55ea3f9b/experiment_config.yaml` — the `pile-4L` config (verified to match the run's W&B eval block, incl. PGD attack `n_steps 20`/`step_size 0.1`) so `pd-speedup-compare` can check T1 eval-config parity |

Compare against it with: `pd-speedup-compare s-55ea3f9b <variant_run_id> [--at_step N]`.

Final quality-bundle values (the bar), by tier:

| tier | metric | value |
|---|---|---|
| primary | PGD recon (PPGD) | 0.65225 |
| primary | L0 total (sparsity) | 201.45 |
| secondary | Stochastic hidden-acts recon | 0.41470 |
| secondary | CI-masked hidden-acts recon | 0.85740 |
| guardrail | CE diff (CI-masked) | 0.28603 |
| guardrail | KL (CI-masked) | 0.34315 |
| guardrail | CE unrecovered (CI-masked) | 0.0042753 |

## T0 — `ss_llama_simple_mlp-2L` — ◐ RUNNING (3 seeds, reduced)

Three seeds of `ss_llama_simple_mlp-2L-baseline.yaml` (50k/b32, current repo) — cheaper than the
canonical 400k/b64, a qualitatively representative comparison point. **The 3 seeds give the noise
floor**: the per-metric band is their spread.

| seed | run short code | W&B | SLURM |
|---|---|---|---|
| 0 | `p-db5adc3b` | https://wandb.ai/goodfire/param-decomp/runs/p-db5adc3b | 644416 (dp2) — done |
| 1 | `p-89a42376` | https://wandb.ai/goodfire/param-decomp/runs/p-89a42376 | 644427 (dp2) |
| 2 | `p-993ae14a` | https://wandb.ai/goodfire/param-decomp/runs/p-993ae14a | 644429 (dp2) |

Compare against it with: `pd-speedup-compare p-db5adc3b,p-89a42376,p-993ae14a <variant> --at_step 50000`.

Quality-bundle values by tier — mean and [min,max] band across the 3 seeds
(_auto-filled on completion of all three seeds_):

| tier | metric | mean | band [min, max] |
|---|---|---|---|
| primary | PGD recon (PPGD) | _TBD_ | _TBD_ |
| primary | L0 total (sparsity) | _TBD_ | _TBD_ |
| secondary | Stochastic hidden-acts recon | _TBD_ | _TBD_ |
| secondary | CI-masked hidden-acts recon | _TBD_ | _TBD_ |
| gate | CE diff (CI-masked) | _TBD_ | _TBD_ |
| gate | KL (CI-masked) | _TBD_ | _TBD_ |
| gate | CE unrecovered (CI-masked) | _TBD_ | _TBD_ |

**Historical reference (not the active baseline):** `s-eab2ace8` — a full 400k SimpleStories
decomposition from an older repo/config (batch 24); metrics cached at `runs/s-eab2ace8/`.
Useful as a sanity reference, but experiments compare against `p-db5adc3b` (current code).
