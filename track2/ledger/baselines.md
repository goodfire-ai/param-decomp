# Locked baselines — READ-ONLY for experiments

The canonical configs run on each tier, used as the fixed comparison point for every
experiment. Agents compare against **these exact artifacts**, never a re-derived baseline.
Re-run a baseline only when a human deliberately rebases it (keep the old one).

**Single-seed policy (2026-06-08):** baselines are one seed each. There is no measured seed
spread, so "within band" is a **fixed relative tolerance** (`pd-speedup-compare --tol_pct`,
default ±2%), not a σ. Escalate to multi-seed only before a final claim.

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

## T0 — `ss_llama_simple_mlp-2L` — ◐ RUNNING (single seed, reduced)

Baseline run on the **current** repo, intentionally cheaper than the canonical 400k/b64
config: `ss_llama_simple_mlp-2L-baseline.yaml` (steps 50k, batch 32). Not a perfect run —
a qualitatively representative comparison point for judging speedup changes.

| field | value |
|---|---|
| run short code | `p-db5adc3b` |
| W&B | https://wandb.ai/goodfire/param-decomp/runs/p-db5adc3b |
| config | `param_decomp_lab/experiments/lm/ss_llama_simple_mlp-2L-baseline.yaml` |
| SLURM job | 644416 (dp2) |
| status | running — fill final quality-bundle values + `metrics.jsonl` path on completion |

Compare against it with: `pd-speedup-compare p-db5adc3b <variant_run_id> [--at_step N]`.

Final quality-bundle values (by tier — _auto-filled on completion of `p-db5adc3b`_):

| tier | metric | value |
|---|---|---|
| primary | PGD recon (PPGD) | _TBD_ |
| primary | L0 total (sparsity) | _TBD_ |
| secondary | Stochastic hidden-acts recon | _TBD_ |
| secondary | CI-masked hidden-acts recon | _TBD_ |
| guardrail | CE diff (CI-masked) | _TBD_ |
| guardrail | KL (CI-masked) | _TBD_ |
| guardrail | CE unrecovered (CI-masked) | _TBD_ |

**Historical reference (not the active baseline):** `s-eab2ace8` — a full 400k SimpleStories
decomposition from an older repo/config (batch 24); metrics cached at `runs/s-eab2ace8/`.
Useful as a sanity reference, but experiments compare against `p-db5adc3b` (current code).
