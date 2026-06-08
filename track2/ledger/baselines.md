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
| local artifact | `runs/s-55ea3f9b/metrics.jsonl` (81 rows, distilled from W&B history) |

Compare against it with: `pd-speedup-compare s-55ea3f9b <variant_run_id>`.

Final quality-bundle values (the bar):

| metric | value |
|---|---|
| CE diff (CI-masked) | 0.28603 |
| CE unrecovered (CI-masked) | 0.0042753 |
| KL (CI-masked) | 0.34315 |
| L0 total (sparsity) | 201.45 |
| Stochastic hidden-acts recon | 0.41470 |
| PGD recon | 0.65225 |

## T0 — `ss_llama_simple_mlp-2L` — ☐ PENDING (single seed)

Measured cost (`pd-speedup-bench`, 1×H100, b64/s512): **~0.29 s/step, 28 GB peak** →
400k steps ≈ **32 GPU-h** single-GPU (≈8 h at `--dp 4`).

To lock (after the launch decision):

```bash
pd-lm param_decomp_lab/experiments/lm/ss_llama_simple_mlp-2L.yaml \
    --group t2-baseline-ss2L --tags baseline   # add --dp 4 to finish in ~8h
```

Then record below and compute the final quality-bundle values.

| seed | run_id | metrics.jsonl | wandb URL |
|---|---|---|---|
| 0 | _TODO_ | | |

Final quality-bundle values: _TODO_
