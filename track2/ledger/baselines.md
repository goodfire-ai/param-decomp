# Locked baseline — READ-ONLY for experiments

The canonical run every experiment compares against. Agents compare against **this exact artifact**,
never a re-derived baseline. Re-run it only when a human deliberately rebases it (keep the old one).

**Band policy:** single baseline (1 seed) → no spread, so the band is just **±`tol_pct`** (default
±2%); treat deltas near that floor as noise. **20k/50k are an early screen, not a converged
judgment:** faithfulness is ~settled by 50k (the gate is meaningful), but the primary metrics
(PGD-recon, `no_beta`) are far from their 400k values. L0 is non-monotonic and demoted to
informational. So a **primary WIN needs a longer (400k) confirmation run** (see `plan_t1.md` §Why
early-step screening).

---

## `pile_llama_simple_mlp-4L` (post-refactor) — ⏳ REGENERATING

`p-20f9fc15` re-runs the 4L config on current code so it logs the **`no_beta`** importance-minimality
term (the predecessor `p-5b17949e` predates that metric). Same config, so its trajectory should match
`p-5b17949e` within noise. It **self-caches** its own `metrics.jsonl` live during training — no W&B
distillation. Judged **early** (20k & 50k). (Pre-refactor predecessor of the config: `s-55ea3f9b`,
whose final 400k metrics `p-5b17949e` matched — PGD recon 0.660 vs 0.652, L0 201 vs 201, CE diff
0.273 vs 0.286.)

| field | value |
|---|---|
| run short code | `p-20f9fc15` (job 644450) |
| W&B | https://wandb.ai/goodfire/param-decomp/runs/p-20f9fc15 |
| steps | 400000 (batch 64, dp 16) |
| wall-clock | ~0.198 s/step (predecessor) → ≈1.1 h to 20k, ≈2.75 h to 50k, ≈22 h to 400k |
| local artifact | `runs/p-20f9fc15/metrics.jsonl` (written live by `pd-lm`; eval every 1k, slow every 10k) |
| eval-ruler | `runs/p-20f9fc15/experiment_config.yaml` = the `pile_llama_simple_mlp-4L` config (eval PGD attack `n_steps 20`/`step_size 0.1`) |
| predecessor | `p-5b17949e` (no `no_beta`; stays cached for the other metrics' reference values) |

Compare with: `pd-speedup-compare p-20f9fc15 <variant> --at_step 20000` and `--at_step 50000`.

Quality-bundle **bar at the early checkpoints**. **@20k and @50k are now read from `p-20f9fc15`**
(the live baseline). **@400k is still the predecessor `p-5b17949e`'s values (†), pending refresh**
once `p-20f9fc15` reaches it (`no_beta` @400k pending too). ⚠ `p-20f9fc15`'s @20k PPGD (1.4774) was
**~2× below** the predecessor's (3.0936) — the early trajectory did **not** match within noise — but
by @50k they reconverge (PPGD 0.872 vs predecessor 0.965).

| tier | metric | @20k | @50k | @400k | early-step note |
|---|---|---|---|---|---|
| primary | PGD recon (PPGD) | 1.4774 | 0.87151 | 0.65964† | monotone↓ — ranks variants |
| primary | Importance-minimality (no_beta) | 275.33 | 213.95 | _pending_ | monotone↓ (275.3→214.0) — well-behaved sparsity proxy |
| secondary | Stochastic hidden-acts recon | 0.53068 | 0.44642 | 0.40159† | monotone↓ |
| secondary | CI-masked hidden-acts recon | 0.90267 | 0.80279 | 0.82510† | ~flat |
| secondary | L0 total (informational) | 1879.5 | 2517.5 | 200.78† | **non-monotone** — informational only |
| gate | CE diff (CI-masked) | 0.30395 | 0.27519 | 0.27273† | settled early |
| gate | KL (CI-masked) | 0.64088 | 0.45218 | 0.33454† | drifting↓ — same-step only |
| gate | CE unrecovered (CI-masked) | 0.0045113 | 0.0040894 | 0.0040755† | settled early |

† predecessor `p-5b17949e` value — pending refresh from `p-20f9fc15` (not yet at this step).
