# Track T0 — speed up the core PD method on one small run

**Goal:** make the PD training loop **faster in wall-clock** on a single small run while keeping the
**same or better** decomposition quality. One model, one config, one baseline. Winners get handed to
Dan to try on the bigger runs.

> Simplified, single-baseline plan. (`plan.md` has the fuller two-tier version; ignore it for this.)
> Harness + ledger already exist: `param_decomp_lab/speedup/` and `track2/ledger/`.

---

## The baseline (the only one)

- **Config:** `param_decomp_lab/experiments/lm/ss_llama_simple_mlp-2L-baseline.yaml` — SimpleStories
  2-layer, **50k steps, batch 32**, current code.
- **Run as 3 seeds** for a noise floor: `p-db5adc3b`, `p-89a42376`, `p-993ae14a`
  (W&B group `t2-baseline-ss2L`, project `param-decomp`). The per-metric **band = the spread across
  the 3 seeds**.
- **Read-only.** Don't re-run or redefine it (except a deliberate rebase). It's cached locally, so
  `pd-speedup-compare` resolves the ids.

## What counts as a win

A change to `param_decomp/` (or the config) that is **both**:

1. **≥10% faster** — wall-clock per step (or total to 50k), from `pd-speedup-bench` on a pinned GPU
   at the baseline's batch/seq, eval excluded.
2. **Quality same-or-better** vs the 3-seed band — `pd-speedup-compare` prints the verdict, by tier:
   - **Gate — faithfulness** (CI-masked CE-diff / CE-unrecovered / KL): must stay **within band**. A
     regression here **fails the result regardless of speed** (otherwise a "win" is just trading
     faithfulness for lower L0 — a worse decomposition).
   - **Primary** — **PPGD recon** (eval `PGDReconLoss`) + **L0** (sparsity): within band or better.
   - **Secondary** — stochastic-mask recon + CI-mask recon: not meaningfully worse.

Below 10%, or any gate regression → not a win (record it as a kill).

## Measure it (two commands)

```bash
source .venv/bin/activate   # console scripts pd-speedup-* exist after `make install-lab`

# Speed: variant vs baseline config (same settings), pinned GPU, eval excluded.
python -m param_decomp_lab.speedup.benchmark <variant-config>.yaml --out bench.md

# Quality: variant run vs the 3-seed baseline band (compares the 50k endpoint).
python -m param_decomp_lab.speedup.compare_runs p-db5adc3b,p-89a42376,p-993ae14a <variant-run-id>
```

## Run one experiment

1. **Change** one thing (config or `param_decomp/`). Work on a branch off `feature/track2-setup`
   (we never touch `main`). For a core edit, a worktree keeps it isolated:
   `git worktree add <path> -b feature/spd-<idea> feature/track2-setup` then
   `uv sync --all-packages` in it.
2. **Smoke:** `python -m param_decomp_lab.experiments.lm.run …/ss_llama_simple_mlp-2L-smoke.yaml`
   — finishes in minutes, must run clean (no NaNs). For an approximation, add an equivalence test.
3. **Bench** the variant vs the baseline config → confirm ≥10%.
4. **Run 50k:** `pd-lm <variant-config>.yaml --dp 2 --job_name ai-pd-lm` (background SLURM; default
   partition; prefix the job name with `ai-`).
5. **Compare** at 50k (command above) → read the overall verdict (faithfulness gate first).
6. **Record** one row in `track2/ledger/README.md` + a short card in
   `track2/ledger/experiments/<id>.md`: speedup, verdict, run_id + W&B URL.

## Rules (keep us honest)

- **Don't edit the ruler** — the `eval:` block (incl. the eval PGD attack `n_steps`/`step_size` and
  eval batch), `param_decomp_lab/speedup/quality_bundle.py`, or the baseline. Weakening the eval is
  cheating; `pd-speedup-compare` flags eval-config drift.
- **Artifacts or it didn't happen** — every number cites a run_id + W&B URL.
- **Asserts stay on** — a speedup that trips `isfinite`/shape checks is a fail, not a finding.
- **Caveat:** this is the 50k regime; L0 and PPGD-recon aren't fully converged at 50k (on the 4L
  baseline they were ~12× / ~50% off their 400k values). Fine for *ranking ideas* here; survivors
  must be confirmed at scale (Dan's job, not this track's).

## Where to look for speed (measure, don't assume)

`pd-speedup-bench`'s profiler shows where the time goes. Three compute centers:

- **PPGD inner loop** — `(n_warmup_steps + n_samples)` masked component forwards
  (`metrics/persistent_pgd_recon.py`). Levers: fewer inner steps (`n_warmup_steps`), cheaper source
  optimizer/scope, lower precision, or an **approximate component forward** (validated by an
  equivalence test). The component forward loops over modules — each module's C components are one
  batched einsum.
- **CI function** (`global_shared_transformer`) — runs every step to produce causal importances.
  Cheaper architecture / fewer blocks / lower precision.
- **Target forward** — the target is **frozen**, so its activations for a fixed batch are constant:
  cache / precompute instead of recomputing each step.
- Plus: the output reconstruction is **logit-KL** (`output_loss_type: kl`) — try hidden-state MSE
  (cheaper, no vocab-size logit matmul). (Note: the *faithfulness warmup* is separate — a one-time
  weight-space pre-train, not per-step.)

## Status (2026-06-08)

- 3 baseline seeds running (~1.5h); band locked on completion.
- First experiment `spd-ppgd-nwarmup0` (PPGD `n_warmup_steps` 2→0): **~33% faster**
  (156→105 ms/step); quality verdict pending the run + band.
