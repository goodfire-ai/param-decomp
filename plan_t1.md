# Track T1 — speed up the core PD method on the big run, judged early

**Goal:** change the core PD method (`param_decomp/`) so the training loop runs **faster in wall-clock**
on the **real 4-layer pile run** — the one that matters — while keeping the **same or better**
decomposition quality. One model, one config, one baseline. To iterate fast we **judge early** (at
20k and 50k steps) instead of waiting for the full 400k.

> The measurement contract (the ruler, what to report) is [`track2/README.md`](track2/README.md);
> harness code is `param_decomp_lab/speedup/`; the ledger is `track2/ledger/`.

---

## Why early-step screening

The full baseline took **22.0 h on 16 GPUs** (400k steps, ~0.198 s/step). Waiting for 400k per idea
is too slow. **Hypothesis (likely-but-not-certain):** a variant's standing at **20k / 50k** is
indicative of where it lands at 400k. If so, iteration time is **~1 h** (20k) instead of a day.

What the baseline's own trajectory says about that hypothesis (read these as the screen's known
biases — see [`track2/ledger/baselines.md`](track2/ledger/baselines.md)):

| metric (tier) | @20k | @50k | @400k | early-step usefulness |
|---|---|---|---|---|
| PGD recon — PPGD (primary) | 3.094 | 0.965 | 0.660 | **monotone↓** — same-step rank is meaningful |
| L0 total (primary) | 1819 | 2465 | 201 | **non-monotone** (rises to 50k, then collapses 12×) — *do not* chase early L0; treat as within-band guardrail only |
| Stochastic hidden-acts recon (secondary) | 0.534 | 0.447 | 0.402 | monotone↓ |
| CI-masked hidden-acts recon (secondary) | 0.906 | 0.807 | 0.825 | ~flat |
| CE diff CI-masked (gate) | 0.286 | 0.280 | 0.273 | **settled early** — gate is meaningful |
| KL CI-masked (gate) | 0.626 | 0.456 | 0.335 | still drifting↓ — compare same-step only |
| CE unrecovered CI-masked (gate) | 0.00425 | 0.00416 | 0.00408 | **settled early** — gate is meaningful |

**Upshot:** at 20k/50k the **faithfulness gate is trustworthy**, **PGD recon ranks variants** (lower
at the same step = good sign), and **L0 is non-monotonic so it only gates within-band** — a primary
*win* on L0/PPGD still has to be confirmed at 400k (see §Rules, "early ≠ converged").

## The baseline (the only one)

- **Run:** `p-5b17949e` — `jose-replica…post-refactor`, the post-large-refactor equivalent of Jose's
  canonical run. `pile_llama_simple_mlp-4L`, **400k steps, batch 64, dp 16**.
  W&B: https://wandb.ai/goodfire/param-decomp/runs/p-5b17949e (project `param-decomp`).
- **Config:** reproduced by `param_decomp_lab/experiments/lm/pile_llama_simple_mlp-4L.yaml` (eval
  block verified identical to the run — it's the ruler). Final 400k metrics match the pre-refactor
  predecessor `s-55ea3f9b` (PGD recon 0.660 vs 0.652, L0 201 vs 201, CE diff 0.273 vs 0.286).
- **Single baseline → band = ±`tol_pct`** (default ±2%); there's no seed spread, so treat deltas
  near the floor as noise.
- **Read-only.** Cached locally at `runs/p-5b17949e/` (`metrics.jsonl` distilled from W&B history,
  every 10k steps; `experiment_config.yaml` for eval-parity), so `pd-speedup-compare` resolves it.
- **Compare points: step 20000 and step 50000.** Both are multiples of `slow_every` (10000), so
  every bundle metric (incl. the slow PGD-recon attack) is present at both.

## What counts as a win

A change to `param_decomp/` (or the config) that is **both**:

1. **≥10% faster** — wall-clock per step, from `pd-speedup-bench` on a pinned GPU at the baseline's
   batch/seq, eval excluded. (Measurable in minutes — no need to wait for 20k.) **Single-GPU memory
   caveat:** the 4L config at batch 64 doesn't bench single-GPU — it pegs an 80 GB H100 (~81 GB,
   the real run spreads it over dp 16 = per-GPU batch 4) and times out. **Bench at batch 16**
   (measured: 766 ms/step, 57.4 GB) via a batch-16 copy of the config — **identical for baseline
   and variant** so the per-step ratio is fair; the 50k quality run still uses real batch 64 / dp 16.
2. **Quality same-or-better** vs the baseline **at both 20k and 50k** — `pd-speedup-compare` prints
   the verdict per tier:
   - **Gate — faithfulness** (CE-diff / KL / CE-unrecovered): must stay **within band** at the same
     step. A regression here **fails the result regardless of speed**.
   - **Primary** — **PPGD recon** (eval `PGDReconLoss`) + **L0**: PPGD within-band-or-better;
     **L0 only needs to be within band** (non-monotonic early — see table). A *primary win* (PPGD
     improved) is a **candidate**, flagged for a 400k confirmation, not a final win on its own.
   - **Secondary** — stochastic / CI-mask recon: not meaningfully worse.

Below 10%, or any gate regression at 20k or 50k → not a win (record a kill).

## Measure it (two commands)

```bash
source .venv/bin/activate   # pd-speedup-* exist after `make install-lab`

# Speed: variant config vs baseline config (same settings), pinned GPU, eval excluded.
python -m param_decomp_lab.speedup.benchmark <variant-config>.yaml --out bench.md

# Quality: variant run vs baseline, at BOTH early checkpoints.
python -m param_decomp_lab.speedup.compare_runs p-5b17949e <variant-run-id> --at_step 20000
python -m param_decomp_lab.speedup.compare_runs p-5b17949e <variant-run-id> --at_step 50000
```

## Run one experiment

1. **Change** one thing. Variants must reproduce `p-5b17949e` except the change — derive the config
   from `pile_llama_simple_mlp-4L.yaml` (one-line diff). Work on a branch off `feature/track2-t1`;
   for a core edit, a worktree keeps it isolated:
   `git worktree add <path> -b feature/spd-<idea> feature/track2-t1` then `uv sync --all-packages`.
2. **Smoke:** `…/pile_llama_simple_mlp-4L-smoke.yaml` (variant change applied) — finishes in minutes,
   must run clean (no NaNs). For an approximation, add an equivalence test.
3. **Bench** the variant vs the baseline config → confirm **≥10%**. (Fast — do this before any 50k.)
4. **Run to 50k:** `pd-lm <variant-config>.yaml --dp 16 --job_name ai-pd-lm` with `pd.steps: 50000`
   (background SLURM, default partition). One run yields **both** the 20k and 50k eval points
   (eval/slow every 1k/10k). ~1.1 h to 20k, ~2.75 h to 50k on 16 GPUs.
5. **Compare** at **20k first** (the fast read, ~1.1 h) then **50k** (confirm). Read the verdict —
   **faithfulness gate first**, then PPGD; L0 within-band only.
6. **Record** one row in `track2/ledger/README.md` + a card in `track2/ledger/experiments/<id>.md`:
   speedup, both-step verdicts, run_id + W&B URL. Mark a PPGD-improving result as a **400k-confirm
   candidate**, not a closed win.

## Running this as a loop

Autonomous, no check-in between iterations. GPUs are abundant — **don't serialize on one in-flight
run**; reap finished runs and launch a new idea each iteration. Every iteration does both, then stops:

1. **Reap** — for each pending run from a prior iteration: if it's **past 20k**, run the §"Measure it"
   compare at 20k (and at 50k once past 50k), read the verdict (**gate first**), record it. A run
   that's failed the gate at 20k can be **cancelled early** (don't wait for 50k) and recorded as a
   kill. Leave still-pre-20k runs for the next iteration.
2. **Launch one** — pick the next idea (see §Where to look for speed), **smoke**, then **bench** vs
   the baseline config. `<10%` faster or anything fails (NaNs, tripped asserts) → record a **kill**.
   `≥10%` → submit the 50k run (`pd-lm <variant-config>.yaml --dp 16 --job_name ai-pd-lm`, background
   SLURM, default partition) and note the run_id for the next iteration to reap.

- **Autonomy:** auto-submit the 50k run when the 10% bench gate passes — don't pause for approval.
- **GPU budget:** **≤16 GPUs per run** (`--dp 16`). Concurrent runs are fine; hundreds of GPUs total
  is OK (per `CLAUDE.md` Track-2 exception).
- **Self-pacing:** a run reaches 20k in ~1.1 h and 50k in ~2.75 h — schedule the next wake-up around
  the nearer of the pending runs' 20k marks.

Invoke with: `/loop work the next Track T1 experiment per plan_t1.md` (no interval — self-paces around
the ~1–3 h runs).

## Rules (keep us honest)

- **Don't edit the ruler** — the `eval:` block (incl. the eval PGD attack `n_steps`/`step_size` and
  eval batch), `param_decomp_lab/speedup/quality_bundle.py`, or the baseline. `pd-speedup-compare`
  flags eval-config drift.
- **Compare same-step** — only ever 20k-vs-20k and 50k-vs-50k (KL and L0 are still moving early;
  cross-step diffs are meaningless).
- **Artifacts or it didn't happen** — every number cites a run_id + W&B URL.
- **Asserts stay on** — a speedup that trips `isfinite`/shape checks is a fail, not a finding.
- **Early ≠ converged** — 20k/50k *screen and rank*; a primary win is a **400k-confirm candidate**,
  not a closed result (esp. L0, which is non-monotonic — see §Why).

## Where to look for speed (measure, don't assume)

`pd-speedup-bench`'s profiler shows where the time goes. Three compute centers — the 4L config has a
large CI transformer (d_model 2048 / 8 blocks), so re-profile rather than assume:

- **PPGD inner loop** — `(n_warmup_steps + n_samples)` masked component forwards
  (`metrics/persistent_pgd_recon.py`). The prime target: an earlier probe on a smaller 2-layer model
  found the PPGD inner loop dominates the step (dropping `n_warmup_steps` 2→0 cut ~33%), while the CI
  fn was only ~5%. Levers: fewer inner steps (`n_warmup_steps`), cheaper source optimizer/scope,
  lower precision, or an **approximate component forward** (validated by an equivalence test).
- **CI function** (`global_shared_transformer`, here d_model 2048 / 8 blocks) — runs every step.
  Cheaper architecture / fewer blocks / lower precision. (Only ~5% on the small model — re-check at
  4L scale before assuming the same.)
- **Target forward** — frozen; per-step it's recomputed for a fresh batch, so caching across steps
  doesn't apply, but within a step its activations are reused.
- Plus: output reconstruction is **logit-KL** — try hidden-state MSE (cheaper, no vocab matmul).

## Status (2026-06-08)

- Baseline `p-5b17949e` cached and verified: `pd-speedup-compare p-5b17949e <variant> --at_step
  {20000,50000}` resolves and self-checks within-band. Bar values are in
  [`track2/ledger/baselines.md`](track2/ledger/baselines.md).
- Baseline bench (1×H100, **b16**/s512, eval excluded): **766 ms/step**, peak mem 57.4 GB
  (`pd-speedup-bench` on a batch-16 copy of `pile_llama_simple_mlp-4L.yaml`). Batch 64 does **not**
  bench single-GPU — it pegs the 80 GB H100 (~81 GB) and times out; use batch 16 for the bench
  ratio (same for baseline + variant), batch 64 / dp 16 for the 50k quality run.
- No experiments launched yet. First idea: drop the PPGD inner warmup steps (`n_warmup_steps` 2→0) —
  the prime PPGD lever from the smaller-model probe.
