# Track 2 — Method Speedups (the measurement contract)

This is the in-repo, re-read-each-session contract for Track 2. The full rationale is in
[`/plan.md`](../plan.md); this file is the operating manual the agents follow.

**Goal:** change the core PD method (`param_decomp/`) so the loop runs much faster for
similar quality, validated **only** on small models. A finished result is a faithful
speedup or simplification that **holds on `pile_llama_simple_mlp-4L` (T1)** under the
rules below. No 3-pool / multipool / 8b work — that is Track 1.

---

## The model ladder

| Tier | Config | Smoke | Role |
|---|---|---|---|
| **T0** | `param_decomp_lab/experiments/lm/ss_llama_simple_mlp-2L.yaml` | `…-2L-smoke.yaml` | Fast iteration; a T0 win is a *hypothesis*. |
| **T1** | `param_decomp_lab/experiments/lm/pile_llama_simple_mlp-4L.yaml` | `…-4L-smoke.yaml` | The bar; the T1 number is the headline. |

Iterate on T0; **confirm on T1**; the T1 result is what we claim. Report speedups at
**both** tiers — never carry a T0 speedup up as if it were the T1 result.

> Real cost numbers (week-1 measurement) go in [`ledger/baselines.md`](ledger/baselines.md).

---

## The ruler — READ-ONLY for an experiment

An experiment may rewrite `param_decomp/` freely. It may **NOT**, as part of an
experiment, edit any of:

- the **eval harness** / eval-metric classes (`param_decomp_lab/eval_metrics/`),
- the **quality bundle** (`param_decomp_lab/speedup/quality_bundle.py`),
- the **baseline pointers** (`ledger/baselines.md`),
- this contract.

Changing any of these is a separate, explicitly-flagged change a human reviews — it's the
ruler, not the thing being measured. (Mitigates goalpost-moving / reward-hacking.)

---

## What every result must report

1. **Speed (the thing we trade):** ≥**10% faster** is the floor for "material" (wall-clock/step or
   GPU-h). Report step time + tokens/sec + peak memory + a profiler op breakdown, on a **pinned GPU
   at fixed batch/seq, eval excluded** — `pd-speedup-bench`.
2. **Quality bundle (the objective), tiered** — all of `QUALITY_BUNDLE` (`quality_bundle.py`), via
   `pd-speedup-compare`, which prints an **overall verdict**:
   - **primary** (must hold or improve): **PPGD recon** (`PGDReconLoss`, the eval-time PGD attack —
     *not* the train-time persistent loss) + **L0** (sparsity);
   - **secondary** (informational): stochastic-mask recon + CI-mask recon;
   - **gate** — CI-masked faithfulness (CE-diff / CE-unrecovered / KL). This is a **hard
     constraint**, not a soft guardrail: regress it past the band and the result **fails regardless
     of PPGD/L0** (else a "win" is just trading faithfulness for lower L0 — a worse decomposition).

   Decision rule: faithfulness gate first, then **primary** drives WIN/NEUTRAL. Report the whole
   vector so nothing hides.
3. **Statistics — band = measured noise floor; judge early.** The **T0 baseline is 3 seeds**; the
   per-metric band is the seed spread (`pd-speedup-compare <seed1>,<seed2>,<seed3> <variant>`),
   floored at `--tol_pct` (default ±2%). **Don't wait for full training**: ~**50k steps is enough
   signal** (incl. 4L) — compare at a matched step against the baseline's cached trajectory
   (`--at_step 50000`, on a slow-eval step). Experiments may be single-seed (judged against the
   3-seed baseline band). **A promising result graduates to a longer run** before *merge* (early
   ranking can cross over). T1's baseline is single-seed → ±`tol_pct` floor only; lean on the T0 band.
4. **Eval is frozen (ruler).** The experiment runs the **same `eval:` block** as the baseline —
   metrics, the **PGD-attack strength** (`PGDReconLoss` `n_steps`/`step_size`), eval batch, cadence.
   Weakening the attack or changing eval batch is a reward-hack; `pd-speedup-compare` flags drift.
5. **Artifacts:** every number cites a `run_id`, its `metrics.jsonl` path, and (if wandb) the URL.
   **No artifact → it didn't happen.**
6. **Asserts stay on.** A speedup that trips an `isfinite`/shape assert is a failure, not a finding.

---

## The harness (base DDP `pd-lm` path)

Activate the venv first (`source .venv/bin/activate`). Console scripts (`pd-speedup-*`)
need `make install-lab`; otherwise use the `python -m` form.

```bash
# 1. Smoke — first thing after editing core. Finishes in minutes, no wandb.
python -m param_decomp_lab.experiments.lm.run \
    param_decomp_lab/experiments/lm/ss_llama_simple_mlp-2L-smoke.yaml

# 2. Benchmark — primary metric (step time, tokens/sec, peak mem, profiler breakdown).
python -m param_decomp_lab.speedup.benchmark \
    param_decomp_lab/experiments/lm/ss_llama_simple_mlp-2L.yaml --out bench.md

# 3. Compare — tiered diff + faithfulness-gated verdict. T0 baseline = 3 seeds (comma-sep) →
#    band from their spread. --at_step judges a partial run vs the baseline at the same step.
python -m param_decomp_lab.speedup.compare_runs <seed1>,<seed2>,<seed3> <variant_run_id> --at_step 50000 --out cmp.md
```

Profiler instrumentation note: the `pd/*` `record_function` labels live on the Track-1
profiling branches, not `main`. The benchmark therefore reports a plain torch.profiler
op breakdown (aten-op level). Add finer labels inside an experiment if a change needs them.

---

## Workflow

`proposed → T0 (iterate) → T1 (confirm) → merged / killed / parked`. No human gate before
T1 — agents promote themselves once T0 looks good. **GPU budget: a single run may use up to
16 GPUs (multi-node `--dp`, multiple of 8); keep ≤16 in flight across all Track-2 runs.**

- **One experiment = one worktree = one branch = one spec.** Branch
  `feature/spd-<short-idea>`. In a worktree, `uv sync` first; never `cd` back to main.
- Copy [`ledger/TEMPLATE.md`](ledger/TEMPLATE.md) → `ledger/experiments/<id>.md`, fill the
  hypothesis / claim type / thresholds (quoted from this contract) / artifact links.
- Add/maintain a row in [`ledger/README.md`](ledger/README.md) — the human's whole report
  surface. Record kills as first-class rows so nobody re-runs a dead end.
- Long runs are **background SLURM jobs** (default partition; no custom CPU/mem;
  `--qos=opportunistic`/`scavenge` for speculative/over-quota). **Prefix every SLURM job name with
  `ai-`** (`pd-lm … --job_name ai-pd-lm`) so AI-launched jobs are identifiable. Poll your own jobs
  (`squeue --me`); never cancel others'. After a crash, verify `squeue --me` for zombie GPU holders.
- `/code-review` the diff before merging.

**Killing is a first-class outcome.** "Tried X, here's the run, it regressed Y" with
artifacts goes in the ledger so nobody re-runs it.

### Promotion criteria

- **→ T0:** smoke passes, unit + equivalence tests green, no NaNs.
- **T0 → T1:** `pd-speedup-compare` overall verdict WIN or NEUTRAL vs the **3-seed T0 band** at
  ~50k (faithfulness gate held, primary within-or-better) **and ≥10% speedup** on `ss-2L`. Single
  experiment seed OK.
- **T1 → merged:** same verdict vs `s-55ea3f9b` at ~50k + ≥10% speedup, **then a longer
  confirmation run** showing the win holds later in training (guards early/late crossover). Flip the
  change to default.
