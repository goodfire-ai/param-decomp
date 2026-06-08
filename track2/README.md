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

1. **Primary metric (the thing we trade):** wall-clock/step + tokens/sec + peak memory +
   a profiler op breakdown, on a **pinned GPU at fixed batch/seq, eval excluded**.
   Produced by the benchmark command below.
2. **Quality bundle (the objective), as a tiered vector** — all of `QUALITY_BUNDLE`
   (`quality_bundle.py`), reported via `pd-speedup-compare` (groups by tier):
   - **primary** (must hold or improve): **PPGD recon** (`PGDReconLoss`, the eval-time PGD
     attack — *not* the train-time persistent loss) + **L0** (sparsity);
   - **secondary** (weighted less): stochastic-mask recon + CI-mask recon;
   - **guardrail** (must not blow up): CI-masked faithfulness (CE-diff / CE-unrecovered / KL).

   A change reports the **whole** vector so it can't hide a regression in one by improving
   another. Promote/kill calls are driven by the **primary** tier.
3. **Statistics — single-seed, judged early.** Iterate at one seed to move fast.
   **Don't wait for full training** (esp. the 4L Pile run): the baselines have their full
   trajectory cached, so compare at a **matched intermediate step**
   (`pd-speedup-compare --at_step N`, with `N` on a slow-eval step). Run experiments to a
   reduced budget and compare the trajectory. Escalate to ≥3 seeds / a longer run only to
   confirm a borderline winner. Because baselines are single-seed, **"within band" is a fixed
   relative tolerance** (`--tol_pct`, default ±2%), not a measured spread.
4. **Artifacts:** every number cites a `run_id`, its `metrics.jsonl` path, and (if wandb)
   the run URL. **No artifact → it didn't happen.**
5. **Asserts stay on.** A speedup that trips an `isfinite`/shape assert is a failure, not
   a finding.

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

# 3. Compare — tiered quality-bundle diff vs the locked baseline. Add --at_step N to judge
#    a partial run against the baseline at the same step (don't wait for full training).
python -m param_decomp_lab.speedup.compare_runs <baseline_run_id> <variant_run_id> [--at_step N] --out cmp.md
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
  `--qos=opportunistic`/`scavenge` for speculative/over-quota). Poll your own jobs with a
  sane cadence (`squeue --me`); never cancel others'. After a crash, verify `squeue --me`
  for zombie GPU holders.
- `/code-review` the diff before merging.

**Killing is a first-class outcome.** "Tried X, here's the run, it regressed Y" with
artifacts goes in the ledger so nobody re-runs it.

### Promotion criteria

- **→ T0:** smoke passes, unit + equivalence tests green, no NaNs.
- **T0 → T1:** primary tier within band (±`tol_pct`) at a matched step + a real measured
  speedup on `ss-2L` (single seed, partial run OK).
- **T1 → merged:** primary tier holds on `pile_llama_simple_mlp-4L` (compared at a matched
  intermediate step — no need to run to convergence), secondary not meaningfully worse,
  faithfulness guardrail intact, measured speedup. Escalate to ≥3 seeds / a longer run only
  if a Δ is near the band edge. Flip the change to default.
