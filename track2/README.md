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
2. **Quality bundle (guardrails), as a vector** — all of `QUALITY_BUNDLE`
   (`quality_bundle.py`): CE-diff / CE-unrecovered / KL (CI-masked), L0-total, stochastic
   hidden-acts recon, PGD recon. A change reports **all** of them vs the baseline so it
   can't hide a regression in one by improving another.
3. **Statistics — single-seed initially.** Iterate (and even confirm on T1) at one seed to
   move fast. Escalate to ≥3 seeds only before *claiming/merging* a result, or when a Δ
   sits near the band edge. Because the baselines are single-seed (the T1 baseline is one
   run we can't repeat), **"within band" is a fixed relative tolerance**
   (`--tol_pct`, default ±2%), not a measured seed spread. Short runs are smoke + speed only.
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

# 3. Compare — quality-bundle diff vs the locked baseline.
python -m param_decomp_lab.speedup.compare_runs <baseline_run_id> <variant_run_id> --out cmp.md
```

Profiler instrumentation note: the `pd/*` `record_function` labels live on the Track-1
profiling branches, not `main`. The benchmark therefore reports a plain torch.profiler
op breakdown (aten-op level). Add finer labels inside an experiment if a change needs them.

---

## Workflow

`proposed → T0 (iterate) → T1 (confirm) → merged / killed / parked`. No human gate before
T1 — agents promote themselves once T0 looks good. Within the **≤8-GPU** project cap.

- **One experiment = one worktree = one branch = one spec.** Branch
  `feature/spd-<short-idea>`. In a worktree, `uv sync` first; never `cd` back to main.
- **The ledger lives in lore** (`pd-lore`, the team's PD knowledge base), not in this repo
  — so parallel agents on separate worktrees/nodes share one coherent, append-only ledger
  instead of colliding on in-repo files. See [`ledger/README.md`](ledger/README.md) for
  setup + the workflow. Per experiment: `kb_append` a spec doc using the
  [`ledger/TEMPLATE.md`](ledger/TEMPLATE.md) body, then `kb_link` it to related docs
  (`builds-on`, `supersedes`, `killed-by`, `contradicts`). Update the same doc as the
  stage changes; record kills as first-class docs so nobody re-runs a dead end.
- **The ruler stays in-repo** (this README, `quality_bundle.py`, `ledger/baselines.md`) —
  version-locked and snapshotted with the code under test.
- Long runs are **background SLURM jobs** (default partition; no custom CPU/mem;
  `--qos=opportunistic`/`scavenge` for speculative/over-quota). Poll your own jobs with a
  sane cadence (`squeue --me`); never cancel others'. After a crash, verify `squeue --me`
  for zombie GPU holders.
- `/code-review` the diff before merging.

**Killing is a first-class outcome.** "Tried X, here's the run, it regressed Y" with
artifacts goes in the ledger so nobody re-runs it.

### Promotion criteria

- **→ T0:** smoke passes, unit + equivalence tests green, no NaNs.
- **T0 → T1:** quality bundle within band (±`tol_pct`) + a real measured speedup on `ss-2L`
  (single seed OK).
- **T1 → merged:** holds on `pile_llama_simple_mlp-4L`, full bundle within band, measured
  speedup. Single seed is enough to propose; escalate to ≥3 seeds before the final claim if
  any Δ is near the band edge. Flip the change to default.
