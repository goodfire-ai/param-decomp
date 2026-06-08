# Track 2 — Method Speedups (the measurement contract)

The in-repo, re-read-each-session contract for Track 2. The operating plan is
[`/plan_t1.md`](../plan_t1.md); this file is the ruler + reporting rules the agents follow.

**Goal:** change the core PD method (`param_decomp/`) so the loop runs much faster for similar
quality, validated on the **`pile_llama_simple_mlp-4L`** run, **judged early at 20k/50k steps** (a
fast proxy for the full 400k). A finished result is a faithful speedup or simplification that holds
under the rules below. No 3-pool / multipool / 8b work — that is Track 1.

---

## The model

| Config | Smoke | Baseline |
|---|---|---|
| `param_decomp_lab/experiments/lm/pile_llama_simple_mlp-4L.yaml` | `…-4L-smoke.yaml` | `p-5b17949e` (see [`ledger/baselines.md`](ledger/baselines.md)) |

Variants reproduce the baseline config except the one change under test. Iterate fast by reading
**20k** (~1.1 h) then confirming at **50k** (~2.75 h); a real WIN is confirmed by a longer (400k) run.

---

## The ruler — READ-ONLY for an experiment

An experiment may rewrite `param_decomp/` freely. It may **NOT**, as part of an experiment, edit any
of:

- the **eval harness** / eval-metric classes (`param_decomp_lab/eval_metrics/`),
- the **quality bundle** (`param_decomp_lab/speedup/quality_bundle.py`),
- the **baseline pointer** (`ledger/baselines.md` / the cached `p-5b17949e` artifacts),
- this contract.

Changing any of these is a separate, explicitly-flagged change a human reviews — it's the ruler, not
the thing being measured. (Mitigates goalpost-moving / reward-hacking.)

---

## What every result must report

1. **Speed (the thing we trade):** ≥**10% faster** is the floor for "material" (wall-clock/step).
   Report step time + tokens/sec + peak memory + a profiler op breakdown, on a **pinned GPU at fixed
   batch/seq, eval excluded** — `pd-speedup-bench`. (The 4L config OOMs at batch 64 single-GPU; bench
   at batch 16, identically for baseline and variant — see `plan_t1.md`.)
2. **Quality bundle (the objective), tiered** — all of `QUALITY_BUNDLE` (`quality_bundle.py`), via
   `pd-speedup-compare`, which prints an **overall verdict**:
   - **primary**: **PPGD recon** (`PGDReconLoss`, the eval-time PGD attack — *not* the train-time
     persistent loss) within-or-better; **L0** within band only (non-monotonic early);
   - **secondary** (informational): stochastic-mask recon + CI-mask recon;
   - **gate** — CI-masked faithfulness (CE-diff / CE-unrecovered / KL). A **hard constraint**:
     regress it past the band and the result **fails regardless of PPGD/L0** (else a "win" is just
     trading faithfulness for lower L0 — a worse decomposition).

   Decision rule: faithfulness gate first, then **primary** drives WIN/NEUTRAL. Report the whole
   vector so nothing hides.
3. **Statistics — single-seed band; early-screen.** The baseline is one seed → band = ±`tol_pct`
   (default ±2%); treat near-floor deltas as noise. Judge **at the same step**: `--at_step 20000`
   then `--at_step 50000` (both slow-eval steps). Faithfulness is settled by 50k (gate is
   meaningful); the primary metrics are not (PGD-recon still high, L0 non-monotonic). So 20k/50k is a
   **screen**; a **primary WIN must be confirmed with a longer 400k run** before merge.
4. **Eval is frozen (ruler).** The experiment runs the **same `eval:` block** as the baseline —
   metrics, the **PGD-attack strength** (`PGDReconLoss` `n_steps`/`step_size`), eval batch, cadence.
   Weakening the attack or changing eval batch is a reward-hack; `pd-speedup-compare` flags drift.
5. **Artifacts:** every number cites a `run_id`, its `metrics.jsonl` path, and (if wandb) the URL.
   **No artifact → it didn't happen.**
6. **Asserts stay on.** A speedup that trips an `isfinite`/shape assert is a failure, not a finding.

---

## The harness (base DDP `pd-lm` path)

Activate the venv first (`source .venv/bin/activate`). Console scripts (`pd-speedup-*`) need
`make install-lab`; otherwise use the `python -m` form.

```bash
# 1. Smoke — first thing after editing core. Finishes in minutes, no wandb.
python -m param_decomp_lab.experiments.lm.run \
    param_decomp_lab/experiments/lm/pile_llama_simple_mlp-4L-smoke.yaml

# 2. Benchmark — primary metric (step time, tokens/sec, peak mem, profiler breakdown).
python -m param_decomp_lab.speedup.benchmark <variant-config>.yaml --out bench.md

# 3. Compare — tiered diff + faithfulness-gated verdict, at both early checkpoints.
python -m param_decomp_lab.speedup.compare_runs p-5b17949e <variant_run_id> --at_step 20000 --out cmp20k.md
python -m param_decomp_lab.speedup.compare_runs p-5b17949e <variant_run_id> --at_step 50000 --out cmp50k.md
```

Profiler instrumentation note: the `pd/*` `record_function` labels live on the Track-1 profiling
branches, not here. The benchmark therefore reports a plain torch.profiler op breakdown (aten-op
level). Add finer labels inside an experiment if a change needs them.

---

## Workflow

`proposed → running → confirmed (≥10% + quality at 20k/50k) → merged / killed / parked`.

- **One experiment = one worktree = one branch = one spec.** `git worktree add <path> -b
  feature/spd-<idea> feature/track2-t1` — branch off `feature/track2-t1`; wins merge back into it. In
  the worktree run `uv sync --all-packages` (plain `uv sync` misses the lab CLIs); never `cd` back to
  the main repo.
- Copy [`ledger/TEMPLATE.md`](ledger/TEMPLATE.md) → `ledger/experiments/<id>.md`, fill the
  hypothesis / claim type / thresholds (quoted from this contract) / artifact links.
- Add/maintain a row in [`ledger/README.md`](ledger/README.md) — the human's whole report surface.
  Record kills as first-class rows so nobody re-runs a dead end.
- Long runs are **background SLURM jobs** (default partition; no custom CPU/mem;
  `--qos=opportunistic`/`scavenge` for speculative/over-quota). **Prefix every SLURM job name with
  `ai-`** (`pd-lm … --job_name ai-pd-lm`) so AI-launched jobs are identifiable. Poll your own jobs
  (`squeue --me`); never cancel others'. After a crash, verify `squeue --me` for zombie GPU holders.
- `/code-review` the diff before merging.

**Killing is a first-class outcome.** "Tried X, here's the run, it regressed Y" with artifacts goes
in the ledger so nobody re-runs it.

### Promotion criteria

- **→ running:** smoke passes, unit + equivalence tests green, no NaNs, **≥10% speedup** (`pd-speedup-bench`).
- **→ confirmed:** `pd-speedup-compare` overall verdict WIN or NEUTRAL vs `p-5b17949e` at **both 20k
  and 50k** (faithfulness gate held, PPGD within-or-better, L0 within band).
- **→ merged:** a primary WIN additionally holds in a **longer 400k confirmation run** (guards
  early/late crossover). Flip the change to default.
