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

**@20k is read from the live baseline `p-20f9fc15`** (step 20000). **@50k and @400k are still the
predecessor `p-5b17949e`'s (†), pending refresh** once `p-20f9fc15` reaches them (`no_beta` @50k/@400k
pending). ⚠ `p-20f9fc15`'s @20k PPGD (1.477) is **~2× below** the predecessor's (3.094) — the early
trajectory does **not** match the predecessor within noise, so don't trust the predecessor's @50k as a
proxy.

| metric (tier) | @20k | @50k | @400k | early-step usefulness |
|---|---|---|---|---|
| PGD recon — PPGD (primary) | 1.477 | 0.965† | 0.660† | **monotone↓** — same-step rank is meaningful |
| Importance-minimality `no_beta` (primary) | 275.3 | _pending_ | _pending_ | pure L_p sparsity term, the penalty the optimizer drives down; characterize from `p-20f9fc15` |
| Stochastic hidden-acts recon (secondary) | 0.531 | 0.447† | 0.402† | monotone↓ |
| CI-masked hidden-acts recon (secondary) | 0.903 | 0.807† | 0.825† | ~flat |
| L0 total (secondary, informational) | 1880 | 2465† | 201† | **non-monotone** (rises then collapses) — *do not* chase; informational only |
| CE diff CI-masked (gate) | 0.304 | 0.280† | 0.273† | **settled early** — gate is meaningful |
| KL CI-masked (gate) | 0.641 | 0.456† | 0.335† | still drifting↓ — compare same-step only |
| CE unrecovered CI-masked (gate) | 0.00451 | 0.00416† | 0.00408† | **settled early** — gate is meaningful |

† predecessor `p-5b17949e` value — pending refresh from `p-20f9fc15` (not yet at this step).

**Upshot:** at 20k/50k the **faithfulness gate is trustworthy** and **PGD recon ranks variants** (lower
at the same step = good sign). Sparsity is now read off **`no_beta`** (the beta-independent
importance-minimality term) rather than L0 — L0 is non-monotone early and demoted to informational. A
primary *win* (PPGD or `no_beta` improved) still has to be confirmed at 400k (see §Rules, "early ≠
converged").

## The baseline (the only one)

- **Run:** `p-20f9fc15` — `pile_llama_simple_mlp-4L`, **400k steps, batch 64, dp 16**, re-run on
  current code so it logs the `no_beta` importance-minimality term (the predecessor `p-5b17949e`
  predates that metric). W&B: https://wandb.ai/goodfire/param-decomp/runs/p-20f9fc15 (project
  `param-decomp`). Same config as `p-5b17949e`, so its trajectory should match within noise.
- **Config:** `param_decomp_lab/experiments/lm/pile_llama_simple_mlp-4L.yaml` (eval block is the
  ruler). `p-5b17949e`'s 400k metrics matched the pre-refactor predecessor `s-55ea3f9b` (PGD recon
  0.660 vs 0.652, L0 201 vs 201, CE diff 0.273 vs 0.286), validating the config.
- **Single baseline → band = ±`tol_pct`** (default ±2%); there's no seed spread, so treat deltas
  near the floor as noise.
- **Self-cached.** `pd-lm` writes `runs/p-20f9fc15/metrics.jsonl` + `experiment_config.yaml` live
  during training (no W&B distillation needed), so `pd-speedup-compare` resolves it directly.
- **Compare points: step 20000 and step 50000.** Both are multiples of `slow_every` (10000), so
  every bundle metric (incl. the slow PGD-recon attack) is present at both.

## What counts as a win

A change to `param_decomp/` (or the config) that is **both**:

1. **≥5% faster** — wall-clock per step, from `pd-speedup-bench` on a pinned GPU at the baseline's
   batch/seq, eval excluded. (Measurable in minutes — no need to wait for 20k.) **Single-GPU memory
   caveat:** the 4L config at batch 64 doesn't bench single-GPU — it pegs an 80 GB H100 (~81 GB,
   the real run spreads it over dp 16 = per-GPU batch 4) and times out. **Bench at batch 16**
   (measured: 766 ms/step, 57.4 GB) via a batch-16 copy of the config — **identical for baseline
   and variant** so the per-step ratio is fair; the quality run still uses real batch 64 / dp 16.
2. **Quality same-or-better** vs the baseline **at both 20k and 50k** — `pd-speedup-compare` prints
   the verdict per tier:
   - **Gate — faithfulness** (CE-diff / KL / CE-unrecovered): must stay **within band** at the same
     step. A regression here **fails the result regardless of speed**.
   - **Primary** — **PPGD recon** (eval `PGDReconLoss`) + **`no_beta`** (eval
     `ImportanceMinimalityLoss/no_beta`, the beta-independent sparsity term): both within-band-or-
     better. A *primary win* (PPGD or `no_beta` improved, neither regressed) is a **candidate**,
     flagged for a 400k confirmation, not a final win on its own.
   - **Secondary** — stochastic / CI-mask recon + **L0** (informational, non-monotone early): not
     meaningfully worse.

Below 5%, or any gate regression at 20k or 50k → not a win (record a kill).

## Measure it (two commands)

```bash
source .venv/bin/activate   # pd-speedup-* exist after `make install-lab`

# Speed: variant config vs baseline config (same settings), pinned GPU, eval excluded.
python -m param_decomp_lab.speedup.benchmark <variant-config>.yaml --out bench.md

# Quality: variant run vs baseline, at BOTH early checkpoints.
# Fails fast if the baseline hasn't reached the step; the variant reads INCOMPLETE until it has.
python -m param_decomp_lab.speedup.compare_runs p-20f9fc15 <variant-run-id> --at_step 20000
python -m param_decomp_lab.speedup.compare_runs p-20f9fc15 <variant-run-id> --at_step 50000
```

Runs live at `$DATA_MOUNT/artifacts/mechanisms/param-decomp/runs/<run_id>/` (`DATA_MOUNT` resolves to
`/mnt/polished-lake`, `/mnt/data`, … per cluster — never hard-code a mount). `pd-speedup-compare`
resolves run-ids through this for you; only reach for the absolute path when tailing `metrics.jsonl`
by hand.

## Run one experiment

1. **Change** one thing. Variants must reproduce `p-20f9fc15` except the change — derive the config
   from `pile_llama_simple_mlp-4L.yaml` (one-line diff). Work on a branch off `feature/track2-t1`;
   for a core edit, a worktree keeps it isolated:
   `git worktree add <path> -b feature/spd-<idea> feature/track2-t1` then `uv sync --all-packages`.
2. **Smoke:** `…/pile_llama_simple_mlp-4L-smoke.yaml` (variant change applied) — finishes in minutes,
   must run clean (no NaNs). For an approximation, add an equivalence test.
3. **Bench** the variant vs the baseline config → confirm **≥5%**. (Fast — do this before any 50k.)
4. **Launch the run:** `pd-lm <variant-config>.yaml --dp 16 --job_name ai-pd-lm` (background SLURM,
   default partition). **Keep `pd.steps: 400000`** and the **fixed shape** (`--dp 16`, train
   `batch_size: 64`, eval `batch_size: 128`) — the lr schedule, faithfulness warmup, β-anneal and
   pnorm-anneal are all parameterized over total steps, so the variant must share the baseline's full
   schedule or the same-step compare is invalid. We screen early by *reading* the run at 20k/50k and
   then cancelling it — **never** by shortening the config. One run yields both the 20k and 50k eval
   points (eval/slow every 1k/10k). ~1.1 h to 20k, ~2.75 h to 50k on 16 GPUs. **Capture the printed
   `Run ID` + `Job ID` into `track2/ledger/inflight.json`** (see §Running this as a loop) so the run is
   reapable by a later iteration.
5. **Compare** at **20k first** (the fast read, ~1.1 h) then **50k** (confirm). Read the verdict —
   **faithfulness gate first**, then PPGD + `no_beta`; L0 informational. After the 50k read, **cancel
   the SLURM job** (a 400k confirm is a separate, deliberate decision — don't let it run on).
6. **Record** one row in `track2/ledger/README.md` + a card in `track2/ledger/experiments/<id>.md`:
   speedup, both-step verdicts, run_id + W&B URL. Mark a PPGD-improving result as a **400k-confirm
   candidate**, not a closed win.

## Running this as a loop

Autonomous, no check-in between iterations. GPUs are abundant — **don't serialize on one in-flight
run**; reap finished runs and launch a new idea each iteration. Every iteration does both, then stops:

**State lives on disk, not in memory.** A fresh loop iteration can't remember what a prior one
launched (and the conversation may have been summarized), so the source of truth for in-flight runs is
`track2/ledger/inflight.json` — a list of records, one per submitted run:

```json
[{"run_id": "p-1a2b3c4d", "job_id": "644454", "idea": "spd-001 n_warmup 2→0",
  "config": "pile_llama_simple_mlp-4L-nwarmup0.yaml", "submitted": "2026-06-08T14:02:00Z",
  "stage": "running", "reaped_20k": false, "reaped_50k": false}]
```

`pd-lm`'s submit output prints both **`Run ID`** and **`Job ID`** — capture both (or pass an explicit
`--run_id` you generate so you never scrape stdout) and append the record *before* moving on. Without
this map you can't tell which `ai-pd-lm-*` job in `squeue` is which idea, nor which one to `scancel`
after its 50k read. Update the record's `stage`/`reaped_*` as you go; move finished ones into the
ledger card and drop them from `inflight.json`.

1. **Reap** — for each record in `inflight.json` with `stage: running`: check progress (read its
   `metrics.jsonl`, or `squeue`). If it's **past 20k**, run the §"Measure it" compare at 20k (and at
   50k once past 50k), read the verdict (**gate first**), record it. A run that's failed the gate at
   20k can be **cancelled early** (`scancel <job_id>` — only ever its own `ai-pd-lm-*` job) and
   recorded as a kill. Cancel a passing run once its 50k read is done (don't run on to 400k). Leave
   still-pre-20k runs in place for the next iteration. The compare **fails fast if the baseline hasn't
   reached the step** and reads **INCOMPLETE** if the variant hasn't — both mean "not ready, leave it",
   not a verdict.
2. **Launch one** — pick the next idea (see §Where to look for speed), **smoke**, then **bench** vs
   the baseline config. `<5%` faster or anything fails (NaNs, tripped asserts) → record a **kill**.
   `≥5%` → submit the run (full `pd.steps: 400000` config — screened early, not shortened; **fixed
   shape: `--dp 16`, train `batch_size: 64`, eval `batch_size: 128`**;
   `pd-lm <variant-config>.yaml --dp 16 --job_name ai-pd-lm`, background SLURM, default partition),
   then **append its `{run_id, job_id, idea, …}` record to `inflight.json`** for the next iteration to
   reap.

- **Autonomy:** auto-submit the run when the 5% bench gate passes — don't pause for approval.
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
- **Fixed run shape** — every quality run is **dp 16, train `batch_size: 64`, eval `batch_size: 128`**.
  These are invariants of the comparison, not levers: a variant changes the *method* (`param_decomp/`)
  or non-batch config fields only, never the parallelism or the batch sizes (changing them makes the
  same-step compare apples-to-oranges and shifts memory/throughput off the baseline). Speedups that
  only manifest at this scale (comms, multi-node) won't show in the single-GPU bench — note them as
  bench-blind and lean on the run.
- **Compare same-step** — only ever 20k-vs-20k and 50k-vs-50k (KL and L0 are still moving early;
  cross-step diffs are meaningless). `pd-speedup-compare` enforces this: it drops any metric whose
  nearest logged row is >`--max_step_distance` (default 2000) from `--at_step`, and **fails fast if
  the baseline hasn't itself reached the step** — so you can never silently rank a variant against an
  earlier baseline checkpoint. A variant that isn't there yet reads `INCOMPLETE`, not a verdict.
- **Artifacts or it didn't happen** — every number cites a run_id + W&B URL.
- **Asserts stay on** — a speedup that trips `isfinite`/shape checks is a fail, not a finding.
- **Early ≠ converged** — 20k/50k *screen and rank*; a primary win is a **400k-confirm candidate**,
  not a closed result (L0 especially is non-monotonic — informational only; see §Why).

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

- **Baseline `p-20f9fc15` (job 644450, 16 GPUs) is live and past 20k** (~step 25k as of this writing),
  self-caching `metrics.jsonl` and logging `no_beta`. The real **@20k bar from the run itself**:
  PPGD `1.4774`, `no_beta` `275.33`, stochastic-recon `0.53068`, CI-mask-recon `0.90267`, L0 `1879.5`,
  CE-diff `0.30395`, KL `0.64088`, CE-unrecovered `0.0045113`. **@50k is not reached yet** (~2.75 h
  mark) — pending. **⚠ The predecessor `p-5b17949e` bar in `baselines.md` does NOT hold at 20k:**
  predecessor PPGD@20k was `3.0936` vs the live `1.4774` (~2×), so the "trajectory matches within
  noise" assumption is false early. Refresh `baselines.md` + the trajectory table's @20k column from
  `p-20f9fc15`, and treat the predecessor's @50k/@400k numbers as provisional until `p-20f9fc15`
  reaches them.
- **Sparsity metric switched** from L0 (non-monotone, now informational/secondary) to
  `ImportanceMinimalityLoss/no_beta` (primary) in `quality_bundle.py`.
- **`pd-speedup-compare` now guards same-step comparisons** (`--max_step_distance`, default 2000):
  metrics whose nearest row is too far from `--at_step` drop out, the baseline must have reached the
  step (else fail-fast), and a not-there-yet variant reads `INCOMPLETE`.
- Baseline bench (1×H100, **b16**/s512, eval excluded): **766 ms/step**, peak mem 57.4 GB
  (`pd-speedup-bench` on a batch-16 copy of `pile_llama_simple_mlp-4L.yaml`). Batch 64 does **not**
  bench single-GPU — it pegs the 80 GB H100 (~81 GB) and times out; use batch 16 for the bench
  ratio (same for baseline + variant), batch 64 / dp 16 for the quality run.
- **In flight (tracked in `track2/ledger/inflight.json`):** the PPGD-warmup idea, split two ways —
  `p-3105a340` (job 644454, `n_warmup_steps` 2→0, **past 20k — reapable now**) and `p-ebc2de5b`
  (job 644568, 2→1, pre-20k). Both were launched by an earlier loop iteration with no ledger card;
  backfilled into `inflight.json` from `squeue`. Next reap should compare `p-3105a340` @20k and write
  its ledger card.
