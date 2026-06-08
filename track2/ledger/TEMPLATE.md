# <id> — <one-line idea>

- **Claim type:** speedup | simplification
- **Stage:** proposed → running → confirmed → merged/killed/parked
- **Branch / worktree:** `feature/spd-<short-idea>` (off `feature/track2-t1`)
- **Owner agent:** <session/agent id>

## Hypothesis
What change to `param_decomp/`, and why it should be faster/simpler for similar quality.

## Diff vs baseline
Config and/or core-code changes relative to the locked baseline (see
[`baselines.md`](baselines.md)). Link the PR/branch.

## Success / kill thresholds (quoted from track2/README.md)
- ≥5% speedup (`pd-speedup-bench`, batch 16, vs the baseline config) — state the ms/step you beat.
- Quality bundle **within band** (±`tol_pct`, single-seed) on every `QUALITY_BUNDLE` metric, at
  **both 20k and 50k**; faithfulness gate is hard. A primary WIN is a 400k-confirm candidate.

## Results (artifacts required — no artifact, didn't happen)

- **benchmark:** step time / tokens-sec / peak mem (vs baseline) — `bench.md`
- **quality @20k / @50k:** paste the `compare_runs` tables — overall verdict, within-band? regressed?
- **run:** `run_id` + `metrics.jsonl` path + wandb URL

## Verdict
merged / killed / parked — and the one-line reason. If killed, what regressed.
