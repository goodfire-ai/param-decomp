# <id> — <one-line idea>

- **Claim type:** speedup | simplification
- **Stage:** proposed → T0 → T1 → merged/killed/parked
- **Branch / worktree:** `feature/spd-<short-idea>`
- **Owner agent:** <session/agent id>

## Hypothesis
What change to `param_decomp/`, and why it should be faster/simpler for similar quality.

## Diff vs baseline
Config and/or core-code changes relative to the locked baseline (see
[`baselines.md`](baselines.md)). Link the PR/branch.

## Success / kill thresholds (quoted from track2/README.md)
- Speedup target (state the number you're trying to beat, at T0 and T1).
- Quality bundle must stay **within band** = within the baseline's seed spread
  (`baselines.md`) on every `QUALITY_BUNDLE` metric.
- ≥3 seeds for any promote/kill decision.

## Results (artifacts required — no artifact, didn't happen)

### T0 (`ss_llama_simple_mlp-2L`)
- benchmark: step time / tokens-sec / peak mem (vs baseline) — `bench.md`
- quality bundle: paste `compare_runs` table — within-band? regressed?
- runs: `run_id` + `metrics.jsonl` path + wandb URL, per seed

### T1 (`pile_llama_simple_mlp-4L`)
- (same structure; **the T1 number is the headline**)

## Verdict
merged / killed / parked — and the one-line reason. If killed, what regressed.
