# `param_decomp_lab/speedup/` — Track-2 measurement harness

Tooling for Track 2 (method speedups). The research contract, workflow, and ledger live in
[`track2/`](../../track2/README.md) at the repo root; this directory is just the code.

## Files

- `quality_bundle.py` — `QUALITY_BUNDLE`: the frozen vector of eval-metric keys every
  experiment reports vs baseline. **The ruler — read-only for an experiment** (see
  `track2/README.md`). Keys are the full `eval/{namespace}/{key}` strings as logged.
- `benchmark.py` (`pd-speedup-bench`) — primary metric: wall-clock/step, tokens/sec, peak
  GPU memory, torch.profiler op breakdown. Base DDP `pd-lm` path, single GPU, eval
  excluded, faithfulness warmup run once and not counted. Drives the public `Trainer.run`
  in phases by re-pointing `trainer.pd_config` to a `model_copy` with a higher `steps`.
- `compare_runs.py` (`pd-speedup-compare`) — reads two+ runs' `metrics.jsonl` and diffs the
  final value of each `QUALITY_BUNDLE` metric into a markdown table. Read-only over
  artifacts.

Console scripts need `make install-lab`; otherwise run `python -m param_decomp_lab.speedup.<mod>`.

## Gotchas

- The benchmark's measure phase includes a small (`warmup_steps`-batch) dataloader replay
  before the timed steps, because each `Trainer.run` call replays `self.step` batches. Keep
  `warmup_steps` modest; replay is data-only and small vs `measure_steps` of compute.
- No `pd/*` `record_function` labels on `main` (they're on Track-1 profiling branches), so
  the profiler table is aten-op level. Add labels inside an experiment if needed.
- `quality_bundle.py`'s `l0` key embeds the rounding threshold (`0.0_total`); if a config
  changes `CI_L0.rounding_threshold`, the key changes too.
