# Rust Acceleration Scoring Seam

This fork keeps Goodfire's PyTorch decomposition/training and post-processing in
Python, then accelerates the repeated bootstrap/component scoring loop through an
optional Rust extension.

## Backend Shape

The v1 backend scores rank-one linear ablations from exported arrays:

- `inputs`: activation rows entering the decomposed linear map.
- `labels`: target class IDs for the sampled rows.
- `reference_logits`: unablated logits for those rows.
- `components_u`, `components_v`: rank-one factors where each ablated component
  subtracts `(inputs @ v)[:, None] * u[None, :]` from `reference_logits`.
- Optional CSR slices: `slice_names`, `slice_offsets`, `slice_indices`.

Python calls `param_decomp.accel.score_rank_one_linear_components(...)` with
`backend="python"`, `"rust"`, or `"auto"`. The default is `"python"` so upstream
behavior stays conservative. `"rust"` opts into the extension and fails if it is
missing. `"auto"` uses the Rust extension when installed and falls back to Python.

## Build

Install the Rust extension from the repo root:

```powershell
python -m pip install maturin
python -m maturin develop --manifest-path crates/param_decomp_accel/Cargo.toml
```

Set `PARAM_DECOMP_ACCEL=rust` to require the Rust backend. Leave it unset, or set
`PARAM_DECOMP_ACCEL=python`, to use the Python fallback.

## Benchmark

Run a synthetic scorer benchmark from the repo root:

```powershell
python scripts/benchmark_accel_scoring.py --backend python --rows 512 --components 256
python scripts/benchmark_accel_scoring.py --backend rust --rows 512 --components 256 --threads 8
```

The benchmark isolates the scorer seam and reports `component_scores_per_sec`.
It does not include model loading, training, JSONL writes, or Goodfire harvest
post-processing.

## Next Integration Step

Wire supported Goodfire/Storyworld bootstrap scoring jobs to export this compact
bundle per target linear map, stream Rust-returned chunks back into the existing
JSONL event/artifact files, and preserve completed-pair resume semantics in
Python.
