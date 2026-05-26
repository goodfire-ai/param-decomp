# `param_decomp_lab/postprocess/`

Unified SLURM submission for the full post-decomposition pipeline. One YAML, one
`pd-postprocess <config.yaml>`, all stages submitted with proper job dependencies.

## What it orchestrates

```
harvest                 (GPU array → merge, PD-only)
├── intruder eval       (CPU, label-free, depends on harvest merge)
├── attributions        (GPU array → merge, PD-only, depends on harvest merge)
│   └── graph interp    (CPU, LLM calls, depends on harvest merge + attribution merge)
└── autointerp          (CPU, LLM calls, resumes via completed keys)
    ├── detection       (label-dependent)
    └── fuzzing         (label-dependent)
```

Each stage's actual logic lives in its own module — postprocess just builds the SLURM
dependency chain. Per-stage detail:

- [`../harvest/CLAUDE.md`](../harvest/CLAUDE.md)
- [`../autointerp/CLAUDE.md`](../autointerp/CLAUDE.md)
- [`../dataset_attributions/CLAUDE.md`](../dataset_attributions/CLAUDE.md)
- [`../graph_interp/CLAUDE.md`](../graph_interp/CLAUDE.md)
- Intruder eval lives inside `harvest/` (see `param_decomp_lab/harvest/intruder.py`)
  because it tests *decomposition* quality, not label quality. Scores go in
  `harvest.db`, not `interp.db`.

## Files

| File | Purpose |
|---|---|
| `cli.py` | `pd-postprocess` entry point. Thin argparse wrapper for fast `--help`; heavy imports deferred. |
| `config.py` | `PostprocessConfig` — composes the per-stage SLURM configs |
| `__init__.py` | `postprocess()` — does the actual submission + dependency wiring |

## Config shape

```yaml
harvest:       { ... HarvestSlurmConfig ... }       # required
autointerp:    { ... AutointerpSlurmConfig ... }    # optional — null skips
intruder:      { ... IntruderSlurmConfig ... }      # optional — null skips
attributions:  { ... AttributionsSlurmConfig ... }  # optional — null skips
graph_interp:  { ... GraphInterpSlurmConfig ... }   # optional — null skips
```

Cross-field invariants (validated in `PostprocessConfig.model_post_init`):

- `attributions` requires `harvest.config.method_config` to be a `ParamDecompHarvestConfig`
  (attributions are PD-specific).
- `graph_interp` requires `attributions` (graph interp consumes attribution data).

## Usage

```bash
# Submit everything
pd-postprocess config.yaml

# Chain off a training job
pd-postprocess config.yaml --dependency 311644_1

# See the resolved config without submitting
pd-postprocess config.yaml --dry_run
```

`--dependency` accepts a SLURM job ID string (note: argparse, not Fire — Fire would
parse `"311644_1"` as the integer `3116441` since `_` is a Python numeric separator).

## How `postprocess()` works

1. Snapshot the current git state to a `postprocess-<hex>` branch for reproducibility.
2. Submit `harvest` (with optional upstream `--dependency`).
3. For each enabled downstream stage, submit with the right `dependency_job_id`(s)
   from the harvest / attribution merge jobs.
4. Write `metadata.yaml` recording the resolved config, snapshot ref, and every
   submitted SLURM job ID.

Output: `PARAM_DECOMP_OUT_DIR/postprocess/pp-<timestamp>/metadata.yaml`.

## Running stages individually

Each stage also has its own `pd-*` CLI (see the root `CLAUDE.md` CLI table) for
running it in isolation when you don't want the whole pipeline. `pd-postprocess` is
just the convenience wrapper for "do all of it, in order, with dependencies."

## Adding a new stage

1. Build the stage's own `submit_<stage>` function in the stage's `scripts/run_slurm.py`.
   It should accept `snapshot_ref`, `dependency_job_id(s)`, and the stage's slurm config,
   and return a `SubmitResult`-like object with `.job_id`.
2. Add the slurm-config field to `PostprocessConfig` (typed `<Stage>SlurmConfig | None`).
3. Add a numbered block in `postprocess()` that submits when the field is non-null,
   wiring the right upstream `dependency_job_id`(s).
4. Record the resulting job ID into the `jobs` dict written into `metadata.yaml`.
5. If the new stage depends on existing stages in a non-obvious way, add the invariant
   check in `PostprocessConfig.model_post_init` (e.g. the existing `graph_interp
   requires attributions` rule).
