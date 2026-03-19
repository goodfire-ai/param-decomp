# Rich Examples Variant Plan

Baseline experiment uses Jose subset:
- subset file: `component_subsets/jose_coherent_100_seed0.txt`
- harvest subrun: `h-20260318_223737`

## Branch Structure

These branch names are intended for SLURM `snapshot_branch` runs.

Important:
- `snapshot_branch` uses the committed tip of the named branch.
- Local uncommitted changes in the current worktree will not be visible to jobs.
- So these branches are placeholders until each variant's code is committed on that branch.

## Proposed Fan-Out

### 1. Baseline

- branch: `exp/rich-base-jose100`
- purpose: current `rich_examples` baseline after no-regrets cleanup

### 2. High-Confidence Union

- branch: `exp/rich-hiconf-union-jose100`
- purpose: combine the next batch of high-confidence prompt changes in one variant
- candidate changes:
  - slightly tighter skepticism / mixed-evidence wording
  - modest cleanup of punctuation-heavy highlighted examples
  - any small prompt clarifications that are clearly beneficial and low-risk

### 3. Delimiter Sweep

- branch: `exp/rich-delim-brackets-jose100`
- purpose: current `[[[token]]] (ci, act)` style

- branch: `exp/rich-delim-angle-jose100`
- purpose: legacy angle-bracket-style delimiter for direct comparison

- branch: `exp/rich-delim-inline-jose100`
- purpose: lighter inline annotation style without heavy wrappers, e.g. `token{ci,act}`

- branch: `exp/rich-delim-prefix-jose100`
- purpose: prefix marker style, e.g. `^token (ci, act)` or similar

## Recommended Execution Order

1. Run `exp/rich-base-jose100`.
2. Run delimiter sweep variants in parallel.
3. Run `exp/rich-hiconf-union-jose100`.
4. Inspect compare view and evals before fanning further.

## Launch Record

Launch date:
- `2026-03-19`

Shared launch config:
- config file: `scratch/autointerp_rich_jose100_qwen_base.yaml`
- decomposition id: `s-55ea3f9b`
- harvest subrun: `h-20260318_223737`
- subset file: `component_subsets/jose_coherent_100_seed0.txt`
- model: `qwen/qwen3-235b-a22b-2507`

Submitted runs:

- branch: `exp/rich-base-jose100`
  - interpret: `358322`
  - detection: `358328`
  - fuzzing: `358333`

- branch: `exp/rich-delim-angle-jose100`
  - interpret: `358320`
  - detection: `358324`
  - fuzzing: `358332`

- branch: `exp/rich-delim-inline-jose100`
  - interpret: `358323`
  - detection: `358326`
  - fuzzing: `358330`

- branch: `exp/rich-delim-prefix-jose100`
  - interpret: `358321`
  - detection: `358325`
  - fuzzing: `358329`

- branch: `exp/rich-hiconf-union-jose100`
  - interpret: `358327`
  - detection: `358331`
  - fuzzing: `358334`

Logs:
- SLURM logs live under `/mnt/polished-lake/artifacts/mechanisms/spd/slurm_logs/`
