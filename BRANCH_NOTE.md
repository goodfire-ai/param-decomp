# Shelved: `ExperimentBundle` + generic runner extraction

This branch is **shelved as of 2026-05-27** — included here so a future
maintainer can pick it back up.

## What's on the branch

One commit on top of `feature/resumption` (PR #520):

- `542960fc` — lifts the per-experiment orchestration out of
  `param_decomp_lab/experiments/lm/run.py` into a generic
  `ExperimentBundle[CfgT]` + `run_fresh` / `run_resumed` drivers in
  `param_decomp_lab/experiments/runner.py`. LM's `_fresh_main` /
  `_resume_main` (~120 lines) collapse to a 17-line bundle literal and
  3-line dispatch.

## Why shelved

PR #520's reviewer (Dan) flagged that resumption lived inside the
lm-specific `run.py`. We tried the extraction (this branch + a follow-up
`feature/resumption-bio-demo` that demoed migrating ESM2 / Carbon /
GPN-MSA scaffolds onto the bundle). The migration worked cleanly across
4 consumers, but with only LM actually needing resume today, we decided
to keep things concrete: live with lm-specificity until a *second*
experiment genuinely needs resume support.

Dan's review explicitly said *"I'm OK with this for now, until we have
another concrete experiment type that we care about, at which point we
can worry about consolidating this stuff."* This matches that framing.

## Open design questions when reviving

1. **The lambda-adapter smell.** 4-of-4 bundles wrote identical lambda
   shapes because the experiment-side build functions take sub-configs
   (`cfg.target`, `cfg.data`) but the bundle wants the full `cfg`. Two
   fixes on the table:
   - (a) Change the experiment-side build function signatures to take
     the full `cfg`. Then `build_target=build_target` works as a literal.
     ~15 LOC across N experiments.
   - (b) Switch the bundle to an ABC so methods take `self + cfg` and
     the build logic can be inlined into methods (no wrapper around a
     free function). ~200 LOC but unlocks inheritance (e.g.
     `HFCausalLMExperiment` base for LM + Carbon) and naturally absorbs
     the `Saved<X>Run` boilerplate.

2. **Speculative fields to drop before shipping.** The bio-demo branch
   already has these removals applied + verified under 4 consumers (418
   tests pass):
   - `refine_cfg` — added speculatively for TMS's `tied_weights`
     post-build hook. TMS isn't migrating; drop it.
   - `uses_distributed: bool` — redundant.
     `init_distributed()` is already a no-op when `WORLD_SIZE` is unset.

3. **`Saved<X>Run` duplication.** Every experiment has ~25 lines of
   identical `from_path` + `load_model` boilerplate. Could become a
   single generic `SavedRun[CfgT]` driven by the bundle. ~125 LOC of
   dup to recover (5 experiments).

## Reviving

```bash
git fetch origin feature/resumption-generic-runner
git switch feature/resumption-generic-runner
git rebase origin/main   # or the current resumption branch HEAD
# Apply the cleanups from feature/resumption-bio-demo (commit 3ef96673).
# Decide (a) signature fix vs (b) ABC for the lambda smell.
```

See also: `feature/resumption-bio-demo` for the 3-bio-experiment migration
that stress-tested the bundle abstraction.
