# Shelved: bio-model demo + `ExperimentBundle` stress-test

This branch is **shelved as of 2026-05-27**, alongside its parent
`feature/resumption-generic-runner`. Both wait for a future need that
justifies reviving them.

## What's on the branch

Stacked on top of `feature/resumption-generic-runner` (PR #523, also
shelved):

1. `fd73a251` — **ESM2 scaffold** (`EsmForMaskedLM`, protein masked LM).
   Cherry-picked from `worktree-agent-af10582dea37def1d`.
2. `308e4c37` — **Carbon-500M scaffold** (Llama-style DNA model).
   Cherry-picked from `worktree-agent-ad506d48ece81ef36`.
3. `1fdccb35` — **GPN-MSA scaffold** (multi-species sequence model with
   dict-input `forward`). Cherry-picked from
   `worktree-agent-a069844e73fc73bbe`.
4. `b930d1ad` — Migrate ESM2 / Carbon / GPN-MSA `main` to use the
   `ExperimentBundle` from the parent branch. Per-experiment savings:
   `main` shrinks from ~40 lines to a 3-line `run_fresh(_BUNDLE, ...)`
   call plus a ~17-line bundle literal.
5. `3ef96673` — Drop `refine_cfg` and `uses_distributed` from the
   bundle (both confirmed dead-weight at 4 consumers).

## Findings from the demo

- The bundle scales cleanly to 4 consumers. Each bio experiment fit
  without bundle-side changes.
- GPN-MSA's structured-input `forward` (`model(**batch).logits`) slotted
  in via the `make_run_batch` callable; no new bundle field was needed.
- 4-of-4 bundles wrote identical `lambda cfg: build_target(cfg.target)`
  shapes. The smell is real but addressable (see the parent branch's
  `BRANCH_NOTE.md` for fix options).
- 418 tests pass with the migration + cleanup applied.

## Why shelved

The bundle is shelved (see `feature/resumption-generic-runner`'s
`BRANCH_NOTE.md`); this branch demos the scaling story but has no
runtime value without the bundle.

The bio scaffolds themselves carry useful per-experiment TODOs that
might be revived independently:
- **Carbon**: its FNS recon loss (factorised nucleotide supervision)
  doesn't fit the current `ReconstructionLoss` Protocol `(pred, target)
  -> (sum, n)`. Scaffold uses vanilla KL as a placeholder.
- **ESM2**: synthetic loader works; the `uniref50` data path is a typed
  stub because `create_lm_data_loader` packs tokens cross-document,
  which is wrong for protein sequences.
- **GPN-MSA**: needs real cross-species MSA data extraction
  (`songlab-cal/gpn` repo) for non-synthetic runs.

## Reviving

Original scaffold commits live on local worktree-agent branches (listed
above) — you can cherry-pick those independently of the bundle work.

The bundle migration + cleanup commits (`b930d1ad`, `3ef96673`) only
make sense alongside `feature/resumption-generic-runner`. If the bundle
is revived, these can be cherry-picked back.
