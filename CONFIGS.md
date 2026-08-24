# Config policy

The repo maintains a small, named set of experiment configs — the **canonical
seats** — and nothing else. Every yaml the repo carries is a maintenance
obligation: each schema change must migrate it, forever. Seats are capped at
**10 LM configs**, each with a stated purpose.

## Why committing sweep configs adds nothing

A launched run's config provenance already lives in three places, none of them
the repo tree:

1. the run dir's pinned `launch_config.yaml` (immutable; resume byte-compares it),
2. the git snapshot ref `refs/runs/snapshot/<id>` taken at submit,
3. the wandb run config.

So a sweep/profile/one-off yaml committed "for the record" records nothing —
it only rots. Launch one-offs from your workspace (`python -m param_decomp.experiments.lm.run <path>`
takes any path); if the sweep matters, record its run IDs and findings externally. The
run directories carry the exact configs.

## The canonical seats

| seat | file | purpose |
|---|---|---|
| llama8b L18 | `param_decomp/experiments/lm/configs/llama8b_l18_C49k_200k.yaml` | the L18-MLP decomposition flagship recipe |
| llama8b full-model | `param_decomp/experiments/lm/configs/llama8b_full32L_HSDP_b64_dp64.yaml` | the full-32L production TP1 recipe; TP variants are derived from the performance-benchmark seat rather than committed as additional configs |
| save-path smoke | `param_decomp/experiments/lm/configs/llama8b_full32L_HSDP_b32_dp32_SAVESMOKE.yaml` | cheap end-to-end save/resume smoke launch |
| config-suite fixture | `param_decomp/experiments/lm/configs/llama8b_l18_b128_cmp32.yaml` | the representative full config the core config/resume tests load (`test_config.py`, `test_finetune_resume.py`, `test_llama_simple_mlp.py`) |
| chunkwise fixture | `param_decomp/experiments/lm/configs/llama8b_l18-26_9layer_chunkwise.yaml` | the 27-site chunkwise CI-fn config `test_config.py` converts |
| ss 2L SimpleMLP | `param_decomp/experiments/lm/configs/ss_llama_simple_mlp-2L.yaml` | current JAX reference for the 2L SimpleStories VPD target (dp=1); reproduces [p-5926d125](https://wandb.ai/goodfire/param-decomp-ss2l-repro/runs/p-5926d125) |
| pile 4L VPD reference | `param_decomp/experiments/lm/configs/pile_llama_simple_mlp-4L.yaml` | current JAX reference for the VPD paper target; reproduces [p-76082aa1](https://wandb.ai/goodfire/param-decomp/runs/p-76082aa1) |
| full32 performance benchmark | `param_decomp/experiments/lm/configs/profile_llama8b_full32_adam.yaml` | canonical faith-on base for derivable, matched H100 communication profiles; the checked-in file is maintained, while generated topology variants remain launch-local |
| tPD L18 arithmetic | `param_decomp/experiments/lm/configs/llama8b_l18_arith_targeted.yaml` | the tPD (SPEC §11) L18 modular-addition recipe; the targeted-schema fixture `test_lm_targeted.py` loads |
| tPD full-model arithmetic | `param_decomp/experiments/lm/configs/llama8b_full32L_arith_targeted.yaml` | the full-32L tPD modular-addition recipe (every layer at low C) |

The toy testbeds (`param_decomp/experiments/tms/configs/`,
`param_decomp/experiments/resid_mlp/configs/`) and the pretrain configs
(`param_decomp/pretrain/configs/`) are separate small schemas, maintained with
their experiments; they are seats too, just not LM-schema ones.

## The pre-JAX archetypes

Three seats reached tip with a `pd.ci_config` asking for `mode: global` /
`fn_type: global_shared_transformer`, a CI function the JAX trainer never
gained (the name survives only in the torch reference,
`nano_param_decomp/run.py`) — not *unmigrated* but **not migratable as
written**. Two are now the JAX reference configs in the table above, each
rewritten onto a `chunkwise_transformer` CI function and revalidated by a
completed run: pile-4L on 2026-07-27, ss-2L on 2026-07-30. Neither reproduces
the paper's own PyTorch decomposition — its Lp importance-minimality objective
is gone from this codebase, so no config can express it.

The third, `jose.yaml` (the original gpt2-arch 4L flagship reference), was
evicted on 2026-07-23 ("Remove unused configs") along with `jose-ish.yaml`
(#917 — the deliberate rewrite of that recipe onto one `chunkwise_transformer`
chunk over all 4 blocks); git history keeps both.

## Rules

1. **Every LM config yaml in the tree parses at tip** — CI-enforced by
   `param_decomp/tests/test_repo_configs_parse.py` (schema parse + the placement
   gate). A schema PR that breaks one migrates
   it **in the same PR**, with an executed in-repo migration (the #966
   pattern) — never a script attached to a PR comment (#939 attached one; it
   never ran, and 97 of 104 stored runs became unopenable before anyone
   noticed).
2. **Sweep / one-off configs are not committed.** Launch generated variants from a
   workspace path. The single registered performance-benchmark seat is the derivation base,
   not a growing family of committed topology variants. Deleting a config is cheap (git
   history keeps it; run dirs pin what actually ran) — un-rotting one is not.
3. **Adding a canonical seat is taking on a maintenance obligation**: add the
   file, a registry row above naming its purpose, and it's covered by the CI
   gate from that commit on. The cap is **10** (the table seats 10 — AT cap:
   the next seat requires an eviction, which is the point). A config a test loads
   is a seat by definition — deleting it is a test change, so grep for the
   basename first.
4. **Stored-run pins are immutable.** Never migrate a run dir's
   `launch_config.yaml` in place (resume byte-compares it; a live old-code run
   whose pin is rewritten refuses its next requeue). Consumers reparse stored
   pins against the full canonical schema
   (`experiments/lm/config.py::load_config`; production consumers read the
   deliverable projection, `experiments/lm/deliverable.py`), so a pin from
   an older schema opens at its original revision or through an explicit
   external converter — tip does not migrate it (dataset-name case history
   below).
5. **Seats carry names, never locations.** A config references the world only
   through names — a dataset name, a run id, an HF hub id, a wandb ref — each
   with exactly one resolver, rooted in a global registry (HF hub, W&B) or the
   explicit `data_root`; names are immutable versions, so a pin means the same
   thing on every machine and every year. User-home-relative paths (`~/...`,
   e.g. `runtime.compilation_cache_dir`) are a sanctioned reference kind of the
   same shape: the name of a per-user spot with exactly one resolver
   (`expanduser`, applied once at the consuming entry). No absolute path
   appears in a committed config outside a tagged escape arm (`kind: dir`) —
   CI-enforced by the parse gate's `test_seats_carry_names_never_locations`.
   Known tracked exception: `resume_provenance.parent_run_dir` is an absolute
   path that should be a run id.

## Case history

- **#939** (2026-07-03): ScheduleConfig unification; migration script attached
  as a PR comment, never executed → 7/104 stored runs parseable at tip.
- **#966**: the counter-example — carve migration shipped as an in-repo,
  executed tool covering every live repo yaml.
- **#982**: 25 sweep yamls in one PR — the accumulation pattern this policy
  ends. The sweeps' findings live in lore; the run dirs pin their configs.
- **dataset-name schema** (2026-07-27): the whole HF-costume `data:` block →
  `data: {kind: name, name} | {kind: dir, dir}` — the block IS the dataset ref
  (2026-07-30: nested to `data: {train, eval}`, each a required dataset ref;
  `eval` is the held-out split the eval pass reads and must resolve to different shards
  than `train`). The dataset's own facts
  (`seq_len`, `tokenizer_name`) moved into its `meta.json` (`infra.dataset_store.DatasetMeta`),
  read at load; every seat re-stamped in the same PR. Older immutable pins require
  their original revision or an explicit external converter; tip does not migrate them.
- **pretrain dataset ref** (2026-07-31): the five `param_decomp/pretrain/configs/`
  seats hardcoded an absolute shard directory — right on one machine,
  wrong on every other — plus a hand-declared `tokenizer_name` nothing read. Both are
  gone: `data:` is the same `DatasetRef` the LM seats carry, moved to
  `infra.dataset_store` so the pretrainer never imports `experiments/lm/`, resolved
  against the run's stamped root. The launcher's `out_dir` stamp (`<data_root>/runs`)
  became a `data_root` stamp, the trainer having already been recovering the root as
  `out_dir.parent`. The tokenizer is the dataset's own fact, in its `meta.json`. The
  parse gate now covers the pretrain seats.
- **#23** (2026-07-16): SwiGLU / FFN-as-its-own-config migrated all 30
  `param_decomp/configs/` yamls in one commit — the same 8-line edit applied
  30 times, ~26 of them to launched one-offs nobody will open again. The tax
  this policy stops, paid once more while the policy sat in review.
