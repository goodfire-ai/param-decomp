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
it only rots. Launch one-offs from your workspace (`pd-lm <path>` takes any
path); if the sweep matters, its lore finding cites the run ids, and the run
dirs carry the exact configs.

## The canonical seats

| seat | file | purpose |
|---|---|---|
| llama8b L18 | `param_decomp/configs/llama8b_l18_C49k_200k.yaml` | the L18-MLP decomposition flagship recipe |
| llama8b full-model | `param_decomp/configs/llama8b_full32L_seq512_b128_dp128.yaml` | the full-32L production recipe |
| pile 4L PPGD testbed | `param_decomp/configs/pile_ppgd_bsc.yaml` | persistent-PGD (bsc) 4L-pile algorithm-research testbed |
| save-path smoke | `param_decomp/configs/llama8b_full32L_HSDP_b32_dp32_SAVESMOKE.yaml` | cheap end-to-end save/resume smoke launch |
| config-suite fixture | `param_decomp/configs/llama8b_l18_b128_cmp32.yaml` | the representative full config the core config/resume tests load (`test_config.py`, `test_finetune_resume.py`, `test_llama_simple_mlp.py`) |
| chunkwise fixture | `param_decomp/configs/llama8b_l18-26_9layer_chunkwise.yaml` | the 27-site chunkwise CI-fn config `test_config.py` converts |
| jose-ish (4L pile flagship) | `param_decomp_lab/experiments/lm/jose-ish.yaml` | the flagship `jose` recipe migrated to the JAX schema — runnable at tip (#917) |
| jose (gpt2-arch 4L) | `param_decomp_lab/experiments/lm/jose.yaml` | the original flagship reference — **unrunnable at tip**, see below |
| ss 2L SimpleMLP | `param_decomp_lab/experiments/lm/ss_llama_simple_mlp-2L.yaml` | small-LM regression archetype — **unrunnable at tip**, see below |
| pile 4L SimpleMLP | `param_decomp_lab/experiments/lm/pile_llama_simple_mlp-4L.yaml` | pretrained-target archetype — **unrunnable at tip**, see below |

The toy testbeds (`param_decomp_lab/experiments/tms/configs/`,
`param_decomp_lab/experiments/resid_mlp/configs/`) and the pretrain configs
(`pretrain/configs/`) are separate small schemas, maintained with their
experiments; they are seats too, just not LM-schema ones.

## The three unrunnable seats

`jose.yaml`, `ss_llama_simple_mlp-2L.yaml` and `pile_llama_simple_mlp-4L.yaml`
all fail to parse at tip for the same structural reason: their `pd.ci_config`
asks for `mode: global` / `fn_type: global_shared_transformer`, and the JAX
trainer has no global-shared-transformer CI function — the name survives only
in the torch reference (`nano_param_decomp/run.py`). They are not *unmigrated*;
they are **not migratable as written**. Making them parse means either
implementing a global CI fn in the JAX stack, or rewriting each onto
`chunkwise_transformer` — which changes what the config means, and is a
research call, not a mechanical migration.

`jose-ish.yaml` is what that second choice looks like when someone makes it
deliberately: one chunk over all 4 blocks as a coarser stand-in for the
paper's global CI fn, with the deviations enumerated in its header. It is the
reason `jose.yaml` is kept rather than deleted — it is the referent those
deviations are stated against.

They are quarantined in `KNOWN_BROKEN` in
`param_decomp_lab/tests/test_repo_configs_parse.py`, which asserts they still
fail (so the list can't silently rot) and must only ever shrink.

## Rules

1. **Every LM config yaml in the tree parses at tip** — CI-enforced by
   `param_decomp_lab/tests/test_repo_configs_parse.py` (schema parse +
   `assert_canonical_algorithm_config`). A schema PR that breaks one migrates
   it **in the same PR**, with an executed in-repo migration (the #966
   pattern) — never a script attached to a PR comment (#939 attached one; it
   never ran, and 97 of 104 stored runs became unopenable before anyone
   noticed).
2. **Sweep / profile / one-off configs are not committed.** Launch them from a
   workspace path. Deleting a config is cheap (git history keeps it; run dirs
   pin what actually ran) — un-rotting one is not.
3. **Adding a canonical seat is taking on a maintenance obligation**: add the
   file, a registry row above naming its purpose, and it's covered by the CI
   gate from that commit on. The table is **at its cap of 10** — the next seat
   requires an eviction, which is the point of the cap. A config a test loads
   is a seat by definition — deleting it is a test change, so grep for the
   basename first.
4. **Stored-run pins are immutable.** Never migrate a run dir's
   `launch_config.yaml` in place (resume byte-compares it; a live old-code run
   whose pin is rewritten refuses its next requeue). Consumers must not
   require a full-schema parse of stored pins — they parse only the fields
   they read (`ConsumerRunConfig`, PR #952).

## Case history

- **#939** (2026-07-03): ScheduleConfig unification; migration script attached
  as a PR comment, never executed → 7/104 stored runs parseable at tip.
- **#966**: the counter-example — carve migration shipped as an in-repo,
  executed tool covering every live repo yaml.
- **#982**: 25 sweep yamls in one PR — the accumulation pattern this policy
  ends. The sweeps' findings live in lore; the run dirs pin their configs.
- **#23** (2026-07-16): SwiGLU / FFN-as-its-own-config migrated all 30
  `param_decomp/configs/` yamls in one commit — the same 8-line edit applied
  30 times, ~26 of them to launched one-offs nobody will open again. The tax
  this policy stops, paid once more while the policy sat in review.
