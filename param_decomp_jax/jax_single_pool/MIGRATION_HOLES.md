# Migration holes — known gaps in the JAX-primary stack

Living list of things deliberately left un-migrated / un-reimplemented during the
torch→JAX migration, so they're tracked rather than silently lost. Append as found.

## Orphaned eval metrics (computed NOWHERE)

These sit in `config.OFFLINE_EVAL_METRIC_TYPES` — the in-loop trainer skips them — and
they were historically "deferred to the offline path". But that torch offline path
(`pd-offline-eval`) was **retired in #2**, and `jsp-slow-eval` (`slow_eval.py`) only
recomputes `CIHistograms` / `ComponentActivationDensity` / `CIMeanPerComponent` (+ the
hidden-acts metrics) natively. So the following are in the config enum, skipped by the
trainer, and recomputed by no live code:

- **`UVPlots`** — V/U weight visualizations. Not reimplemented.
- **`PermutedCIPlots`** — permute the CI matrix to identity (linear-sum-assignment) and
  plot. Not reimplemented. Its substrate was `param_decomp_lab/toy_models/` (the
  `permute_to_identity_*` + `_linear_sum_assignment` helpers), now also dead/deletable.
- **`IdentityCIError` (general / LM)** — the discrete CI-vs-target-pattern distance.
  Reimplemented for the toy targets only (`tms.identity_ci_error`,
  `resid_mlp.identity_ci_error`, wired into `train_tms`/`train_resid_mlp`); NOT as a
  general eval metric.
- **`AutointerpLabels`** — already dead pre-migration (registered, zero configs, no driver).

To re-add: implement as JAX-native `jsp-slow-eval` metrics in `slow_eval.py`.
`PermutedCIPlots` can reuse a JAX linear-sum-assignment; the general `IdentityCIError`
generalizes the tms/resid `identity_ci_error`; `UVPlots` plots V/U directly. The config
classes (in torch-free `param_decomp_config.eval_metrics`) were KEPT, so a config can
still name them — they just no-op until reimplemented.

## Deferred to a follow-up push (not in the current push)

- **#10 torch→jax run adapter** — loading OLD torch PD runs (`model_*.pth`) into the JAX
  consumers (autointerp/intruder/app). The torch-run-loading surface (`adapters/pd`,
  `component_model_io`, vendored Llama) was dropped (#872); re-add as a JAX-native loader
  off `open_jax_run`/orbax. Until then autointerp/intruder work on JAX runs only.
- **App** (`param_decomp_lab/app/`) — temporarily removed (#868); slated for a JAX-native
  re-add. `pd-investigate` subprocess-launches it, so it's broken until the app returns.

## `pretrain/` is DELETED — reimplement in JAX when next needed

`param_decomp_lab/experiments/lm/pretrain/` (the in-house target model defs + training
loop + `pd-pretrain` CLI) was **deleted** with the rest of torch — the repo is now
torch-free (zero `import torch`). When we next need to pretrain a target, write it in JAX
from scratch. This costs nothing in the meantime: the base models we currently decompose
(Llama-3.1-8B, the pile `LlamaSimpleMLP` `t-9d2b8f02`) are already pretrained on disk, and
the trainer loads them through its OWN torch-free loaders
(`llama_simple_mlp.load_target_from_pretrain_cache` / `load_prefix_from_pretrain_cache`)
reading the on-disk weight cache — never through `pretrain/` code. The one-off torch
checkpoint→safetensors converter (`tools/convert_llama_simple_mlp_checkpoint.py`) was
deleted too; the existing caches already hold their converted safetensors.

## Imp-min token-count reparameterization (deferred, Oli)

The imp-min entropy term carries a `log2(batch·seq)` coupling — a per-token-batch
artifact. A token-count-invariant reparameterization would remove the (currently
ignored) batch sensitivity. See `project_impmin_scaling` memory.
