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

## Review blind spots — dropped/changed with no prior tracking note

Surfaced by the `main`-vs-`feature/jax` recursive taxonomy (`MIGRATION_TAXONOMY.md`,
repo root). All are deliberate consequences of the torch shed; listed here so they're
tracked rather than silently absent. None block the squash.

Intentional drops (confirm scope, no re-add planned this push):

- **Component types: `EmbeddingComponents`, Radford-`Conv1D`, `Identity`** — the JAX stack
  decomposes `nn.Linear`-equivalents only (Llama MLP). The factory's conv1d/identity/
  embedding dispatch has no JAX path; folds into the deferred eqx-auto-decompose (#11).
- **`identity_insertion` / `identity_decomposition_targets`** — refused via assert; config
  field retained but inert.
- **LM-path component weight tying (`tie_component_weights`)** — refused via
  `assert tied_weights is None` on the LM path (TMS/ResidMLP keep embedding ties).
- **`ci_sigmoids` registry** — only `leaky_hard` (split lower/upper) survives; `normal` /
  `hard` / standalone `leaky_hard` / `swish_hard` are unreachable schema literals.
- **`mlp_scalar` CI-fn arch** — torch's scalar `get_component_acts(x)=x@V` couples CI-fn
  input to trained components; doesn't fit the generic `ci_fn(site_inputs)` waist. Replaced
  by the vector-input `LayerwiseMLPCIFn`. (Rationale in `CLAUDE.md`.)
- **`PersistentPGDReconSubsetLoss`** — dropped from the config union; a future composition
  per `LOSS_PARITY_DESIGN.md`.
- **CLT/transcoder adapters + `_vendor` models (#863)** — comparison-method tooling; can't
  harvest CLT/transcoder runs until re-added.
- **`editing/` + `generate_token_divergence.py`** — model-editing + token-divergence viz
  (also noted in `TRANSITION.md §1/§6`).
- **toy-models target-CI pattern framework** — `DenseCIPattern` / `TargetCISolution` /
  fnmatch expansion / greedy permutation gone; only per-target `identity_ci_error` survives.

Benign mechanism changes (no behavior risk):

- **`component_acts` cache-mode coupling removed** — autointerp/harvest recompute `x@V`
  inline; no per-component pre/post-detach cache.
- **`wandb.finish()` never called** — relies on process exit.
- **torch DDP `with_distributed_cleanup` / `ensure_cached_and_call`** — os._exit SIGABRT
  guard + download-once-per-node helper have no JAX analog.
- **imp-min `world_size` scaling** — explicit `log2(1+sum·world_size)` replaced by GSPMD's
  in-graph global sum (see `imp_min_world_size_noop` memory).

## Imp-min token-count reparameterization (deferred, Oli)

The imp-min entropy term carries a `log2(batch·seq)` coupling — a per-token-batch
artifact. A token-count-invariant reparameterization would remove the (currently
ignored) batch sensitivity. See `project_impmin_scaling` memory.
