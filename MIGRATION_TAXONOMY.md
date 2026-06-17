# Migration Taxonomy — `main` (torch) → `feature/jax`

A recursive, per-feature map of the torch→JAX migration: what was **kept**, **changed**,
**replaced**, **dropped**, or **deferred**. Structured by subsystem → module → feature
(hierarchy from each feature's dotted path). Every leaf carries the disposition, a detail
note, and the feature/jax location of its successor (blank when nothing survives).

Disposition vocabulary:

- **kept** — survives essentially verbatim (possibly relocated; behavior unchanged).
- **changed** — survives with modified semantics/shape/mechanism, same role.
- **replaced** — the role survives but the implementation was rewritten under a different
  paradigm (torch nn.Module → JAX pure-fn/pytree; DDP → GSPMD; .pth → orbax).
- **dropped** — gone, no successor, not tracked as a follow-up (e.g. torch-only plumbing,
  unused variants, deliberate cuts).
- **deferred** — intentionally not-yet-reimplemented, tracked as a follow-up (the
  torch→JAX run-adapter, the app re-add, attribution graphs backlog, pretrain-in-JAX).

---

## Summary

### Counts per disposition

| Disposition | Count (approx) | What it means here |
|---|---:|---|
| **kept** | ~165 | verbatim survivors — mostly the torch-free config schema (`param_decomp_config`), pure utilities (`base_config`, `log`, `sqlite`, `git`, `sampling`/`analysis`/`db` numpy ports), the papers, and the harvest/autointerp/clustering read pipelines |
| **changed** | ~200 | same role, modified mechanism — config relocations, numpy ports of harvest/clustering, GSPMD-isms, recon-term refactors |
| **replaced** | ~225 | rewritten under JAX paradigm — the trainer core (ComponentModel→DecomposedModel, Trainer→jsp-train, DDP→GSPMD, .pth→orbax), the loss/metric machinery, harvest worker |
| **dropped** | ~210 | no successor — torch-only plumbing (hooks, device-movers, DDP collectives), unused loss/CI variants, identity-targets, attribution-graph internals, model-editing, toy-model target-CI framework |
| **deferred** | ~120 | tracked follow-ups — the web app + frontend (~90 leaves), torch→JAX run-adapter, attribution graphs (dataset_attributions + graph_interp), pretrain-in-JAX, circuit-opt/model-editing-in-JAX |

(Counts are aggregate tallies over the per-module diffs below; the per-feature dispositions
in the tree are the authoritative source. The web-app removal alone — backend routers +
`compute`/`optim_cis`/`database` + the whole Svelte `components`/`lib` tree + `run_app` —
contributes the bulk of `deferred`, and is one PR `#868` "remove-now-re-add-later".)

### Headline narrative — what the migration did

**The training core was retired and rebuilt, not ported.** The torch `Trainer`,
`ComponentModel`, `Components`, the CI-fn nn.Modules, the mask payload structs, DDP, and
`.pth` checkpointing were all **replaced** by a functional JAX stack in
`param_decomp_jax/jax_single_pool/`: `DecomposedModel` (ordered `sites` + pure fns over
`(frozen, vu)` pytrees), `jsp-train` (the composition root + only I/O layer), GSPMD
sharding in place of DDP, and orbax sharded checkpoints. The torch trainer survives only
as the *semantic oracle* at git tag `torch-oracle`; `SPEC.md` is the normative contract.

**The config schema was lifted out of torch and kept verbatim.** Everything in
`param_decomp_config/` (`base`, `schedule`, `routing`, `ci_fn`, `decomposition_target`,
`losses`, `pd`, `experiment`, `lm`, `eval_metrics`, `autointerp`) is **kept**/**changed**
(relocated, occasionally hardened) so the JAX trainer reads the same YAML the torch trainer
wrote. This is the migration's load-bearing continuity: config is the shared contract.

**The 13 torch recon-loss classes collapsed into one factorization.** `build_recon_terms`
expresses every recon loss as `plan (chunking) × routing × mask-source strategy`
(`ConstantSources`/`StochasticSources`/`FreshPGDSources`/`PersistentSources`). Faithfulness
and importance-minimality became pure fns. The `Metric` ABC, `MetricContext`,
`dispatch`, and the metric output-normalization layer were all dropped — there are no
metric objects in JAX; the jitted step returns a flat scalar dict.

**What was genuinely dropped (no plan to return):** identity-decomposition targets;
Embedding/Conv1D/Identity component types (only Linear-shaped matrices decompose); the
scalar/parallel/shared/global-shared-mlp CI-fn variants (only the global transformer +
layerwise MLP survive); the `swish_hard`/`normal`/`hard` sigmoids (only `leaky_hard`);
component-weight tying; the `CLT`/`transcoder` comparison adapters + vendored models;
the toy-models target-CI pattern framework; the editing module; the token-divergence
script; per-module hook/cache/device plumbing.

**What was deferred (tracked in `MIGRATION_HOLES.md`):** the **#10 torch→JAX run adapter**
(loading old `model_*.pth` runs into JAX consumers); the **web app** (backend + Svelte
frontend, removed wholesale in #868 pending a JAX-native read-only viewer); **attribution
graphs** (`dataset_attributions/`, `graph_interp/`, `gradient_connectivity`, the app's
graph routers) as a JAX backlog item; **pretrain-in-JAX**; **circuit-opt / model-editing**
in JAX; and several orphaned eval metrics (`UVPlots`, `PermutedCIPlots`, general
`IdentityCIError`, `AutointerpLabels`, `StochasticHiddenActsRecon` as a training loss).

---

## Deferred (post-merge) — torch-bridge + auto-decompose + app + attribution

These are the tracked follow-ups, not silent drops. `MIGRATION_HOLES.md` is the living list.

- **Torch→JAX run adapter (#10).** The torch consumer-bridge surface — `RunBatch` /
  `ReconstructionLoss` protocols (`param_decomp/batch_and_loss_fns.py`),
  `component_model_io.load_component_model`, `adapters/pd.py`, the vendored Llama load path
  — was deleted in #872 ("torch-run loading is being deferred"). JAX consumers
  (harvest/autointerp/intruder/clustering) currently load **JAX-native orbax runs only**
  via `jax_single_pool/load_run.py` (`open_jax_run` / `run_metadata`) and
  `param_decomp_lab/adapters/jax_pd.py`. Re-adding the ability to read OLD torch `.pth`
  runs into the JAX consumers is the deferred adapter.

- **Auto-decompose any `eqx.Module`.** The torch `make_components` / `Components` ABC could
  decompose any nn.Linear/Embedding/Conv1D module by introspection. JAX targets define
  their own `sites` + `weight_deltas` explicitly per vendored arch
  (`llama_simple_mlp.py`, `llama8b.py`, `resid_mlp.py`, `tms.py`). A generic
  "decompose any eqx.Module" follow-up would restore the introspective path (and is where
  Embedding/Conv1D/Identity decomposition would re-land).

- **Web app + frontend.** `param_decomp_lab/app/` (FastAPI backend + Svelte frontend,
  ~248 files) was removed in #868 as an explicit remove-now-re-add-later, slated for a
  JAX-native read-only viewer. Within it, **attribution graphs / intervention / circuit-opt**
  routers were *dropped* (TRANSITION §1 backlog), while the read-only browse surfaces
  (runs, prompts, activation-contexts, correlations, clusters, autointerp-compare,
  dataset-explorer, data-sources, investigations) are *deferred* with the viewer. Caveat:
  `pd-investigate` subprocess-launches the app backend, so it is runtime-broken until the
  app returns.

- **Attribution graphs (backlog).** `dataset_attributions/` + `graph_interp/` +
  `topology/gradient_connectivity.py` + the app graph/attribution routers were deleted in
  #848. TRANSITION §1/§6 mark these "revisit in JAX later, as backlog — possibly easier in
  JAX (vmap + grad)", recoverable from git history. Classified `dropped`/`deferred` in the
  tree below depending on the per-module author's read; all share this one backlog.

- **Pretrain-in-JAX.** `experiments/lm/pretrain/` (model defs + training loop + `pd-pretrain`)
  was deleted with torch; reimplement in JAX when next needed. The currently-decomposed
  targets are already pretrained on disk and load through torch-free cache loaders.

---

## ⚠️ Review blind spots — `dropped`/`changed` features with NO MIGRATION_HOLES note

`MIGRATION_HOLES.md` covers: the #10 torch→JAX run-adapter; the app; `pretrain/`; the
orphaned eval metrics (`UVPlots`, `PermutedCIPlots`, general `IdentityCIError`,
`AutointerpLabels`); the imp-min token-count reparam. The attribution-graphs backlog and
circuit-opt/editing are documented in `TRANSITION.md §1/§6` (a sibling doc, not
MIGRATION_HOLES). Everything below is a **dropped or changed** feature **not** covered by
either doc — i.e. the review's blind spots:

**Genuine drops with no tracking doc (verify these were intended):**

- **`component_model.forward.caching` → `component_acts` cache mode (dropped)** — the
  per-component pre/post-detach activation cache (torch `fn_type=mlp` `get_component_acts`
  coupling) has no JAX analog and isn't in any holes doc. Autointerp/harvest recompute
  `x@V` inline instead, so likely fine — but the *coupling removal* is an undocumented
  design decision.
- **`ci_sigmoids` registry + `normal`/`hard`/`leaky_hard`(standalone)/`swish_hard`
  sigmoids (dropped)** — only `leaky_hard` (as the split lower/upper pair) survives;
  4 of 6 squashing fns are unreachable schema literals. Not in holes.
- **`pd_config`/component weight tying — `tie_component_weights` (dropped)** — refused via
  `assert tied_weights is None` on the LM path; TMS/ResidMLP use it only for embedding
  ties. The LM-decomposition weight-tying capability is silently gone. Not in holes.
- **`identity_insertion` (entire feature, dropped)** — all identity-op machinery refused
  via `assert identity_decomposition_targets is None`. The config field still exists.
  Mentioned in TRANSITION only obliquely; not in MIGRATION_HOLES.
- **`components.factory.conv1d_dispatch` + `identity_dispatch` (dropped)** —
  Radford-Conv1D and Identity decomposition targets have no JAX path. Would fold into the
  deferred eqx-auto-decompose, but not currently noted.
- **`components.embedding.*` (deferred per author, but NOT in holes)** — Embedding
  decomposition (`EmbeddingComponents`) is absent; the per-module entries call it
  "deferred" but no holes doc lists embedding decomposition specifically.
- **`ci_fns.modules.mlp_scalar` (dropped)** — the scalar `get_component_acts(x)=x@V` CI-fn
  arch was deliberately not ported (breaks the generic `ci_fn(site_inputs)` waist). Design
  rationale lives only in `jax_single_pool/CLAUDE.md`, not holes.
- **`metric.subset` / `PersistentPGDReconSubsetLoss` (dropped)** — the persistent-PGD
  subset variant was dropped from the config union entirely (only `LOSS_PARITY_DESIGN.md`
  notes it as a future composition). Not in MIGRATION_HOLES.
- **`adapters` CLT/transcoder + `_vendor` models (dropped)** — the entire comparison-method
  adapter family (PR #863). Comparison runs against CLTs/transcoders can't be harvested.
  Not in holes (only stale doc-string mentions remain).
- **`editing/` + `scripts/generate_token_divergence.py` (dropped)** — model editing and the
  token-divergence visualization. Covered by TRANSITION §1 (circuit-opt/editing), NOT
  MIGRATION_HOLES.
- **`toy_models/target_ci.py` + `_linear_sum_assignment.py` (dropped/replaced)** — the
  general target-CI *pattern framework* (DenseCIPattern, TargetCISolution, fnmatch
  expansion, `compute_target_metrics`, greedy permutation) is gone; only a per-target
  `identity_ci_error` survives. MIGRATION_HOLES mentions `PermutedCIPlots`/`IdentityCIError`
  but NOT the dense-pattern / multi-module-solution drop.
- **`run_sink.output.finish` / `wandb.finish()` (dropped)** — the JAX trainer never calls
  `wandb.finish()` (relies on process exit). Not in holes.
- **`distributed.*` torch DDP collectives, `with_distributed_cleanup`,
  `ensure_cached_and_call` (dropped)** — the os._exit SIGABRT-avoidance decorator and the
  download-once-per-node helper have no JAX analog. Not in holes (mechanism-change, mostly
  benign).

**Notable `changed` semantics worth a second look (not blind spots per se, but
undocumented behavior shifts):**

- **`importance_minimality` world_size scaling dropped** — the explicit `log2(1+sum*world_size)`
  factor is replaced by GSPMD's in-graph global sum. Guarded by
  `test_imp_min_global_reduction.py`; the residual `log2(batch·seq)` coupling IS the one
  imp-min item that IS in holes.
- **`fused_minimality` factoring** — imp-min + freq-min computed over `ci_upper` per site
  rather than the torch fused-over-shared-sum `s_c` form (same value, different factoring).
- **`recon` semantics narrowed to KL-on-final-logits only** — MSE/hidden-acts recon paths
  dropped as *training* losses; hidden-acts moved to offline eval (in holes via
  `StochasticHiddenActsRecon` deferral, but the CI-masked MSE-recon micro-tests were
  silently dropped).
- **`ce_kl` cross-batch averaging** — torch token-weighted accumulate-then-compute replaced
  by uniform per-batch mean (proven equivalent for uniform (B,T); documented in `eval.py`).

---

## Subsystem → Module → Feature

Notation per leaf: `[disposition]` detail — `jax: <location>` (omitted when empty).

---

## param_decomp core (torch library)

### `param_decomp/component_model.py`

- `component_model` — **[replaced]** torch ComponentModel (nn.Module wrapping frozen target + Components + CI fn) deleted in #872; role → JAX functional `DecomposedModel` (frozen dataclass of pure fns over `(frozen, vu)`). jax: `jax_single_pool/lm.py (DecomposedModel)`
  - `.construction` — **[replaced]** nn.Module-init → `init_train_state` / `init_ci_fn`; sharding explicit, not DDP. jax: `run_state.py`, `ci_fn.py`, `lm.py`
    - `.frozen_target_assertion` — **[replaced]** freezing is structural (separate frozen pytree), not a requires_grad flag. jax: `lm.py`
    - `.target_resolution` — **[changed]** `module_to_c` → ordered `SiteC`/`SiteSpec` tuple. jax: `lm.py`
    - `.component_registration` — **[replaced]** `make_components`+ModuleDict → V/U arrays in vu pytree; no dot→dash mangling. jax: `run_state.py`, `lm.py`
    - `.ci_fn_registration` — **[changed]** `make_ci_fn_wrapper` → `init_ci_fn` over CIArch/MLPCIArch. jax: `run_state.py`, `ci_fn.py`, `ci_fn_mlp.py`
    - `.sigmoid_selection` — **[changed]** selection collapsed; hardcoded leaky_hard pair. jax: `ci_fn.py`
  - `.target_weight` — **[changed]** [d_out,d_in] convention kept; Conv1D-transpose + Identity-synthesis dropped. jax: `llama_simple_mlp.py`, `lm.py`
  - `.forward` — **[replaced]** one overloaded forward → distinct pure fns (clean_output/masked_output/site_inputs/masked_site_outputs). jax: `lm.py`, `llama_simple_mlp.py`
    - `.plain_passthrough` — **[replaced]** → `clean_output` (all-frozen, SPEC S3). jax: `lm.py`
    - `.hook_setup` — **[replaced]** forward-hooks → functional dataflow in `masked_output`. jax: `llama_simple_mlp.py`
    - `.component_replacement` — **[changed]** `torch.where` routing → `jnp.where`, no hook. jax: `llama_simple_mlp.py (~286-290)`
    - `.caching` — **[replaced]** 4 cache modes → pure `site_inputs`/`masked_site_outputs`; **`component_acts` cache DROPPED** (no JAX analog). jax: `lm.py`, `hidden_acts_eval.py`
    - `.hook_lifetime_management` — **[dropped]** no hooks to register/remove.
  - `.calc_causal_importances` — **[changed]** → `CIFn.__call__` → `CIValues(lower,upper)`; binomial noise is a separate adversary concern. jax: `ci_fn.py`, `ci_fn_mlp.py`
    - `.input_detach` — **[replaced]** detach flag → `stop_gradient` at loss-term call sites. jax: `losses.py`/`train.py`
    - `.sigmoid_squash` — **[changed]** `CIValues`+`site_logits`; custom lower_leaky VJP bit-matched (S6). jax: `ci_fn.py`
    - `.binomial_noise` — **[deferred]** binomial sampling not ported (★ continuous is production; out of spec scope).
  - `.calc_weight_deltas` — **[changed]** → `weight_deltas` (fp32 W−V@U, N2). jax: `lm.py:127`, `llama_simple_mlp.py:287`
  - `.data_types` — **[changed]** OutputWithCache dropped; CIOutputs → CIValues + flat site-keyed dicts. jax: `ci_fn.py`, `lm.py`
- `component_grad_norms` — **[changed]** reimplemented as `_grad_norm_metrics` (same key families; NaN-for-unpopulated unneeded). jax: `train.py (~93-111)`

### `param_decomp/components.py`

- `components` — **[replaced]** whole module deleted (#872); V/U live in `DecompVU`; only Linear-shaped matrices decompose. jax: `llama8b.py (DecompVU)`, `lm.py`, per-target forwards
  - `.weight_init` — **[replaced]** `init_param_` not ported for V/U (`init_decomp_vu`); Kaiming-fan-in style survives for CI-fn weights. jax: `llama8b.py:268`, `ci_fn.py:155`
  - `.abc` — **[replaced]** Components ABC → concrete `DecompVU`; abstract methods → free fns. jax: `llama8b.py:258`
    - `.component_activations` — **[changed]** `get_component_acts` method → inline `x@V`. jax: `load_run.py:203-207`
  - `.linear` — **[replaced]** LinearComponents → generic per-site V/U path; frozen bias on frozen target. jax: `llama8b.py:284`, `llama_simple_mlp.py:267`
    - `.effective_weight` — **[changed]** einsum property → inline `(V@U).T` only where needed. jax: `lm.py:127`
    - `.component_acts` — **[changed]** → inline `x@V`. jax: `llama8b.py:298`, `load_run.py:206`
    - `.forward` — **[replaced]** → pure `_site_out`. jax: `llama8b.py:284`, `llama_simple_mlp.py:267`
      - `.masking` — **[kept]** `acts*mask` verbatim. jax: `llama_simple_mlp.py:283`
      - `.weight_delta` — **[kept]** masked delta term survives; `has_delta` static flag. jax: `llama_simple_mlp.py:286-288`, `lm.py:64-69`
      - `.acts_cache` — **[replaced]** detach/requires_grad_ → `stop_gradient` + `jax.grad` over sources. jax: `train.py:277-402`, `adversary.py`
  - `.embedding` — **[deferred]** no EmbeddingComponents; embedding is frozen prefix; belongs to eqx-auto-decompose follow-up.
    - `.effective_weight` — **[deferred]**
    - `.component_acts` — **[deferred]**
    - `.forward` — **[deferred]**
  - `.target_introspection` — **[replaced]** `get_module_input_dim` → static `SiteSpec` dims. jax: `lm.py:49`, `llama_simple_mlp.py:133`
  - `.factory` — **[replaced]** `make_components` → `init_decomp_vu(sites,key)`; no type dispatch. jax: `llama8b.py:268`
    - `.linear_dispatch` — **[replaced]** single Linear path. jax: `llama8b.py:268`
    - `.conv1d_dispatch` — **[dropped]** GPT-2 weights pre-transposed; no Conv1D in live targets.
    - `.identity_dispatch` — **[dropped]** no Identity target decomposed.
    - `.embedding_dispatch` — **[deferred]** with embedding-decomposition follow-up.

### `param_decomp/decomposition_targets.py`

- `decomposition_target_resolution` — **[replaced]** torch model-walk → torch-free string site resolution. jax: `config.py`, `llama8b.py`, `llama_simple_mlp.py`
  - `.config_schema` — **[kept]** `DecompositionTargetConfig` moved verbatim. jax: `param_decomp_config/decomposition_target.py`
  - `.resolved_target_dataclass` — **[replaced]** `DecompositionTarget` → `SiteC`. jax: `lm.py`
  - `.pattern_to_module_matching` — **[changed]** fnmatch-over-modules → string `_site_cs`/`expand_wildcard_site_cs`; canonical order (layer-asc + KIND_ORDER), set-equality test. jax: `config.py`, `llama_simple_mlp.py`, `tests/test_fnmatch_site_order.py`
    - `.unmatched_pattern_guard` — **[changed]** ValueError → SITE_NAME_PATTERN assert; oracle in test. jax: `config.py`
    - `.overlap_guard` — **[changed]** ValueError → duplicate-site assert in `canonical_site_cs`. jax: `llama8b.py`/`llama_simple_mlp.py`
- `identity_insertion` — **[dropped]** all identity-shim machinery refused (`assert identity_decomposition_targets is None`). Config field still exists.
  - `.identity_shim_module` — **[dropped]**
  - `.forward_pre_hook` — **[dropped]**
    - `.input_shape_assertions` — **[dropped]**
  - `.in_place_attachment` — **[dropped]**
    - `.duplicate_pattern_guard` — **[dropped]**
    - `.unmatched_pattern_guard` — **[dropped]**
    - `.input_dim_inference` — **[dropped]**
    - `.unsupported_module_rejection` — **[dropped]**

### `param_decomp/ci_fns.py`

- `ci_fns` — **[replaced]** deleted #872; split into config + two JAX arches. jax: `param_decomp_config/ci_fn.py`, `ci_fn.py`, `ci_fn_mlp.py`
  - `.config` — **[changed]** extracted to torch-free package, hardened (#856). jax: `param_decomp_config/ci_fn.py`
    - `.discriminated_union` — **[changed]** CiConfig kept; global arm now itself a union. jax: `param_decomp_config/ci_fn.py:124`
    - `.layerwise` — **[changed]** kept; only `fn_type=mlp` honored. jax: `param_decomp_config/ci_fn.py:12`
    - `.global` — **[changed]** split into discriminated union; only transformer has a runtime. jax: `param_decomp_config/ci_fn.py:115`
      - `.transformer_cfg` — **[kept]**. jax: `param_decomp_config/ci_fn.py:47`
        - `.attention` — **[kept]** AttnConfig verbatim. jax: `param_decomp_config/ci_fn.py:30`
  - `.modules` — **[replaced]** 5 torch arches → 2 (global transformer + layerwise MLP). jax: `ci_fn.py`, `ci_fn_mlp.py`
    - `.mlp_scalar` — **[dropped]** scalar `get_component_acts` coupling deliberately not ported.
    - `.vector_mlp` — **[replaced]** → per-site `SiteMLP`/`LayerwiseMLPCIFn`. jax: `ci_fn_mlp.py`
    - `.shared_mlp` — **[dropped]** inert config literal, no JAX module.
    - `.global_shared_mlp` — **[dropped]** config exists, no JAX module.
    - `.global_transformer` — **[changed]** → `CIFn`+`CIBlock`; exact-erf GELU, RMS eps. jax: `ci_fn.py:100`
      - `.seq_dim_adaptation` — **[replaced]** unsqueeze/squeeze → `leading_axes`/`expects_axes`. jax: `ci_fn.py:110`, `run_state.py:104`
      - `.target_metadata` — **[replaced]** TargetLayerConfig → generic SiteSpec. jax: `ci_fn.py:154`
  - `.wrappers` — **[replaced]** dict-in/out wrapper collapsed into the arches. jax: `ci_fn.py:136`, `ci_fn_mlp.py:71`
    - `.layerwise` — **[replaced]** ModuleDict → plain `{site:SiteMLP}`. jax: `ci_fn_mlp.py:57`
    - `.global` — **[changed]** folded into `CIFn.__call__`; embedding float-coercion dropped. jax: `ci_fn.py:136`
  - `.builders` — **[replaced]** → `init_*` factories dispatched in `init_train_state`. jax: `run_state.py:97`
    - `.make_ci_fn_wrapper` — **[replaced]** → `match cfg.ci_fn`. jax: `run_state.py:97`, `config.py:349`
    - `.make_layerwise_ci_fn` — **[replaced]** → `init_layerwise_mlp_ci_fn`. jax: `ci_fn_mlp.py:79`, `config.py:366`
    - `.make_global_ci_fn` — **[changed]** → `init_ci_fn`; transformer-only. jax: `ci_fn.py:154`, `config.py:349`

### `param_decomp/ci_sigmoids.py`

- `sigmoid_registry` — **[dropped]** config-driven lookup gone; JAX hardcodes leaky_hard pair.
  - `.type_literal` — **[changed]** SigmoidType moved to config, narrowed to 5 (lower_leaky_hard dropped from enum); only leaky_hard accepted. jax: `param_decomp_config/pd.py:145`
- `squashing.normal` — **[dropped]** unreachable schema literal.
- `squashing.hard` — **[dropped]** unreachable schema literal.
- `squashing.leaky_hard` — **[dropped]** standalone leaky_hard not reimplemented; replaced by the two-view treatment.
- `squashing.upper_leaky_hard` — **[changed]** → `upper_leaky_hard_sigmoid`, ordinary autodiff. jax: `ci_fn.py:55`
- `squashing.lower_leaky_hard` — **[changed]** → custom-VJP `lower_leaky_hard_sigmoid`. jax: `ci_fn.py:39`
  - `.custom_autograd` — **[changed]** torch.autograd.Function → `jax.custom_vjp` (`_lhs_f`/`_lhs_b`). jax: `ci_fn.py:42`
- `squashing.swish_hard` — **[dropped]** unreachable schema literal.
  - `.swish_primitives` — **[dropped]**

### `param_decomp/ci_nn_blocks.py`

- `linear_layers` — **[replaced]** custom Linear modules → inline matmuls in eqx CI fns. jax: `ci_fn.py`, `ci_fn_mlp.py`
  - `.single` — **[replaced]** → inline `x@w+b`, Kaiming init reproduced. jax: `ci_fn_mlp.py (SiteMLP)`, `ci_fn.py`
  - `.parallel` — **[replaced]** ParallelLinear (C-stacked) → per-site MLP (per site, not per component). jax: `ci_fn_mlp.py`
- `positional_encoding.rope` — **[replaced]** RoPEEmbedding module → free fns `rope_cos_sin`/`apply_rope`. jax: `ci_fn.py`, `vendored_jax/llama.py`
  - `.inv_freq_buffer` — **[changed]** registered buffer → eqx field + `stop_gradient`. jax: `ci_fn.py`
  - `.angle_construction` — **[replaced]** → `rope_cos_sin`. jax: `vendored_jax/llama.py`
  - `.rotate_half_application` — **[replaced]** → `rotate_half`+`apply_rope`. jax: `vendored_jax/llama.py`
- `attention.self_attention` — **[replaced]** SelfAttention module → inline in `CIBlock`. jax: `ci_fn.py`
  - `.projections` — **[changed]** raw weight arrays; attn uses torch-default init (`attn_default`). jax: `ci_fn.py`
  - `.head_reshape` — **[replaced]** inlined in CIBlock. jax: `ci_fn.py`
  - `.sdpa` — **[replaced]** torch SDPA → `jax.nn.dot_product_attention`; dropout dropped. jax: `ci_fn.py`, `vendored_jax/llama.py`
- `transformer_block` — **[replaced]** → eqx `CIBlock`. jax: `ci_fn.py`
  - `.rmsnorm_prenorm` — **[changed]** weightless `_weightless_rms_norm`, eps `CI_FN_RMS_EPS`. jax: `ci_fn.py`
  - `.mlp` — **[changed]** Sequential-of-Linear → fixed single-hidden inline; multi-hidden survives in `SiteMLP`. jax: `ci_fn.py`, `ci_fn_mlp.py`
  - `.residual_stream` — **[replaced]** inlined residual adds. jax: `ci_fn.py`

### `param_decomp/masks.py`

- `mask_payload` — **[replaced]** bundle-then-apply → plain pytree dicts + ReconForward/ReconPlan. jax: `recon.py`, `train.py`, `lm.py`
  - `.components_mask_info` — **[dropped]** ComponentsMaskInfo struct gone; 3 fields → positional pytree args.
  - `.weight_delta_and_mask_type` — **[replaced]** → separate `weight_deltas` + `delta_masks` + static `has_delta`. jax: `lm.py:127`, `train.py:206-219`, `recon.py:107`
  - `.routing_masks_type` — **[replaced]** `dict|"all"` → `dict|None`. jax: `recon.py:51`
- `routing` — **[replaced]** Router ABC → pure `RoutingSampler` closures. jax: `recon.py:52`
  - `.router_interface` — **[replaced]** ABC → `Callable[[key,shape], tuple[Routes,...]]`. jax: `recon.py:52-62`
  - `.all_layers` — **[replaced]** → `route_all_n`. jax: `recon.py:195`
  - `.static_probability` — **[replaced]** → `static_probability_routing`. jax: `recon.py:177`
  - `.uniform_k_subset` — **[replaced]** → `uniform_k_routing` (production path). jax: `recon.py:152,165`
  - `.single_layer` — **[replaced]** LayerRouter → `per_site` chunking. jax: `recon.py:228`
  - `.config` — **[changed]** moved to torch-free package; gained `AllRoutingConfig`. jax: `param_decomp_config/routing.py`
    - `.uniform_k_subset` — **[kept]**. jax: `param_decomp_config/routing.py:8`
    - `.static_probability` — **[kept]**. jax: `param_decomp_config/routing.py:14`
  - `.factory` — **[replaced]** `get_subset_router` → `routing_sampler_from_config`. jax: `recon.py:204`
  - `.sampling_helpers` — **[replaced]** torch samplers → pure JAX-PRNG. jax: `recon.py:152`
    - `.rand_perm` — **[replaced]** double-argsort → single argsort inline. jax: `recon.py:160`
    - `.uniform_k_subset_masks` — **[replaced]** → `uniform_k_subset_routes`. jax: `recon.py:152`
- `component_masking` — **[replaced]** → `MaskSourceStrategy` family + `source_masks` dispatch. jax: `recon.py:64-105`, `train.py:196-232`
  - `.sampling_type` — **[kept]** `SamplingType` moved to config. jax: `param_decomp_config/routing.py:31`
  - `.interpolate` — **[replaced]** formula kept, inlined per strategy. jax: `train.py:217,230`
  - `.stochastic_from_ci` — **[replaced]** decomposed across strategy×sampler×forward. jax: `recon.py:69`, `train.py:196-221`
- `assembly` (`make_mask_infos`) — **[dropped]** no bundle to assemble; separate dict args.

### `param_decomp/optimize.py`

- `trainer` — **[replaced]** stateful Trainer class → `jsp-train` function composition. jax: `run.py::train`
  - `.construction` — **[replaced]** split run.py / run_state.py builders. jax: `run.py`, `run_state.py`
    - `.identity_insertion` — **[dropped]** refused (`assert identity_decomposition_targets is None`).
    - `.target_freeze` — **[changed]** frozen pytree runtime arg + stop_gradient; no eval()/requires_grad. jax: `train.py::step`, `lm.py`
    - `.decomposition_target_resolution` — **[changed]** → ordered SiteC at config-build time. jax: `config.py`, `lm.py`
    - `.component_model_build` — **[replaced]** ComponentModel → DecomposedModel + sharded V/U. jax: `lm.py`, `run_state.py`, `llama8b_sharding.py`
    - `.weight_tying` — **[dropped]** `tie_component_weights` not ported; refused (`assert tied_weights is None`).
    - `.param_partition` — **[kept]** two-pool split → typed `TrainState` (components/ci_fn). jax: `train.py::TrainState`
    - `.dual_optimizers` — **[kept]** two optax optimizers (opt_vu/opt_ci); wd zeroed (N1). jax: `run_state.py::build_optimizers`
    - `.pgd_scope_validation` — **[replaced]** → supported-scope asserts + sharding divisibility. jax: `recon.py`, `config.py:436`, `llama8b_sharding.py`
  - `.distributed` — **[changed]** DDP → SPMD (mesh + sharded arrays). jax: `sharding.py`, `run.py::train`
    - `.ddp_wrap` — **[replaced]** no DDP; GSPMD collectives. jax: `train.py::make_train_step`
    - `.seeding` — **[changed]** seed_all/per_rank → single PRNGKey + fold_in. jax: `run.py`, `train.py`
    - `.metric_averaging` — **[dropped]** values already global (sharded). 
    - `.grad_sync` — **[changed]** explicit barrier/hooks → implicit XLA collectives. jax: `train.py::step`
  - `.persistence` — **[changed]** TrainingState round-trip → orbax sharded TrainState. jax: `checkpoint.py`, `train.py`
    - `.snapshot` — **[changed]** → orbax StandardSave of TrainState. jax: `checkpoint.py::save_state`
    - `.from_snapshot` — **[changed]** → restore-onto-reference; mid-run config edits refused. jax: `checkpoint.py::restore_latest/init_from_parent`
    - `.load_state` — **[changed]** → `restore_step`. jax: `checkpoint.py`
    - `.topology_independent_optimizer_state` — **[replaced]** name-keying unnecessary (orbax restore-onto-reference). jax: `checkpoint.py`
    - `.named_param_ordering` — **[dropped]** no name-mapping step.
  - `.train_loop` — **[changed]** → `for step` loop calling pure jitted step. jax: `run.py::train`
    - `.faithfulness_warmup` — **[kept]** runs once when fresh; `make_faith_warmup_step`. jax: `run.py`, `train.py`
    - `.loader_replay` — **[replaced]** replay eliminated; O(1) deterministic resume. jax: `run.py`, `data.py`
    - `.lr_scheduling` — **[kept]** optax-applied (`torch_cosine_schedule`, S20). jax: `run_state.py`, `run.py`
    - `.metric_context` — **[replaced]** no MetricContext; inline in step. jax: `train.py::step`
    - `.context_invariants` — **[changed]** → jaxtyped/beartype + axes assert + finiteness asserts. jax: `train.py`, `run_state.py`, `run.py`
    - `.weight_deltas_fp32` — **[kept]** fp32 masters; faith on fp32 components. jax: `train.py::loss_fn`
    - `.bf16_autocast` — **[changed]** toggle → unconditional explicit-cast bf16 (N1). jax: `train.py::cast_floating`
    - `.loss_aggregation` — **[changed]** → recon-term aggregation; start_frac where-gated. jax: `train.py::loss_fn`, `recon.py`
    - `.backward_hooks` — **[replaced]** before/after_backward → fused single backward + inline ascent. jax: `train.py::step`, `adversary.py`
    - `.grad_clip` — **[changed]** components-only via optax.chain (S19); ci unclipped. jax: `run_state.py`
    - `.optimizer_step` — **[changed]** both step every step; final-step carve-out dropped. jax: `train.py::step`
    - `.train_logging` — **[kept]** extended (grad norms, LRs, tok/s). jax: `run.py`, `train.py`
  - `.eval_loop` — **[changed]** EvalLoop → in-loop FAST + separate offline SLOW tiers. jax: `run.py`, `eval.py`, `slow_eval.py`
    - `.config` — **[changed]** EvalLoop dataclass → EvalConfig; slow gating offline. jax: `run.py`, `config.py`
    - `.metric_merge` — **[dropped]** no Metric registry merge.
    - `.slow_fast_gating` — **[replaced]** slow tier is a separate binary. jax: `run.py`, `eval.py`, `slow_eval.py`
    - `.eval_logging_and_cleanup` — **[changed]** empty_cache/gc dropped (JAX manages buffers). jax: `run.py`
  - `.checkpointing` — **[kept]** save-final/cadence/SIGTERM; sink.checkpoint → orbax `save_state`. jax: `run.py`, `checkpoint.py`
  - `.sigterm_preemption` — **[kept]** module-global flag + synchronous save. jax: `run.py::_install_sigterm_flag`

### `param_decomp/schedule.py`

- `schedule` — **[changed]** relocated verbatim to config; JAX loop reimplements LR in optax. jax: `param_decomp_config/schedule.py`
  - `.config` — **[kept]** ScheduleConfig identical; live type. jax: `param_decomp_config/schedule.py:9`
    - `.warmup` — **[kept]**. jax: `param_decomp_config/schedule.py:16`
    - `.decay_function_selection` — **[kept]**. jax: `param_decomp_config/schedule.py:27`
    - `.final_value_fraction` — **[kept]**. jax: `param_decomp_config/schedule.py:20`
    - `.validation` — **[kept]**. jax: `param_decomp_config/schedule.py:31`
  - `.value_lookup` — **[changed]** kept as config oracle; runtime LR is optax. jax: `param_decomp_config/schedule.py:37`, `run_state.py:36`
    - `.bounds_assertions` — **[kept]**. jax: `param_decomp_config/schedule.py:39`
    - `.warmup_phase` — **[kept]**. jax: `param_decomp_config/schedule.py:46`
    - `.degenerate_decay_guard` — **[kept]** mirrored in `torch_cosine_schedule`. jax: `param_decomp_config/schedule.py:49`, `run_state.py:44`
    - `.decay_dispatch` — **[changed]** oracle kept; JAX hardcodes cosine-to-0.1x. jax: `param_decomp_config/schedule.py:54`, `run_state.py:36`

### `param_decomp/faithfulness_warmup.py`

- `faithfulness_warmup` — **[replaced]** deleted; → `make_faith_warmup_step` + run.py loops. jax: `train.py`, `run.py`
  - `.run_faithfulness_warmup` — **[replaced]** → inlined per-target warmup blocks. jax: `run.py:277`, `train.py:551`
    - `.preconditions` — **[dropped]** no list to assert non-empty.
    - `.optimizer_setup` — **[changed]** lr honored; weight_decay pinned to 0.0. jax: `run.py:278`, `config.py:507,525`
    - `.optimization_loop` — **[replaced]** → jitted warmup_step; SIGTERM check added. jax: `train.py:551`, `run.py:286-298`
      - `.weight_delta_objective` — **[kept]** faith on weight_deltas, no data forward. jax: `train.py:559-561`, `losses.py`, `lm.py`
    - `.progress_logging` — **[changed]** → single final summary print. jax: `run.py:307-310`
    - `.teardown` — **[dropped]** no cache-clear/gc.

### `param_decomp/training_state.py`

- `training_state` — **[replaced]** deleted; → `TrainState` pytree; import-cycle rationale moot. jax: `train.py:80`
  - `.schema` — **[replaced]** → orbax-saveable TrainState; no configs inside. jax: `train.py:80`, `run_state.py`, `checkpoint.py`
    - `.step` — **[changed]** int filename → jnp scalar in pytree. jax: `train.py`, `run.py:256`
    - `.pd_config` — **[replaced]** → config.yaml byte-compare. jax: `config.py:757`, `run.py`
    - `.runtime_config` — **[dropped]** no RuntimeConfig stored (torch-numerics concern).
    - `.component_model` — **[replaced]** state_dict → V/U pytree. jax: `train.py`, `run_state.py`, `llama8b_sharding.py`
    - `.components_optimizer` — **[replaced]** name-keyed → optax OptState pytree. jax: `train.py`, `run_state.py`
    - `.ci_fn_optimizer` — **[replaced]** → optax OptState. jax: `train.py`, `run_state.py`
    - `.loss_metrics` — **[replaced]** → `sources`/`sources_opt_state` per persistent term (S23). jax: `train.py`, `run_state.py`, `recon.py`
  - `.ddp_canonical_artifact` — **[changed]** rank-0-canonical → orbax sharded all-ranks. jax: `checkpoint.py`
  - `.cycle_free_bridge` — **[dropped]** both endpoints deleted; no cycle.

### `param_decomp/batch_and_loss_fns.py`

- `protocols` — **[deferred]** deleted #872; part of torch→JAX run-adapter (#10).
  - `.run_batch` — **[deferred]** RunBatch protocol; JAX uses DecomposedModel pure fns.
  - `.reconstruction_loss` — **[deferred]** ReconstructionLoss protocol; JAX has native recon.
- `batch_device_transfer` — **[dropped]** `move_batch_to_device` cascade-pruned; JAX uses device_put/sharding.
  - `.nested_container_recursion` — **[dropped]**
  - `.passthrough_other_types` — **[dropped]**

### `param_decomp/run_sink.py`

- `run_sink_protocol` — **[replaced]** Protocol deleted; → concrete `MetricsSink`. jax: `run.py (MetricsSink)`
  - `.runtime_checkable` — **[dropped]** plain concrete class, no isinstance check.
  - `.log_metrics` — **[changed]** → `MetricsSink.log(step,record)`; flat float dict, re-namespaced; CommError swallow present. jax: `run.py:186-209`
  - `.console_lines` — **[changed]** → bare print(flush) inline. jax: `run.py:197-200`
  - `.checkpoint` — **[replaced]** → orbax `save_state` decoupled from sink. jax: `checkpoint.py:48-50`
  - `.finish` — **[dropped]** no finish/close; `wandb.finish()` NOT called by trainer.

### `param_decomp/configs.py`

- `pd_config` — **[changed]** moved to torch-free package, near-identical. jax: `param_decomp_config/pd.py:PDConfig`
  - `.reproducibility` — **[kept]**. jax: `param_decomp_config/pd.py`
  - `.stochastic_masking` — **[changed]** kept; SamplingType moved. jax: `param_decomp_config/pd.py`, `routing.py`
  - `.causal_importance` — **[changed]** kept; CiConfig moved. jax: `param_decomp_config/pd.py`, `ci_fn.py`
  - `.decomposition_targets` — **[changed]** kept; config moved. jax: `param_decomp_config/pd.py`, `decomposition_target.py`
    - `.identity_target_expansion` — **[kept]** byte-identical. jax: `param_decomp_config/pd.py`
  - `.delta_component` — **[kept]**. jax: `param_decomp_config/pd.py`
  - `.tied_weights` — **[kept]** (field; but LM path refuses non-null). jax: `param_decomp_config/pd.py`
  - `.loss_metrics` — **[kept]**. jax: `param_decomp_config/pd.py`
    - `.validation` — **[kept]**. jax: `param_decomp_config/pd.py`
  - `.training` — **[kept]**. jax: `param_decomp_config/pd.py`
    - `.optimizers` — **[kept]**. jax: `param_decomp_config/pd.py`
  - `.faithfulness_warmup` — **[kept]** defaults unchanged. jax: `param_decomp_config/pd.py`
- `optimizer_config` — **[kept]** byte-identical. jax: `param_decomp_config/pd.py:OptimizerConfig`
- `any_loss_metric_config` — **[changed]** membership changed (PersistentPGDSubset dropped, ChunkwiseSubset added); moved to losses.py. jax: `param_decomp_config/pd.py`, `losses.py`
  - `.faithfulness` — **[changed]** schema kept; impl native. jax: `losses.py:FaithfulnessLossConfig` + `jax_single_pool/{recon,losses}.py`
  - `.importance_minimality` — **[changed]** schema kept; impl native. jax: `losses.py` + `jax_single_pool/{recon,losses}.py`
  - `.recon_unmasked` — **[changed]** → ConstantSources(1.0). jax: `losses.py` + `recon.py`
  - `.recon_ci_masked` — **[changed]** 3 configs kept → ReconLossTerms. jax: `losses.py` + `recon.py`
  - `.recon_stochastic` — **[changed]** 3 of 4 kept+impl; HiddenActs deferred. jax: `losses.py` + `recon.py`
  - `.recon_pgd` — **[changed]** 3 kept → FreshPGDSources. jax: `losses.py` + `recon.py`
  - `.recon_persistent_pgd` — **[changed]** 2 classes → 1 + scope; sc/bsc only. jax: `losses.py` + `recon.py`
- `runtime_config` — **[changed]** moved; `dp` broadened; `remat_recon_forwards` added. jax: `param_decomp_config/pd.py:RuntimeConfig`
  - `.validation` — **[kept]** byte-identical. jax: `param_decomp_config/pd.py`
- `cadence` — **[changed]** moved; `dense_log_phase` added. jax: `param_decomp_config/pd.py:Cadence`
  - `.checkpoint_retention` — **[changed]** docstring → orbax. jax: `param_decomp_config/pd.py`
  - `.schedule_predicates` — **[changed]** should_log_train honors dense_log_phase. jax: `param_decomp_config/pd.py`

### `param_decomp/base_config.py`

- `base_config` — **[kept]** relocated verbatim. jax: `param_decomp_config/base.py`
  - `.config_base_class` — **[kept]**. jax: `param_decomp_config/base.py`
    - `.strict_immutable_model` — **[kept]**. jax: `param_decomp_config/base.py`
    - `.load_from_file` — **[kept]**. jax: `param_decomp_config/base.py`
      - `.format_dispatch` — **[kept]**
      - `.validation_error_annotation` — **[kept]**
    - `.serialize_to_file` — **[kept]**
      - `.format_dispatch` — **[kept]**
  - `.probability_type` — **[kept]**. jax: `param_decomp_config/base.py`
  - `.runtime_cast` — **[kept]**. jax: `param_decomp_config/base.py`

### `param_decomp/distributed.py`

- `distributed` — **[replaced]** whole file deleted (#876); DDP → GSPMD/SPMD. jax: `sharding.py`
  - `.state` — **[replaced]** → `init_distributed` + process_index/count. jax: `sharding.py`
    - `.snapshot_type` — **[dropped]** DistributedState dataclass; facts via jax APIs.
    - `.module_cache` — **[dropped]**
    - `.import_time_gate` — **[replaced]** → runtime SLURM check. jax: `sharding.py`
    - `.get_distributed_state` — **[dropped]**
  - `.role_queries` — **[replaced]** → inline process_index comparisons. jax: `run.py`
    - `.is_distributed` — **[replaced]** → init_distributed bool. jax: `sharding.py`
    - `.is_main_process` — **[replaced]** → `is_main = process_index()==0`. jax: `run.py`
    - `.is_local_main_process` — **[dropped]** no surviving consumer.
  - `.collectives` — **[replaced]** → jit-inserted GSPMD collectives. jax: `sharding.py`
    - `.sync_across_processes` — **[dropped]**
    - `.all_reduce` — **[replaced]** implicit dp all-reduce. jax: `sharding.py`
    - `.broadcast_tensor` — **[dropped]** NamedSharding replicate.
    - `.gather_all_tensors` — **[dropped]** no multi-pool reduction-group.
  - `.metric_reduction` — **[replaced]** mean over sharded batch in-graph. jax: `sharding.py`
    - `.sum_metrics_across_ranks` — **[dropped]**
    - `.avg_metrics_across_ranks` — **[dropped]**
  - `.seeding` — **[replaced]** → functional PRNG (PRNGKey/fold_in). jax: `run.py`
    - `.seed_all_ranks` — **[replaced]** → PRNGKey(cfg.seed). jax: `run.py`
    - `.seed_per_rank` — **[replaced]** → key splitting + ShardServer process sharding. jax: `run.py`

### `param_decomp/torch_helpers.py`

- `autocast` — **[dropped]** file deleted (#872); JAX is bf16-native.
  - `.bf16_autocast` — **[dropped]** consumers gone.
- `dataloader` — **[dropped]** torch DataLoader utils; JAX has own pipeline.
  - `.loop_dataloader` — **[dropped]**
    - `.epoch_advance_on_exhaustion` — **[dropped]**
- `device_resolution` — **[dropped]** meaningless under JAX.
  - `.CanGetDevice_type` — **[dropped]**
  - `.collect_devices` — **[dropped]**
  - `.get_obj_device` — **[dropped]**

### `param_decomp/log.py`

- `logger` — **[kept]** byte-identical (blob 518a242e). jax: `param_decomp/log.py`
  - `.custom_class` — **[kept]**
    - `.values` — **[kept]**
    - `.section` — **[kept]**
    - `.set_format` — **[kept]**
- `setup` — **[kept]**
  - `.handlers` — **[kept]**
  - `.logfile_path` — **[kept]**
- `formats` — **[kept]**

### `param_decomp/tests`

- `component_model` — **[replaced]** test_component_model.py deleted; → per-target JAX tests + goldens. jax: `tests/test_llama8b.py`
  - `.target_wrapping` — **[replaced]** frozen/vu split structural. jax: `tests/test_llama8b.py`
  - `.unsupported_module` — **[dropped]** no generic make_components raise.
  - `.weight_deltas` — **[replaced]** → `weight_deltas` pinned by stacked_parity. jax: `tests/stacked_parity/test_stacked_parity.py`
  - `.forward_equivalence` — **[replaced]** → clean==all-frozen contract. jax: `tests/test_llama8b.py`
  - `.input_cache` — **[dropped]** subsumed by pure `site_inputs`.
  - `.replacement_scaling` — **[dropped]** covered by perfect-init recon checks.
  - `.routing` — **[replaced]** → recon plan tests. jax: `recon.py`, `tests/equivalence/`
  - `.identity_replacement` — **[dropped]** no Identity concept.
  - `.checkpoint_io` — **[replaced]** → orbax round-trip. jax: `tests/test_checkpoint.py`
    - `.ci_config_mismatch` — **[replaced]** → fine-tune structural-compat guard. jax: `tests/test_finetune_resume.py`
- `ci_fns` — **[replaced]** torch CI-fn family deleted. jax: `ci_fn.py`
  - `.parallel_linear` — **[dropped]** ParallelLinear building block gone.
  - `.layerwise_mlp` — **[replaced]** → layerwise MLP shape tests. jax: `tests/test_resid_mlp.py`
  - `.global_shared_mlp` — **[dropped]**
    - `.gradient_flow` — **[dropped]** → custom-VJP grad check. jax: `tests/test_lower_leaky_hard_grad.py`
  - `.global_shared_transformer` — **[dropped]**
  - `.global_ci_integration` — **[replaced]** piecewise across JAX suite. jax: `tests/test_checkpoint.py`
- `identity_insertion` — **[dropped]** test_identity_insertion.py deleted.
  - `.layer_creation` — **[dropped]**
  - `.hooks` — **[dropped]**
  - `.output_preservation` — **[dropped]**
  - `.error_handling` — **[dropped]**
- `masks` — **[replaced]** test_masks.py deleted; sampling moved to recon.py. jax: `recon.py`
  - `.rand_perm` — **[dropped]**
  - `.uniform_k_subset_routing` — **[replaced]** → equivalence/structural coverage. jax: `recon.py`
- `schedule` — **[kept]** ONLY main test surviving in place; import path only. jax: `param_decomp/tests/test_schedule.py`
  - `.constant_linear_cosine` — **[kept]**
  - `.warmup` — **[kept]**
  - `.edge_cases` — **[kept]**
- `optimize` — **[replaced]** test_optimize.py deleted. jax: `tests/test_checkpoint.py`
  - `.config_validation` — **[kept]** → JAX config-build tests. jax: `tests/test_config.py`
  - `.grad_norm_logging` — **[replaced]** NaN-logging case not carried; clip parity pinned. jax: `tests/test_optim_torch_parity.py`
  - `.eval_loop` — **[replaced]** → eval-step tests. jax: `tests/test_eval.py`
  - `.seeding` — **[replaced]** → PRNGKey determinism. jax: `tests/test_eval.py`
  - `.snapshot_resume` — **[replaced]** → orbax exact-trajectory continuation. jax: `tests/test_checkpoint.py`
- `loss_metrics` — **[replaced]** all torch metric tests deleted; → equivalence/stacked_parity goldens. jax: `tests/equivalence/test_equivalence.py`
  - `.fixtures` — **[replaced]** → per-target builders + committed npz fixtures. jax: `tests/equivalence/gen_fixtures.py`
  - `.faithfulness` — **[replaced]** parity pinned ~1e-4. jax: `losses.py`
  - `.importance_minimality` — **[replaced]** Jensen + device-invariance pinned. jax: `tests/test_imp_min_global_reduction.py`
  - `.ci_masked_recon` — **[replaced]** KL parity golden. jax: `recon.py`
  - `.ci_masked_recon_layerwise` — **[replaced]** chunk-plan structural. jax: `recon.py`
  - `.ci_masked_recon_subset` — **[replaced]** routing structural. jax: `recon.py`
  - `.stochastic_recon` — **[replaced]** parity pinned (S10). jax: `tests/equivalence/test_equivalence.py`
  - `.stochastic_recon_layerwise` — **[replaced]** chunked-plan coverage. jax: `recon.py`
  - `.stochastic_recon_subset` — **[replaced]** static-live gate. jax: `recon.py`
  - `.recon_loss_sanity` — **[replaced]** subsumed by goldens. jax: `tests/equivalence/test_equivalence.py`
  - `.pgd_recon` — **[replaced]** cscope DP-invariance + source-grad-mean tests. jax: `tests/test_fresh_pgd_cscope_dp_invariance.py`
  - `.hidden_acts_recon` — **[deferred]** recon is KL-on-logits only; hidden-acts a deferred recon variant.
  - `.persistent_pgd_recon` — **[replaced]** mask/broadcast pinned by equivalence. jax: `adversary.py`
  - `.persistent_pgd_state` — **[replaced]** → resume Adam bias-correction tests. jax: `tests/test_persistent_ascent_resume_bias_correction.py`

---

## Loss metrics (`param_decomp/metrics`)

### `base.py`

- `metric_abc` — **[replaced]** Metric[TConfig] ABC deleted; → pure fns + static plan. jax: `losses.py`, `recon.py`, `train.py`
  - `.class_metadata` — **[replaced]** ClassVars → data/conventions (_METRIC_KEYS, slow_eval path, name field). jax: `run.py`, `slow_eval.py`, `losses.py`
  - `.lifecycle.construct` — **[dropped]** no metric object.
  - `.lifecycle.bind` — **[dropped]** closure capture at step-factory time.
  - `.lifecycle.reset` — **[replaced]** → functional accumulators (slow_eval). jax: `slow_eval.py`, `hidden_acts_eval.py`
  - `.lifecycle.update` — **[replaced]** → per-term loss in jitted loss_fn + offline accumulate. jax: `train.py`, `slow_eval.py`
  - `.lifecycle.compute` — **[replaced]** → step dict + SlowEvalOutput. jax: `train.py`, `slow_eval.py`
  - `.backward_hooks.before_backward` — **[replaced]** sources as differentiated leaf (S14). jax: `train.py`
  - `.backward_hooks.after_backward` — **[replaced]** → explicit post-grad ascent (S13'/S14'/S23). jax: `train.py`, `adversary.py`
  - `.checkpoint_state.state_dict` — **[replaced]** → unified TrainState.sources. jax: `train.py`, `checkpoint.py`, `run_state.py`
  - `.checkpoint_state.load_state_dict` — **[replaced]** → orbax restore-onto-reference. jax: `checkpoint.py`, `run_state.py`
- `loss_metric_config` — **[kept]** already in torch-free package; `name` field added. jax: `param_decomp_config/losses.py`
- `metric_result_type` — **[dropped]** Tensor|Mapping union; → concrete dict[str,Array] / SlowEvalOutput.

### `dispatch.py`

- `registry` — **[replaced]** metrics/ tree deleted; → `build_recon_terms` structural mapping. jax: `recon.py`, `losses.py`
  - `.class_name_keying` — **[dropped]** → structural match on config type; keyed by instance_key.
  - `.recon_loss_family` — **[replaced]** → strategy×plan×routing product. jax: `recon.py`
    - `.ci_masked` — **[replaced]** → ConstantSources(0.0). jax: `recon.py:331-345`
    - `.pgd_masked` — **[replaced]** → FreshPGDSources. jax: `recon.py:377-388`
    - `.persistent_pgd` — **[changed]** → PersistentSources; subset variant NOT wired. jax: `recon.py:389-401`, `adversary.py`
    - `.stochastic` — **[changed]** → StochasticSources; ChunkwiseSubset config has no torch metric. jax: `recon.py:346-376`
      - `.hidden_acts` — **[deferred]** raises as a training loss; → standalone offline eval (S31). jax: `hidden_acts_eval.py`
    - `.unmasked` — **[replaced]** → ConstantSources(1.0). jax: `recon.py:331-336`
  - `.non_recon_losses` — **[changed]** faith/imp-min → pure fns in LossSpec. jax: `losses.py`, `recon.py`, `train.py:415-416,559`
- `instantiate_metrics` — **[changed]** → `build_recon_terms(...) -> LossSpec`; no model/bind. jax: `recon.py`
  - `.loss_metric_construction` — **[changed]** → match-over-cfg; eager config validation. jax: `recon.py`, `config.py:_losses`
    - `.duplicate_guard` — **[kept]** re-keyed by instance_key (`assert_unique_instance_key`). jax: `recon.py`
    - `.model_binding` — **[dropped]** no bind/device.
  - `.eval_metric_binding` — **[dropped]** no caller-supplied eval-Metric set.
    - `.optional_eval_set` — **[dropped]**
    - `.duplicate_guard` — **[dropped]**
  - `.collision_rejection` — **[dropped]** no merged loss+eval set.
  - `.result_assembly` — **[changed]** → LossSpec(faith/imp/recon_terms/persistent). jax: `recon.py`

### `context.py`

- `metric_context` — **[replaced]** MetricContext bundle deleted; → explicit step args. jax: `train.py`, `eval.py`
  - `.model_and_batch_carriers` — **[replaced]** → closed-over lm + frozen/residual args. jax: `train.py`, `eval.py:136,190`
  - `.activation_and_ci_state` — **[replaced]** → recomputed inline (site_inputs/CIValues/weight_deltas). jax: `train.py`, `ci_fn.py`, `losses.py`
  - `.masking_config` — **[changed]** use_delta hardwired on; n_mask_samples/sampling → build_recon_terms. jax: `config.py:505`, `recon.py`, `train.py`
  - `.reconstruction_loss_fn` — **[changed]** → `lm.recon_loss_fn` + LossSpec. jax: `train.py`, `recon.py`, `losses.py`
  - `.training_progress` — **[changed]** step→TrainState; is_eval flag dropped (separate entry points). jax: `train.py`, `losses.py`, `eval.py`
    - `.current_frac_of_training` — **[changed]** → inline step/total_steps; 0-guard not reproduced. jax: `losses.py`, `train.py`

### `faithfulness.py`

- `faithfulness` — **[changed]** split config + native fn. jax: `param_decomp_config/losses.py` + `jax_single_pool/losses.py`
  - `.config` — **[kept]** relocated. jax: `param_decomp_config/losses.py:40-41`
  - `.pure_loss_fn` — **[changed]** reimplemented (S17), fp32 cast, jaxtyped. jax: `losses.py:28-37`
    - `.device_inference` — **[dropped]** GSPMD placement.
    - `.nonempty_assert` — **[dropped]** non-empty implicit via sites.
    - `.global_param_normalization` — **[kept]** per-param normalization. jax: `losses.py:36`
  - `.metric` — **[replaced]** Metric class gone; inline in step, logged `train/loss/FaithfulnessLoss`. jax: `train.py:415,474`, `run.py:146`
    - `.reset` — **[dropped]**
    - `.update` — **[replaced]** → direct call. jax: `train.py:415`
    - `.compute` — **[dropped]**
    - `.distributed_aggregation` — **[replaced]** → GSPMD in-graph reduction. jax: `train.py`, `losses.py:48-50`

### `importance_minimality.py`

- `importance_minimality` — **[changed]** → `importance_minimality_terms` pure fn. jax: `losses.py`
  - `.config` — **[kept]**. jax: `param_decomp_config/losses.py`
    - `.lp_penalty_params` — **[kept]**
    - `.p_annealing_schedule` — **[kept]**
  - `.p_annealing` — **[changed]** → `annealed_pnorm`; fail-fast on missing final_p (S9). jax: `losses.py:64-70`
  - `.per_component_sums` — **[changed]** folded inline; n-examples implicit via GSPMD (S7/S8). jax: `losses.py:53-60`
  - `.finalize` — **[changed]** world_size factor dropped (GSPMD global sum, D2). jax: `losses.py:59-60`
    - `.lp_term` — **[kept]**. jax: `losses.py`
    - `.entropy_like_term` — **[changed]** world_size arg dropped. jax: `losses.py:60`
  - `.functional_api` — **[changed]** split into annealed_pnorm + terms; no wrapper. jax: `losses.py`
  - `.metric` — **[dropped]** stateful Metric subclass gone.
    - `.reset` — **[dropped]**
    - `.update` — **[changed]** → inline call. jax: `train.py:416-417`
    - `.compute` — **[dropped]**
  - `.distributed` — **[replaced]** → GSPMD; guarded by test. jax: `losses.py` + `tests/test_imp_min_global_reduction.py`

### `output.py`

- `metric_output_normalization` — **[replaced]** whole package deleted; flat dict from jitted step. jax: `train.py:531-543`, `run.py:412`
  - `.single_result_cleaning` — **[dropped]** no per-metric dispatch.
    - `.scalar_tensor_case` — **[replaced]** → uniform float(v). jax: `run.py:412`, `train.py:536`
    - `.dict_case` — **[dropped]** keys emitted pre-namespaced.
    - `.non_tensor_passthrough` — **[dropped]** homogeneous scalar map.
    - `.unsupported_type_rejection` — **[dropped]**
  - `.multi_metric_collection` — **[replaced]** → dict-literal assembly. jax: `train.py:531-543`
    - `.key_collision_detection` — **[dropped]** no disjoint-key assert (last-wins).
  - `.type_alias` — **[replaced]** MetricOutType → concrete dict[str,Array]/dict[str,float]. jax: `train.py:93`, `run.py:195`

### `unmasked_recon.py`

- `unmasked_recon_loss` — **[replaced]** deleted; → ConstantSources(1.0) recon term. jax: `recon.py`, `train.py`
  - `.config` — **[kept]** identity unchanged. jax: `param_decomp_config/losses.py`
  - `.all_ones_masking` — **[changed]** → CI-derived value=1 (no module_to_c buffer). jax: `train.py`, `recon.py`
    - `.device_placement` — **[dropped]** GSPMD.
  - `.forward_and_loss` — **[changed]** → checkpointed_masked_forward + kl_per_position; one recon semantics. jax: `train.py`, `losses.py`
  - `.metric_lifecycle` — **[replaced]** → ReconLossTerm; key loss/UnmaskedReconLoss. jax: `recon.py`, `train.py`, `tests/test_recon_log_keys.py`
    - `.reset` — **[dropped]**
    - `.update` — **[replaced]** → per-step mean. jax: `train.py`
    - `.compute` — **[dropped]**
  - `.distributed_reduction` — **[replaced]** → GSPMD. jax: `losses.py`, `train.py`

### `ci_masked_recon.py` (+ `_layerwise`, `_subset`)

- `ci_masked_recon` — **[replaced]** module gone; → ReconLossTerm ConstantSources(0.0). jax: `recon.py`, `train.py`, `LOSS_PARITY_DESIGN.md`
  - `.config` — **[kept]**. jax: `param_decomp_config/losses.py`
  - `.core_update` — **[replaced]** inlined in jitted loss_fn. jax: `train.py`
    - `.mask_construction` — **[changed]** → constant_entry_masks; has_delta=False static skip. jax: `train.py`, `recon.py`
    - `.masked_forward` — **[changed]** → `masked_output` pure fn. jax: `lm.py`, `train.py`
    - `.recon_scoring` — **[changed]** → `recon_loss_fn`=kl_per_position; one semantics. jax: `losses.py`, `lm.py`, `train.py`
  - `.direct_helper` — **[dropped]** convenience wrapper not ported.
  - `.metric` — **[replaced]** → ReconLossTerm; loss/<name> key. jax: `recon.py`, `train.py`, `ci_fn.py`
    - `.reset` — **[dropped]**
    - `.update` — **[changed]** inline per-term. jax: `train.py`
      - `.mask_source_selection` — **[kept]** ci.lower (lower_leaky). jax: `ci_fn.py`, `train.py`
    - `.compute` — **[replaced]** → GSPMD in-graph. jax: `losses.py`, `train.py`

### `stochastic_recon.py` (+ `_layerwise`, `_subset`, `_hidden_acts`)

- `stochastic_recon` — **[replaced]** module gone; → StochasticSources recon term. jax: `recon.py`, `train.py`
  - `.config` — **[kept]**. jax: `param_decomp_config/losses.py:77`
  - `.core_accumulation` — **[replaced]** Python loop → per-draw loop in jitted step. jax: `train.py:419-465`, `recon.py:347-354`
    - `.mask_sampling` — **[replaced]** → `stochastic_entry_masks`. jax: `train.py:197-221`
    - `.masked_forward` — **[replaced]** → `masked_forward`/`checkpointed_masked_forward`. jax: `train.py:171-195`
    - `.weight_delta_masking` — **[changed]** delta always live for stochastic (strategy_has_delta). jax: `recon.py:107-112`, `train.py:218-220`
    - `.recon_loss_eval` — **[replaced]** pluggable → fixed kl_per_position. jax: `losses.py:14`, `train.py`
  - `.standalone_helper` — **[dropped]** no functional entry; covered by goldens.
  - `.metric` — **[replaced]** → ReconLossTerm. jax: `recon.py:136`, `train.py:536`
    - `.reset` — **[dropped]**
    - `.update` — **[replaced]** inline. jax: `train.py:419-472`
      - `.delta_gating` — **[replaced]** → static has_delta. jax: `recon.py:107-129`
    - `.compute` — **[changed]** → mean + dp-mesh reduction. jax: `train.py:465`

### `pgd_masked_recon.py` (+ `_layerwise`, `_subset`)

- `pgd_recon_loss_metric` — **[changed]** survives as fresh-PGD recon TERM, not a metric object. jax: `recon.py`, `adversary.py`, `train.py`
  - `.config` — **[changed]** mask_scope renamed (unique_per_datapoint→bsc, shared_across_batch→c); siblings added. jax: `param_decomp_config/losses.py`
  - `.metric_class` — **[dropped]** Metric subclass + lifecycle gone.
    - `.reset` — **[dropped]**
    - `.update` — **[changed]** inline term. jax: `train.py`
      - `.delta_component_gating` — **[changed]** → static has_delta. jax: `recon.py`, `adversary.py`
    - `.compute` — **[dropped]** GSPMD reduction.
  - `.direct_helper` — **[dropped]** no wrapper.
  - `.all_layers_routing` — **[changed]** AllLayersRouter → AllRoutingConfig. jax: `recon.py`
  - `.per_step_pgd` — **[changed]** → `sign_ascend_body` lax.scan + init_fresh_pgd_sources. jax: `train.py`, `adversary.py`
    - `.adversarial_source_init` — **[changed]** → init_fresh_pgd_sources (+1 delta channel always). jax: `adversary.py:55-86`
      - `.init_strategy` — **[changed]** inlined match. jax: `adversary.py:79-86`
      - `.mask_scope` — **[changed]** 2-way → c/bc/bsc; broadcast via GSPMD. jax: `adversary.py`, `losses.py (MaskScope)`
    - `.inner_pgd_loop` — **[changed]** → lax.scan; grad-clean assert dropped. jax: `train.py`
      - `.shared_grad_reduce` — **[changed]** explicit AVG → GSPMD sign-invariance. jax: `train.py`, `tests/test_fresh_pgd_cscope_dp_invariance.py`
    - `.mask_construction` — **[changed]** → `source_masks` plain dicts. jax: `adversary.py`, `train.py`
      - `.weight_delta_split` — **[changed]** unconditional delta channel. jax: `adversary.py`
    - `.masked_forward_and_recon` — **[changed]** → masked_output + kl_per_position. jax: `train.py`, `lm.py`, `losses.py`

### `persistent_pgd_recon.py`

- `config` — **[replaced]** config classes → torch-free package. jax: `param_decomp_config/losses.py:253`
  - `.shared_base` — **[changed]** base dissolved; subset dropped. jax: `losses.py:253-274`
    - `.optimizer` — **[kept]** PGDOptimizerConfig; only AdamPGD supported. jax: `losses.py:263,177-194`
    - `.scope` — **[kept]** shape-spelled union; sc/bsc only. jax: `losses.py:264,197-246`
    - `.sigmoid_parameterization` — **[deferred]** kept in config, refused at runtime. jax: `losses.py:265`, `recon.py:293`
    - `.warmup_steps` — **[kept]** lax.scan warmup. jax: `losses.py:266`, `train.py:282-345`
    - `.start_frac` — **[changed]** None-return → in-graph where-gating (S32). jax: `losses.py:273`, `train.py`
    - `.n_samples` — **[kept]** → per-chunk draw count. jax: `losses.py:274`, `recon.py:393-398`
  - `.all_layers_variant` — **[kept]** sole PPGD config. jax: `losses.py:253-274`, `pd.py:62`
  - `.subset_variant` — **[dropped]** gone entirely; composition-only future.
- `router_selection` — **[replaced]** → routing-sampler dispatch. jax: `recon.py:204-213,393-395`
- `scope_validation` — **[replaced]** → GSPMD sharding + scope refusal asserts. jax: `run_state.py:108-126`, `recon.py:289-297`
- `metric.base` — **[replaced]** generic base → ReconLossTerm + PersistentSources. jax: `recon.py`, `train.py`
  - `.lazy_state_construction` — **[replaced]** eager init at run start. jax: `run_state.py:108-134`
  - `.reset` — **[dropped]** pure step, no accumulators.
  - `.update` — **[replaced]** → term contribution in jitted step. jax: `train.py`, `recon.py:135-146`
    - `.start_frac_gate` — **[changed]** → in-graph where (S32). jax: `train.py:288-298,466-471,502-508`
    - `.lr_schedule_step` — **[changed]** → pure `warmup_then_constant_lr`. jax: `losses.py:73-77`, `train.py`
    - `.warmup_drive` — **[kept]** lax.scan warmup, route-all (Q1). jax: `train.py:306-345`
    - `.delta_component_routing` — **[changed]** → static has_delta; delta always live for PPGD. jax: `recon.py:107-129`, `adversary.py:128-141`
    - `.live_recon_loss` — **[changed]** → kl_per_position over plan draws. jax: `losses.py:13-25`, `train.py`
  - `.eval_accumulation` — **[replaced]** → offline slow-eval. jax: `slow_eval.py`
    - `.hidden_acts_mse` — **[dropped]** from recon path; → standalone offline metric (S31). jax: `hidden_acts_eval.py`
  - `.compute` — **[replaced]** → step metrics dict. jax: `train.py`, `recon.py:138-139`
    - `.distributed_reduce` — **[replaced]** → GSPMD in-graph. jax: `losses.py:13-25,48-50`
  - `.backward_orchestration` — **[replaced]** → single fused value_and_grad. jax: `train.py:404-509`
    - `.source_grads` — **[changed]** → differentiated leaf (S14'). jax: `train.py:404-484`
    - `.source_step` — **[changed]** → inline final ascent /coeff (S14/S23). jax: `train.py:485-509`, `adversary.py:97-125`
  - `.checkpointing` — **[changed]** → orbax of TrainState.sources/opt_state. jax: `run_state.py:127-134`, `adversary.py:29-34`
    - `.deferred_resume` — **[dropped]** eager state, no defer dance.
- `metric.all_layers` — **[replaced]** → ReconLossTerm one_chunk+AllRouting+PersistentSources. jax: `recon.py:388-399`
- `metric.subset` — **[dropped]** dropped with its config; composition-only.

### `persistent_pgd_state.py`

- `optimizer_config` — **[kept]** PGDOptimizerConfig in config package. jax: `param_decomp_config/losses.py`
  - `.sign` — **[kept]** (deferred for persistent; live for fresh). jax: `param_decomp_config/losses.py`
  - `.adam` — **[kept]**. jax: `param_decomp_config/losses.py`
- `source_scope` — **[changed]** members renamed shape-spelled + legacy aliases. jax: `param_decomp_config/losses.py:197-251`
  - `.single_source` → **[changed]** CScope; deferred for persistent. jax: `losses.py (CScope)`
  - `.broadcast_across_batch` → **[changed]** SCScope; replicated over dp. jax: `losses.py (SCScope)`, `llama8b_sharding.py:138-175`
  - `.repeat_across_batch` → **[deferred]** NSCScope; refused at runtime. jax: `losses.py (NSCScope)`
  - `.per_batch_per_position` → **[changed]** BSCScope; batch-sharded. jax: `losses.py (BSCScope)`, `llama8b_sharding.py`
- `optimizer` — **[replaced]** ABC gone; → pure fns over SourcesAdamState. jax: `adversary.py`
  - `.sign_impl` — **[replaced]** → `sign_ascend_body`. jax: `train.py`
  - `.adam_impl` — **[replaced]** → `sources_adam_ascend_project`. jax: `adversary.py`
    - `.checkpoint` — **[replaced]** → TrainState leaf via orbax. jax: `adversary.py`, `checkpoint.py`
  - `.factory` — **[replaced]** dispatch collapsed (Adam persistent, sign fresh). jax: `train.py`, `recon.py`
- `state` — **[replaced]** mutable state → TrainState + pure fns. jax: `train.py`, `adversary.py`, `recon.py`
  - `.source_init` — **[changed]** → init_persistent/fresh; U[0,1] only (sigmoid deferred). jax: `adversary.py`, `llama8b_sharding.py`
  - `.gradient` — **[replaced]** autograd+AVG → jax.grad + GSPMD. jax: `train.py`, `llama8b_sharding.py`
  - `.step` — **[replaced]** → pure ascend-project. jax: `adversary.py`, `train.py`
  - `.effective_sources` — **[dropped]** sigmoid mapping deferred; clamp-only.
  - `.lr_schedule` — **[changed]** → `warmup_then_constant_lr`. jax: `losses.py`, `train.py`
  - `.checkpoint` — **[replaced]** → structural orbax. jax: `checkpoint.py`, `train.py`
  - `.warmup` — **[kept]** → lax.scan warmup_body, route-all (S24). jax: `train.py`
  - `.recon_scoring` — **[changed]** (sum,n) → batch-mean. jax: `train.py`
- `mask_construction` — **[changed]** → `source_masks` plain dicts; routing separate. jax: `adversary.py`, `train.py`, `recon.py`
  - `.source_expansion` — **[changed]** → pure broadcasting; nsc-repeat not implemented. jax: `adversary.py`
  - `.delta_channel_split` — **[kept]** trailing-channel split verbatim. jax: `adversary.py`
  - `.interpolation` — **[kept]** ci+(1-ci)*source verbatim. jax: `adversary.py`
- `recon_forward` — **[changed]** → source_masks + masked_forward + recon_loss_fn; scalar mean. jax: `train.py`, `lm.py`

### `pgd_utils.py`

- `per_step_pgd` — **[replaced]** reimplemented in adversary.py + train.py. jax: `adversary.py`, `recon.py`, `train.py:349-402`
  - `.config` — **[kept]** PGDConfig in config package. jax: `param_decomp_config/losses.py:153`, `recon.py:377-387`
    - `.init_strategy` — **[kept]** PGDInitStrategy. jax: `param_decomp_config/losses.py:125`, `adversary.py:79-86`
    - `.mask_scope` — **[changed]** shape-spelled c/bc/bsc + legacy alias. jax: `param_decomp_config/losses.py:125-159`
  - `.init_tensor` — **[replaced]** folded into init_fresh_pgd_sources. jax: `adversary.py:79-86`
  - `.init_sources` — **[replaced]** → init_fresh_pgd_sources (+1 always). jax: `adversary.py:55-86`
    - `.scope_unique_per_datapoint` — **[changed]** → bsc. jax: `adversary.py:68-74`
    - `.scope_shared_across_batch` — **[changed]** → c (+ new bc). jax: `adversary.py:73-74`
  - `.inner_loop` — **[replaced]** → lax.scan; (sum,n) re-eval dropped. jax: `train.py:385-402`
    - `.grad_step` — **[kept]** sign-ascent. jax: `train.py:391-399`
    - `.projection` — **[kept]** clip[0,1] (S15). jax: `train.py:393-397`
    - `.grad_reduction_shared` — **[replaced]** → GSPMD. jax: `train.py:264-273`
    - `.fresh_grad_assert` — **[dropped]** functional grad has no stale .grad.
  - `.mask_construction` — **[replaced]** → `source_masks` + Routes. jax: `adversary.py:128-141`
    - `.broadcast_expand` — **[replaced]** → jax broadcasting. jax: `adversary.py:128-141`
    - `.weight_delta_split` — **[kept]** trailing channel. jax: `adversary.py:139-140`
  - `.forward` — **[replaced]** → entry_loss_for_sources + masked_forward. jax: `train.py:235-262,442-462`, `lm.py`
  - `.entrypoint` — **[replaced]** no metric-update driver; split across step phases. jax: `train.py:349-402,420-477`, `recon.py:300-414`

---

## Experiments (decomposition drivers)

### `experiments/tms`

- `target_model` — **[replaced]** torch autoencoder → eqx `TMSTarget` + DecomposedModel. jax: `tms.py`
  - `.config_schema` — **[changed]** split torch-free config + flat dataclass; device/init_bias dropped. jax: `param_decomp_config/tms.py`, `config.py`
  - `.tied_weights` — **[changed]** tie reconstructed functionally; decomposition untied. jax: `tms.py`
  - `.hidden_layers` — **[dropped]** n_hidden_layers>0 refused. 
  - `.bias_init` — **[dropped]** b2 always zero.
  - `.checkpoint_loading` — **[replaced]** → in-process pretrain. jax: `tms.py`
  - `.run_info` — **[dropped]** no run-dir resolution.
- `dataset` — **[replaced]** IterableDataset → pure `sample_sparse_features`. jax: `tms.py`, `run.py::train_tms`
  - `.independent_sparse` — **[kept]** at_least_zero_active. jax: `tms.py`
  - `.exactly_n_active` — **[changed]** only exactly_one_active survives. jax: `tms.py`, `param_decomp_config/tms.py`
  - `.synced_inputs` — **[dropped]**
  - `.no_zero_sample_rejection` — **[dropped]**
- `target_training` — **[replaced]** standalone driver → `pretrain_tms_target` in-process. jax: `tms.py`
  - `.config_schema` — **[changed]** → minimal TMSPretrainConfig. jax: `param_decomp_config/tms.py`
  - `.fixed_hidden_layers` — **[dropped]**
  - `.run_naming_and_artifacts` — **[dropped]** ephemeral pretrain.
  - `.feature_representation_analysis` — **[dropped]**
  - `.visualization.polygon` — **[dropped]**
  - `.visualization.cosine_similarity` — **[dropped]**
- `decomposition_run` — **[replaced]** pd-tms → `train_tms` over unified core. jax: `run.py::train_tms`
  - `.config_schema` — **[changed]** target pretrained from scratch. jax: `param_decomp_config/tms.py`
  - `.build_target` — **[replaced]** → tms_decomposed_model + pretrain. jax: `run.py`, `tms.py`
  - `.data_loader` — **[replaced]** → in-loop sample_residual. jax: `run.py::train_tms`
  - `.run_batch` — **[replaced]** identity residual. jax: `tms.py::tms_input_residual`
  - `.tied_weights_wiring` — **[changed]** always untied; tied_weights:null. jax: `configs/tms_5-2.yaml`
  - `.eval_loop` — **[replaced]** → in-loop identity_ci_error. jax: `run.py::train_tms`, `tms.py`
  - `.saved_run_reload` — **[deferred]** torch→JAX run-adapter.
- `configs` — **[changed]** 4 YAMLs → tms_5-2 + tms_5-5_SMOKE; -id variants dropped, 40-10 not ported. jax: `configs/tms_5-2.yaml`, `configs/tms_5-5_SMOKE.yaml`
  - `.decomposition_targets` — **[changed]** 5-2 C=20 ported; 40-10/id dropped. jax: `configs/tms_5-2.yaml`
  - `.loss_metrics` — **[changed]** imp-min+stochastic kept; Faithfulness ADDED to list. jax: `configs/tms_5-2.yaml`
  - `.eval_metrics` — **[changed]** battery not wired; only identity_ci_error survives. jax: `tms.py::identity_ci_error`

### `experiments/resid_mlp`

- `target_model` — **[changed]** torch ResidMLP → frozen eqx `ResidMLPTarget` + DecomposedModel. jax: `resid_mlp.py`
  - `.architecture` — **[changed]** → pure fns, right-mult oriented, no resid-vs-readoff toggle. jax: `resid_mlp.py`
    - `.mlp_block` — **[kept]** ResidMLPLayer with optional biases. jax: `resid_mlp.py`
  - `.config` — **[changed]** split config; fixed_random/identity → one bool. jax: `param_decomp_config/resid_mlp.py`, `config.py`
  - `.init` — **[kept]** Kaiming-uniform reimplemented. jax: `resid_mlp.py`
  - `.persistence` — **[replaced]** → in-process pretrain; torch reloads = deferred adapter. jax: `resid_mlp.py`, `param_decomp_config/resid_mlp.py`
    - `.filenames` — **[dropped]**
- `dataset` — **[changed]** → pure `sample_sparse_features`; value_range [-1,1]. jax: `resid_mlp.py`, `run.py::train_resid_mlp`
  - `.batch_generation` — **[changed]** only exactly_one/at_least_zero; synced groups dropped. jax: `resid_mlp.py`
  - `.labels` — **[dropped]** only act_plus_resid for pretrain; abs dropped.
    - `.coeffs` — **[dropped]** trivial unit coeffs.
- `feature_importances` — **[dropped]** geometric importance weighting not ported.
- `target_training` — **[replaced]** standalone → `pretrain_resid_mlp_target`. jax: `resid_mlp.py`, `run.py`
  - `.config` — **[changed]** → minimal ResidMLPPretrainConfig. jax: `param_decomp_config/resid_mlp.py`
  - `.loss` — **[changed]** readoff-only; resid branch dropped. jax: `resid_mlp.py`
  - `.loop` — **[changed]** → jit adamw, no schedule/logging/save. jax: `resid_mlp.py`
  - `.embedding_modes` — **[changed]** learned mode dropped; fixed only. jax: `resid_mlp.py`
  - `.logging` — **[dropped]**
  - `.presets` — **[dropped]** → wrapper YAMLs.
- `pd_decomposition` — **[replaced]** pd-resid-mlp → `train_resid_mlp`. jax: `run.py`
  - `.config` — **[changed]** target carries arch + pretrain block. jax: `param_decomp_config/resid_mlp.py`, `config.py`
  - `.build_target` — **[changed]** → build + pretrain-from-scratch. jax: `run.py`, `resid_mlp.py`
  - `.data_loader` — **[replaced]** → in-step sample_residual. jax: `run.py::train_resid_mlp`, `resid_mlp.py`
  - `.run_batch` — **[replaced]** W_E embedding as prefix-residual. jax: `run.py`, `resid_mlp.py`
  - `.train_loop` — **[changed]** → unified core. jax: `run.py::train_resid_mlp`
  - `.eval_loop` — **[replaced]** → in-loop identity_ci_error. jax: `run.py::train_resid_mlp`, `resid_mlp.py`
  - `.saved_run` — **[deferred]** torch→JAX run-adapter (load_run asserts unwired). jax: `load_run.py`
  - `.cli` — **[replaced]** → jsp-train. jax: `run.py`
  - `.configs` — **[changed]** only 1-layer ported (+SMOKE); 2/3L + global not ported. jax: `configs/resid_mlp_1l.yaml`
    - `.ci_variants` — **[changed]** only layerwise mlp wired; shared_mlp/global unwired. jax: `config.py:366`, `run_state.py`

### `experiments/lm`

- `decomposition_driver` — **[replaced]** Trainer-glue → jsp-train. jax: `run.py`
  - `.config_schema` — **[kept]** LMTargetConfig/LMExperimentConfig read directly; gained fields. jax: `param_decomp_config/lm.py`, `config.py`
    - `.target_spec` — **[changed]** kept; gained HFWeightsInVendored. jax: `param_decomp_config/lm.py`
  - `.target_loading` — **[replaced]** → per-vendored-model loaders. jax: `llama_simple_mlp.py`, `llama8b.py`
  - `.run_batch` — **[dropped]** RunBatch obsolete; logits direct; output_extract knowingly ignored.
  - `.fresh_run` — **[replaced]** → train()/train_tms/train_resid_mlp. jax: `run.py::train`
  - `.resume` — **[replaced]** → orbax restore_latest + O(1) schedule fast-forward. jax: `run.py`, `checkpoint.py`
  - `.eval_loop` — **[replaced]** → jitted fast eval + jsp-slow-eval. jax: `eval.py`, `slow_eval.py`
  - `.slurm_submission` — **[replaced]** --dp → pd-jax-lm (snapshot workspace). jax: `experiments/lm/jax_launch.py`
  - `.cli` — **[replaced]** pd-lm → pd-jax-lm. jax: `experiments/lm/jax_launch.py:cli`
  - `.saved_run_reload` — **[deferred]** SavedLMRun gone; metadata-only via jax_pd adapter; trained-component reload = #10. jax: `adapters/jax_pd.py`, `load_run.py`
- `layerwise` — **[dropped]** pd-lm-layerwise deleted; JAX decomposes all blocks in one run.
  - `.config_expansion` — **[dropped]**
  - `.array_submission` — **[dropped]**
  - `.wandb_grouping` — **[dropped]**
  - `.cli` — **[dropped]**
- `data_loading` — **[replaced]** HF DataLoader → ShardServer over parquet. jax: `data.py`
  - `.config` — **[kept]** LMDataConfig; some fields knowingly ignored. jax: `param_decomp_config/lm.py`, `config.py`
  - `.tokenize_and_chunk` — **[changed]** → public `tokenize_and_concatenate`, offline-only. jax: `experiments/lm/data.py`
  - `.pretokenized` — **[dropped]** runtime branching gone; offline tokenize.
  - `.dataloader` — **[replaced]** → ShardServer/BatchSchedule. jax: `data.py`, `run.py`
  - `.distributed_sharding` — **[replaced]** → deterministic process-index sharding. jax: `data.py`
  - `.collate` — **[dropped]** parquet yields stacked arrays.
- `pretrain` — **[dropped]** entire subtree deleted (#876); reimplement in JAX when needed.
  - `.train_loop` (+ config/lr_schedule/checkpointing/validation/logging) — **[dropped]**
  - `.slurm_submission` — **[dropped]** pd-pretrain gone.
  - `.run_info` (+ wandb_download/legacy_migration/tokenizer_resolution) — **[dropped]**
  - `.models` (gpt2/gpt2_simple/llama/llama_simple) — **[dropped]**; `llama_simple_mlp` — **[replaced]** as ground-truth ref → JAX `llama_simple_mlp.py`. jax: `jax_single_pool/llama_simple_mlp.py`
- `configs.decomposition` — **[changed]** pile/ss edited (scope renames); HF configs orphaned; canonical configs under jax_single_pool/configs. jax: `experiments/lm/*.yaml`, `jax_single_pool/configs/*.yaml`
  - `.layerwise` — **[changed]** orphaned (expander deleted).
- `configs.pretrain` — **[dropped]** all pretrain YAMLs deleted.

### `experiments/lm/layerwise.py`

- `layerwise_launcher` — **[dropped]** deleted #814; replacement model is whole-model single-run via pd-jax-lm.
  - `.cli` — **[dropped]** pd-lm-layerwise console script removed.
  - `.cli.arg_parsing` — **[dropped]**
  - `.config_expansion` (+ pattern_substitution/filtering/block_selection/identity_guard/per_target_rebuild) — **[dropped]**
  - `.run_workspace` — **[replaced]** → single immutable workspace. jax: `experiments/lm/jax_launch.py`
  - `.git_snapshot` — **[replaced]** → mandatory single-run snapshot. jax: `jax_launch.py:84`
  - `.shared_subrun_coordination` — **[replaced]** → single srun command. jax: `jax_launch.py`
  - `.command_generation` (+ single_gpu/ddp) — **[dropped/replaced]** GSPMD srun. jax: `jax_launch.py`
  - `.slurm_submission` (+ array_script/concurrency_cap) — **[replaced]** SlurmConfig single job. jax: `jax_launch.py`
  - `.wandb_workspace_view` — **[dropped]** single run, no sweep view.
  - `.reporting` — **[replaced]** single-run summary. jax: `jax_launch.py`

### `experiments/lm/data.py`

- `config` — **[changed]** LMDataConfig moved to torch-free package. jax: `param_decomp_config/lm.py`
  - `.hub_resolution` — **[changed]** fields kept; runtime HF fetch gone (offline prestage). jax: `param_decomp_config/lm.py`, `prestage_tokenized.py`
  - `.streaming_controls` — **[changed]** kept; buffer/shuffle knowingly ignored (deterministic schedule). jax: `param_decomp_config/lm.py`, `config.py`
  - `.tokenization_mode` — **[changed]** max_seq_len read; is_tokenized dispatch gone. jax: `config.py`
- `dataset_prep` — **[dropped]** `_prepare_lm_dataset` deleted; always pre-tokenized parquet.
  - `.column_pruning` — **[kept]** `_keep_single_column`. jax: `experiments/lm/data.py:15`
  - `.pretokenized_path` — **[replaced]** → ShardServer width-assert+truncate. jax: `data.py:114-130`
  - `.tokenize_and_concatenate` — **[changed]** → public, offline-only. jax: `experiments/lm/data.py:27`
    - `.eos_join_chunking` — **[kept]**
    - `.fixed_length_reshape` — **[kept]**
    - `.bos_prefix` — **[kept]**
    - `.streaming_aware_map` — **[changed]** simplified.
    - `.simplestories_lowercasing` — **[changed]** mechanism kept; auto-enable dropped.
- `dataloader` — **[replaced]** create_lm_data_loader → ShardServer/BatchSchedule. jax: `data.py`, `run.py`
  - `.dataset_loading` — **[replaced]** HF load → scan_shards/pyarrow. jax: `data.py:39,114`
  - `.tokenizer_loading` — **[dropped]** no run-time tokenizer (offline). jax: `prestage_tokenized.py:64`
  - `.shuffling` — **[replaced]** → seeded BatchSchedule permutations. jax: `data.py:67-90`
  - `.distributed_sharding` (+ streaming_shard/map_style_sampler) — **[replaced/dropped]** → per-process slice. jax: `data.py:93,132`
  - `.reproducibility` — **[replaced]** → pure (seed,step) schedule. jax: `data.py:70-90`
  - `.batching_options` — **[replaced]** → consecutive row-windows. jax: `data.py:53,86-90,104`
- `collation` — **[dropped]** ShardServer reads fixed-width matrix.
- `rank_batch_size` — **[replaced]** → ShardServer per_process. jax: `data.py:104-109`

### `experiments/utils.py`

- `experiment_config_schema` — **[changed]** moved to torch-free package; only filename const kept in lab. jax: `param_decomp_config/experiment.py`
  - `.generic_experiment_config` — **[changed]** gained run_name/run_id/out_dir. jax: `param_decomp_config/experiment.py:ExperimentConfig`
    - `.optional_eval_block` — **[kept]**
    - `.optional_wandb_block` — **[kept]**
  - `.wandb_config` — **[changed]** gained group/tags. jax: `param_decomp_config/experiment.py:WandbConfig`
  - `.eval_config` — **[changed]** slow_every multiple rule codified. jax: `param_decomp_config/experiment.py:EvalConfig`
    - `.eval_metric_list` — **[kept]**. jax: `param_decomp_config/experiment.py`
- `run_initialization` (`init_pd_run`) — **[replaced]** dissolved across jax_launch (submit) + run.py (runtime). jax: `jax_launch.py`, `run.py`
  - `.run_id_and_output_dir_allocation` — **[changed]** → submit-time mint+stamp. jax: `jax_launch.py`
  - `.config_persistence` — **[changed]** → _pin_config_copy (config.yaml). jax: `run.py`, `jax_launch.py`, `utils.py`
  - `.ddp_rank_awareness` — **[changed]** → is_main = process_index()==0. jax: `run.py`
  - `.sink_selection.local_sink` — **[replaced]** → single MetricsSink. jax: `run.py`
  - `.sink_selection.wandb_sink` — **[replaced]** → MetricsSink wandb.init; artifact upload dropped. jax: `run.py`
    - `.tag_parsing` — **[changed]** → submit-time parse. jax: `jax_launch.py`, `param_decomp_config/experiment.py`
    - `.group_label` — **[changed]** → config field. jax: `jax_launch.py`, `param_decomp_config/experiment.py`, `run.py`

---

## LM pretraining (`experiments/lm/pretrain`)

(Whole subtree deleted #876 — see also `experiments/lm.pretrain`.)

### `cli.py`

- `pretrain_cli` — **[deferred]** deleted; reimplement in JAX when next needed.
  - `.fire_entrypoint` — **[deferred]** pd-pretrain console script removed.
  - `.config_validation` — **[deferred]**
  - `.dispatch` — **[deferred]**
  - `.local_run` — **[deferred]**
    - `.multi_gpu_torchrun` — **[dropped]** torchrun mechanism gone for good.
    - `.single_gpu_python` — **[dropped]**
  - `.slurm_submit` (+ git_snapshot_stamp/log_dir_prep/script_generation/job_submission_report/partition_time_control) — **[deferred]** infra helpers retained; pretrain call sites deferred. jax: `experiments/lm/jax_launch.py`, `infra/run_files.py`, `infra/slurm.py`

### `train.py`

- All features (`config`, `distributed`, `model`, `data`, `optimization`, `precision`, `reproducibility`, `training_loop`, `evaluation`, `sampling`, `logging`, `checkpointing`, `io`, `entrypoint`, `modes`) — **[deferred]** deleted with the torch trainer; reimplement in JAX. Exceptions:
  - `config.schema` shape fields survive as `LlamaSimpleMLPConfig`. jax: `jax_single_pool/llama_simple_mlp.py`
  - `model.from_pretrained` → JAX pretrain-cache loader. jax: `jax_single_pool/llama_simple_mlp.py`
  - `distributed.*`, `model.compile`, `precision.*` torchrun/NCCL/autocast — **[dropped]** torch-specific mechanisms.
  - `checkpointing.save_model` write side **[deferred]**; the load/read of cached `.pt`/`model_config.yaml` survives. jax: `jax_single_pool/llama_simple_mlp.py`

### `run_info.py`

- `run_info` — **[deferred]** PretrainRunInfo bundle; consumers assume pre-populated cache.
  - `.resolve` — **[deferred]** survives only as referenced mechanism in docstrings. 
  - `.resolve.wandb_path_detection` — **[kept]** `parse_wandb_run_path` survives. jax: `infra/wandb.py:48`
  - `.resolve.wandb_download` — **[deferred]** delegated to external torch repo.
    - `.cache_dir` — **[changed]** layout re-derived torch-free. jax: `jax_single_pool/llama_simple_mlp.py:550`
    - `.locate_configs` — **[changed]** only model_config.yaml read. jax: `llama_simple_mlp.py:574`
    - `.locate_tokenizer` — **[dropped]** tokenizer by name from config.
    - `.latest_checkpoint` — **[changed]** → single safetensors invariant. jax: `llama_simple_mlp.py:580`
    - `.race_safe_fetch` — **[deferred]** no in-tree fetch.
  - `.resolve.local_path` — **[replaced]** → cache-dir-rooted loaders. jax: `llama_simple_mlp.py:550-585`, `load_run.py:75`
  - `.config_parsing` — **[changed]** only model_config; refuses torch keys. jax: `llama_simple_mlp.py:83,574`
    - `.legacy_data_migration` — **[dropped]** no migration shim.
    - `.hf_tokenizer_extraction` — **[replaced]** → direct typed access. jax: `prestage_tokenized.py:64`, `tokenizer_display.py:69`
  - `.tokenizer_loading` — **[replaced]** → AutoTokenizer by name. jax: `tokenizer_display.py:69`

### `models`

- `registry` (+ config_discrimination/class_lookup) — **[dropped]** ModelConfig union + MODEL_CLASSES gone.
- `gpt2` (+ config/fused_qkv/flash_or_manual/gelu/tied_weights/residual_scaled init/lm_with_loss/from_pretrained_hf/configure_optimizers/generate) — **[dropped]** fused-QKV GPT2 has no JAX counterpart.
- `gpt2_simple` — **[replaced]** split-QKVO survives as vendored JAX target. jax: `vendored_jax/gpt2.py`
  - sub-features: split_qkvo **[replaced]**; layernorm.frozen_std **[dropped]**; mlp.gelu **[replaced]**; forward **[changed]** (logits-only); loaders.from_run_info **[replaced]**; from_pretrained_runinfo **[dropped]**; optimizer/generate **[dropped]**. jax: `vendored_jax/gpt2.py`
- `llama` (+ config/rope/gqa_fused_kv/position_ids/rmsnorm/swiglu/module_dict/lm_with_loss/import paths/optimizer/generate) — **[dropped]** from-scratch fused-KV Llama; 8b target is a fresh impl. jax: `llama8b.py`, `vendored_jax/llama.py`
  - `.norm.rmsnorm` — **[replaced]** → `rms_norm`. jax: `vendored_jax/llama.py`
  - `.mlp.swiglu` — **[replaced]** survives on 8b path. jax: `vendored_jax/llama.py`
- `llama_simple` — **[replaced]** split-QKV/flat basis of LlamaSimpleMLP target. jax: `jax_single_pool/llama_simple_mlp.py`
  - `.mlp.swiglu` — **[dropped]** live target uses GELU.
  - others **[replaced/changed/dropped]** per training-vs-decomposition concern. jax: `llama_simple_mlp.py`
- `llama_simple_mlp` — **[replaced]** THE second live JAX target (read-only torch ground truth). jax: `jax_single_pool/llama_simple_mlp.py`
  - `.config`/`.mlp.gelu`/`.attention_rope_norm` — **[replaced]**; `.loaders_forward_optimizer_generate` — **[changed]** (logits-only; optimizer/generate dropped). jax: `llama_simple_mlp.py`

### `configs`

- `pretrain_recipes` (+ all architectures/scales/datasets/data_pipeline/optimization/compute/eval_logging) — **[deferred]** catalog deleted; reimplement in JAX. (`compute.torch_compile` **[deferred]** — obsolete torch.compile.)

---

## Harvest

### `pipeline.py`

- `harvest` — **[changed]** collection loop reimplemented torch-free (numpy accumulator + JAX forward). jax: `harvest/scripts/run_worker_jax.py`
  - `.method_agnostic_entry` — **[replaced]** harvest_fn indirection dropped; single PD path. jax: `run_worker_jax.py`
  - `.accumulator_construction` — **[changed]** device dropped; `collect_component_cooccurrence` added. jax: `run_worker_jax.py`, `accumulator.py`
  - `.device_resolution` — **[dropped]** numpy accumulator.
  - `.batch_budget` — **[changed]** explicit n_batches required; whole_dataset dropped. jax: `run_worker_jax.py`
  - `.dataset_exhaustion` — **[dropped]** ShardServer indexed by fixed range.
  - `.batch_processing` — **[changed]** no_grad/autocast removed; JAX forward → NumPy batch. jax: `run_worker_jax.py`
  - `.distributed_sharding` — **[changed]** modulo-stripe → ShardServer process slice. jax: `run_worker_jax.py`
  - `.progress_reporting` — **[changed]** tqdm/throttled-log → plain per-batch log. jax: `run_worker_jax.py`
  - `.completion_logging` — **[changed]** worker tag dropped. jax: `run_worker_jax.py`
  - `.output.worker_state` — **[changed]** `.pt` → `.npz`. jax: `run_worker_jax.py`, `accumulator.py`
  - `.output.final_results` — **[kept]** HarvestRepo.save_results. jax: `run_worker_jax.py`, `repo.py`
- `merge_harvest` — **[changed]** kept; loads `.npz` not `.pt`; torch-free. jax: `harvest/pipeline.py`
  - `.discover_states` / `.fold_reduce` — **[changed]** extension/device only. jax: `pipeline.py`
  - `.persist_and_verify` / `.cleanup` — **[kept]**. jax: `pipeline.py`

### `harvest_fn`

- `harvest_fn` — **[replaced]** subdir deleted; → JAX forward + `harvest_batch_from_forward`. jax: `run_worker_jax.py`, `load_run.py`, `schemas.py`
  - `.protocol` — **[dropped]** no HarvestFn Protocol (single method).
  - `.uniform_output_contract` — **[kept]** HarvestBatch shape (np.ndarray). jax: `schemas.py`, `accumulator.py`
  - `.dispatch` — **[dropped]** method-config union collapsed to ParamDecomp.
  - `.param_decomp` — **[replaced]** → JAX forward over orbax run. jax: `load_run.py`, `run_worker_jax.py`
    - `.causal_importance_firings` — **[changed]** lower-leaky CI threshold in JAX. jax: `load_run.py`, `run_worker_jax.py`
    - `.u_norm_scaled_component_activation` — **[changed]** reimplemented (‖U‖·(x@V)). jax: `load_run.py`
  - `.transcoder` (+ hooks/encode) — **[dropped]** SAE comparison removed.
  - `.clt` (+ hooks/encode) — **[dropped]** CLT comparison removed.
  - `.device_and_eval_setup` — **[replaced]** → mesh/jit. jax: `load_run.py`, `run_worker_jax.py`
  - `.output_probs` — **[changed]** PD source only; fp32 softmax. jax: `load_run.py`, `run_worker_jax.py`

### `accumulator.py`

- `harvester` — **[changed]** torch tensors → NumPy host arrays; device dropped; cooccurrence opt-in + seed. jax: `accumulator.py`
  - all state/process_batch/persistence/merge/build_results sub-features — **[changed]** torch→numpy ports (semantics preserved); `cooccurrence_counts` **[changed]** made optional; `persistence.load` legacy fallback **[changed]** removed (fail-fast). jax: `accumulator.py`

### `reservoir.py`

- `reservoir` — **[changed]** torch→numpy port; Algorithm-R + Efraimidis-Spirakis semantics preserved; rng param threaded. jax: `reservoir.py`
  - `.device_transfer` (`to()`) — **[dropped]** numpy has no device. (all other sub-features **[changed]** ports.)

### `sampling.py`

- `sampling` — **[changed]** torch→numpy port; same 3 public fns. jax: `sampling.py`
  - all sub-features **[changed]**; `.pmi_ranking.valid_k_clamping` / `.infinite_value_filtering` — **[kept]**. jax: `sampling.py`

### `analysis.py`

- `component_correlation` / `token_statistics` / `result_types` — **[kept]** (numpy-backed storage); most leaves **[changed]** torch→numpy ports, AppTokenizer import relocated. jax: `analysis.py`, `tokenizer_display.py`

### `intruder.py`

- `intruder_scoring` (all sub-features) — **[kept]** byte-identical except AppTokenizer import relocated. jax: `harvest/intruder.py`, `tokenizer_display.py`

### `db.py` / `storage.py` / `repo.py`

- `harvest_db` (all sub-features) — **[kept]** byte-identical. jax: `harvest/db.py`

### `config.py` / `schemas.py`

- `harvest_config_schema` — **[kept]** schema survives; method_config narrowed to ParamDecomp. jax: `harvest/config.py`
  - `.method_configs` — **[changed]** union → single ParamDecomp. jax: `harvest/config.py:56`
    - `.param_decomp` — **[kept]**. jax: `harvest/config.py:19`
    - `.clt` — **[dropped]**; `.transcoder` — **[dropped]**
    - `.run_id` — **[changed]** only PD id survives.
  - `.pipeline_tuning` — **[changed]** narrowed + `collect_component_cooccurrence`. jax: `harvest/config.py:56-67`
    - dataset_extent/activation_budgets/pmi_topk — **[kept]**
  - `.intruder_eval` (+ all) — **[kept]** (LLMConfig import relocated). jax: `harvest/config.py:37-45`
  - `.slurm_submission` (+ harvest/intruder/partition) — **[kept]**. jax: `harvest/config.py:48-78`

### `scripts`

- `harvest_collection` — **[replaced]** run_worker.py → run_worker_jax.py (JAX-native). jax: `run_worker_jax.py`
  - `.worker_execution`/`.distributed_sharding`/`.worker_command_generation` — **[replaced/changed]**; `.subrun_id_minting` — **[kept]**. jax: `run_worker_jax.py`
- `harvest_merge` (+ all) — **[kept]** run_merge.py byte-identical. jax: `run_merge.py`
- `harvest_orchestration` — **[changed]** imports run_worker_jax; threads run_dir. jax: `run_slurm.py`
  - most sub-features **[kept]**; `.array_plus_merge_submission` — **[changed]**. jax: `run_slurm.py`
- `intruder_eval` (+ all incl. cross_method_comparison) — **[kept]**. jax: `run_intruder*.py`, `compare_intruder_scores.py`

---

## Autointerp

### `interpret.py`

- `interpret` (all sub-features) — **[kept]** byte-identical except AppTokenizer + strategy-config imports relocated. jax: `autointerp/interpret.py`, `tokenizer_display.py`, `param_decomp_config/autointerp.py`

### `llm_api.py` / `providers.py`

- `batch_llm_orchestration` + all (rate_limiting/retry/json_parsing/cost_tracking/budget/provider/timeout/teardown) — **[kept]** byte-identical. jax: `autointerp/llm_api.py`

### `strategies`

- `strategies` (canon/compact_skeptical/dual_view/rich_examples/dispatch) — **[kept]** subsystem intact; cross-cutting **[changed]**: config classes from torch-free package, AppTokenizer relocated, `model_class` removed, `dataset_description()` helper, `decomposition_method_descriptions` narrowed to pd-only. jax: `autointerp/strategies/`, `param_decomp_config/autointerp.py`

### `scoring`

- `scoring` (detection/fuzzing/llm_execution/cli) — **[kept]** intact; `delimit_tokens` import relocated. jax: `autointerp/scoring/`, `tokenizer_display.py`

### `prompt_helpers.py`

- `prompt_helpers` — **[kept]** function bodies byte-identical; config/AppTokenizer imports relocated, `dataset_description()` accessor added, `dataset` table extended. jax: `autointerp/prompt_helpers.py`

### `subsets.py`

- `component_subset_persistence` — **[kept]** (functions renamed: load/save/get_subrun_component_keys_path). jax: `autointerp/subsets.py`

### `db.py` / `repo.py` / `schemas.py` / `config.py`

- `interp_db` (all sub-features) — **[kept]** byte-identical. jax: `autointerp/db.py`

### `scripts`

- `run_interpret` / `run_slurm` / `run_slurm_cli` (all sub-features) — **[kept]** byte-identical; `adapter_resolution` consumes the changed adapters module underneath. jax: `autointerp/scripts/`
- `render_prompt` — **[changed]** AppTokenizer/RichExamplesConfig imports relocated; dropped `model_class`/`include_dataset_description` fixture fields. jax: `autointerp/scripts/render_prompt.py`

---

## Graph interp — **[dropped / deferred: attribution-graphs backlog (TRANSITION §1)]**

Entire `param_decomp_lab/graph_interp/` deleted in #848. All modules and features
(`interpret.py`, `graph_context.py`, `ordering.py`, `prompts.py`, `db.py`/`repo.py`/
`schemas.py`/`config.py`, `scripts/`) and every leaf within them are **dropped** (most
per-module authors say "dropped"; `ordering.py`/`storage.py` say "deferred") — recoverable
from git, revisit-in-JAX backlog. No successor on feature/jax.

---

## Clustering

### `merge.py` / `merge_config.py` / `merge_history.py`

- `merge_iteration` — **[kept]** driver intact; torch→numpy backend port. jax: `clustering/merge.py`
  - `.device_placement` (`_choose_coact_device`) — **[dropped]** numpy on CPU.
  - `.loop.early_stop` — **[changed]** warnings.warn → logger.info.
  - `.loop.mdl_metrics` / `.matrix_shrink` / `.log_callback` — **[changed]** numpy ports. jax: `merge.py`
  - all other sub-features **[kept]**. jax: `merge.py`

### `memberships.py` / `sample_membership.py`

- `memberships` — **[changed]** numpy-fed; preview mechanism dropped. jax: `clustering/memberships.py`
  - `.processed_memberships.{save,load}` — **[changed]** preview branch removed. jax: `memberships.py`
  - `.builder.add_batch.preview_capture` — **[dropped]**; `.finalize.preview_assembly` — **[dropped]**
  - `.lm_token_sampling.*` — **[changed]** np.random.Generator; flatten public. jax: `memberships.py`
  - `.collect` — **[replaced]** → `harvest_jax_run`. jax: `clustering/scripts/run_worker_jax.py`
    - `.token_budget_control` — **[replaced]**; `.progress_reporting` — **[changed]**. jax: `run_worker_jax.py`
  - all other sub-features **[changed]** numpy ports. jax: `memberships.py`

### `activations.py`

- `activation_extraction` — **[changed]** re-scoped to numpy reference path. jax: `clustering/activations.py`
  - `.single_batch_ci` — **[replaced]** → JAX forward lower_leaky_ci. jax: `run_worker_jax.py`
- `coactivation` — **[replaced]** → CSR `compute_coactivation_matrix`. jax: `sample_membership.py`
- `dead_component_filtering` — **[kept]** (numpy + streaming builder). jax: `activations.py`, `memberships.py`
- `multi_module_processing` — **[changed]** numpy reference/test path. jax: `activations.py`
- `processed_result` — **[changed]** slimmed to data container; query helpers dropped:
  - `.validation` — **[changed]** → ProcessedMemberships.validate. jax: `memberships.py`
  - `.label_lookup` — **[dropped]**; `.module_views` — **[dropped]** (no dense concat tensor).

### `math`

- `distance_metrics` / `group_assignment` / `merge_pair_sampling` / `scaling.semilog` — **[changed]** torch→numpy ports (most leaves), with several `group_assignment` ops dropped (`components_in_group_mask`, `random()`, `matrix_conversion`, `all_downstream_merged`, batched `from_list`/`from_matrix`/`to_matrix`) as unused. `perm_invariant_hamming` / `semilog` — **[kept]** (already torch-free). jax: `clustering/math/`

### `plotting` — **[dropped]**

Entire `clustering/plotting/` (activations.py, merge.py) deleted #850 — only used by the
deleted torch ensemble pipeline. All leaves **dropped**, no successor.

### `compute_costs.py` / `formatting.py` / `paths.py` / `types.py`

- `mdl_cost` / `coactivation_update` (all sub-features) — **[changed/kept]** numpy ports; MDL formula + merge logic preserved. jax: `clustering/compute_costs.py`

### `harvest_config.py` / `clustering_run_config.py`

- `harvest_config` (all sub-features) — **[kept]** schema verbatim; BaseConfig/Probability import relocated. jax: `clustering/harvest_config.py`

### `scripts`

- `harvest` — **[replaced]** run_harvest.py → run_worker_jax.py (JAX-native). jax: `clustering/scripts/run_worker_jax.py`
  - `.load_decomposition` — **[replaced]** → open_jax_run. jax: `run_worker_jax.py`, `load_run.py`
  - `.collect_memberships` — **[replaced]** inline JAX. jax: `run_worker_jax.py`, `memberships.py`
  - `.cli` — **[changed]** argparse over run_dir. jax: `run_worker_jax.py`
- `merge` (all sub-features) — **[kept]** run_merge.py + pd-cluster-merge byte-identical. jax: `clustering/scripts/run_merge.py`
- `single_run` (run_clustering.py + all sub-features) — **[dropped]** deleted #850.
- `ensemble_pipeline` (run_pipeline.py + all sub-features) — **[dropped]** pd-clustering deleted.
- `distances` (calc_distances.py + all sub-features) — **[dropped]** script deleted (math survives, no driver).
- `cluster_mapping_export` — **[kept]** get_cluster_mapping.py; one `.numpy()` removed. jax: `clustering/scripts/get_cluster_mapping.py`

### `configs`

- `clustering_configs` (pipeline + run + crc presets) — **[dropped]** dir deleted #850; harvest/merge schemas survive as code (`harvest_config`/`merge_config`), but stored config files and the ensemble orchestration are gone; per-run wandb/logging-interval config **[dropped]**.

---

## Dataset attributions — **[dropped / deferred: attribution-graphs backlog (TRANSITION §1/§6)]**

Entire `param_decomp_lab/dataset_attributions/` deleted in #848. All modules
(`pipeline.py`, `accumulator.py`, `storage.py`/`repo.py`, `config.py`, `scripts/`) and
every leaf — **dropped** (pipeline/accumulator/scripts/config authors say "dropped";
storage/config authors say "deferred"). Recoverable from git; revisit-in-JAX backlog
("possibly easier in JAX, vmap + grad"). No successor on feature/jax.

---

## Postprocess orchestration

### `cli.py`

- `cli` (all sub-features) — **[kept]** pd-postprocess byte-identical. jax: `postprocess/cli.py`

### `config.py`

- `postprocess_config` — **[changed]** BaseConfig from torch-free package; stages trimmed. jax: `postprocess/config.py`
  - `.stage_composition.harvest` — **[kept]**; `.autointerp`/`.intruder` — **[changed]** gained `=None` default. jax: `postprocess/config.py`
  - `.stage_composition.attributions` — **[dropped]**; `.graph_interp` — **[dropped]** (modules deleted).
  - `.stage_skipping` — **[kept]**
  - `.cross_stage_validation` (+ attributions_require_pd / graph_interp_requires_attributions) — **[dropped]** (model_post_init removed with the stages).
  - `.schema_export` — **[changed]** generator kept; committed schema.json removed.

---

## Investigate

### `agent_prompt.py`

- `agent_prompt` (all sub-features) — **[kept]** byte-identical. jax: `investigate/agent_prompt.py`

### `schemas.py`

- `investigation_output_schemas` (all sub-features) — **[kept]** byte-identical. jax: `investigate/schemas.py`

### `scripts`

- `investigate` (cli/submit/worker + all sub-features) — **[kept]** byte-identical; decoupled from torch/JAX (loads decomposition via app backend HTTP API). **Caveat:** `run_agent.py` still launches `app.backend.server`, so `pd-investigate` is runtime-broken until the app returns (deferred). jax: `investigate/scripts/`

---

## Eval metrics

### `ce_and_kl_losses.py`

- `ce_and_kl_metric` — **[replaced]** torch Metric class deleted; → jitted `eval_step`. jax: `eval.py`
  - `.config` — **[kept]** CEandKLLossesConfig in torch-free package. jax: `param_decomp_config/eval_metrics.py`
    - `.rounding_threshold` — **[kept]** → EvalConfig.rounding_threshold. jax: `eval.py`
  - `.metric_lifecycle` — **[replaced]** stateless eval_step + run.py aggregation. jax: `eval.py`, `run.py`
    - `.reset` — **[dropped]**; `.update` — **[replaced]**; `.compute` — **[replaced]** GSPMD. jax: `eval.py`, `run.py`
  - `.mask_variants` (ci/unmasked/stoch/random/rounded/zero) — **[kept]** all six; stoch reimplemented inline. jax: `eval.py`
  - `.loss_computation` — **[kept]**; `.ce_vs_labels` — **[changed]** slicing-based ignore; `.kl_vs_target` — **[changed]** → kl_per_position; `.target_ce_baseline` — **[kept]**. jax: `eval.py`, `losses.py`
  - `.derived_metrics` (ce_difference/ce_unrecovered) — **[kept]** all keys. jax: `eval.py`
  - `.result_assembly` — **[changed]** same 16 keys; loss_keys order-check dropped. jax: `eval.py`

### `ci_histograms.py` / `ci_l0.py` / `ci_mean_per_component.py`

- `ci_histograms` — **[replaced]** torch metric deleted; → offline `slow_eval` reductions. jax: `slow_eval.py`
  - `.config` — **[changed]** kept; recovered from raw config.yaml. jax: `param_decomp_config/eval_metrics.py`, `slow_eval.py`
  - `.metric_registration` — **[dropped]** no Metric class.
  - `.accumulation` — **[replaced]** → `accumulate_site_reductions`. jax: `slow_eval.py`
    - `.batch_cap` — **[kept]**; `.dual_ci_form` — **[kept]**. jax: `slow_eval.py`
  - `.compute` — **[replaced]** split accumulate/render; cross-rank gather dropped. jax: `slow_eval.py`
    - `.distributed_gather` — **[dropped]** per-process offline pass.
    - `.plot`/`.result` — **[kept]** PNG figures under slow_eval/figures/. jax: `slow_eval.py`

### `ci_hidden_acts_recon_loss.py` / `attn_patterns_recon_loss.py`

- `ci_hidden_acts_recon_loss` — **[changed]** reimplemented JAX-native over `masked_site_outputs` seam; deterministic torch parity. jax: `hidden_acts_eval.py`, `slow_eval.py`
  - `.config` — **[kept]**. jax: `param_decomp_config/eval_metrics.py:26`
  - `.metric_class` — **[replaced]** functional; slow=True implicit. jax: `hidden_acts_eval.py`
  - sub-features **[changed/replaced]**: target-capture via all-frozen masked_site_outputs; CI mask direct; per-site fp32 sum-MSE; host-side fold; `distributed_reduction` **[changed]** process-local under GSPMD. jax: `hidden_acts_eval.py`, `slow_eval.py`

### `component_activation_density.py` / `identity_ci_error.py`

- `component_activation_density` — **[replaced]** → offline slow-eval reduction. jax: `slow_eval.py`
  - `.config` — **[kept]**; `.ci_alive_threshold` — **[changed]** sourced from EvalConfig. jax: `param_decomp_config/eval_metrics.py:85`, `config.py:242`
  - `.metric`/`.reset`/`.update`/`.compute` — **[replaced/changed]** functional reduction; `.distributed_reduction` — **[dropped]** offline single-process. jax: `slow_eval.py`
  - `.plotting` — **[changed]** ported into slow_eval.py (numpy/matplotlib). jax: `slow_eval.py`

### `plotting.py` / `uv_plots.py` / `permuted_ci_plots.py`

- `plotting` — **[replaced]** dir deleted; reduction-based plots reimplemented in slow_eval.py. jax: `slow_eval.py`
  - `.causal_importance_heatmaps` (+ probe/permutation/matshow/dual) — **[dropped]** no heatmap render; scalar identity_ci_error only.
  - `.mean_component_ci_scatter` — **[replaced]** → `plot_mean_component_cis_both_scales`. jax: `slow_eval.py`
  - `.uv_matrices` — **[dropped]** (UVPlots orphaned; in MIGRATION_HOLES).
  - `.component_activation_density_histograms` — **[replaced]**. jax: `slow_eval.py`
  - `.ci_value_histograms` — **[replaced]**. jax: `slow_eval.py`
  - `.grid_layout` — **[changed]** → `_grid_dims`. jax: `slow_eval.py`

---

## App (web visualization) — **[deferred: app re-add (#868) / dropped: attribution+intervention+circuit-opt (TRANSITION §1)]**

Entire `param_decomp_lab/app/` removed in #868 (deliberate remove-now-re-add-later; a
JAX-native read-only viewer is slated to replace it). Within the per-module diffs:

- **`backend/server.py`** (app_factory, middleware, error_handling, routing.*, cli) —
  **[deferred]** the read-only surfaces (runs, prompts, activation_contexts, correlations,
  clusters, autointerp_compare, dataset_search, data_sources, pretrain_info, agents,
  investigations, mcp); **[dropped]** the attribution/intervention surfaces (graphs,
  graph_interp, intervention, dataset_attributions).
- **`backend/routers`** — same split: browse/read routers **[deferred]**; graphs /
  graph_interp / intervention / dataset_attributions routers + the optimize_graph /
  run_ablation / probe_component / save_graph_artifact MCP tools **[dropped]** (TRANSITION §1).
  `key_translation` **[changed]** (AppTopology built torch-free, removed with app);
  `tokenizer_display_helpers` **[kept]** (relocated to `param_decomp_lab/tokenizer_display.py` — the one genuine survivor).
- **`backend/compute.py`** (attribution_graph, intervention_eval, result_types) —
  **[dropped]** (attribution/PGD-intervention/circuit-opt, TRANSITION §1); the read-only
  forward helpers (ci_only, node_statistics, node_keys) **[deferred]** with the viewer.
- **`backend/optim_cis.py`** (all features) — **[deferred]** circuit-opt backlog
  (TRANSITION §6, "possibly easier in JAX (vmap + grad)").
- **`backend/database.py` / `state.py` / `schemas.py`** (all features) — **[deferred]**
  PromptAttrDB returns with the viewer.
- **`frontend/src/components`** (run_selection, display_settings, activation_contexts,
  component_stats, autointerp_compare, clusters, dataset_explorer, data_sources,
  model_graph, prompt_attributions, investigations, shared_ui) — **[deferred]** read-only
  views with the viewer; **[dropped]** the attribution/intervention/circuit-opt views
  (model_graph, prompt_attributions.*, graph_interp_badge, dataset_attributions) per
  TRANSITION §1.
- **`frontend/src/lib`** (api, state, types, utils, registry, components) — **[deferred]**
  with the viewer.
- **`run_app.py`** (launcher + all sub-features) — **[deferred]** dev-server launcher
  returns with the viewer.

`tokenizer_display.py` (AppTokenizer, escape_for_display, delimit_tokens) is the
load-bearing residue: **[kept]**, relocated out of the app for harvest/autointerp consumers.

---

## Editing — **[dropped: model-editing (TRANSITION §1)]**

Entire `param_decomp_lab/editing/` deleted in #848. `editable_model.py` and all features
(component_key, search, loading, geometry, activations, circuit_discovery, editing,
measurement, result_types) — **dropped**. Backlog per TRANSITION §1 (circuit-opt /
model-editing), recoverable from git. No JAX successor.

---

## Adapters (decomposition-method abstraction)

### `base.py`

- `adapter_protocol` — **[changed]** ABC slimmed to metadata-only; `dataloader` removed (torch-free). jax: `adapters/base.py`
  - `.identity`/`.vocab_size`/`.layer_activation_sizes`/`.tokenizer_name`/`.model_metadata` — **[kept]** abstract props; impl in JaxPDAdapter. jax: `adapters/jax_pd.py`, `load_run.py`
  - `.dataloader` — **[replaced]** data sourcing moved into the JAX harvest worker (ShardServer). jax: `harvest/scripts/run_worker_jax.py`
- `pretrain_dataloader` (+ config_reconstruction/input_ids_collation/streaming_lm_loader) — **[replaced/dropped]** torch helper deleted; corpus served by ShardServer. jax: `harvest/scripts/run_worker_jax.py`, `jax_single_pool/data.py`

### `pd.py`

- `pd_adapter` — **[replaced]** torch PDAdapter → torch-free `JaxPDAdapter` (loads JAX run; loading OLD torch runs = deferred #10). jax: `adapters/jax_pd.py`
  - `.construction`/`.lazy_run_loading.*` — **[changed/replaced]** → run-dir config + RunMetadata (no orbax restore for metadata). jax: `jax_pd.py`, `load_run.py`
    - `.component_model` — **[replaced]** torch ComponentModel → RunMetadata + JAX HarvestForward. jax: `load_run.py`, `run_worker_jax.py`
    - `.topology` — **[replaced]** → torch-free path schema. jax: `topology/path_schemas.py`, `load_run.py`
  - `.identity` — **[kept]**; `.vocab_size`/`.layer_activation_sizes` — **[changed]** from RunMetadata; `.tokenizer_name` — **[kept]**. jax: `jax_pd.py`, `load_run.py`
  - `.model_metadata` — **[changed]** RunMetadata + torch-free schema; `model_class` dropped; `_semantic_dataset_name`. jax: `jax_pd.py`
    - `.layer_descriptions` — **[changed]** → torch-free `parse_target_path().canonical_str()`. jax: `jax_pd.py`, `topology/path_schemas.py`
  - `.dataloader` — **[replaced]** removed from interface; → ShardServer. jax: `harvest/scripts/run_worker_jax.py`

### `clt.py` / `transcoder.py` — **[dropped]**

Both comparison-method adapters deleted (#863). All features (clt_adapter,
construction, base_model_resolution, clt_loading, identity, vocab_size,
layer_activation_sizes, tokenizer_name, model_metadata, dataloader; and the analogous
transcoder ones) — **dropped**. The abstract `dataloader` method + `pretrain_dataloader`
helper also removed from the base.

### `_vendor` — **[dropped]**

Vendored CLT/transcoder torch models (clt_model.py, transcoder_model.py) deleted (#863).
All features — **dropped**; comparison runs can't be harvested.

---

## Topology

### `topology.py` — **[dropped (torch-coupled half)]**

`TransformerTopology` (nn.Module-bound) deleted #876. Path-mapping survives in
`path_schemas.py`/`canonical.py` (the torch-free half); live-module resolution
(`module_resolution`, `embedding_module`, `unembed_module`, `get_unembed_weight`,
`get_module`, `is_cross_seq_pair`) — **dropped**. `n_blocks`/`structure_query` —
**[replaced]** → config-derived RunMetadata. `canon_to_target`/`target_to_canon` —
**[changed]** survive as `_PathSchema.render_canonical_weight`/`parse_target_path`.
jax: `topology/path_schemas.py`, `load_run.py`, `adapters/jax_pd.py`

### `canonical.py`

- `canonical_weight` (all sub-features) — **[kept]** byte-identical (torch-free pure data types). jax: `topology/canonical.py`

### `gradient_connectivity.py` — **[deferred: attribution-graphs backlog]**

`get_sources_by_target` and all sub-features deleted #848. **[deferred]** (some leaves
"dropped"); revisit-in-JAX backlog (vmap+grad). No successor.

### `path_schemas.py`

- `path_schemas` — **[changed]** torch-free; selection by model-type string. jax: `topology/path_schemas.py`
  - `.component_schemas` (separate_attn/fused_attn/glu_mlp/ffn_mlp + parse/render/lookup) — **[kept]**. jax: `path_schemas.py`
  - `.abstract_schema` (+ parse_target_path/render_canonical_weight/block_path_parse/regex_cache/layer_weight_render) — **[kept]**. jax: `path_schemas.py`
  - `.family_schemas` — **[changed]** 5 → 4 (HFGpt2 dropped); registry by model-type string. jax: `path_schemas.py`
    - `.llama_simple`/`.llama_simple_mlp`/`.gpt2_simple`/`.gpt2` — **[kept]**. jax: `path_schemas.py`
    - `.hf_gpt2` — **[dropped]** (no JAX GPT2LMHeadModel target).
  - `.schema_selection` — **[replaced]** `get_path_schema(nn.Module)` → `path_schema_for_model_type(str)`. jax: `path_schemas.py:216`, `adapters/jax_pd.py:71`

---

## Infra

### `settings.py` / `paths.py`

- All features (repo_root_resolution, cluster_detection, output_dir, slurm_partition, app_default_run) — **[kept]** byte-identical (`PARAM_DECOMP_APP_DEFAULT_RUN` now orphaned). jax: `infra/settings.py`

### `slurm.py`

- `slurm_submission` (all sub-features) — **[kept]** purely additive (gained `torchrun_command`, qos/ntasks_per_node/signal/requeue fields, optional setup override; git_snapshot source uses shell-time $HOME). jax: `infra/slurm.py`

### `ddp_launch.py` — **[replaced/dropped]**

File deleted #876. `DDPLaunch`/`build_ddp_launch` (training launcher) **[replaced]** — JAX
training launches via `jax_launch.py` (srun + jsp-train, no torchrun). A generic
torchrun/srun builder survives as `slurm.py:torchrun_command` (uncalled, for torch
consumers). `master_port` (`_choose_master_port`) — **[dropped]**; `nccl_env` (`DDP_ENV`)
— **[changed]** → `slurm.py:CUDA_FLAGS`; `launch_descriptor` (`DDPLaunch`) — **[dropped]**.
jax: `experiments/lm/jax_launch.py`, `infra/slurm.py`

### `wandb.py` / `wandb_tensor_info.py`

- `run_reference_parsing` (+ all) — **[kept]**. jax: `infra/wandb.py`
- `config_flattening` — **[changed]** moved to torch-free `param_decomp_config/wandb_config.py`; metric short-names static dict (`short_name_caching` **[dropped]**). jax: `param_decomp_config/wandb_config.py`
- `entity_resolution` — **[kept]**. jax: `infra/wandb.py`
- `run_initialization` (init_wandb + sub-features) — **[changed]** orphaned (no caller); live wandb-init is `MetricsSink.__init__`. `view_meta`/`code_logging` **[dropped]** from the live path. jax: `jax_single_pool/run.py:159`
- `checkpoint_retrieval` (+ all) — **[kept]**. jax: `infra/wandb.py`
- `fault_tolerance` (try_wandb) — **[kept]**. jax: `infra/wandb.py`

### `sqlite.py` / `run_files.py`

- `nfs_sqlite_connection` (all sub-features) — **[kept]** byte-identical. jax: `infra/sqlite.py`

### `git.py` / `markdown.py` / `pydantic.py`

- `git_utils` (all sub-features) — **[kept]** byte-identical; `create_git_snapshot` now also the JAX launch snapshot mechanism. jax: `infra/git.py`, `experiments/lm/jax_launch.py`

---

## Lab top-level helpers

### `run_sink.py` — **[replaced]**

Concrete torch `RunSink` deleted; → JAX `MetricsSink` (run.py) + orbax checkpointing.
- construction (local/with_wandb) — **[replaced]** single MetricsSink; `silent` — **[dropped]**.
- output.log — **[changed]** scalar dict; jsonl kept; figures side-channel **[dropped]**; wandb coercion **[dropped]**.
- output.console — **[changed]** → print; output.checkpoint — **[replaced]** orbax sharded; checkpoint upload + `model_<step>.pth` artifact — **[deferred/dropped]** (interop gap → #10 adapter).
- output.finish — **[dropped]**. rank_awareness — **[kept]** (is_main guards). jax: `jax_single_pool/run.py (MetricsSink)`, `checkpoint.py`

### `component_model_io.py` — **[deferred: torch→JAX run-adapter (#10)]**

`load_component_model` + all sub-features deleted #872; loading OLD torch runs into JAX
consumers is the deferred adapter. The JAX-native analog is `open_jax_run`. `get_all_component_acts`
— **[changed]** → harvest forward's `component_acts` (‖U‖·(x@V)). jax: `jax_single_pool/load_run.py`

### `batch_and_loss_fns.py`

- `run_batch` (+ passthrough/first_element/factory) — **[deferred]** RunBatch family = #10 adapter.
- `recon_loss` (mse/kl/mean_lm) — **[replaced]** → `kl_per_position`/`tms_mse` native fns. jax: `jax_single_pool/losses.py`, `tms.py`

### `distributed.py` / `seed.py` — **[replaced/dropped]**

Whole torch DDP plumbing deleted #876. process_group_lifecycle/init/cleanup/device_selection/
seeding — **[replaced]** → `sharding.py` init_distributed + GSPMD + functional PRNG.
`with_distributed_cleanup` (os._exit SIGABRT-avoidance) — **[dropped]**;
`ensure_cached_and_call` (download-once-per-node) — **[dropped]**; `log0` — **[replaced]**
→ inline is_main print; `is_local_main_process` — **[dropped]**. jax: `sharding.py`, `run.py`

### `resumption`

- `resumption` — **[replaced]** torch package deleted; split into requeue-resume (restore_latest) + fine-tune-from-parent (init_from_parent, S33). jax: `run.py`, `checkpoint.py`
  - `.config.resume_config_schema` — **[replaced]** → `ResumeProvenance` field on ExperimentConfig. jax: `param_decomp_config/experiment.py:46`
  - `.config.resolve_step` — **[replaced]** → orbax latest_step / step-existence assert. jax: `checkpoint.py`
  - `.config.read_training_snapshot` — **[replaced]** → orbax sharded restore. jax: `checkpoint.py`
  - `.provenance` (schema/write/read) — **[changed]** inline on ExperimentConfig (in config.yaml + wandb), not a sibling file. jax: `param_decomp_config/experiment.py`, `run.py`

### `scripts`

- `token_divergence_data_generation` (generate_token_divergence.py + all sub-features) — **[dropped]** deleted #848 (editing consumer; TRANSITION §1, not in MIGRATION_HOLES).

---

## Toy models

### `target_ci.py`

- `target_ci` — **[replaced]** general framework deleted; per-target `identity_ci_error` only. jax: `tms.py::identity_ci_error`, `resid_mlp.py::identity_ci_error`
  - `.column_permutation` — **[replaced]** inline Hungarian; `.to_identity_greedy`/`.to_identity_auto`/`.to_dense` — **[dropped]**; `.to_identity_hungarian` — **[replaced]**; `.vendored_hungarian` — **[replaced]** → direct scipy. jax: `tms.py`, `resid_mlp.py`
  - `.patterns` — **[dropped]** ABC gone; `.identity` — **[replaced]** → free fn; `.dense` — **[dropped]** (DenseCIPattern not ported; NOT in MIGRATION_HOLES).
  - `.solution` (+ pattern_expansion) — **[dropped]** multi-module aggregator gone.
  - `.metrics` (`compute_target_metrics`) — **[dropped]** (`total_0p2` etc.; IdentityCIError no-ops per holes).
  - `.config_builder` — **[dropped]** (config specs survive torch-free, no-op).

### `_linear_sum_assignment.py` — **[dropped → replaced by scipy]**

Vendored Kuhn-Munkres deleted #876; scipy is now a declared dep, so all features
**[dropped]** as vendored code / **[replaced]** by `scipy.optimize.linear_sum_assignment`
(used in tms.py, resid_mlp.py, clustering/math/perm_invariant_hamming.py).

---

## nano_param_decomp (standalone reference)

### `run.py`

- All features (config, sigmoids, component_linear, ci_transformer, losses, ppgd,
  lr_schedule, distributed, train_loop, eval) — **[kept/changed/replaced]** as the
  single-file torch paper reference; the *production* path is the JAX single-pool trainer
  (`jax_single_pool/`). sigmoids/ci_transformer/ppgd/faith_warmup/checkpointing largely
  **[kept]** (reimplemented faithfully); the nn.Module/DDP/install-components machinery
  **[replaced]** by pure-fn/pytree + GSPMD. jax: `jax_single_pool/` (run.py, train.py, lm.py, ci_fn.py, adversary.py, recon.py, eval.py, slow_eval.py)

### `pile_4L.py` — **[replaced]**

Entry-point wiring file deleted #876; the concrete pile-4L run is now a YAML config
(`configs/pile_pgd1.yaml`) consumed by jsp-train, launched via pd-jax-lm. All features
(component_count_spec, target_model_loading, data_loading, run_composition, launch_modes)
**[replaced]**; the logits-only forward patch + EOS-packing + on-the-fly tokenization
**[dropped]** (logits direct; offline parquet). jax: `jax_single_pool/configs/pile_pgd1.yaml`

### `simplestories_2L.py` — **[dropped]**

Torch SimpleStories-2L wiring deleted #876; NO SimpleStories target exists in the JAX
production path (only LlamaSimpleMLP pile / llama8b / resid_mlp / tms). All features
**dropped/replaced**; the model_type shim, forward adapter, on-the-fly tokenization, and
EOS-packing **[dropped]**.

---

## Papers — **[kept, all]**

All three papers are static research artifacts, byte-identical between branches:

- **`Attribution_based_Parameter_Decomposition/`** (apd_paper.md + figures) — every feature **[kept]**.
- **`Stochastic_Parameter_Decomposition/`** (spd_paper.md + figures) — every feature **[kept]**.
- **`Interpreting_Language_Model_Parameters.md`** (VPD) — the method is **[kept]** (reimplemented
  end-to-end in JAX, SPEC.md normative). Within it: `fused_minimality` **[changed]** (factoring);
  `evaluation.recon_vs_sparsity_tradeoff` / `feature_splitting` / `cross_seed_consistency`
  **[dropped]** (paper-only offline analyses, no code on either branch);
  `attention_analysis.*` **[dropped]** (paper-only, never had code);
  `circuit_analysis.*` + `model_editing` **[deferred]** (attribution-graphs / editing backlog);
  `subcomponent_analysis.interactive_browser` **[dropped]** (app feature, removed with app).
