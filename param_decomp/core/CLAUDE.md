# param_decomp — agent notes

Single-pool VPD trainer in JAX, **generic over vendored targets**: the engine sees a
target only through the `DecomposedModel` protocol (`model.py`) and the `ArchFamily`
grammar contract (`family.py`). The concrete targets — LM and toy alike — are the
sibling subpackage `param_decomp.targets` (one slice per architecture); the library's
layering (core imports NO target, targets import core only) is pinned by
`tests/test_runtime_standalone.py`. The semantics source of truth is `SPEC.md`
(normative pseudocode + numbered invariants, grounded in the stable torch
`param_decomp` impl). See `README.md` for the file map.

Open items: the persistent-source shape `nsc` and sigmoid parameterization are
deliberately refused. SPEC S24's two torch-parity quirks (PPGD warmup route-all,
fresh-PGD single routing draw) are pinned pending a team decision.

## The one rule

**Every change is checked against SPEC.md, by invariant ID.** If a change deviates
from an invariant, either fix the change or (deliberately, with Oli) amend the spec —
never silently diverge. Cite IDs (`S14`, `N1`, …) in commit messages and reviews.

## Architecture in one breath

`model.py` defines `DecomposedModel` — a `@runtime_checkable Protocol`: ordered `sites` +
`has_position_axis` + the methods `clean_output`, `read_activations`,
`clean_output_and_activations` (both from ONE frozen forward — the GLU target emits
the taps as ys of its clean-forward `lax.scan`; what the train + fast-eval steps call),
`masked_output`, `masked_site_outputs`, `weight_deltas`, and a `recon_loss_fn`
(LM: `kl_per_position`). The
concrete impl per target is an `eqx.Module` (`GLUDecomposedModel`,
`SimpleMLPDecomposedModel`, `TMSDecomposedModel`, `ResidMLPDecomposedModel`) carrying its
FROZEN target weights as ARRAY FIELDS; the TRAINABLE V/U (`vu: ComponentStacks`) stays an explicit
METHOD ARG (separate lifecycle — own optimizer + checkpoint, C-sharded while the frozen
weights replicate). Flat site-name-keyed dicts at the boundary; the model threads into the
jitted step as a pytree ARG (never a jit-closure constant — an 8B target becomes a multi-GB
HLO constant; see "HLO-baking rule" below). The activation waist comes in EXACTLY TWO
shapes — positionless `[B, d]` (masks/CI `[B, C]`; the toys) or one position axis
`[B, P, d]` (masks/CI `[B, P, C]`; an LM, whose position axis is the token sequence).
`DecomposedModel.has_position_axis` declares which; the run threads the extents as
`positions: Positionless | Positioned(n_positions)` (matched exhaustively wherever
shapes are built — never a rank branch). Inside the step, masking / routing / sources /
imp-min read an opaque `leading = residual.shape[:-1]`; reductions are
`math.prod(shape[:-1])` / `axis=tuple(range(ndim-1))`. CI is independent over every
leading axis. `CIFn.has_position_axis` mirrors the model's, and `init_train_state`
asserts they're equal (early fail) so the CI fn stays per-domain (RoPE over positions)
without the core adapting.
Adversarial sources (both adversaries) are configured by `source_shape: c | bc | sc | bsc`
(`configs.SourceShape`): each letter names a waist axis the source keeps full, missing
letters are stored size-1 broadcast axes, rank always matches the waist. Batch-B shapes
(`bc`/`bsc`) are batch-sharded, no cross-replica sync; batch-1 shapes are shared across
the batch (SPEC S16/D1). `sc`/`bsc` on a positionless target raise; per-step (fresh) PGD
rejects `sc` at validation; legacy spellings (`scope: {type: ...}`, `mask_scope`, the
verbose value names) are rejected at parse — no config aliases remain. Every (positions x source_shape) persistent
shape is written out in `init_sources_sharded`, so persistent PGD runs on the toys too.
Batch size is `pd.batch_size` uniformly — `DataConfig` carries no batch.
`CIHiddenActsReconLoss` / `StochasticHiddenActsReconLoss` are standalone eval metrics
(`hidden_acts_eval.py`, in-loop on `eval.slow_every`) over a fifth model fn
`masked_site_outputs` — NOT recon-grid training terms (the recon loss stays
KL-on-final-logits; SPEC S31). CI-fn numerics: GELU is exact-erf (`approximate=False`),
RMSNorm eps is `finfo(fp32).eps` (`CI_FN_RMS_EPS`) — SPEC §4.6. The
three EDGES are generic so non-LM (bio-style) targets fit (#828): the model INPUT
(the opaque batch `clean_output` / `read_activations` / `masked_output` consume, typed
`Any` — token ids for an LM, a dict for bio), the model OUTPUT (`clean_output`/`masked_output` return `Any` — logits, a tuple
of heads, coords; field NAMES stay `*_logits` pending a deferred rename), and the recon
comparison (`recon_loss_fn(clean_output, masked_output) -> scalar`, default
`kl_per_position` so the LM path is byte-identical). The waist shape contract (all per-site
tensors in one forward share one `*leading` prefix) is enforced at trace time by
`@jaxtyped(typechecker=beartype)` on the core `step`, `masked_forward`, and the loss fns.
`train.py` is the generic step factory
(fp32 masters / bf16 compute) over a flat tuple of self-describing loss TERMS
(`recon.LossTerms` — faithfulness, importance-minimality, and the recon terms, iterated
uniformly; S10′ — the recon loss-class cartesian product factored as chunking × routing ×
mask-source strategy: a live-set helper (`all_sites_live`/`each_site_live`/`live_groups`)
feeds the single `make_plan` constructor, built from the shared configs by
`recon.build_loss_terms`;
see LOSS_PARITY_DESIGN.md),
consuming `losses.py` (pure loss terms + schedules) and `adversary.py` (persistent
vs fresh source machinery — semantically distinct adversaries sharing only
`source_masks`); `ci_fn.py` the shared CI transformer. The targets are the sibling
distribution: `param_decomp/targets/glu_transformer.py` the SHARED HF GLU-transformer
target machinery (site grammar, `FrozenAttn`/`GLULayer`/`GLUDecomposedModel`, the
scan/masked-forward engine, HF loading, and the target's own placement via
`.shardings(mesh)`; the generic seeded-init-placed helpers are engine-side,
`init_placed.py`), with the model FAMILIES in their own files:
`param_decomp/targets/llama8b.py` (vendored `LlamaConfig`, llama3 rope) and
`param_decomp/targets/qwen3_8b.py` (`Qwen3FrozenAttn` — REQUIRED `q_norm`/`k_norm`
fields applied in the `_prep_qk` pre-RoPE hook; Qwen3's one structural delta). Nothing
in the shared file switches on a family; the model-name → family registry is composition-side
(`experiments/lm/config.py::HF_MODEL_FAMILIES`). Qwen3 JAX↔HF parity is pinned DIRECTLY
by `param_decomp/targets/tests/qwen3_hf_parity/` (a tiny-random `Qwen3ForCausalLM`
golden at fp32 tolerance + a slow real-weights logits check; goldens regenerate via its
torch-env `gen_hf_fixtures.py`). There is ONE
recon semantics: masks thread through the full token-input forward, loss is KL on final logits
(SPEC §2.3–2.5). Site-local recon is a conceptual no-no, not a "simplification".
`param_decomp/targets/llama_simple_mlp.py` is the second target (the pile-pretrained `LlamaSimpleMLP`,
t-9d2b8f02; sites `h.{i}.attn.{q,k,v,o}_proj` / `h.{i}.mlp.{c_fc,down_proj}`) —
config dispatch is `TargetConfig` (the HF GLU families) vs `LlamaSimpleMLPTargetConfig`, both composition-side
(`param_decomp/experiments/lm/config.py`, which reads the canonical schema DIRECTLY —
`build_experiment_config`/`load_config` — resolving each target's tiled
`decomposition.sites` via its `ArchFamily`), target build in the LM composition root
`param_decomp/experiments/lm/run.py::main`. The slow plot metrics are computed
NATIVELY in JAX (`slow_eval.py`) — no torch export round-trip (the torch offline-eval
bridge `jsp-export` / `pd-offline-eval` was retired). They run IN-LOOP ONLY on
`eval.slow_every` next to the fast pass (SPEC S28/S29; there is NO offline/retrospective
CLI — `slow_eval.py` is a pure library): the collective
forward + device→host pull in lockstep on all ranks, then a pure matplotlib renderer on a
rank-0 background thread (`run.py::BackgroundRenderer`). The renderer returns encoded media
plus its semantic step; the shared `MetricsSink` serializes W&B transport against the dedicated
`slow_eval/figure_step` axis, so late renders are not rejected by W&B's monotonic `_step`. The config-gated position-CI metrics
(`PermutedCIPlots` / CI heatmaps + `IdentityCIError`) ALSO run in-loop off the cheap
`(T, C)` position-CI matrix (`accumulate_position_ci`, collective; the heatmap figures on
the background thread, the `IdentityCIError` scalars synchronously on `_step`). `UVPlots`
is a config-gated figure metric usable for ANY decomposition (the torch `Metric` pattern —
returns a wandb figure): for the LM in-loop tier the LM composition's `UVPlots` operation does a
NAIVE host gather of the C-sharded V/U (gated on `want_uv_plots`) and passes `components` to
`render_permutation_figures` — it OOMs / breaks at production C BY DESIGN (per Oli), no
special handling; for the positionless toys (TMS/ResidMLP) `toy_uv_eval.render_uv_metric`
renders it off the small on-host V/U + the probe CI as permutation source (cheap, no
gather), sharing `slow_eval.render_uv_figure` / `plot_uv_matrices` with the LM path.

`experiments/lm/arithmetic_eval.py` is a config-gated LM-only figure tier (`ArithmeticCIGrid`, on
`eval.slow_every`) for inspecting how the decomposition reconstructs a `target model`'s
modular-arithmetic mechanism (Feucht et al.'s L18 addition neurons). The probe is a FIXED
`a x b` operand grid of `"<a><op><b>="` prompts (one prompt per row, all one token length,
the `=` answer at a constant position) — NOT the streaming corpus, so it brings its own
batch. The probe is a SPEC (`operation` + `a_range`/`b_range` on the metric config), not a
filesystem artifact: `experiments/lm/arithmetic_eval_operation.py::make_arithmetic_operation` builds it in-memory at
startup from the target's tokenizer (`experiments/lm/arithmetic_probe.py`, deterministic —
every rank builds the identical grid, no rank-0 write or barrier), so configs stay
cluster-portable. The ONE fused `make_arithmetic_grid_step` slices, at the answer position with
the BATCH axis KEPT as the grid, each component's lower-leaky CI (from the CI fn) and its
pre-mask activation `x@V` (from the decomposed forward under all-ones masks — the
`masked_component_activations` seam, GLU-target-only, narrowed via the `ComponentActivationModel`
Protocol). The device→host pull is TWO-PHASE (`compute_arithmetic_selection`), sized to what
the figures need — never the full `(n_prompts, C)` grids (~GBs/site at production C): the
step's replicated per-component max CI (over REAL rows only — the sharding-pad tail is
masked) drives the host-side selection, identically on every rank, then only the ≤`top_k`
selected columns are gathered. The active set per threshold (max CI > threshold) is selected
ONCE (`select_active`, one stable descending ordering per site, so a higher threshold's set
is a PREFIX of a lower's) and drives both the `n_alive` scalars and the CI + activation
heatmaps; figures render off-loop on rank 0 (`run.py::BackgroundRenderer`). The probe's
CE/KL/L0/PGD scalars compose the independent kernels from `experiments/lm/eval.py`
with `n_valid_rows=n_prompts`, so pad rows carry zero weight. The complete typed
operation lives in `experiments/lm/arithmetic_eval_operation.py`.

**Every target lives in `param_decomp.targets` — the toys (TMS, ResidMLP) included.**
The core trainer carries ZERO target-specific code — the toy *targets*
(`DecomposedModel`s, pretrain, identity-CI eval) are `param_decomp/targets/{tms,resid_mlp}.py`,
peers of the LM slices; their composition roots (`experiments/{tms,resid_mlp}/run.py`) stay
composition-side. CI-fn *architectures* are NOT toy-specific code: core owns every CI-fn arch
regardless of which experiments use it. The positionless MLPs and the sequence transformer
are peers in `ci_fn.py` (differing by domain, not status), not a toy carve-out. The
generic engine is `run.py::run_decomposition_training(pd, cadence, run, model, ci_fn,
positions, remat_recon_forwards, remat_ci_fn, ascend_replicate, compiler_options,
sample_batch, evaluation, mesh)` — the ONE train loop every target runs through (init/restore/finetune/faith-warmup
via `_init_or_restore_state`, the recon-grid step factory, orbax checkpointing, schedules,
SIGTERM-save). It reads the pydantic `PDConfig` / `Cadence` (`param_decomp.core.configs`)
DIRECTLY — optimizers / loss metrics / faith warmup / seed / steps — so there is
NO flattened mirror dataclass; the run identity rides in
`built_run.RunInstance`, and the composition-built objects (`ci_fn` arch, `data`, the decomposed target)
pass alongside. A target injects exactly three seams: the data source
(`sample_batch(step) -> residual`), the domain-bound `Evaluation` (typed operations + context factory, scheduled directly by core), and (for the LM) the perf token count.
`param_decomp/experiments/lm/run.py::train` is the thin LM caller (parquet
`sample_batch` + domain-bound CEandKL/CI-L0/PGD/attention operations; that
LM composition root is LM-ONLY (`experiments.lm.config.build_from_schema` validates
`LMExperimentConfig` and returns a `config.BuiltRun`; `main`'s `match built.target` covers
only `TargetConfig` / `LlamaSimpleMLPTargetConfig`). `BuiltRun.target` is typed by the core
`built_run.TargetSites` protocol (just `.sites`), `BuiltRun.data` is generic; LM binds it to `experiments.lm.built.DataConfig | None` (None
for a toy run). The shared run-identity / CI-fn-arch helpers are public
composition-side for the toys to reuse: `experiments.config.run_instance` / `ci_arch`.

The TMS + ResidMLP targets live at `param_decomp/targets/{tms,resid_mlp}.py` (the JAX
`DecomposedModel` + frozen target + in-process pretrain + identity-CI eval); each
`param_decomp/experiments/{tms,resid_mlp}/run.py` is the toy CPU composition root
(a module main) that builds the `ExperimentConfig` from the canonical schema and calls
`run_decomposition_training`. They are positionless and use the MLP CI fns. All CI-fn architectures live together in
`ci_fn.py`: `LayerwiseMLPCIFn` (positionless, one independent MLP per site mapping
`site_input [B,d_in] -> [B,C]`), `GlobalMLPCIFn` (positionless, one shared MLP over all
sites jointly, concat/split in canonical site order), and the LM `ChunkwiseTransformerCIFn`
(positioned, per-chunk transformers reading residual taps, stacked +
`lax.scan`'d with per-chunk remat, and **N per-site output heads** (one `[d_model, C_j]` per
site-slot). `n_blocks=0` degenerates that arch to `RMS-normed taps → in_proj → heads`:
position-LOCAL and attention-free, still `has_position_axis=True`, but affine on the NORMALIZED
tap — no hidden layer and blind to tap magnitude. A positioned target whose position count
makes O(P²) attention infeasible (pairwise positions) should take
`LayerwiseMLPCIArch(has_position_axis=True)`; the blockless chunk is a baseline.
The mesh is `(replicate, fsdp, tp)`: `replicate` spans nodes, while each
node is an `(fsdp, tp)` plane. `tp` shards declared target dimensions and the CI output C
axis; the per-site heads keep site boundaries explicit rather than slicing a glued-ΣC head
mid-site. **Persistence layouts (÷N)**: the trainable V/U masters AND their
optimizer moments persist as same-shape STACKS (`ComponentStacks.stacks`, owner-partitioned: stack
axis ÷`replicate` — whole matrices owned per node-group, zero cross-node weight collectives,
muon NS node-local — matrix d dims ÷`fsdp`, C ÷`tp`; SPEC D4 amendments 2026-07-15 +
2026-07-21; a stack that doesn't tile `replicate` is an ERROR under strict `owner` — the
per-group fallback to intra-matrix data sharding is the config-opt-in `owner+zero1`
preset / an explicit table's `params.zero1` row. The per-group assignment is resolved
ONCE, at `placement.from_config(spec, mesh, sites)` during config build — bidirectional
claim included: declaring `zero1` when every group tiles is equally an error, refused
during config build for `dp: N` — and flows down as data; the consumer boundary
(`placement.component_stacks_shardings`) only validates the received assignment, never
re-decides. See PLACEMENT_DESIGN.md "Decision at build time"). The CI-fn
masters + moments keep intra-matrix ZeRO-1 (`("fsdp","replicate")` on d_model — fsdp-major,
so the ÷N→÷fsdp reconstruct is a pure all-gather over `replicate`; replicate-major cost a
~13 GiB/rank/step grid-transpose collective-permute, PR #927). Either way
the dominant optimizer-state memory scales 1/N, not the fixed 1/fsdp. The bf16
COMPUTE weights are reconstructed to the `fsdp`-sharded (÷fsdp) layout ONCE per step in ENTRY
(the cross-`replicate` gather, off the hot path — `glu_transformer._reconstruct_compute_weights` /
`ci_fn._reconstruct_ci_compute_weights` pin `P(None,"fsdp",...)` BEFORE the per-layer /
per-chunk scan), landing a SMALL ÷fsdp-resident stack; the scan body then gathers ONE layer's
`fsdp` shard to full d_in transiently (NVLink, freed each iteration) — NEVER a
full-model `[n_layer, full_d_in, C]` weight stack resident.
`run_state.init_train_state` dispatches CI-fn construction on `cfg.ci_fn`
(`LayerwiseMLPCIArch` / `GlobalMLPCIArch` / `ChunkwiseTransformerCIArch`) and uses replicated (not
C-sharded) V/U + CI for the tiny toys; the core `ci_fn.CIFnArch` admits all three and the
composition-side `experiments.config.ci_arch` builds the layerwise / global arch from the toy
`decomposition.ci` (validated end-to-end on CPU via
the ResidMLP composition root). Harvest / slow-eval / export over the toys are NOT wired
(`experiments.lm.load_run.build_target` / `run_metadata` are LM-only).

## The HLO-baking rule (filter_jit discipline)

The decomposed model is an `eqx.Module` whose frozen target weights are ARRAY FIELDS — a
Llama-8B target is multi-GB. Therefore:

- **Every `jax.jit` that touches a model is an `eqx.filter_jit` with the model as a TRACED
  ARG** — never `@jax.jit` over a function that CLOSES OVER an array-bearing model. A closed-
  over model is a jit *constant*: its arrays bake into the HLO (multi-GB constant tensors,
  recompiled per concrete model). As a traced arg, the array leaves are dynamic inputs and
  the static fields (`sites`, `eps`, `has_position_axis`) bake harmlessly.
- **Step factories read only STATIC config off the closed-over `model_static` at trace-setup**
  (`model_static.site_names`, `model_static.sites`, `model_static.recon_loss_fn` — `recon_loss_fn` is a `@staticmethod`,
  pure, holds no arrays, so closing over it is safe). All ARRAY access goes through the
  model ARG (named `model` inside the jitted fn). `make_train_step`, `make_eval_step`,
  `make_*_hidden_acts_step`, `make_slow_eval_step`, `make_position_ci_step`,
  `make_*_attn_patterns_step`, and `make_faith_warmup_step` all follow this; each carries a
  comment at the step factory. The toy `run.py`s and `load_run.py`'s harvest `forward`
  thread the model as a filter_jit arg too.
- This is why the methods take only the *runtime-varying* args (`vu`, `resid`, masks, …) and
  the frozen weights ride on `self`: `self` reaches the trace as the traced model arg.

## Invariants with sharp teeth (the ones that have actually bitten)

- **S3**: the recon target is the FROZEN-path forward (`clean_output`), never the
  `mask=1` decomposed identity (bf16 rounding + V/U in the stopped graph).
- **S13/S15**: source updates go through the persistent Adam AND project to [0,1]
  after EVERY ascent — an unprojected drift past 1 has zero `clip` gradient and the
  entry dies.
- **S14**: the final source ascent reuses the main backward's source-grad
  (pre-update θ), unscaled by the ppgd coeff. No extra forward.
- **N1**: fp32 masters everywhere (`optax.adamw(..., weight_decay=0.0)` — optax's
  default wd is 1e-4, torch's is 0; this was audit finding A7).
- **`inv_freq` is a buffer, not a param** — `stop_gradient` in `CIFn.__call__`.
- **S10/S11**: chunking is sequential `sites_per_chunk` groups in canonical site
  order; routing is uniform-k over the chunk's sites only.

## Validation stack (run all before claiming correctness)

1. `pytest param_decomp/core/tests/ param_decomp/targets/tests/` — at the default device
   count AND `XLA_FLAGS="--xla_force_host_platform_device_count=4"`.
2. `param_decomp/targets/tests/equivalence/` — fixture-driven JAX-vs-frozen-golden
   per-term numeric equivalence (fp32, no RNG, zeroed attn). The torch references are
   FROZEN committed goldens (`torch_reference.json`, `simple_mlp_equivalence/*.npz`,
   `tools/export_fixtures/*`); the torch generators/verifier that produced them are
   deleted so the runtime imports no torch (push-1). Regenerate goldens only when
   the MATH changes: redraw fixtures JAX-side with `gen_fixtures.py`, then check out the
   `torch-oracle` git tag in a torch-venv worktree and run that revision's
   `torch_reference.py` / `gen_torch_fixtures.py` / `gen_export_fixture.py`, copying the
   emitted goldens back here.
3. `param_decomp/targets/invariance_check.py` at 4 sim devices — trajectory invariant
   to device count up to float reassociation (SPEC D4).

`basedpyright` over the whole workspace must be clean (run `make type`); `param_decomp`
is in the root `[tool.pyright]` include and is checked in the one venv, one pass,
alongside the rest of the workspace.

## The training pipeline

The generic ENGINE `run.py::run_decomposition_training` is a pure library (no `main`, no
YAML). The composition root + only I/O layer lives in `param_decomp.experiments`:
`python -m param_decomp.experiments.lm.run <config.yaml>` reads the YAML, builds the
target + data loader + `ExperimentConfig`, and calls the engine; the step stays pure. Data
reaches the engine as `DataConfig.dir` — a directory of pre-tokenized parquet shards;
the loader takes `seq_len` as an explicit parameter (`ShardServer`), and the dir is not
required to exist until load time. How that directory is named, resolved, described,
and populated is not core's business — see `experiments/CLAUDE.md` and the root
CLAUDE.md. Training
NEVER streams or tokenizes at run time: the pure-function batch schedule needs a fixed,
enumerable local artifact, and startup keeps external services out of the N-rank
collective bring-up (history: the 2026-06-09 thunderherd docs in lore). The batch schedule is a pure
function of `(seed, step)` (O(1) resume, no replay); checkpoints are orbax sharded
saves (no on-loop full-gather), TWO items per step — `decomposition` (V/U + ci_fn, the
product every consumer restores alone) and `training` (opt states + adversaries + step,
trainer-only) — a clean break, no in-code compat for pre-split `default`-item runs (SPEC
S22); SIGTERM → save → SLURM requeue → resume from latest.
Resume with a changed config is refused (byte-compare). Smokes before a long run
MUST exercise save AND resume at the production per-rank shape.

A run config is ONE self-contained yaml: the experiment schema
(`param_decomp.experiments.config.ExperimentConfig` over the core
`param_decomp.core.configs` pieces — `pd`/`data`/`eval`/`cadence`/`target`/`wandb`, plus
`runtime` on `LMExperimentConfig`: the compute substrate is per-domain, and a toy — single
device by construction — declares no `runtime:` section at all)
plus the run-instance fields
the schema now also carries — top-level `run_name`/`run_id`/`out_dir`, the
`runtime.remat_recon_forwards` memory/compute knob, and `wandb.group`/`wandb.tags`.
`run_id`/`out_dir` are absent in a hand-authored config; the composition root mints and
pins them at startup (a launcher may stamp them in beforehand instead, and pass
`--run-id`).

**Fine-tune from a parent checkpoint** (`resume_provenance`, SPEC S33, LM-only). A fresh
run can initialize its trained decomposition (V/U + ci_fn) from a PARENT run's checkpoint
and continue under a DIFFERENT config (changed LR / coeffs / eps / seq / batch / steps —
NOT changed C / sites / ci-fn arch). Add to the config:

```yaml
resume_provenance:
  # ABSOLUTE path — the trainer's cwd is the code it was launched from, not the output
  # root, so a relative path would resolve against the wrong directory.
  parent_run_dir: /abs/path/to/runs/p-xxxxxxxx
  parent_step: 175000
```

On the FIRST entry (own `ckpts/` empty) the trainer restores the `decomposition` item of
`parent_run_dir/ckpts/175000` onto the fresh reference; the optimizer states,
persistent sources, and `step` are FRESH (`step = 0`, no faith warmup) so the new LR /
p-anneal schedule recomputes over the new `cfg.steps` from 0. A subsequent SLURM requeue
(own `ckpts/` now non-empty) resumes from the run's own dir and ignores provenance.
`run.py::assert_finetune_structural_compat` reads the parent's pinned `launch_config.yaml` and
asserts matching sites (names + C) + ci-fn arch before the restore. Provenance flows into
`config.yaml` + `wandb.config`. Run it exactly like any other config.

**Topology is CONFIG-DERIVED; submission is a verb.** `runtime.dp` + `runtime.gpus_per_node`
fully determine process bring-up — there is no launch field:
- `dp <= gpus_per_node` → ONE process over exactly `dp` local devices, asserted at
  startup (`sharding.assert_inline_topology`), never absorbed from the ambient
  allocation. `dp: 1` is the smoke/debug mode; `dp: 8` an external scheduler's own
  whole-node job running the trainer inside its allocation.
- `dp > gpus_per_node` → one process per node (`nodes = dp // gpus_per_node`), brought
  up via `jax.distributed`'s own cluster auto-detection (`init_distributed` — the jax
  ecosystem's contract). Multiple processes on one node is deliberately unrepresentable.

`python -m param_decomp.experiments.lm.run <config> --data-root …` runs HERE, in the
current allocation, minting and pinning its own identity when `--run-id` is absent. A
launcher that wants to own the identity mints the `p-` run id itself, stages
`<data_root>/runs/<id>/` with the pinned config, and passes `--run-id <id>`.

A restart re-reads the run's PINNED launch config, never the live checkout — so a
requeued job resumes the run it started, whatever the working tree says by then.

`main` enables JAX's persistent compilation cache
(`_enable_persistent_compilation_cache`) at `<data_root>/xla_compilation_cache`
— a SIBLING of `runs/` (derived from `out_dir.parent`), shared across all runs and all
8N ranks, NOT per-run. The multi-minute chunkwise-step compile is keyed by HLO + backend +
topology + jax/xla version, so a requeue/resume or a fresh run at the same config+topology
loads the executable from disk in seconds. Set after `init_distributed` (the write gate
reads the distributed state) and before the first compile; threshold 60s
(`jax_persistent_cache_min_compile_time_secs`) so only the big compiles cache. Multi-host
safe on jax 0.10.1: jax gates the cache WRITE on `process_id == 0` (`compiler.py` — "Only
write cache entries from the first process … contention for writes on some filesystems"),
so all ranks read but only rank 0 writes — no shared-FS race. Requires the cache dir on a
shared FS — a requirement on the deployment: a multi-node run must pass a `data_root`
on one.

### Compile time (measured 2026-07-06 probe grid; full data in PR #956)

- **Keep seeded inits few-outputs-under-jit**: a jit returning n_sites (hundreds of)
  sharded outputs — or n_chunks unrolled RNG bodies — is a multi-minute SPMD/layout
  compile. vmap-stack over the same per-site/per-chunk keys (bit-identical values),
  then fan out with a trivial slice jit. `init_component_stacks_placed` is the template;
  `init_ci_fn_placed` / `init_sources_sharded` follow it.
- **The `jit_step` compile (~5 min at dp32) is FLAT across graph structure**: recon
  chunk count, C, CI-fn depth, and PPGD warmup all measured within noise (~83%
  priority-fusion). Don't chase graph-shrink refactors for compile time without new
  evidence.

## Gotchas

- **Process bring-up is config-derived, NEVER SLURM-sniffing** (`sharding.py`): the LM
  composition root branches on `runtime.distributed` (= `dp > gpus_per_node`) —
  distributed → `init_distributed(dp, gpus_per_node)` (`jax.distributed.initialize` +
  assert the realized device count equals `dp`); otherwise →
  `assert_inline_topology(dp)` (one process, exactly `dp` local devices).
  Once the config has decided we're distributed, the rank comes from jax's own SLURM
  bring-up (`jax.distributed.initialize` auto-detects it); the library reads no SLURM
  rank var itself — only the generic `PD_RANK` hint, and only to pick the HLO-dump
  writer (`experiments/lm/run.py::_enable_hlo_dump`). Do NOT revert to inferring
  distributedness from ambient `SLURM_PROCID` — that
  env is present in EVERY process on a SLURM box (incl. a pytest worker), so sniffing it
  wrongly fires `jax.distributed.initialize` mid-suite (the `test_pretrain` smoke
  failure).
- **`shard_batch` topology** (`sharding.py`): uses `make_array_from_process_local_data`
  so it's correct for BOTH single-process-many-devices and multi-process-1-device.
  Do NOT revert to the per-`process_index()`-slice idiom — it silently replicates one
  slice on single-process multi-device CPU.
- **`vendored_jax` is `param_decomp/vendored_jax` — a subpackage of the same `param-decomp` distribution**;
  no `sys.path` hacks anywhere. If an import fails, the install is broken — fix the env
  (`make install-dev`), don't add a path shim.
