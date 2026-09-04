# param_decomp — agent notes

Single-pool VPD trainer in JAX, **generic over vendored targets**: the engine sees a
target only through the `DecomposedModel` protocol (`model.py`) and the `ArchFamily`
grammar contract (`family.py`). The concrete targets — LM and toy alike — are the
sibling subpackage `param_decomp.targets` (one slice per architecture); the library's
layering (core imports NO target, targets import core only) is pinned by
`param_decomp/tests/core/test_runtime_standalone.py`. The semantics source of truth is `SPEC.md`
(normative pseudocode + numbered invariants, grounded in the stable torch
`param_decomp` impl). See `README.md` for the file map.

Open items: the persistent-source shape `nsc` and sigmoid parameterization are
deliberately refused. SPEC S24's two torch-parity quirks (PPGD warmup route-all,
fresh-PGD single routing draw) are pinned pending a team decision. tPD's target-pass
delta polarity (SPEC T10) and whether T3's faithfulness exclusion is definitional await
the algorithm's author-definition.

## Step-machinery rules

- **Atoms/vocabulary functions take no per-caller mode flags.** A step variant that
  needs different behavior gets its own draw dispatcher or its own step body — never a
  boolean or `isinstance` branch inside shared machinery. The step factories enumerate
  the shapes that exist; shared pieces stay shape-blind.
- **Why tidiness IS flexibility now**: with agents doing the rewriting, restructuring
  is cheap — the scarce resource is verifiability of intent. Narrow typed nouns, SPEC
  rows, RNG-chain pins and trajectory goldens are what make the next reshape an
  afternoon of bounded agent work instead of archaeology. Flexibility is maintained
  through pins and types, not through anticipatory parameterization.

## The one rule

**Every change is checked against SPEC.md, by invariant ID.** If a change deviates
from an invariant, either fix the change or (deliberately, with Oli) amend the spec —
never silently diverge. Cite IDs (`S14`, `N1`, …) in commit messages and reviews.

## Architecture in one breath

`model.py` defines `DecomposedModel[PreparedT]` — a `@runtime_checkable Protocol` with
ordered `sites`, `has_position_axis`, `site_output_keys`, one `clean_forward`, one
`masked_forward`, `component_activation_forward`, `target_weight_sq_norms`, `weight_deltas`, and
`recon_loss_fn` (LM:
`kl_per_position`). Activation identity is target-owned: core passes immutable frozensets of
canonical names, and each target parses and deterministically orders them into its private
sparse slot layout on first trace. No capture-plan type crosses the protocol, and core never
imports or
interprets a residual/block/matrix vocabulary. `ForwardResult.captures` carries one array per
requested physical activation. Transformer site names are not aliases for matrix inputs;
consumers ask directly for the vectors they need. An empty key set takes the target's
untouched no-capture path. Both forwards return one `ForwardResult`; masked execution receives
the exhaustive
`MaterializedMasking | StochasticMasking` union, preserving in-block stochastic mask
rebuilds without a deterministic/stochastic method grid. `PreparedT` is the target-private
compute layout returned by `prepare_compute_weights`; generic code may transport it only
back into that same target's methods, so cross-target prepared-weight mixing is type-invalid.
Component-linear execution receives the run's resolved `PlacementRules` explicitly:
`components.operands` controls each selected V/U matrix, `activations.external` the public
linear input/output waist, and `activations.component` both `x@V` and CI squashings. The
model travels paired with those rules as ONE `PlacedModel` bundle (model = pytree child,
rules = static on the treedef), assembled exactly once at run assembly — no downstream code
holds an unresolved (model, rules) combination, and the mesh is the rules' own
(`PlacementRules.mesh`), never a second threaded copy. `placement is None` on the bundle is
the decided unplaced (CPU/test) execution. The `DecomposedModel` protocol itself keeps its
per-call `placement` params — targets are untouched; the bundle is the single supplier.

The concrete implementation per target is an `eqx.Module` (`GLUDecomposedModel` — hosting
every LM family, GLU and SimpleMLP alike — `TMSDecomposedModel`, `ResidMLPDecomposedModel`) carrying its
FROZEN target weights as ARRAY FIELDS; the TRAINABLE V/U (`vu: ComponentStacks`) stays an
explicit METHOD ARG (separate lifecycle — own optimizer + checkpoint, C-sharded while the
frozen weights replicate). Flat site-name-keyed dicts remain the decomposition boundary;
the model threads into the jitted step as a pytree ARG (never a jit-closure constant — an
8B target becomes a multi-GB HLO constant; see "HLO-baking rule" below). The activation
waist comes in EXACTLY TWO shapes — positionless `[B, d]` (masks/CI `[B, C]`; the toys) or
one position axis `[B, P, d]` (masks/CI `[B, P, C]`; an LM, whose position axis is the
token sequence). `DecomposedModel.has_position_axis` declares which; the run threads the
extents as `positions: Positionless | Positioned(n_positions)` (matched exhaustively
wherever shapes are built — never a rank branch). Inside the step, masking / routing /
sources / imp-min read an opaque `leading = residual.shape[:-1]`; reductions are
`math.prod(shape[:-1])` / `axis=tuple(range(ndim-1))`. CI is independent over every
leading axis. `CIFn.has_position_axis` mirrors the model's, and `run_state.init_decomposition` asserts
they're equal (early fail) so the CI fn stays per-domain (RoPE over positions) without the
core adapting.
Adversarial sources (both adversaries) are configured by `source_shape: c | bc | sc | bsc`
(`configs.SourceShape`): each letter names a waist axis the source keeps full, missing
letters are stored size-1 broadcast axes, rank always matches the waist. Batch-B shapes
(`bc`/`bsc`) are batch-sharded, no cross-replica sync; batch-1 shapes are shared across
the batch (SPEC S16/D1). `sc`/`bsc` on a positionless target raise; per-step (fresh) PGD
rejects `sc` at validation; legacy spellings (`scope: {type: ...}`, `mask_scope`, the
verbose value names) are rejected at parse — no config aliases remain. Every (positions x source_shape) persistent
shape is written out in `init_sources_sharded`, so persistent PGD runs on the toys too.
Batch size is `pd.batch_size` uniformly — `DataConfig` carries no batch.
Each configured recon term retains its end-to-end comparison and may add S35's
`HiddenActsReconstruction` at explicit target-owned capture points, currently measured as
positive-coefficient relative MSE. The clean forward
captures the order-preserving union of CI inputs and every term's points; each masked draw
captures only its own points. Persistent adversaries default to `adversary_objective: e2e`,
which excludes S35 hidden-activation reconstruction from every source ascent while the outer
components/CI objective keeps it; explicit `term` mode makes the adversary ascend the
complete loss. CI-fn
numerics: GELU is exact-erf (`approximate=False`),
RMSNorm eps is `finfo(fp32).eps` (`CI_FN_RMS_EPS`) — SPEC §4.6. The
three EDGES are generic so non-LM (bio-style) targets fit: the model INPUT
(the opaque batch `clean_forward` / `masked_forward` consume, typed `Any` — token ids for
an LM, a dict for bio), the model OUTPUT (`ForwardResult.output: Any` — logits, a tuple of
heads, coords; field NAMES stay `*_logits` pending a deferred rename), and the recon
comparison (`recon_loss_fn(masked_output, clean_output) -> scalar`, default
`kl_per_position` so the LM path is byte-identical). The waist shape contract (all per-site
tensors in one forward share one `*leading` prefix) is enforced at trace time by
`@jaxtyped(typechecker=beartype)` on the core `step`, `masked_recon`, and the loss fns.
`train.py` is the generic step factory
(fp32 masters / bf16 compute) over the explicit-role loss surface
(`objective.LossSurface` — faithfulness, importance-minimality, the recon terms, and an
optional nonlinearity-locality term;
S10′ — each authored recon config compiles to ONE `ReconLossTerm` = routing sampler ×
mask-source strategy, every draw routing over ALL the model's sites, built from the
shared configs by `objective.build_objective`;
see LOSS_PARITY_DESIGN.md),
consuming `losses.py` (pure loss terms + schedules), `adversary.py` (persistent
adversarial state and optimization), and `masking.py` (mask construction shared across
source strategies and targets); `ci_fn.py` the shared CI transformer. The targets are the sibling
distribution: `param_decomp/targets/glu_transformer.py` the SHARED HF GLU-transformer
target machinery (site grammar, `FrozenAttn`/`GLULayer`/`GLUDecomposedModel`, the
scan/masked-forward engine, HF loading, and the target's own placement via
`.shardings(placement)`; the generic seeded-init-placed helpers are engine-side,
`init_placed.py`), with the model FAMILIES in their own files:
`param_decomp/targets/llama31.py` (vendored `LlamaConfig`, llama3 rope) and
`param_decomp/targets/qwen3.py` (`Qwen3FrozenAttn` — REQUIRED `q_norm`/`k_norm`
fields applied in the `_prep_qk` pre-RoPE hook; Qwen3's one structural delta). Nothing
in the shared file switches on a family; the model-name → family registry is composition-side
(`experiments/lm/config.py::HF_MODEL_VARIANTS`). Qwen3 JAX↔HF parity is pinned DIRECTLY
by `param_decomp/tests/targets/qwen3_hf_parity/` (a tiny-random `Qwen3ForCausalLM`
golden at fp32 tolerance + a slow real-weights logits check; goldens regenerate via its
torch-env `gen_hf_fixtures.py`). There is ONE
recon semantics: masks thread through the full token-input forward, loss is KL on final logits
(SPEC §2.3–2.5). Site-local recon is a conceptual no-no, not a "simplification".
`param_decomp/targets/llama_simple_mlp.py` is the second target (the pile-pretrained `LlamaSimpleMLP`,
t-9d2b8f02; sites `h.{i}.attn.{q,k,v,o}_proj` / `h.{i}.mlp.{c_fc,down_proj}`) —
config dispatch is `TargetConfig` (the HF GLU families) vs `LlamaSimpleMLPTargetConfig`, both composition-side
(defined in `experiments/lm/resolved.py`; `param_decomp/experiments/lm/config.py` reads the canonical schema DIRECTLY —
`build_experiment_config`/`load_config` — resolving each target's tiled
`decomposition.sites` via its `ArchFamily`), target build in the LM composition root
`param_decomp/experiments/lm/training.py::main` (`experiments/lm/run.py` is the pre-JAX
env bootstrap deferring to it). The slow plot metrics are computed
NATIVELY in JAX (`slow_eval.py`) — no torch export round-trip. They run IN-LOOP ONLY on
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
`render_permutation_figures` — it OOMs / breaks at production C BY DESIGN, no
special handling; for the positionless toys (TMS/ResidMLP) `toy_uv_eval.render_uv_metric`
renders it off the small on-host V/U + the probe CI as permutation source (cheap, no
gather), sharing `slow_eval.render_uv_figure` / `plot_uv_matrices` with the LM path.

`well_temperedness.py` measures whether components with higher causal importance
preactivations have a greater effect on reconstruction loss when ablated one at a time. It
compares components across all heads and layers, separately below 0, between 0 and 1, and
above 1. Named groups add the same measurement for subsets such as attention and MLP sites.

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
positions, remat_recon_forwards, remat_ci_fn, compiler_options,
sample_batch, evaluation, sink, profiling)` — the ONE train loop every target runs through (init/restore/finetune/faith-warmup
via `_init_or_restore_state`, the recon-grid step factory, orbax checkpointing, schedules,
SIGTERM-save). It reads the pydantic `PDConfig` / `Cadence` (`param_decomp.core.configs`)
DIRECTLY — optimizers / loss metrics / faith warmup / seed / steps — so there is
NO flattened mirror dataclass; the run identity rides in
`built_run.RunInstance`, and the composition-built objects (`ci_fn` arch, `data`, the decomposed target)
pass alongside. A target injects exactly three seams: the data source
(`sample_batch(step) -> residual`), the domain-bound `Evaluation` (typed operations + context factory, scheduled directly by core), and (for the LM) the perf token count.
`param_decomp/experiments/lm/training.py::train` is the thin LM caller (parquet
`sample_batch` + domain-bound CEandKL/CI-L0/PGD/attention operations; that
LM composition root is LM-ONLY (`experiments.lm.config.build_from_schema` validates
`LMExperimentConfig` and returns the built `LMRun`; target dispatch
(`load_run.load_target`) covers only `TargetConfig` / `LlamaSimpleMLPTargetConfig`).
`BuiltRun.target` is typed by the core
`built_run.TargetSites` protocol (just `.sites`), `BuiltRun.data` is generic; LM binds it
to `experiments.lm.resolved.ResolvedLMData`. The shared run-identity / CI-fn-arch helpers are public
composition-side for the toys to reuse: `experiments.config.run_instance` /
`experiments.toy_config.build_toy_ci_arch`.

The TMS + ResidMLP targets live at `param_decomp/targets/{tms,resid_mlp}.py` (the JAX
`DecomposedModel` + frozen target + in-process pretrain + identity-CI eval); each
`param_decomp/experiments/{tms,resid_mlp}/run.py` is the toy CPU composition root
(a module main) that builds the `ExperimentConfig` from the canonical schema and calls
`run_decomposition_training`. They are positionless and use the MLP CI fns. All CI-fn architectures live together in
`ci_fn.py`: `LayerwiseMLPCIFn` (positionless, one independent MLP per site mapping
`site_input [B,d_in] -> [B,C]`), `GlobalMLPCIFn` (one shared MLP over explicit input taps —
`TapSpec` keys + widths, DECOUPLED from the output sites, so several sites may share one
physical tap — concat/split pointwise over every leading axis: positionless on the toys,
per-token on an LM), and the LM `ChunkwiseTransformerCIFn`
(positioned, per-chunk transformers reading residual taps, stacked +
`lax.scan`'d with per-chunk remat, and **N per-site output heads** (one `[d_model, C_j]` per
site-slot). `n_blocks=0` degenerates that arch to `RMS-normed taps → in_proj → heads`:
position-LOCAL and attention-free, still `has_position_axis=True`, but affine on the NORMALIZED
tap — no hidden layer and blind to tap magnitude. A positioned target whose position count
makes O(P²) attention infeasible (pairwise positions) should take
`LayerwiseMLPCIArch(has_position_axis=True)`; the blockless chunk is a baseline.
The mesh is `(replicate, fsdp, tp)`. No axis is required to coincide with a hardware
boundary (`sharding.py`); the maintained seats AUTHOR `replicate` across nodes with each
node an `(fsdp, tp)` NVLink plane, and owner placement's zero-cross-node weight
collectives + node-local muon NS hold under exactly that authoring. `tp` shards declared
target dimensions and the CI output C
axis; the per-site heads keep site boundaries explicit rather than slicing a glued-ΣC head
mid-site. **Persistence layouts (÷N)**: the trainable V/U masters AND their
optimizer moments persist as target-declared semantic stacks (`ComponentStacks.stacks`; LM
targets group by matrix kind). Under owner placement, the stack
axis ÷`replicate` — whole matrices owned per node-group, zero cross-node weight collectives,
muon NS node-local — matrix d dims ÷`fsdp`, C ÷`tp`; SPEC D4. Placement is fallback-free:
one set of component rows places EVERY semantic group, and a stack that doesn't tile a
stack-sharded row refuses at `placement.from_config(spec, mesh, sites)` during config build
(pre-submit for a submitted run), naming the groups and the remedies — a tiling mesh, or a
placement with no stack sharding (`zero1` rests every master intra-matrix; its faithfulness
rows ARE that master layout). Mixed per-group placement is unrepresentable (no fallback
preset, no fallback rows in the schema). The resolved group census flows down as data; the
consumer boundary (`placement.component_stacks_shardings`) only validates the received
census, never re-decides. See PLACEMENT_DESIGN.md, "Presets"). The CI-fn
masters + moments keep intra-matrix ZeRO-1 (`("fsdp","replicate")` on d_model — fsdp-major,
so the ÷N→÷fsdp reconstruct is a pure all-gather over `replicate`; replicate-major would
cost a per-step grid-transpose collective-permute). Either way
the dominant optimizer-state memory scales 1/N, not the fixed 1/fsdp. The bf16
COMPUTE weights are materialized to the `fsdp`-sharded (÷fsdp) layout ONCE per step in ENTRY
(the cross-`replicate` gather, off the hot path — `placement.materialize_reduced_weights`, via
`component_stacks_to_compute_weights` / `ci_fn._reconstruct_ci_compute_weights`, BEFORE the
per-layer / per-chunk scan), landing a SMALL ÷fsdp-resident stack typed `reduced` over the
gathered axes (the mesh is jax-Explicit; the reduced typing defers the weight-grad reduction
to this boundary's transpose — one exit reduce-scatter, no in-loop cross-replicate weight
collectives); the scan body then reshards ONE layer's `fsdp` shard to full d_in transiently
(NVLink, freed each iteration) — NEVER a full-model `[n_layer, full_d_in, C]` weight stack
resident.
`run_state.init_train_state` takes the resolved `ci_fn_arch: CIFnArch`
(`LayerwiseMLPCIArch` / `GlobalMLPCIArch` / `ChunkwiseTransformerCIArch`; construction
dispatched by `ci_fn.build_ci_fn`) and uses replicated (not
C-sharded) V/U + CI for the tiny toys; the core `ci_fn.CIFnArch` admits all three and the
composition-side `experiments.toy_config.build_toy_ci_arch` builds the layerwise / global
arch from the toy `decomposition.ci` (validated end-to-end on CPU via
the ResidMLP composition root). Offline consumer runs over the toys are NOT wired
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
  `make_slow_eval_step`, `make_position_ci_step`,
  `make_*_attn_patterns_step`, and `make_faith_warmup_step` all follow this; each carries a
  comment at the step factory. The toy `run.py`s follow the same rule.
- This is why the methods take only the *runtime-varying* args (`vu`, `resid`, masks, …) and
  the frozen weights ride on `self`: `self` reaches the trace as the traced model arg.

## Invariants with sharp teeth (the ones that have actually bitten)

- **S3**: the recon target is the FROZEN-path `clean_forward(...).output`, never the
  `mask=1` decomposed identity (bf16 rounding + V/U in the stopped graph). An empty
  capture-key set selects the target's compact no-capture path.
- **S13/S15**: source updates go through the persistent Adam AND project to [0,1]
  after EVERY ascent — an unprojected drift past 1 has zero `clip` gradient and the
  entry dies.
- **S14**: the default `e2e` final ascent uses the source gradient of output
  reconstruction only. When hidden-activation reconstruction makes the outer objective differ,
  it retakes that gradient with pre-update θ and the same draws; otherwise it reuses the
  main backward's source-grad. Explicit `term` mode always reuses the complete term's
  source-grad. Neither source objective is scaled by the ppgd coeff.
- **N1**: fp32 masters everywhere (`optax.adamw(..., weight_decay=0.0)` — optax's
  default wd is 1e-4, torch's is 0).
- **`inv_freq` is a buffer, not a param** — `stop_gradient` in
  `ChunkwiseTransformerCIFn.__call__`.
- **S11**: uniform-k routing is per position over ALL the model's sites —
  `k ~ U{1..|sites|}`, then a uniform k-subset routes True; draws are fresh per step.

## Validation stack (run all before claiming correctness)

1. `pytest param_decomp/tests/core/ param_decomp/tests/targets/` — at the default device
   count AND `XLA_FLAGS="--xla_force_host_platform_device_count=4"`.
2. `param_decomp/tests/targets/equivalence/` — fixture-driven JAX-vs-frozen-golden
   per-term numeric equivalence (fp32, no RNG, zeroed attn). The torch references are
   FROZEN committed goldens (`equivalence/torch_reference.json` + `*.npz`, the sibling
   `simple_mlp_equivalence/*.npz`); the torch generators/verifier that produced them live
   only at the `torch-oracle` tag, so the runtime imports no torch. Regenerate goldens only when
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
reaches the engine as `ResolvedLMData.dir` — a directory of pre-tokenized parquet shards;
the loader takes `seq_len` as an explicit parameter (`ShardServer`), and the dir is not
required to exist until load time. How that directory is named, resolved, described,
and populated is not core's business — see `experiments/CLAUDE.md` and the root
CLAUDE.md. Data loading
obeys three invariants, not a format: (1) the batch schedule is a pure function of
`(seed, step)` (O(1) resume, no replay, prefetch-safe); (2) rank bring-up depends on no
external service; (3) by train time the data is a local, enumerable artifact. Any
loader satisfying all three is welcome — the parquet `ShardServer` does today.
Checkpoints are orbax sharded saves (no on-loop full-gather), TWO items per step —
`decomposition` (V/U + ci_fn, the
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
plus the run-instance fields —
top-level `run_name`, the
`runtime.remat_recon_forwards` memory/compute knob, and `wandb.group`/`wandb.tags`.
`run_id`/`out_dir` are NOT config fields: the launcher mints the id and passes it as
`--run-id`, and the run dir is a pure function of `data_root` + id
(`experiments.config.run_instance`).

**Fine-tune from a parent checkpoint** (`resume_provenance`, SPEC S33, LM-only). A fresh
run can initialize its trained decomposition (V/U + ci_fn) from a PARENT run's checkpoint
and continue under a DIFFERENT config (changed LR / coeffs / gamma / seq / batch / steps —
NOT changed C / sites / ci-fn arch). Add to the config:

```yaml
resume_provenance:
  # ABSOLUTE path — the trainer runs with cwd = the node workspace (a repo checkout), so
  # a relative path would resolve under the workspace, not the output runs dir.
  parent_run_dir: /abs/path/to/runs/p-xxxxxxxx
  parent_step: 175000
```

On the FIRST entry (own `ckpts/` empty) the trainer restores the `decomposition` item of
`parent_run_dir/ckpts/175000` onto the fresh reference; the optimizer states,
persistent sources, and `step` are FRESH (`step = 0`, no faith warmup) so the new LR /
gamma-anneal schedule recomputes over the new `cfg.steps` from 0. A subsequent SLURM requeue
(own `ckpts/` now non-empty) resumes from the run's own dir and ignores provenance.
`experiments/lm/training.py::assert_finetune_structural_compat` reads the parent's pinned
`launch_config.yaml` and
asserts matching sites (names + C) + ci-fn arch before the restore. Provenance flows into
`launch_config.yaml` + `wandb.config`. Run the module entry point with the new config.

**Mesh topology is explicit; allocation topology belongs to the launcher.**
`runtime.{replicate,fsdp,tp}` names the logical mesh directly, with world size their
product. No axis is required to coincide with a process or node boundary. The process
entry receives `local_device_count` from whoever allocated it; `initialize_topology`
uses that fact only for JAX process bring-up and asserts the realized world matches the
authored mesh.

`python -m param_decomp.experiments.lm.run <config> --data-root … --local-device-count N`
runs in the current allocation, minting and pinning its own identity when `--run-id` is
absent. An external launcher may mint the identity, pin the config and code revision, and
start the matching process topology; those deployment choices are not part of the library.

`main` enables JAX's persistent compilation cache
(`enable_persistent_compilation_cache`) at the config-authored
`runtime.compilation_cache_dir` (`~`-expanded; the seats author the per-user
`~/.cache/param-decomp/xla` — XLA's autotune subdir is unsafe for unrelated Unix users to
share, and per-user is an AUTHORED seat value, not a code default), shared across all
runs and all 8N ranks of one user, NOT per-run. The multi-minute train-step compile is
keyed by HLO + backend +
topology + jax/xla version, so a requeue/resume or a fresh run at the same config+topology
loads the executable from disk in seconds. Set after `initialize_topology` (the write gate
reads the distributed state) and before the first compile; threshold 60s
(`jax_persistent_cache_min_compile_time_secs`) so only the big compiles cache. Multi-host
safe on jax 0.10.1: jax gates the cache WRITE on `process_id == 0` (`compiler.py` — "Only
write cache entries from the first process … contention for writes on some filesystems"),
so all ranks read but only rank 0 writes — no shared-FS race. A multi-node run must
author the cache dir on a filesystem all its nodes share.

### Compile time (2026-07-06 probe grid)

- **Keep seeded inits few-outputs-under-jit**: a jit returning n_sites (hundreds of)
  sharded outputs — or n_chunks unrolled RNG bodies — is a multi-minute SPMD/layout
  compile. vmap-stack over the same per-site/per-chunk keys (bit-identical values),
  then fan out with a trivial slice jit. `init_component_stacks_placed` is the template;
  `init_ci_fn_placed` / `init_sources_sharded` follow it.
- **The `jit_step` compile (~5 min at dp32) is FLAT across graph structure**: C,
  CI-fn depth, and PPGD warmup all measured within noise (~83% priority-fusion).
  Don't chase graph-shrink refactors for compile time without new evidence.

## Gotchas

- **Process bring-up never infers mesh geometry from SLURM** (`sharding.py`).
  `initialize_topology(world_size, local_device_count)` receives the logical world from
  config and process-local allocation size from the entry boundary; when the world is
  larger it invokes `jax.distributed.initialize`, then asserts the realized device count.
  Rank comes from JAX's own cluster bring-up; the library reads no SLURM
  rank var itself — only the generic `PD_RANK` hint, and only to pick the HLO-dump
  writer (`experiments/lm/training.py::enable_hlo_dump`). Do NOT revert to inferring
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
