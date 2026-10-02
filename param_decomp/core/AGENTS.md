# param_decomp

A JAX implementation of the **single-pool** Parameter Decomposition (VPD) training
loop — the four-term loss (faithfulness + importance-minimality + stochastic subset
recon + persistent-PGD adversarial recon) as one `jax.jit` step, GSPMD-sharded,
**generic over vendored targets**: the engine sees a target only through the
`DecomposedModel` protocol; the concrete targets live in the sibling subpackage
[`param_decomp.targets`](../targets/README.md).

The trainer realizes the
"single-pool SPMD collapse" hypothesis: XLA + whole-step `jit` + GSPMD sharding replaces
the hand-written-NCCL multi-pool design with zero manual collectives.

`param_decomp/core/` is the engine layer of the `param-decomp` library, beside its
sibling subpackages `param_decomp.targets` (the concrete targets),
`param_decomp.pretrain` (the target-LM pretrainer), `param_decomp.target_ports`
(bit-parity JAX architectures), and the composition layer (`experiments`). Imports point
only downward — pinned by `param_decomp/tests/core/test_runtime_standalone.py`. Install the
library and development tools with `make install-dev`.

## What's here

| file | what |
|---|---|
| `model.py` | `DecomposedModel` — the interface a vendored LM target implements (ordered sites, flat site-keyed dicts, frozen pytree as runtime arg) |
| `train.py` | the step factory: one fused jit step over faith + imp-min + recon + optional nonlinearity, per-persistent-term fused final ascents, fp32 masters + bf16 compute |
| `losses.py` | pure faithfulness, importance-minimality, reconstruction-comparison, and nonlinearity losses + the jnp schedule evaluators every in-step scheduled quantity uses |
| `adversary.py` | adversarial source machinery: persistent state + Adam ascents, fresh sign-PGD init |
| `masking.py` | construction and materialization of explicit/stochastic component and weight-delta masks |
| `recon.py` | the recon term vocabulary (LOSS_PARITY_DESIGN.md): `ReconLossTerm` (one all-sites forward family = routing sampler × mask-source strategy), the mask-source strategies, the routing samplers, and the reconstruction specs |
| `objective.py` | `PDObjective` (`FaithfulnessTerm` / `MinimalityTerm` / the `ReconLossTerm` tuple / optional `NonlinearityTerm`) and `build_objective` — the shared loss configs compiled onto the explicit-role surface |
| `nonlinearity_eval.py` | standing per-component nonlinearity-unit statistics |
| [`ci_fn/`](ci_fn/README.md) | CI interface, separate transformer/MLP implementations, shared clipping and compute lifecycle |
| `checkpoint.py` | orbax sharded save/resume of plain and targeted PD states (adversary sources + moments included, no full-gather on the loop) |
| `ci_l0_eval.py` | target-generic `CI_L0`: components whose causal importance clears `ci_alive_threshold`, per site and per authored group, with the unselected term of a selected emission priced analytically |
| `hardware_utilization.py` | Compiled `StepCost` for HFU; analytical `ModelCost` for MFU and ideal compute time |
| `flops/` | Analytical model, CI, target, and optimizer FLOP rules, with shared work counts and parameter shapes in `types.py` |
| `recon_eval.py` | target-generic fresh-PGD reconstruction eval: opaque model inputs/outputs, model-owned `recon_loss_fn`, arbitrary leading axes |
| `slow_eval.py` | LIBRARY for the in-loop slow (plot) tier (in-loop only — no offline CLI): the `CIHistograms` / `ComponentActivationDensity` / `CIMeanPerComponent` reductions + renders, the config-gated `PermutedCIPlots` / `IdentityCIError` (off the `(T, C)` position CI), the `UVPlots` figure (`render_uv_figure` / `plot_uv_matrices`, shared by the LM in-loop naive-gather path and the toy `toy_uv_eval` cheap path), and the hidden-acts recon scalars. Torch-free numpy/matplotlib; logged under `slow_eval/figures/*` |
| `components.py` | the decomposition representation: `ComponentStacks` — V/U masters in target-declared semantic stacks, `site(name)` per-site views, `activation_axes` the one spelling of the waist's semantic axes |
| `gauge.py` | a component's two activations: the gauge-variant `x·V_c`, which depends on how `U` and `V` are scaled, and the gauge-invariant `(x·V_c)·‖U_c‖`; `u_norms_of` is the one computation of `‖U_c‖` |
| `decomposed_linear.py` | the placed decomposed-linear primitive: `site_forward`/`site_out` executing one site from its `SiteWeights` and per-forward mask, delta and route (each absent input the identity, skipping its work), under `PlacementRules`, a precompiled `PlannedComponentLinear`, or unplaced `None`; `constrain_component_activation` pins `[*leading, C]` tensors to the component-waist row |
| `placement.py` | `PlacementRules` — the typed placement table (components / activations / target rows, plus the preset name the CI architecture resolves its own rows from), the generic CI row binding (`bind_ci_fn_weight_rows`), preset resolution (`from_config`: `owner` / `zero1` / `ddp` on the three-axis mesh, the `*-replicated-resident` pair, the `*-replicated-resident-moe` pair and `zero1-replicated-resident-moe-replicated-ns` on the `(data, tp)` mesh), the compute-weight materialization (`materialize_reduced_weights`), and the stacked-muon staging claims (see PLACEMENT_DESIGN.md) |
| `muon_stacked.py` | the muon optimizer: per-kind batched Newton-Schulz at the `ns_compute` waypoint; `staging_hops`, the one-axis-per-reshard waypoint chain |
| `optimizer.py` | typed scheduled optimizer and its learning-rate state |
| `run_state.py` | optimizer construction and separate decomposition / training initializers; tracing them supplies checkpoint restore shapes |
| `tools/` | debug tools (`memreport.py` — proto memory-report + live-range peak attribution, `memory_ledger.py` — per-program buffer table + peak snapshot from a proto dump, `ledger_diff.py` — provenance-grouped diff of two ledger roots, `hlo_census.py`, `fit_check.py` — the AOT GPU-fit check) |
| `sharding.py` | generic GSPMD helpers (`initialize_topology`, `hsdp_mesh`, `place_cpu_arrays`, `place_target`, `shard_batch`) |
| `init_placed.py` | seeded init → placed arrays with no host-side full tree (`init_component_stacks_placed` / `init_sources_sharded`; `CIFnInitializer`, which `run_state.init_decomposition` places, and the architecture's own `seeded_ci_fn_initializer`; the few-outputs-under-jit compile doctrine) |
| `family.py` | `ArchFamily` (a target's matrix grammar as data: vocabulary + `name_of`/`parse`) + the family-parameterized `canonical_site_cs`/`site_specs` the targets delegate to. The block-structured `SiteTree` + `resolve_site_tree` (tiled c-spec → tree) live composition-side with the LM schema (`param_decomp/experiments/lm/config.py`) |
| `run.py` | training and evaluation kernels compile before the loop, which executes only native JAX executables; the generic ENGINE `run_decomposition_training` (pure library, no `main`/YAML): faith warmup, loop, metrics jsonl/wandb, in-loop slow renderer, orbax checkpoints, SIGTERM-save + requeue-resume. The LM composition root that reads YAML + builds the target lives composition-side (`param_decomp/experiments/lm/run.py`) |
| `built_run.py` | generic `BuiltRun[DataT, TargetT, PDT, CIFnArchT]`, `RunInstance`, and the target-sites protocol; domain data/eval plans live composition-side |
| `configs.py` | the torch-free pydantic config SCHEMA: routing + the `explicit` (toy) site spec + loss-metric + eval-metric configs, `PDConfig` / `Cadence` / `WandbConfig` / `ResumeProvenance` / `PlacementTableConfig`, and the `wandb.config` shaping helpers. The authored `decomposition.ci` configs, the tiled LM site specs (`GluTransformerCSpec`/`SimpleMlpCSpec`, `LayerSelection`) AND the LM's compute substrate (`RuntimeConfig` / `LaunchEnv`) speak each domain's vocabulary and live with the domain schemas, composition-side (`experiments/lm/config.py` chunkwise + tiled sites, `experiments/lm/runtime.py` the `runtime:` section, `experiments/toy_config.py` toy MLPs); core carries only the RESOLVED CI-fn arches (`ci_fn/`) and resolved flat sites |
| `base_config.py` | `BaseConfig` (frozen `extra=forbid` pydantic `BaseModel` + YAML/JSON round-trip), `Probability` |
| `schedule.py` | `ScheduleConfig` + the host evaluator `get_scheduled_value` (the traced twin `scheduled_value_traced` lives in `losses.py`) — a knot-based piecewise curve `max_val × frac(t)`, interp linear/cosine/hold; every scheduled quantity — LRs, gamma, merged-loss `adv_fraction` — routes through here |
| `../experiments/*/configs/` | the domain-owned self-contained run YAMLs (one file per run) |
| `../tests/core/` | tiny-target engine tests (incl. attention sites + heterogeneous per-site C), checkpoint resume, sharding, and the layering test (`test_runtime_standalone.py`, pinning composition → targets → core and the forbidden-import rule) |
| `../tests/targets/` | per-target parity/golden suites (torch↔JAX equivalence, stacked parity, Qwen3 HF parity, SimpleMLP torch fixtures) |

## Run

```bash
# From the repo root — one venv for the whole workspace:
make install-dev && source .venv/bin/activate

pytest param_decomp/tests/core/ param_decomp/tests/targets/

# GSPMD device-count invariance (simulated devices on CPU):
XLA_FLAGS="--xla_force_host_platform_device_count=4" \
  python -m param_decomp.targets.invariance_check --steps 3
```

## Design

- **Generic over vendored targets.** The trainer sees only the `DecomposedModel` fn-table
  (`model.py`): ordered `sites`, one `clean_forward`, one `masked_forward`, and
  `weight_deltas`. Both forwards accept an immutable frozenset of canonical activation keys
  and return `ForwardResult`; its `.captures` dictionary contains exactly one value per
  requested physical activation. Each target validates and deterministically orders those
  keys privately on first trace;
  core owns no activation grammar or capture-plan type. Every method is pure and the frozen
  pytree remains a *runtime arg* (a frozen 8B target closed over as a jit constant bakes
  multi-GB weights into the HLO). An empty key set takes the untouched no-capture path.
- **Generic over CI implementations.** `CIFnArchitecture` declares the target capture
  request and owns initialization and cost accounting. Each implementation resolves
  its own placement from the run's rules at initialization and holds it statically;
  core never names a CI placement type. Optimizer adapters consume declared parameter
  matrices and their Muon staging without inspecting concrete CI implementations. The engine forwards that request to the target, then hands the
  requested captures unchanged to CI evaluation. Capture names are opaque to the
  engine. CI owns input preparation, including component-activation projections.
  `CIFn.prepare` casts the parameters and relayouts them into their compute layout,
  returning the implementation's own type; differentiation maps the gradients back.
  Evaluation also receives the target's prepared components, whose
  `component_activations(site, x)` is `x @ V` in the target's own compute layout; a CI
  conditioned on component activations reads it, and its V cotangent joins the
  reconstruction's in the prepared components.
- **One jit'd step, functional minimax.** The persistent adversary (source stacks +
  their source optimizer state) lives in the algorithm’s training state and is threaded through; `n_warmup`
  supplemental ascents + one final ascent whose gradient comes from the same backward
  as the param grads.
- **GSPMD, not pools.** Data over the mesh's batch axes (`placement.batch_axes`), params placed by the target's sharding plan,
  `jax.jit` inserts every collective. The torch `reduce_source_grads` dance is absorbed
  by autodiff of the global-mean loss. Validated by `invariance_check.py`: the
  trajectory is device-count-invariant up to float reassociation.

## param_decomp — agent notes

Single-pool VPD trainer in JAX, **generic over vendored targets**: the engine sees a
target only through the `DecomposedModel` protocol (`model.py`) and the `ArchFamily`
grammar contract (`family.py`). The concrete targets — LM and toy alike — are the
sibling subpackage `param_decomp.targets` (one slice per architecture); the library's
layering (core imports NO target, targets import core only) is pinned by
`param_decomp/tests/core/test_runtime_standalone.py`. See `README.md` for the file map.

Open items: the persistent-source shape `nsc` and sigmoid parameterization are
deliberately refused. Two torch-parity quirks (PPGD warmup route-all,
fresh-PGD single routing draw) are kept for now. tPD's target pass
treats the delta exactly as plain VPD does; whether its faithfulness exclusion
is definitional stays open.

## Sequence inputs

A sequence is one LM batch row. It may contain several source documents or a fragment
of one; a document may continue across sequences. `lm.batch.LMBatch` contains token IDs.
The shared transformer accepts `LMBatchWithDocuments(batch, sequence)`, which adds a `SequenceLayout`.
Qwen3.6 accepts only `LMBatch` and treats each sequence as uninterrupted and unpadded.
Routing is independent of document metadata: `LMBatchWithRouting[Batch]` retains either
batch representation alongside the clean expert selection.
The engine transports each target's complete conditioning without interpreting it.
`ForwardResult.sequence` supplies the same document layout to CI; token-only and
positionless targets return `None`. Target attention is causal and CI attention selects
bidirectional or causal within the supplied boundaries. Llama accepts packed documents; dense
Qwen3 and Qwen3.6 reject document-aware datasets.

## Step-machinery rules

- **Atoms/vocabulary functions take no per-caller mode flags.** A step variant that
  needs different behavior gets its own draw dispatcher or its own step body — never a
  boolean or `isinstance` branch inside shared machinery. The step factories enumerate
  the shapes that exist; shared pieces stay shape-blind.
- **Why tidiness IS flexibility now**: with agents doing the rewriting, restructuring
  is cheap — the scarce resource is verifiability of intent. Narrow typed nouns, RNG-chain pins and trajectory goldens are what make the next reshape an
  afternoon of bounded agent work instead of archaeology. Flexibility is maintained
  through pins and types, not through anticipatory parameterization.

## Architecture in one breath

`model.py` defines `DecomposedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT]` — a `@runtime_checkable Protocol` with
ordered `sites`, `has_position_axis`, `site_output_keys`, one `clean_forward`, one
`masked_forward`, `component_activation_forward`, `target_weight_sq_norms`, `weight_deltas`, and
the two output operations on the target's declared output type `Out` — `recon_loss_fn`
(LM: `lm_output_kl_per_position`) and `pin_output_batch` — both pure `@staticmethod`s.
Activation identity is target-owned: core passes immutable frozensets of
canonical names, and each target parses and deterministically orders them into its private
sparse slot layout on first trace. No capture-plan type crosses the protocol, and core never
imports or
interprets a residual/block/matrix vocabulary. `ForwardResult[Out, Conditioning].captures` carries one array per
requested physical activation. Transformer site names are not aliases for matrix inputs;
consumers ask directly for the vectors they need. An empty key set takes the target's
untouched no-capture path. Both forwards return one `ForwardResult[Out, Conditioning]`. Core constructs the logical
per-site `MaterializedMasking | StochasticMasking | SourceMasking` union; the target's
`prepare_masking` translates it into its own `PreparedMaskingT`, which only that target's
`masked_forward` consumes. All layer grouping, padding, draw keys, and token routes are
prepared before crossing back into core. Core transports this private layout opaquely.
`prepare_stochastic_masking(ci)` returns a draw callable with shared prepared CI; training
constructs it once per stream inside the traced step and reuses it across stochastic
recon terms. Draws prepare only their own keys. Eager callers materialize a
complete logical recipe with `materialize_masking`; component masks and delta masks
remain one `MaterializedMasking` value through reconstruction callbacks. Routes are a
separate axis: the routing sampler produces them, and `masked_forward` takes them beside
the prepared masking (`routes: SiteRoutes | None`). Fresh PGD
materializes before target preparation, while persistent-source recipes retain the
target's existing composition boundary. Checkpointed targets retain
mask recomposition in their blocks; positionless targets prepare concrete masks eagerly.
`PreparedT` is the independent target-private weight layout returned by
`prepare_compute_weights`; generic code transports it only back into that same target's
methods and into CI evaluation, whose only view of it is the `ComponentActivations`
protocol, so cross-target prepared layouts cannot be mixed.
`Conditioning` is the complete input to masked execution: the original input for dense
and positionless targets, or `lm.batch.LMBatchWithRouting[Batch]` for a routed target. The latter pairs
the complete LM batch with the clean pass's `BlockSelection` (per-layer top-k indices and weights), so a
masked forward and its generic `CIFn[Conditioning]` receive the same routing and input.
The target returns it as `ForwardResult.conditioning`; `masked_forward(prepared, conditioning, ...)`
and `CIFn.__call__(taps, conditioning, ...)` consume it. Core never looks inside.
Qwen masked execution preserves the selected indices and recomputes mixing weights from
its own residual at those indices. Captures and CI values never supply the routing.
Component-linear execution receives the run's resolved `PlacementRules` explicitly:
`components.operands` controls each selected V/U matrix, `activations.external` the public
linear input/output waist, and `activations.component` both `x@V` and CI squashings. The
TP-sharded presets reuse one default C assignment across component operands,
CI output computation and component activations; their owned-expert variants reuse it
for whole experts. Explicit tables retain independent row choices, subject to the
chosen execution strategy's constraints. Persistence and Muon staging remain
separately declared. The
model travels paired with those rules as ONE `PlacedModel` bundle (model = pytree child,
rules = static on the treedef), assembled exactly once at run assembly — no downstream code
holds an unresolved (model, rules) combination, and the mesh is the rules' own
(`PlacementRules.mesh`), never a second threaded copy. `placement is None` on the bundle is
the decided unplaced (CPU/test) execution. The `DecomposedModel` protocol itself keeps its
per-call `placement` params — targets are untouched; the bundle is the single supplier.
`PlacedModel.prepare_compute_weights` casts master components and forwards that placement
through the target's materialization path. A `CIFn` carries the placement its
architecture resolved at initialization; callers explicitly prepare it (`prepare`) and
call the prepared fn, which casts its taps and squashes its own preactivations.

Selected CI, masks and captured component activations use token-ordered
`SelectedCI`: values `[batch, position, selected_slot * components_per_expert]`
and their pinned block indices. Under placement, batch axes follow data placement;
selected values and indices replicate over TP. Expert-sharded weights and routed
computation remain local to their owners: selected heads unsort their outputs at
this boundary, and masked expert forwards sort token masks into their jobs. Dense
sites retain ordinary C-sharded arrays. Frequencies reduce over the global token
population before their declared statistics placement is established. Source draws
retain token/selected-slot/component identity; delta draws remain in token coordinates.

Replicated expert compute uses a resident MoE table with `tp: 1`; the singleton
expert-owner axis still carries selected CI without distributing experts. Batch
parallelism remains on `data`, independently of the resident weights. Optimizer
storage may remain data-sharded. A small CI chunk stack need not divide the data
mesh under `zero1-replicated-resident-moe-replicated-ns` (named directly, or as an
explicit table's `ci_fn` preset): every Newton–Schulz staging row is replicated, so
each data replica orthogonalizes whole matrices. Optimizer storage remains separately
declared. An explicit table makes the same choice for component Newton–Schulz with
`components.ns_compute: {}`.

Persistent masks carry `source_mask.SourceMaskIngredients`: one CI routing frame,
its aligned source payload, and the independent token-coordinate delta source.
`masking.read_source_mask(ci, source)` gathers source storage into that frame;
`source_masking` returns complete pairs keyed by decomposed site. The target's preparation
boundary owns site-to-layer grouping and padding; GLU zero-fills absent layers leaf by
leaf, while positionless targets materialize the unchanged per-site recipe.
Composition accepts this pair, never a second routed CI bundle. Storage partitioning
and independent producer order remain separate from the chosen mask frame.
Merged persistent/stochastic terms select each document's source family in that same
frame before stacking. Fresh selected-source draws use canonical token/slot coordinates;
expert routing changes their representation, not their RNG identity. Delta and site
routing stay in logical token coordinates, and source gradients reach only adversarial
documents. Qwen recomposes each pair inside checkpointed blocks; GLU/Llama
composes its stacked pairs eagerly, preserving its existing memory policy.

The expert execution strategy is independent of CI emission. `expert_implementation`
selects dense masked token/expert matmuls or a routed grouped-matmul backend in both
target and CI banks. Dense computation consumes pinned token indices and mixing weights directly and still
emits `SelectedCI` at its public boundaries; only routed computation builds job schedules.
Internal dense coefficient tables do not change the selected interface or the global
component partition. Dense execution is the numerical
reference for routed implementations.

`PlacementRules.frequency_sharding` derives the `[C]` statistics layout from the public
component row. Frequency EMA initialization, runtime frequency statistics and
the deviceless fit check consume that same declaration. `per_component_frequencies`
first reduces each representation in its natural layout, then
`ForwardSubstrate.component_frequencies` explicitly establishes the declared layout. Dense C-sharded
CI naturally retains its C partition; token-ordered selected CI produces a replicated
`[C]` vector whose final placement partitions the small statistic. This conversion is
not asserted to be a no-op. CI internal computation, public activation rows and source
storage retain independent placement choices. Generic blocked linears likewise convert
from their internal expert/component layout at the public flat-C boundary.

The concrete implementation per target is an `eqx.Module` (`TransformerDecomposedModel` — hosting
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
Training adversarial sources always own independent state for every batch row.
Persistent configs, fresh training configs, and resolved strategies carry `configs.BatchSourceShape`
(`bc | bsc`): `bc` shares a row's source across positions, while `bsc` keeps every
position. Both keep the full batch axis and shard it over the mesh's data axes.
`bsc` on a positionless target raises. Each source has the waist's rank; omitted
position axes are size-1 broadcast axes. Fresh evaluation PGD configs require shared
`c` sources and random initialization. Training and evaluation use separate config
types with the same `PGDReconLoss` YAML tag, selected by their containing union;
neither config can enter the other role's union. Fast and slow eval configs are
siblings with distinct tags and a shared attack schema. `sc` is not configurable. Every (positions x source_shape)
persistent layout is enumerated in `init_placed._source_leading`.
Persistent sources PERSIST as target-declared semantic stacks (`adversary.SourceStacks`,
the same `SiteSpec.group` grouping `ComponentStacks` uses: one stack-major
`[stack, *leading, C]` / `[stack, *leading, block, c]` stack per group plus
`[stack, *leading]` deltas; stack axis replicated, leading batch axis on `data`, C / the
block axis (`expert`) on `tp`). The source optimizer state and checkpoint tree mirror that
layout; every CONSUMER (mask formation, eval probes) reads the site-keyed view
(`site(name)` / `per_site()`), whose stack-index slices are views — core never sees a
kind vocabulary.
Pooled sources declare `pool: {size_per_batch_element: K}`. Each global batch
index owns K cross-site particles, stored with logical `[batch, particle]` leading
axes. Minipools co-shard with the batch; the particle axis replicates and the
within-minipool gather and its gradient stay local. Particle count and sampling
are independent of mesh shape. Sampling maps over the stored stacks before
reading the per-site view; only the consumer's batch/position shape crosses into the
sampler. Particle count and placement come from storage.
Batch size is `pd.batch_size` uniformly — `DataConfig` carries no batch.
Each reconstruction term retains its model-output comparison and an ordered tuple of
`AuxiliaryReconstructionConfig` groups. A group defines a name, coefficient, and explicit
`CaptureReconstruction(capture, distance)` entries; admitted distances are relative squared
error and categorical KL from logits. Core treats capture names and metric labels as
opaque data. Every group averages its own comparisons before applying its coefficient. Captures that
masking cannot change remain in that authored mean; the engine does not judge their usefulness.
The clean forward captures the union of CI inputs and every requested value; each masked
forward captures only its term's values. Capture pairs have matching shapes, independently
per point, and scoring preserves the target's opaque output representation. Persistent
`adversary_objective: e2e` excludes all auxiliary groups from every source ascent while
the components/CI objective retains them; explicit `term` includes them.
CI-fn
numerics: GELU is exact-erf (`approximate=False`),
RMSNorm eps is `finfo(fp32).eps` (`CI_FN_RMS_EPS`). The
three EDGES are generic so non-LM (bio-style) targets fit: the model INPUT
(the opaque batch `clean_forward` / `masked_forward` consume, typed `TargetIn` — token ids for
an LM, a dict for bio), the model OUTPUT (`ForwardResult[Out, Conditioning].output` — `Out` is the
target's declared type, named at every core seam that carries an output: `Array` logits or
the factored `StreamedLinearOutput` for an LM (`targets.lm_output.LMOutput`), a tuple of
heads, coords), and the recon comparison (`recon_loss_fn(masked_output, clean_output) ->
scalar`; the LM binds `lm_output_kl_per_position`). Core never inspects an output: the
two things it does with one — compare two, pin one's batch axis — are the target's own
`recon_loss_fn` / `pin_output_batch`, composed via `ForwardSubstrate` and the eval step
factories, and everything else core knows about an output derives from those.
Code in core that carries a model or a forward result without caring about its output is
generic in `Out` (`def f[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT](model: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT], ...)`); a bare
`PlacedModel` / `ForwardResult` / `ForwardObservations` is a type error. The waist shape contract (all per-site
tensors in one forward share one `*leading` prefix) is enforced at trace time by
`@jaxtyped(typechecker=beartype)` on the core `step`, `masked_recon`, and the loss fns.
`train.py` is the generic step factory
(fp32 masters / bf16 compute) over the explicit-role loss surface
(`objective.PDObjective` — faithfulness, importance-minimality, the recon terms, and an
optional nonlinearity-locality term;
each authored recon config compiles to ONE `ReconLossTerm` = routing sampler ×
mask-source strategy, one forward per step routing over ALL the model's sites, built from the
shared configs by `objective.build_objective`;
see LOSS_PARITY_DESIGN.md),
consuming `losses.py` (pure loss terms + schedules), `adversary.py` (persistent
adversarial state and optimization), and `masking.py` (mask construction shared across
source strategies and targets); `ci_fn/` the CI interfaces and implementations. The targets are the sibling
distribution: `param_decomp/targets/transformer.py` the SHARED HF GLU-transformer
target machinery (site grammar, `FrozenAttn`/`TransformerLayer`/`TransformerDecomposedModel`, the
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
recon semantics: masks thread through the full token-input forward, loss is KL on final logits. Site-local recon is a conceptual no-no, not a "simplification".
`param_decomp/targets/llama_simple_mlp.py` is the second target (the pile-pretrained `LlamaSimpleMLP`,
t-9d2b8f02; sites `h.{i}.attn.{q,k,v,o}_proj` / `h.{i}.mlp.{c_fc,down_proj}`) —
config dispatch is `TargetConfig` (the HF GLU families) vs `LlamaSimpleMLPTargetConfig`, both composition-side
(defined in `experiments/lm/resolved.py`; `param_decomp/experiments/lm/config.py` reads the canonical schema DIRECTLY —
`build_experiment_config`/`load_config` — resolving each target's tiled
`decomposition.sites` via its `ArchFamily`), target build in the LM composition root
`param_decomp/experiments/lm/training.py::main` (`experiments/lm/run.py` is the pre-JAX
env bootstrap deferring to it). The slow plot metrics are computed
NATIVELY in JAX (`slow_eval.py`) — no torch export round-trip. They run IN-LOOP ONLY on
`eval.slow_every` next to the fast pass (there is NO offline/retrospective
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

`experiments/lm/arithmetic_eval.py` is a config-gated LM-only figure tier (`ArithmeticCIGrid`, on
`eval.slow_every`) for inspecting how the decomposition reconstructs a `target model`'s
modular-arithmetic mechanism (Feucht et al.'s L18 addition neurons). The probe is a FIXED
`a x b` operand grid of `"<a><op><b>="` prompts (one prompt per row, all one token length,
the `=` answer at a constant position) — NOT the streaming corpus, so it brings its own
batch. The probe is a specification (`operation` + `a_range`/`b_range` on the metric config), not a
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
are peers in `ci_fn/` (differing by domain, not status), not a toy carve-out. The
generic engine is `run.py::run_decomposition_training(pd, cadence, run, model, ci_fn,
positions, remat_recon_forwards, remat_ci_fn, compiler_options,
sample_batch, evaluation, sink, profiling)` — the ONE train loop every target runs through (init/restore/finetune/faith-warmup
via `_start_training`, the recon-grid step factory, orbax checkpointing, schedules,
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
`ci_fn/`: `LayerwiseMLPCIFn` (positionless, one independent MLP per site mapping
`site_input [B,d_in] -> [B,C]`), `GlobalMLPCIFn` (one shared MLP over explicit input taps —
`TapSpec` keys + widths, DECOUPLED from the output sites, so several sites may share one
physical tap — concat/split pointwise over every leading axis: positionless on the toys,
per-token on an LM), and the LM `BackboneCIFn` over a `ChunkwiseTransformerBackbone`
(positioned, per-chunk transformers reading residual taps, stacked +
`lax.scan`'d with per-chunk remat, and **N per-site output heads** (one `[d_model, C_j]` per
site-slot). `n_blocks=0` degenerates that arch to `RMS-normed taps → in_proj → heads`:
position-LOCAL and attention-free, still `has_position_axis=True`, but affine on the NORMALIZED
tap — no hidden layer and blind to tap magnitude. A positioned target whose position count
makes O(P²) attention infeasible (pairwise positions) should take
`LayerwiseMLPCIFnArch(has_position_axis=True)`; the blockless chunk is a baseline.
Both transformer CI attention variants carry a required `implementation: flash | xla`.
The shared dense/MoE attention half uses that choice directly, independently of the
target's attention implementation. cuDNN requires a GPU and half-precision operands;
unsupported execution fails instead of changing the requested backend.
The mesh is `(replicate, fsdp, tp)` — or, for the `*-replicated-resident` placements
(whose bf16 working copy is resident whole, so no fsdp axis exists), the two-axis
`(data, tp)`. No axis is required to coincide with a hardware
boundary (`sharding.py`); maintained multi-node configurations assign `replicate`
across nodes with each node an `(fsdp, tp)` device plane. Owner placement's
cross-node weight-collective and node-local Muon properties depend on that assignment. `tp` shards declared
target dimensions and the CI output C
axis; the per-site heads keep site boundaries explicit rather than slicing a glued-ΣC head
mid-site. **Persistence layouts (÷N)**: the trainable V/U masters AND their
optimizer moments persist as target-declared semantic stacks (`ComponentStacks.stacks`; LM
targets group by matrix kind). Under owner placement, the stack
axis ÷`replicate` — whole matrices owned per node-group, zero cross-node weight collectives,
muon NS node-local — matrix d dims ÷`fsdp`, C ÷`tp`. Placement is fallback-free:
one set of component rows places EVERY semantic group. A stack that doesn't tile a
stack-sharded row is placed by PADDING the persist stack with trailing all-zero matrices
(`StackCensus.stack_pad` — an enumerated fact, never shape-inferred: the V/U groups'
`GroupCensus`, mirrored on `ComponentStacks.stack_pads`; the chunkwise CI fn's chunk
stack at its resolved placement's `chunks`, resolved where the CI arch meets its rows
(the arch's `resolve_placement` over
`ci_fn/implementations/chunkwise/placement.resolve_chunk_census`, at its `initialize`) and read back as the fn's derived `census`. The V/U entry
strips pads BEFORE its gather — through `placement.padded_entry_waypoint`, so the
cross-`data` all-gather moves the real stack only — and the layer scan never sees them.
The CI fn gathers its padded stack whole into the compute layout, where the stack axis
is unsharded, and slices the real chunks off locally before the chunk scan; the
faithfulness lane rides the V/U pads as exact
zeros, and wd=0 keeps every pad at zero); anything else
the rows cannot place refuses at `placement.from_config(spec, mesh, sites)` /
the chunkwise archs' `resolve_placement` during config
build (before execution) with the remedies named. Mixed per-group placement
is unrepresentable (no fallback preset, no fallback rows in the schema). The resolved
censuses flow down as data; the consumer boundaries (`placement.component_stacks_shardings`,
the CI fn's `shardings` / `materialize_ci_compute_weights`)
only validate the received census, never re-decide. See PLACEMENT_DESIGN.md, "Presets"
and "Persist-stack padding"). Under `zero1` the CI-fn
masters + moments keep intra-matrix ZeRO-1 (`("fsdp","replicate")` on d_model — fsdp-major,
so the ÷N→÷fsdp reconstruct is a pure all-gather over `replicate`; replicate-major would
cost a per-step grid-transpose collective-permute); under `owner` they stack-cut like the
V/U masters. Either way
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
`run_state.init_pd_state` and `init_targeted_pd_state` take a
`ci_fn_initializer: CIFnInitializer[Conditioning]`, which the composition root builds —
the architecture's own `seeded_ci_fn_initializer`, or a domain initializer that reads data
(the LM's is `experiments.lm.ci_fn_init`). Its CI fn is the resolved
`CIFnArchitecture[Conditioning]`'s (`LayerwiseMLPCIFnArch` / `GlobalMLPCIFnArch` /
`ChunkwiseTransformerCIFnArch` / `GlobalTransformerCIFnArch` — one transformer over every
decomposed block's taps / `BlockSelectedChunkwiseTransformerCIFnArch` — the block-selected
sibling: per-stage chunks of
concat-wide selected block banks dispatched by the clean forward's pinned selection
(`components.BlockSelection`, the CI fn's `pinned` argument — narrowed at entry, failing
closed on any other pinned type; the selection is never read from taps), block-factored
sites emitting `components.SelectedCI` bundles (values + block indices as ONE value; the
full C axis unconstructible outside test oracles) through per-block heads fused into the
last CI block's bank entries; each constructed by its own `initialize`) and uses
replicated (not C-sharded) V/U + CI for the tiny toys; the
composition-side `experiments.toy_config.build_toy_ci_arch` builds the layerwise / global
`ToyCIFnArch` from the toy `decomposition.ci` (validated end-to-end on CPU via
the ResidMLP composition root). Offline consumer runs over the toys are not wired
(`experiments.lm.load_run.build_target` / `run_metadata` are LM-only).

## The HLO-baking rule (traced model arguments)

The decomposed model is an `eqx.Module` whose frozen target weights are ARRAY FIELDS — a
Llama-8B target is multi-GB. Therefore:

- **Every compiled function that touches a model receives the model as a TRACED
  ARG** — never `@jax.jit` over a function that CLOSES OVER an array-bearing model. A closed-
  over model is a jit *constant*: its arrays bake into the HLO (multi-GB constant tensors,
  recompiled per concrete model). As a traced arg, the array leaves are dynamic inputs and
  the static fields (`sites`, `eps`, `has_position_axis`) bake harmlessly.
- **Step factories read only STATIC config off the closed-over `model_static` at trace-setup**
  (`model_static.site_names`, `model_static.sites`, and the two output operations
  `model_static.recon_loss_fn` / `pin_output_batch` — each a `@staticmethod`, pure,
  holding no arrays, so closing over it is safe). All ARRAY access goes through the
  model ARG (named `model` inside the jitted fn). `make_train_step`, `make_eval_step`,
  `make_lm_batch_context_step`, `make_*_attn_patterns_step`, and `make_faith_warmup_step`
  all follow this; each carries a comment at the step factory. The toy `run.py`s thread
  the model as a traced arg too. Training and evaluation factories return pure functions;
  the run boundary lowers and compiles them with native JAX before entering the loop. The
  loop calls those compiled executables directly.
- This is why the methods take only the *runtime-varying* args (`vu`, `resid`, masks, …) and
  the frozen weights ride on `self`: `self` reaches the trace as the traced model arg.

Main optimizer LR schedules live in `ScheduledOptimizerState`, alongside the applied
rate, integer update count, and inner Optax state. Loss
terms own their `RuntimeSchedule` magnitudes as fp32 array leaves of
each algorithm's objective; schedule knots and loss structure stay static. Config
resolution validates an objective under `eqx.filter_eval_shape` because it precedes
`initialize_topology`, and a concrete magnitude would initialize a backend, which forbids
the multi-node `jax.distributed.initialize`; the state initializers build eagerly after it.
`PDState` carries `PDTrainingState`; `TargetedPDState` carries
`TargetedPDTrainingState`, including optional CI-scaled decay. Objectives describe the
penalties; training states own their frequency estimators. An EMA estimator carries
its running estimate and observation count. Target and non-target estimates evolve
independently. Both algorithms admit batch or EMA frequency estimation. Their initializers and steps preserve these concrete types. Changing these
magnitudes can reuse the persistent executable cache at the same shapes, backend,
topology, and compiler options. Both plain and targeted steps use the ordinary
filtered JIT boundary. Each optimizer evaluates its curve once per update; CI-scaled decay and
LR logs read the applied rate from that same optimizer state. Training checkpoints
include the objective's scalar values.

## Invariants with sharp teeth (the ones that have actually bitten)

- The recon target is the FROZEN-path `clean_forward(...).output`, never the
  `mask=1` decomposed identity (bf16 rounding + V/U in the stopped graph). An empty
  capture-key set selects the target's compact no-capture path.
- Source updates use the configured optimizer and project to [0,1]
  after EVERY ascent — an unprojected drift past 1 has zero `clip` gradient and the
  entry dies.
- The default `e2e` final ascent uses the source gradient of output
  reconstruction only. When a reconstruction auxiliary makes the outer objective differ,
  it retakes that gradient with pre-update θ and the same draws; otherwise it reuses the
  main backward's source-grad. Explicit `term` mode always reuses the complete term's
  source-grad. Neither source objective is scaled by the ppgd coeff.
- FP32 masters everywhere (`optax.adamw(..., weight_decay=0.0)` — optax's
  default wd is 1e-4, torch's is 0).
- **`inv_freq` is a buffer, not a param** — `stop_gradient` in
  `ChunkwiseTransformerBackbone.preactivations`.
- Uniform-k routing is per position over ALL the model's sites —
  `k ~ U{1..|sites|}`, then a uniform k-subset routes True; draws are fresh per step.

## Validation stack (run all before claiming correctness)

1. `pytest param_decomp/tests/core/ param_decomp/tests/targets/` at the default device
   count, then the same paths with `-m multidevice --runmultidevice` and
   `XLA_FLAGS="--xla_force_host_platform_device_count=8"`. This matches the selection
   and topology of `make test-multidevice`: three-axis placement fixtures need eight
   devices, while single-device entry-point tests require the default process.
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
   to device count up to float reassociation.

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
Checkpoints are Orbax sharded
saves (no on-loop full-gather), TWO items per step — `decomposition` (V/U + ci_fn, the
product every consumer restores alone) and `training` (opt states + adversaries + step,
trainer-only) — a clean break, with no in-code compatibility reader for pre-split
`default`-item runs. Existing checkpoints require a one-off external migration.
Resume with a changed config is refused by byte comparison. Validation before a long
run must exercise save and resume at the intended per-process shape.

A run config is ONE self-contained yaml: the experiment schema
(`param_decomp.experiments.config.ExperimentConfig` over the core
`param_decomp.core.configs` pieces — `pd`/`data`/`eval`/`cadence`/`target`/`wandb`, plus
`runtime` on `LMExperimentConfig`: the compute substrate is per-domain, and a toy — single
device by construction — declares no `runtime:` section at all)
plus the run-instance fields —
top-level `run_name`, the
`runtime.remat_recon_forwards` memory/compute knob, and `wandb.group`/`wandb.tags`.
`run_id`/`out_dir` are NOT config fields: the entry point mints the id unless the caller
passes `--run-id`, and the run dir is a pure function of `data_root` + id
(`experiments.config.run_instance`).

**Fine-tune from a parent checkpoint** (`resume_provenance`, LM-only). A fresh
run can initialize its trained decomposition (V/U + ci_fn) from a PARENT run's checkpoint
and continue under a DIFFERENT config (changed LR / coeffs / gamma / seq / batch / steps —
NOT changed C / sites / ci-fn arch). Add to the config:

```yaml
resume_provenance:
  # ABSOLUTE path — the trainer's working directory need not be the runs dir, so a
  # relative path would resolve under the working directory, not the output runs dir.
  parent_run_dir: /abs/path/to/runs/p-xxxxxxxx
  parent_step: 175000
```

On the FIRST entry (own `ckpts/` empty) the trainer restores the `decomposition` item of
`parent_run_dir/ckpts/175000` directly into the step's input layout; the optimizer states,
persistent sources, and `step` are FRESH (`step = 0`, no faith warmup) so the new LR /
gamma-anneal schedule recomputes over the new `cfg.steps` from 0. A later entry with an
existing checkpoint resumes from the run's own directory and ignores provenance.
`experiments/lm/training.py::assert_finetune_structural_compat` reads the parent's pinned
`launch_config.yaml` and asserts matching sites (names + C) + ci-fn arch before the restore.
Provenance flows into `launch_config.yaml` and `wandb.config`. Run the LM module entry point
with the new config.

**Mesh topology is explicit; allocation topology belongs to the caller.**
`runtime.mesh` names the logical mesh directly — `{replicate: R, fsdp: F, tp: T}`, or
`{data: D, tp: T}` for a `*-replicated-resident` run — with world size the axes'
product. No axis is required to coincide with a process or node boundary. The process
entry receives `local_device_count` from whoever allocated it; `initialize_topology`
uses that fact only for JAX process bring-up and asserts the realized world matches the
authored mesh.

`python -m param_decomp.experiments.lm.run <config> --data-root … --local-device-count N`
runs in the current allocation, minting and pinning its own identity when `--run-id` is
absent. A caller may instead supply the identity, pin the config and code revision, and
start the matching process topology; those deployment choices are not part of the library.

The LM entry enables JAX's persistent compilation cache through
`enable_persistent_compilation_cache` at the config-authored
`runtime.compilation_cache_dir`. The path is `~`-expanded and should be private to one
user because XLA's autotune subdirectory is not safe for unrelated users to share. It may
be shared across runs and processes that use the same environment. Cache keys include HLO,
backend, topology, and JAX/XLA version, so a later run at the same config and topology can
reuse an executable. The cache is enabled after `initialize_topology` and before the first
compile; `jax_persistent_cache_min_compile_time_secs` is 60 so only expensive compiles are
stored. JAX permits only process 0 to write, while every process may read. A multi-node
caller must place the cache directory on storage visible to every node.

### Compile time (2026-07-06 probe grid)

- **Keep seeded inits few-outputs-under-jit**: a jit returning n_sites (hundreds of)
  sharded outputs — or n_chunks unrolled RNG bodies — is a multi-minute SPMD/layout
  compile. vmap-stack over the same per-site/per-chunk keys (bit-identical values),
  then fan out with a trivial slice jit. `init_component_stacks_placed` is the template;
  `run_state.init_decomposition`'s CI fn init / `init_sources_sharded` follow it.
- **The `jit_step` compile (~5 min at dp32) is FLAT across graph structure**: C,
  CI-fn depth, and PPGD warmup all measured within noise (~83% priority-fusion).
  Don't chase graph-shrink refactors for compile time without new evidence.

## Gotchas

- **Process bring-up never infers mesh geometry from ambient scheduler variables**
  (`sharding.py`). `RuntimeConfig.world_size` is
  `SingleNode(n_gpus=1..8) | MultiNode(n_nodes>=2)`; each multi-node world uses eight
  devices per node. Runtime parsing rejects meshes whose device count cannot form either
  world. `initialize_topology(world_size, local_device_count)` checks the process-local
  count, initializes distributed JAX when needed, and verifies the realized total device
  count. Rank comes from JAX's own cluster bring-up; the library reads only the generic
  `PD_RANK` hint, and only to pick the HLO-dump writer
  (`experiments/lm/training.py::enable_hlo_dump`).
- **`shard_batch` topology** (`sharding.py`): uses `make_array_from_process_local_data`
  so it's correct for BOTH single-process-many-devices and multi-process-1-device.
  Do NOT revert to the per-`process_index()`-slice idiom — it silently replicates one
  slice on single-process multi-device CPU.
- **`target_ports` and `routed` are `param_decomp/` subpackages of the same `param-decomp` distribution**;
  no `sys.path` hacks anywhere. If an import fails, the install is broken — fix the env
  (`make install-dev`), don't add a path shim.
