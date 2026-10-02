# Placement rules

Status: implemented. `placement.py` is the single source of truth for model-state,
operand, and activation placement. Run configuration names the logical mesh explicitly as
`runtime.mesh` — three-axis `{replicate, fsdp, tp}`, or two-axis `{data, tp}` for the
`*-replicated-resident` presets; no axis is required to coincide with a node boundary.
Which shape a table runs on is the table's own knowledge — the mesh axes its rows name
(`PlacementTable.mesh_axes`) — checked ONCE at construction (`_bind`) for presets and
explicit tables alike: a row naming an axis the mesh lacks refuses, and so does a
multi-device mesh axis no row shards over (a size-1 axis is exempt — the mesh schema
requires every axis of a shape, so `tp: 1` is how a run says "no tensor parallelism").

Companion prose, each canonical for its piece: `sharding.py`'s module docstring — the
mesh axes and the authored (not required) hardware alignment; `muon_stacked.py`'s
module docstring — why Newton-Schulz stages at a waypoint at all; `checkpoint.py`'s
module docstring — why checkpoints are topology-free; `CLAUDE.md` (this directory) — the agent-facing summary.

## Invariants

1. Under the HSDP presets, no GPU materializes a fully replicated model. Between forwards,
   BF16 target, component, and CI weights remain sharded over their declared `fsdp` and `tp`
   dimensions. At TP1, full matrix replication exists only for the current linear operand,
   inside that linear's execution boundary; a TP operand retains its Megatron shard. The
   `*-replicated-resident` presets deliberately trade this working-copy sharding away: the
   BF16 weights rest whole (÷tp only), buying loop-free weight collectives at the memory
   cost of the resident model. FP32 masters and optimizer state stay ÷N everywhere
   (invariant 2 is unconditional).
2. FP32 trainable parameters and optimizer moments use their declared `optimizer_state` rows.
   Adam never requires parameter replication.
3. Every reshard follows from two declared forward placements. Gradient communication is the
   ordinary transpose of those forward transitions: there are no gradient-placement rows,
   per-linear custom VJPs, or custom scan backwards.
4. Semantic axes are authored by the code that owns a tensor. Unlisted semantic axes replicate;
   unknown rows, a mesh outside the table's axis vocabulary, rank mismatches, and non-tiling
   matrix axes fail closed.
   Placement is total and fallback-free: one set of component rows places every group. A stack
   length that does not tile the stack-sharding component rows is placed by PADDING the persist
   stack with trailing all-zero slots (an enumerated `GroupCensus.stack_pad`, stripped at the
   entry gather — see "Persist-stack padding" below); anything else those rows cannot place
   refuses at construction with the remedies spelled out.
5. Nested mesh-axis order is semantic. For example, `("fsdp", "replicate")` and
   `("replicate", "fsdp")` are different linearizations and may lower to different collectives.

## Vocabulary

There are three distinct vocabularies:

- **Semantic axes** describe tensor meaning: `stack`, `d_in`, `d_out`, `C`, `batch`, `q_head`,
  `kv_head`, `ffn_hidden`, and so on. Block-factored component stacks add `expert` (the
  block axis) and `C_block` (per-block component axis); their `d_in`/`d_out` name one
  block's dims. The component rows' legal keys derive from the run's factorizations, so a
  key no group consumes refuses at construction.
- **Mesh axes** describe the logical device grid: `replicate`, `fsdp`, and `tp` on the
  three-axis HSDP mesh; `data` (the merged data axis) and `tp` on the resident two-axis
  mesh.
- **Placement rows** map semantic axes to ordered mesh axes for one lifecycle phase or activation
  boundary. A rule value has ONE in-code form, the ordered tuple of mesh axes
  (`axes.MeshAssignment`; `()` = replicated) — the config's `tp` / `[tp, data]` / `null`
  spellings are authoring sugar folded at the parse boundary (`placement._rule`), so no
  consumer branches on a value's shape. A `PartitionSpec` is always derived from a typed
  row plus a tensor's semantic axes.

`PlacementRules` contains three closed sections and the CI preset name:

- `components`: `optimizer_state`, `compute_weights`, faithfulness weight/delta rows,
  `operands`, the muon-NS staging waypoint `ns_compute`, and the resolved semantic-group
  census (each group's factorization and stack length; the transitions read per-group
  leaf axes from it);
- `activations`: the target/component external waist, the masked passes' between-blocks
  residual (`masked_external` — the external row itself unless the run authors
  `runtime.sequence_sharding: sequence_parallel`, which adds `position -> tp`), and the
  `C`-sharded internal waist;
- `target`: persist/operand rows for every frozen weight role, Megatron column/row activation
  contracts, normalization/position buffers, and the component-replaced public interface;
- `ci_fn`: the preset name the CI architecture resolves its own rows from. Each transformer
  CI architecture owns a typed row set (`TransformerCIFnRows`, `BlockSelectedCIFnRows`) and a
  `preset_rows` match over every preset name, refusing the presets it does not support;
  its per-family lifecycle rows (`optimizer_state`, `compute_weights`, `operands`,
  `ns_compute`) bind through `placement.bind_ci_weight_rows`, which fails closed on keys
  its leaves do not consume and on mesh axes the mesh lacks.

The explicit config model mirrors these typed fields. It is not a string-keyed escape hatch.
An explicit table names a preset for its CI rows (`ci_fn: <preset>`); CI rows are never
authored.

## Weight lifecycle

### Trainable components

Component masters are FP32 semantic stacks. The selected ownership row may shard the stack axis
(`owner`) or intra-matrix over the full mesh (`zero1`: `d_in`/`d_out` on `fsdp`, `C` on
`("tp", "replicate")` — Adam is elementwise, so the master layout is free to park `replicate`
minor on `C`, where the compute-entry gather wants it). The mesh axes are jax `Explicit`, so
every traced array carries its sharding in its type and transitions are `jax.sharding.reshard`.

Before target execution, `component_stacks_to_compute_weights` performs the declared
cross-`replicate` gather once (`materialize_reduced_weights`), typing the gathered mesh axes
`reduced` on the resident:

```text
V: [stack, d_in, C]   e.g. P(None, "fsdp", "tp", reduced={"replicate"})
U: [stack, C, d_out]  e.g. P(None, "tp", "fsdp", reduced={"replicate"})
```

The `reduced` typing is the chained-deferral contract: cotangents flow back `unreduced` over
those axes — replica-local through the whole target scan — and reduce exactly once, at the
transpose of this materialization (the masters' exit reduce-scatter). jax's dot transpose
demands the operand's reduced set EQUAL the batch contraction's mesh-axis set, which is why
`LinearPlan.weight_reduced` (the master provenance) rides on the plan and is unioned with the
per-linear gather axes; a provenance-free (frozen) weight stays untagged.

After the target scan slices one layer, `placed_linear` reshards only the FSDP shards required
by that linear into the operand. It never gathers a complete component stack. Its ordinary
transpose emits the matching reduce-scatter.

### Frozen target

Every frozen target matrix persists BF16-sharded over `fsdp` and replicated over `replicate`.
Column and row roles additionally preserve Megatron TP layouts. The scanned target slices one
layer before `LinearPlan` materializes its operand, so only the current linear is gathered across
FSDP; it remains TP-sharded when TP is enabled.
Embedding and output weights have the same declared persist-to-operand lifecycle; norms and RoPE
buffers are explicitly replicated.

### CI transformer

CI FP32 masters and optimizer moments use their family-specific `optimizer_state` rows. The local
shards are cast to BF16 before the declared gather into FSDP-resident compute weights
(`materialize_reduced_weights` again — the gathered axes typed `reduced`): chunk-weight
cotangents stay replica-local through the chunk scan and exit once through the materialization's
transpose (the residency-bounds test pins the in-loop cross-replica collectives to the
sanctioned smalls — replicated-persisted bias/norm-scale grads' whole-batch sums).
Attention, FFN, input, and output linears each derive resident, operand, input, and output specs
from their typed rows. The transformer uses Megatron-style alternating column/row TP and exposes a replicated
model-width waist between blocks. Biases and learned norm scales use the `vectors` row, which
shards `ffn_hidden` and `C` while leaving the model-width waist replicated.

### Faithfulness

Faithfulness consumes exact FP32 component masters, not the BF16 resident representation. In
both sharded presets the faithfulness weight rows ARE the master layout, so the weights
transition is the identity: `owner` keeps the stack rows (delta row `stack` on `replicate`,
`d_out` on `fsdp`), `zero1` the matrix rows (`d_in`/`d_out` on `fsdp`, `C` on
`("tp", "replicate")`; the delta row scatters both `C` contractions onto `d_in`, so no
full-rank delta matrix is materialized). An explicit table may declare a different
faithfulness pair; a stack-sharded faithfulness row participates in the persist-stack
pad resolution, like every stack-sharded component row (the faithfulness lane rides the
pads as exact zeros). Transitions are typed `reshard`s —
semantics-preserving by construction (an axis permutation is unrepresentable); the
anti-collective-permute claim moved from a construction-time allowlist to census-based
tests over the compiled HLO.

### Muon NS

Stacked muon's Newton-Schulz is choreographed by the same table. Each muon leaf is one
semantic kind's `[g, rows<=cols]` stack and gets its own batched NS — grouping by shape
across kinds is banned by design decision, so two kinds that coincidentally share a shape
never merge, and no padding concept exists. The leaf casts to `ns_dtype` and stages at
its family's `ns_compute` waypoint verbatim (`ns_staging_sharding`), where NS executes;
only `stack` may carry an assignment: whole matrices per device is what keeps the NS
loop collective-free (a matrix-sharded operand is an explicit-mode type error on the
Gram contraction, and matrix-axis staging — the persist row verbatim included —
re-triggers the SPMD full-rematerialization fallback). A kind whose stack length does
not tile the declared split refuses at the stacked-muon consumer's claim
(`assert_stacked_muon_component_staging` at optimizer build; for CI,
`ci_fn.optimizer.assert_ci_muon_staging_tiles` over every declared matrix during LM
validation and again when the waypoints stage — only a stacked-muon run consumes the row, so non-muon runs keep any-stack-length
placement) — nothing hunts for an alternative split, and nothing is ever gathered whole
or padded. The waypoint is reached
by `muon_stacked.staging_hops` — one mesh axis moved per reshard, since a combined
move-and-gather reshard also trips the fallback. Every preset declares the same staging,
the stack split over `replicate` (the node axis under the seats' authored convention —
see `sharding.py`; on the resident mesh the same split respells to `data`): under owner persistence the ingress is the identity on the stack
axis; under intra-matrix (zero1) or replicated (ddp)
persistence it is a shard-to-shard hop chain. The NS redundancy within a replicate
group (a kind's shard replicated across that group's fsdp × tp plane — intra-node as
authored) is accepted — NS is a sliver of the step, and comms-free beats FLOP-optimal.

The redundancy is deliberate, and it has a knob. An `ns_compute` row admits any stack
assignment (only matrix-axis assignments are refused), so an explicit table can widen
the split — `{stack: [replicate, fsdp]}` spreads each kind's NS over more devices.
The presets stay at `{stack: replicate}` because widening tightens the tiling
constraint (every kind's stack length must divide the larger split,
`_assert_ns_row_tiles`) and reopens the trade the split exists to close: under owner
masters the write-back stops being communication-free — egress from a wider split is
an intra-node collective. NS is off the critical path, so comms-free wins over
FLOP-optimal until profiling says otherwise.

The pairing is STRUCTURAL for the components group: a stacked-muon run whose master
layout splits a within-block matrix axis across a staging axis (zero1's intra-matrix
cut — `C`/`C_block` riding the ns stack axes) refuses at the components claim
(`assert_stacked_muon_component_staging`), naming the owner-flavored remedies.
Newton-Schulz needs whole matrices per device, and such a layout silently pays a
per-step staging round-trip of the full master bytes; Adam-family optimizers stay
layout-agnostic. The CI families keep their own claims (hop-chain staging remains
legal there).

`owner` staging is Distributed Muon in the sense of Moonshot's Moonlight paper (*Muon
is Scalable for LLM Training*: ZeRO-1-partitioned optimizer states, whole-matrix
ownership, local Newton-Schulz, redistribute) — re-expressed as declarative placement
rows with node-local ownership, so every collective stays compiler-inserted rather
than hand-written.

## Linear lowering

`LinearPlan` is data: mesh, input placement, resident-weight placement, operand placement, and
output placement. Its implementation mechanically:

1. reshards the input to the operand-input row;
2. reshards the weight to the operand row, typing the dropped resident axes (plus the
   plan's master provenance) `reduced`;
3. contracts with the output typed to the public output row (`einsum out_sharding` — a
   contracted TP axis lowers to the reduction the compiler picks).

The implementation contains no optimizer or target special cases. Rematerialization policy stays
at the enclosing scanned forward. With `nothing_saveable`, backward recreates per-linear operands;
with `dots_saveable`, a gathered operand is not itself a saved dot residual.

## Presets

- `zero1`: globally matrix-sharded FP32 trainable state; HSDP-resident BF16 compute weights. No
  row shards the component stack axis, so every semantic group — any stack length — is placeable.
- `owner`: semantic stacks sharded over `replicate`, matrix dimensions over `fsdp`. A group whose
  stack does not tile `replicate` pads its persist stack to the next multiple
  (`GroupCensus.stack_pad`; see "Persist-stack padding" below). There is no fallback
  preset and no fallback row — mixed per-group placement is unrepresentable.
- `zero1-replicated-resident` / `owner-replicated-resident`: the resident twins, on the
  two-axis `(data, tp)` mesh. The BF16 working copy is RESIDENT whole (÷tp only): every
  compute-weight and target-persist row IS its operand row, so the once-per-step
  masters→resident entry gather is the step's only weight collective and no while body
  gathers weights. Residency leaves fsdp nothing to shard, so the axis does not exist;
  the tables are the zero1/owner rules under a mechanical respelling — `fsdp`
  assignments drop, `replicate` renames to `data` (`_resident_table`). Masters keep
  their base persistence (zero1: intra-matrix, `C` on `("tp", "data")`; owner:
  stack-cut, `stack` on `data`), so the faithfulness transition stays the identity in
  both, and owner's stack cut carries over (÷data, padded where the stacks don't tile it).
- `zero1-replicated-resident-moe` / `owner-replicated-resident-moe`: the resident
  twins extended with the rows block-factored component groups need — in both, V/U
  blocks co-locate with the frozen weight blocks they factor (`expert: tp` on residents
  and operands), the faithfulness rows ARE the master layout (identity transition), and
  the component waist additionally keys `expert`, so a component activation's
  block-factored view derives from the same row as its flat `C` view (the flat axis is
  block-major — one layout, two spellings). Keyed on `expert` (+`C_block` where used),
  they bind only site sets that actually hold block-factored groups; a dense-only run
  refuses at the rule-key check and uses the dense twins.
  The zero1 flavor rests masters intra-matrix ÷N (`expert: tp`, `C_block: data`) — any
  stack length placeable; its costs: the faithfulness V·U contraction runs over the
  data-cut `C_block` (the delta lands through a cross-`data` reduce-scatter at entry),
  and the layout is stacked-muon-incompatible (below). One faithfulness-delta row
  serves both factorization kinds, and a block-factored delta carries `expert` and `d_in`
  together, so `d_in` cannot also ride `tp`: the dense (shared-kind) deltas' tp
  contraction all-reduces at entry instead of scattering — small matrices, off the
  hot loop.
  The owner flavor rests COMPONENT masters stack-cut — `{stack: data, expert: tp}`,
  every V/U block whole on one device on both axes. The entry gather becomes the
  layout-preserving stack-axis all-gather over `data`, the faithfulness V·U
  contraction is fully rank-local (the delta path carries no cross-`data` collective),
  and stacked-muon NS staging keeps owner's story (identity on the cross-node stack
  axis; intra-node tp gathers of co-located block axes only). Stack-cut ÷data like
  every owner, padded where the stacks don't tile it. The flavor is the components masters' —
  the block-selected CI rows stay the zero1 twin's (owner-cut CI masters would bind
  `n_chunks` to ÷data for nothing).
  Under both flavors the block-selected CI arch adds two weight families
  (`ci_fn/expert_ffn`, `ci_fn/expert_head`): CI block banks and fused
  selected-emission heads rest whole per block shard at `expert: tp` — co-located with
  the target's frozen blocks and the V/U blocks, so the CI fn's compute over the selected
  blocks rides the same block-sharded schedule with zero weight movement — masters
  ÷(tp·data) in the zero1 spirit under EITHER flavor (`ffn_hidden`/`C_block` on `data`):
  an owner-style stack-cut would demand the CI chunk stack tile `data` (n_chunks = 10
  does not tile 8), and the intra-matrix cut costs owner nothing it claims — CI weights
  have no faithfulness row, and the seats keep the CI group on adamw. Entry is a pure
  all-gather over `data`; NS staging `{stack: data}` (the 4D leaves fold layer/block
  into the canonical stack). Dense presets bind `moe = None`;
  the block-selected arch's `resolve_placement` refuses a table without them.
- `ddp`: replicated model state for small-model and single-node work only.

Unrepresentable is the point. One row set placing every group is a claim a reader can
hold whole: every consumer, checkpoint reader, and profile analysis reasons about ONE
layout per run. A per-group fallback would make each group's layout a build-time
decision every downstream reader must re-derive — every tolerated fallback is another
reachable state multiplying what the placement claim has to cover, and the claim stops
being total. The refusal-with-remedies costs one config edit before submission; the
multiplication would be paid on every read, forever.

## Persist-stack padding

A stack-sharding persist row demands the stack extent tile its cut. Rather than
refusing node counts the layer grid does not divide (40 layers at data=16, or the
36-chunk CI fn at data=64), construction PADS each persist stack to the next common
multiple of its stack-sharding rows' extents. The pad is an enumerated fact, never an
inference — one census type (`StackCensus`: real length + pad) for every persist stack:

- Each V/U semantic group resolves a `GroupCensus` in `from_config`, from the site set,
  over the component rows that cut the stack (`optimizer_state`, `faithfulness_weights`,
  `faithfulness_deltas`); `ComponentStacks.stack_pads` mirrors it on the value tree.
- The chunkwise CI fn's chunk stack resolves a `StackCensus` in `resolve_chunk_census`
  — where the CI arch first meets the rows, at run assembly and at the config-build
  gate — over every row its stacked leaves rest at (the
  four families' `optimizer_state` plus `vectors`, and the expert families' masters for
  the block-selected arch). The arch's resolved placement carries it (`.chunks`), and
  `ChunkwiseTransformerBackbone.census` / `BlockSelectedChunkwiseTransformerCIFn.census`
  mirror it on the value tree.

Both value trees are boundary-validated against their census
(`_validate_component_stacks`, `_validate_chunk_stack`), and no consumer ever derives a
pad from a shape — the muon 96-stack zero-padding saga is the cautionary tale this
design refuses to repeat.

Pad slots are trailing all-zero stack entries — (V, U) slots for a group, one slot on
EVERY leaf of the stacked chunk module for the CI fn (weights and vector leaves alike,
so the stack stays rectangular) — and exist in exactly two places:

- The persist layer — fp32 masters and optimizer moments. The seeded init appends them
  (`pad_component_stacks`, `pad_ci_fn`); wd=0 plus exactly-zero gradients keep them at
  zero through adamw and stacked muon alike (NS on a zero matrix is zero — the wasted
  optimizer FLOPs are the pad fraction, `stack_pad / padded_stack_len`). Muon's
  canonical NS stack length is the padded one (`GroupCensus.ns_stack_len`; the CI
  staging claims count the padded stored matrices).
- The faithfulness lane, whose rows ARE the master layout: pads ride the identity
  weights transition as exact zeros, targets extend the frozen stack with zero
  matrices (`weight_deltas` — pad deltas are exactly `0 − 0·0 = 0`), and
  `make_faithfulness_loss` keeps the site mean over the REAL sites. (No faithfulness
  row exists for CI weights; the CI pads have only the persist layer.)

The V/U compute never sees a pad, and the V/U entry gather never moves one: the entry
(`component_stacks_to_compute_weights`) strips the pads BEFORE the cross-`data` gather
(`materialize_reduced_weights`), so the V/U residents, the forwards, and every mask
surface carry only real stacks and the all-gather's result shapes read `stack_len`, never `padded_stack_len`.
The typed slice cannot drop slots from a cut stack axis (a real length that does not
tile the cut has no sharded spelling — the very reason the pad exists), so a padded
stack first hops to its entry waypoint (`padded_entry_waypoint`: the compute layout with
the stack cut re-parked minor on the leaf's last axis — an all-to-all moving one padded
slot per device, after which the stack rests whole and every device still holds ÷cut of
the bytes), the pads exit there device-local (`strip_stack_pad`), and the all-gather
carries only real slots. The transpose runs the route backwards: a real-slot
reduce-scatter, the slice's transpose writing exact zeros into the pad slots, the
all-to-all returning them to their owners.

The chunkwise CI fns take the plain route instead: `prepare` gathers the
padded stack into the compute layout (`gather_reduced_weights`), so compute values carry
the same `stack_pad` as stored ones, and the forward slices the real chunks off right
before the chunk scan. The compute rows (and the vectors row) refuse `stack`, so that
slice is device-local; its transpose writes exact zeros into the pad cotangents, and the
reduce-scatter back to the persist layout keeps them zero. The pad chunks ride the CI
entry gather. The V/U waypoint's tiling — the leaf's last dim against
its compute assignment with the stack cut nested minor — is validated where the rows
are bound (`validate_stacked_leaf`), like every other matrix-axis tiling. The costs are
the pad fraction on persist bytes and optimizer work, plus the entry all-to-all's one
padded slot per device — nothing on the hot loop, and nothing padded in the step's
largest collective or transient. Checkpoints persist the PADDED stacks, so a consumer
re-placing a padded run must resolve the same pad counts (a different mesh that
resolves different pads refuses at restore shape validation).

Owner vs `zero1` in magnitude: under elementwise optimizers (Adam) the two are
~equivalent per-step communication — entry gather and exit reduce-scatter move the same
bytes either way, and the faithfulness transition is the identity in both. Under
stacked muon the whole difference is NS staging: owner's ingress/egress are the
identity on the stack axis (masters already rest at the waypoint's split), while zero1
masters take the `staging_hops` shard-to-shard chain, in `ns_dtype` bytes. Both are
bounded, off-critical-path transfers; neither preset is a memory class apart — the
per-rank whole-fp32-stack peak is what the hop chain excludes, in every preset.

`from_config` resolves the semantic-group census once from the concrete site set — pad counts
included — and refuses any group the rows cannot place; `resolve_chunk_census` does the same
for the chunk stack from the CI arch (and refuses, at the same construction time, every
arch-known CI master leaf the rows cannot tile). Consumers validate those censuses against
their arrays and never re-decide them; a consumer re-placing a finished run on one device tiles
trivially (every stack length divides 1, so no pads resolve — which is exactly why a PADDED
run's checkpoint must be re-placed on a mesh resolving the same pads).

## Sequence parallelism on masked forwards

`runtime.sequence_sharding: sequence_parallel` (default `replicate`) retypes the MASKED
passes' between-blocks residual to `position -> tp` (`activations/masked_external`) —
Megatron's sequence parallelism, scoped to the masked forwards. The block interiors are
untouched — mixers and the MoE (jobs schedule, routing, shared sites) always run at the
full external width: each block entry gathers the normed carry (one all-gather whose
transpose is that block input's ONE cotangent reduction, where the replicated residual
pays a per-consumer tp all-reduce), and each block-exit reduction — the row linears'
output contraction and the expert combine — lands position-sharded (half the ring bytes
of the replicated arm's all-reduce), with the between-blocks residuals and norms resting
and computing at 1/tp. The final residual gathers back to `external` before the output
edge, and clean forwards, CI-fn taps, and eval comparisons keep the replicated residual
everywhere, so nothing outside the masked engine sees the sharded type. It is a
resharding of the same math (numerics move at reassociation level); captures
under sequence parallelism are an enumerated gap and refuse. Sequence length must tile
tp, and only targets implementing it accept the row (qwen36_moe; others refuse at their
masked forward).

## Performance evidence and profiling validity

Startup prints the mesh, every placement row, target role declarations, and the derived placement
of persistent leaves. Tests cover numerical forward/gradient parity, real multi-device topology,
checkpoint round trips, TP boundaries, and forbidden collective patterns. Scale claims require
the post-SPMD HLO, compiled memory report, and an uncapped XPlane with explicit step boundaries;
small simulated meshes are necessary but not sufficient.

Every performance claim must be reproducible from an evidence record carrying: the pushed
commit; the exact invocation, including any config-derivation command; the pinned resolved
`launch_config.yaml`; hardware, process topology, run id, mesh, batch, objective, and warmup count; the
exact profiled step ranges and paths to the uncapped XPlane and optimized HLO protobuf; and the
parser command/version that produced the numbers. A run name or prose description is never a
substitute. Classify a run by its pinned `launch_config.yaml` and startup placement dump —
names, comments, and copied filenames are labels, not configuration evidence.

### Timing validity

- The uncapped `.xplane.pb` is the timing and kernel-count source of truth. Perfetto/Chrome
  exports cap at one million events and can silently hold only a prefix of the requested
  window: never infer step time from an export's span, and verify a window's step boundaries
  and per-kind kernel counts against the complete XPlane.
- Every analyzed step needs an explicit host range enclosing its final device synchronization;
  the requested profile step count is not a boundary oracle. Average only explicitly identified
  steady-state steps, and say when fewer clean steps remain than the requested window.
- The first logged step is a startup measurement (compilation + initialization), and one
  unprofiled execution is not enough: merely dispatching warmups lets asynchronous work spill
  into the marked window. The harness requires two unprofiled, device-synchronized updates
  before opening the trace; still inspect every marked host range and reject any containing
  compilation or first-execution autotuning.
- Verify algorithmic workload knobs (warmup counts, ablated losses) before comparing
  throughput: an ablated cell is matched topology evidence, not a production-shaped step time.
  A comparison across commits is not a one-variable experiment even when the YAMLs match.
- Kernel-duration unions are intentionally non-additive because streams overlap: report total
  collective union and exposed collective time separately, never a per-kind sum as wall time.
  Treat profiler-derived speedups as provisional until the full XPlane and an independent
  timing source agree.

### Attribution validity

- Never classify collectives by CUDA/NCCL kernel-name substrings. XPlane events carry native
  `(program_id, hlo_op)` identities and dumped HLO protobufs carry module ids and opcodes:
  join those records exactly and fail on missing or ambiguous identities. A text-only HLO dump
  cannot support the join — profiling runs enable `xla_dump_hlo_as_proto` before backend init
  and copy the protos to durable run artifacts before the allocation exits.
- An XLA dump-name regex can omit auxiliary programs whose kernels still run inside a marked
  step. Treat an entirely absent program as an unattributed compute blocker; fail if an event
  names an absent instruction within a loaded program; never manufacture an HLO match from a
  kernel name.
- NVTX projection rows are not one row per physical kernel (nested launch ranges double
  counts): check physical kernel counts in the XPlane or the CUDA timeline before interpreting
  projected counts.
- Nsight Systems: capture only an explicitly marked steady-state range at default NCCL detail
  (full-process `--nccl-trace=all` capture can destabilize ranks). Repeated-capture reports
  describe the same process — analyze them separately, never as one multi-report input. A
  rank-zero process trace covers that process's GPUs, not global communicator metadata.
  Instrumentation can radically perturb distributed wall time while individual kernel
  durations stay representative: use such captures for HLO identity, message shapes, and
  cross-profiler kernel-duration checks, never for throughput.

### Placement-claim validity

- A declared resident placement is not proof that residency survives a scan boundary: GSPMD
  can sink a pre-scan gather through the scan into each use, and ordinary autodiff then emits
  in-loop cross-`replicate` reductions. Inspect the optimized HLO around the actual production
  scan before claiming a pre-scan transition is step-resident.
- A synthetic gradient probe is not a placement proof: every placement regression must
  differentiate through the actual target forward with its complete site set, remat policy,
  masks, and routing inputs.
- Small-mesh CPU proofs do not cover GPU-only custom derivatives (fused attention's custom
  VJP) or backend bugs that appear only at the production mesh (partially manual FSDP/TP
  `shard_map` scopes have aborted jax 0.10.1's CPU backend). Exercise every backend-selected
  custom primitive on the production accelerator and mesh.
- Backend substitution is not a harmless correctness workaround: swapping fused attention for
  the ordinary XLA lowering can change compiled memory far beyond the model-state estimate.
  Inspect the full compiled memory plan.

### Cache validity

- XLA's persistent-cache autotune subdir is unsafe for unrelated Unix users to share; the
  cache dir is the config-authored per-user `runtime.compilation_cache_dir`.

Keep dated measurements, topology sweeps, and reference validations with the experiment
records that support them; this document carries only the resulting rules.

## Known frontiers

- The step-boundary weight lifecycle is unscheduled: the entry owner-to-resident gathers and
  exit gradient reduce-scatters run at the cross-node wire floor with zero compute overlap and
  can occupy a material fraction of the measured step. The fix is scheduling — a staged
  per-group owner/resident stream — not byte reduction.
- Sitewise source, mask, routing, and importance work still grows compiler IR with site count even
  though target and CI depth are scanned.
