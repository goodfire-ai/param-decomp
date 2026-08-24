# Placement rules

Status: implemented. `placement.py` is the single source of truth for model-state,
operand, and activation placement. Run configuration names the logical mesh explicitly as
`runtime.{replicate,fsdp,tp}`; none of those axes is required to coincide with a node boundary.

Companion prose, each canonical for its piece: `sharding.py`'s module docstring — the
mesh axes and the authored (not required) hardware alignment; `muon_stacked.py`'s
module docstring — why Newton-Schulz stages at a waypoint at all; `checkpoint.py`'s
module docstring — why checkpoints are topology-free; `SPEC.md` D4/S20 — the layout
and optimizer invariants; `CLAUDE.md` (this directory) — the agent-facing summary.

## Invariants

1. No GPU materializes a fully replicated model. Between forwards, BF16 target, component, and
   CI weights remain sharded over their declared `fsdp` and `tp` dimensions. At TP1, full matrix
   replication exists only for the current linear operand, inside that linear's execution
   boundary; a TP operand retains its Megatron shard.
2. FP32 trainable parameters and optimizer moments use their declared `optimizer_state` rows.
   Adam never requires parameter replication.
3. Every reshard follows from two declared forward placements. Gradient communication is the
   ordinary transpose of those forward transitions: there are no gradient-placement rows,
   per-linear custom VJPs, or custom scan backwards.
4. Semantic axes are authored by the code that owns a tensor. Unlisted semantic axes replicate;
   unknown rows, unknown mesh axes, rank mismatches, and non-tiling semantic groups fail closed.
   Placement is total and fallback-free: one set of component rows places every group, and a
   group those rows cannot place refuses at construction with the remedies spelled out.
5. Nested mesh-axis order is semantic. For example, `("fsdp", "replicate")` and
   `("replicate", "fsdp")` are different linearizations and may lower to different collectives.

## Vocabulary

There are three distinct vocabularies:

- **Semantic axes** describe tensor meaning: `stack`, `d_in`, `d_out`, `C`, `batch`, `q_head`,
  `kv_head`, `ffn_hidden`, and so on.
- **Mesh axes** describe the logical device grid: `replicate`, `fsdp`, and `tp`.
- **Placement rows** map semantic axes to ordered mesh axes for one lifecycle phase or activation
  boundary. A `PartitionSpec` is always derived from a typed row plus a tensor's semantic axes.

`PlacementRules` contains four closed sections:

- `components`: `optimizer_state`, `compute_weights`, faithfulness weight/delta rows,
  `operands`, the muon-NS staging waypoint `ns_compute`, and the resolved semantic-group
  census (stack lengths);
- `ci_fn`: `optimizer_state`, `compute_weights`, `operands`, and `ns_compute` for attention,
  FFN, input, and output weights, plus vector-state and activation rows;
- `activations`: the target/component external waist and the `C`-sharded internal waist;
- `target`: persist/operand rows for every frozen weight role, Megatron column/row activation
  contracts, normalization/position buffers, and the component-replaced public interface.

The explicit config model mirrors these typed fields. It is not a string-keyed escape hatch.

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
faithfulness pair; a stack-sharded faithfulness row refuses non-tiling groups at
construction, like every stack-sharded component row. Transitions are typed `reshard`s —
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
(`assert_stacked_muon_*_staging`, fired at optimizer build and at the LM pre-submit
gate — only a stacked-muon run consumes the row, so non-muon runs keep any-stack-length
placement) — nothing hunts for an alternative split, and nothing is ever gathered whole
or padded. The waypoint is reached
by `muon_stacked.staging_hops` — one mesh axis moved per reshard, since a combined
move-and-gather reshard also trips the fallback. Every preset declares the same staging,
the stack split over `replicate` (the node axis under the seats' authored convention —
see `sharding.py`): under owner persistence the ingress is the identity on the stack
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
  stack does not tile `replicate` refuses at construction; the refusal names the groups and the
  remedies (a tiling mesh, or a stack-free placement such as `zero1`). There is no fallback
  preset and no fallback row — mixed per-group placement is unrepresentable.
- `ddp`: replicated model state for small-model and single-node work only.

Unrepresentable is the point. One row set placing every group is a claim a reader can
hold whole: every consumer, checkpoint reader, and profile analysis reasons about ONE
layout per run. A per-group fallback would make each group's layout a build-time
decision every downstream reader must re-derive — every tolerated fallback is another
reachable state multiplying what the placement claim has to cover, and the claim stops
being total. The refusal-with-remedies costs one config edit before submission; the
multiplication would be paid on every read, forever.

Owner vs `zero1` in magnitude: under elementwise optimizers (Adam) the two are
~equivalent per-step communication — entry gather and exit reduce-scatter move the same
bytes either way, and the faithfulness transition is the identity in both. Under
stacked muon the whole difference is NS staging: owner's ingress/egress are the
identity on the stack axis (masters already rest at the waypoint's split), while zero1
masters take the `staging_hops` shard-to-shard chain, in `ns_dtype` bytes. Both are
bounded, off-critical-path transfers; neither preset is a memory class apart — the
per-rank whole-fp32-stack peak is what the hop chain excludes, in every preset.

`from_config` resolves the semantic-group census once from the concrete site set and refuses any
group the rows cannot place. Consumers validate that census against their arrays and never
re-decide it; a consumer re-placing a finished run on one device tiles trivially (every stack
length divides 1).

## Performance evidence and profiling validity

Startup prints the mesh, every placement row, target role declarations, and the derived placement
of persistent leaves. Tests cover numerical forward/gradient parity, real multi-device topology,
checkpoint round trips, TP boundaries, and forbidden collective patterns. Scale claims require
the post-SPMD HLO, compiled memory report, and an uncapped XPlane with explicit step boundaries;
small simulated meshes are necessary but not sufficient.

Every performance claim must be reproducible from an evidence record carrying: the pushed
commit; the exact invocation, including any config-derivation command; the pinned resolved
`launch_config.yaml`; cluster, job id, run id, mesh, batch, objective, and warmup count; the
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

### Cluster and cache validity

- A whole-node batch allocation does not give a one-task-per-node `srun` step the node's
  memory; the launcher pins `--mem=0`. Inspect the step's `AllocTRES` before diagnosing a
  compiler or GPU-memory failure.
- XLA's persistent-cache autotune subdir is unsafe for unrelated Unix users to share; the
  cache dir is the config-authored per-user `runtime.compilation_cache_dir`.

Keep dated measurements, topology sweeps, and reference validations with the experiment
records that support them; this document carries only the resulting rules.

## Known frontiers

- The step-boundary weight lifecycle is unscheduled: the entry owner-to-resident gathers and
  exit gradient reduce-scatters run at the cross-node wire floor with zero compute overlap, a
  material fraction of the measured step (campaign log). The fix is scheduling — a staged
  per-group owner/resident stream — not byte reduction.
- Sitewise source, mask, routing, and importance work still grows compiler IR with site count even
  though target and CI depth are scanned.
