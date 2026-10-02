# param_decomp.targets

The vendored decomposition targets: **every `DecomposedModel` implementation lives
here**, one slice per architecture, between the generic engine (`param_decomp.core`)
and the composition layers above.

The dependency direction is `lab → targets → engine`, pinned by
`param_decomp/tests/core/test_runtime_standalone.py`. The engine never imports a target — it
sees only the `DecomposedModel` protocol (`param_decomp.core.model`) and the `ArchFamily`
grammar contract (`param_decomp.core.family`). The lab composes: the model-name → family
registry and all authoring vocabulary stay composition-side
(`param_decomp/experiments/lm/config.py`).

Targets declare weight placement as a `ShardingTree` (a checked pytree of `NamedSharding` leaves) with the same structure
as their array tree. This is a sharding description, not an executable model.
`place_cpu_arrays` checks matching structures and CPU-resident array leaves before
JAX distributes the complete process-local weights. Fit checks use the same declarations
to construct abstract arrays without transferring data.

This layer exists so that *what a target is* and *what we distribute* are independent
decisions: a public release selects packages (engine + whichever slices are shareable);
internal-only slices simply aren't in the set. No slice is ever special to the engine.

Each forward supplies `ForwardResult.leading_shape`: the batch and optional
position dimensions of CI values, masks and sources. Core uses this declaration
without inspecting either the input representation or a particular activation tap.

## Sequence inputs

A sequence is one batch row; a document is a semantic source unit. Sequences may contain
multiple documents, a complete document, or only a fragment. Documents may span sequences.

`LMBatch(token_ids)` contains only tokens. Shared transformer forwards take
`LMBatchWithDocuments(batch, sequence)`, adding per-token document IDs and padding through
`SequenceLayout`. The complete input is retained as conditioning, and
`ForwardResult.sequence` supplies its layout to CI and evaluation. An explicitly
unsegmented input uses `LMBatchWithDocuments.from_unsegmented_sequences(tokens)`; this
allows attention across the row without claiming it contains exactly one source document.

Qwen3.6-MoE forwards accept only `LMBatch`. Attention and DeltaNet recurrence operate
across each full sequence. Token-only readers preserve concat-and-slice rows, including
cross-document spans, and reject document-aware artifacts. The clean pass returns
`LMBatchWithRouting[LMBatch](batch, selection)`. This wrapper is generic: adding routing
to a document-aware batch is equally representable, independent of Qwen's input contract.

Llama attention is causal within each document, and rotary positions restart at document
boundaries and sequence starts, even when a sequence begins partway through a document.
Attention-pattern evaluation uses the same layout as the forward it explains.
Dense Qwen3 and Qwen3.6-MoE reject document-aware datasets.

## The slices

| Slice | Target |
|---|---|
| `transformer` | Shared dense transformer machinery (site grammar, `FrozenAttn`/`TransformerLayer`, `TransformerDecomposedModel`, the scan/masked-forward engine, HF safetensors loading) |
| `llama31` | Llama-3.1 architecture (vendored `LlamaConfig`, llama3 rope); concrete support: 8B — a `glu_transformer` family |
| `qwen3` | Qwen3 architecture (`Qwen3FrozenAttn`: required QK-norm via the `_prep_qk` hook); concrete support: dense 0.6B/1.7B/4B/8B/14B Base and post-trained — a `glu_transformer` family |
| `qwen36_moe` | Qwen3.6-MoE architecture (HF `qwen3_5_moe`: hybrid gated-DeltaNet/gated-attention mixers + per-layer MoE MLP) on its OWN stage-scan engine; sites are the mixer and expert projections — the MoE matrices (expert axis structural inside fused per-layer sites) on every layer, the gated-DeltaNet projections on the linear-attention layers, the gated-attention projections on the full-attention layers; concrete support: Qwen3.6-35B-A3B — its own `qwen36_moe` family (kernels: `target_ports/qwen3_5_moe.py`) |
| `llama_simple_mlp` | The pile-pretrained `LlamaSimpleMLP` (loads from the `pretrain/` cache) — its own `simple_mlp` family, hosted on the shared `transformer` engine (GELU MLP, tied head) |
| `transformer_taps` | The transformer families' activation-tap vocabulary (opaque strings to the engine) |
| `tms` | Toy: TMS (positionless, in-process pretrain) |
| `resid_mlp` | Toy: residual MLP (positionless, in-process pretrain) |

Masking enters target preparation as a logical per-site recipe. Each target declares
its own prepared masking type: Dense transformer and Qwen MoE implementations group complete mask ingredients
into per-kind layer stacks; the toys materialize concrete per-site masks. Only this
prepared type enters `masked_forward`, beside the forward's routes, which each target stacks
into its own layout inside the forward. Stochastic preparation returns a draw callable
that shares one prepared CI layout across the step's reconstruction terms. Preparation
happens inside tracing, and targets retain their own eager or checkpointed composition
policy.

Shared transformer stage inputs likewise pair V/U weights with a materialized mask or stochastic recipe.
Frozen and decomposed layer inputs form a closed union consumed by one scan body.
An empty capture layout contributes no array leaves to the scan carry.

A slice owns everything about its architecture: the frozen modules, the decomposed
forward (including its sharding/remat strategy behind the protocol), its `ArchFamily`,
and its weight loading. `param_decomp/tests/targets/` holds the per-target parity/golden
suites; engine behavior tests that merely use a target as a fixture live under
`param_decomp/tests/core/`.
`invariance_check.py` is the device-count invariance harness (a tiny GLU target
driven through the engine at simulated device counts).

## Weight loading

Checkpoint loaders assemble the ordinary executable model with CPU-resident JAX arrays.
Readers use NumPy only at the checkpoint I/O boundary. A CPU staging context keeps
stacking and generated constants on CPU even when the default backend is CUDA.
`place_target` transfers only addressable slices to the declared device layout;
`abstract_placed_model` constructs shape leaves without device allocation.
Runtime target fields and numerical operations accept JAX arrays exclusively.

## LM output edges

The target projection requires `[batch, seq, d_model]` activations for both output edges.
The streamed package and shared comparison kernels retain arbitrary leading dimensions
for sliced outputs and transformations such as `vmap`.

Every LM family supports `MaterializedOutputEdge` and `StreamedOutputEdge`, declared in
`lm_output.py`. The shared `linear_output` operation either projects the final residual
to logits or returns a `StreamedLinearOutput` containing that residual and the head in
its operand layout. Tied heads use the same path. Model builders require the output
edge explicitly. Weight loaders also require the attention implementation, so both
run settings reach construction without patching the completed model.

The streamed comparison kernels in `losses.py` compute KL and CE one vocabulary chunk
at a time and rematerialize chunks during backpropagation. Chunk logits accumulate in
fp32, so BF16 targets avoid the materialized edge's final logit rounding. The authored
chunk count must divide the vocabulary size; Llama-3.1-8B admits 32 chunks of 4008 tokens.

## MoE expert computation

Qwen3.6-MoE authors `expert_implementation` independently from its selected CI
interface. `dense_masked` computes ordinary dense matmuls over token/expert grids
and applies the router selection when combining outputs. The grouped alternatives
compute only routed jobs. Both frozen and decomposed forwards, including component
activation captures, retain the same pinned routing and selected CI interface.

Qwen stage inputs carry a typed recipe per decomposed kind: materialized masks,
stochastic draw keys, or aligned `SourceMaskIngredients`. Each recipe travels with
its V/U weights and optional route. Checkpointed stage bodies resolve recipes;
materialized delta presence and the two source recipes determine the delta channel
without a separate mode flag. Expert ownership and source sampling remain unchanged.


### Nonlinearity alignment

`SiteSpec.alignment` declares architectural geometry as
`NonlinearityAlignment(side, partition)`: input-side `V` columns or output-side `U` rows,
partitioned into nonlinearity units. Initialization and locality are independent config
choices that consume this geometry; declaring it enables neither. `None` means neither
side directly faces a nonlinearity, as in the intermediate linear chain of TMS.
Head divisibility is checked on the declared side within each expert block.
GDN q/k/v/z use output-side head partitions; GDN out uses input-side value heads,
and a/b use output-side scalar neurons. Attention output projections use input-side
query heads, and MLP down projections use input-side hidden neurons. When configured,
the locality objective pools components across all declared sites within each trained
unit kind, including both input- and output-side factors.

## Router captures

Qwen exposes `router_logits.<layer>` (fp32 scores before softmax),
`router_probs.<layer>` (the full expert distribution), and `router_weights.<layer>`
(the normalized weights applied at selected expert identities). The raw score tap
shares the forward's router matmul and preserves its compute dtype before the fp32
cast. `router_logits_capture_keys` enumerates the captures an LM experiment can select for
a categorical-KL comparison group. Masked forwards keep clean expert identities and recompute
mixing weights from their own router scores.
