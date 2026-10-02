# Causal-importance functions

A CI architecture declares the named target activations it needs. Run assembly
validates those captures against the target; training forwards the capture request
and returned arrays without interpreting their names. Input normalization and
readout belong to CI implementations.

`CIFnArchitecture[Conditioning]` initializes a `CIFn[Conditioning]`, the conditioning
being whatever the paired target hands its CI. Cost models read only its
conditioning-free `CIFnArchitectureFootprint`: captures, parameter counts, and useful
FLOPs. Initialization receives the run's `PlacementRules` (or `None` for unplaced
execution); each implementation resolves its own placement from them and binds it
into the parameters it constructs as static state. Core never names a CI placement
type: the rules carry only the preset name (`PlacementRules.ci_fn`). The transformer
arches share one row set, a lifecycle table per part of the layers plus the vector and
activation rules (`implementations/transformer/placement.TransformerCIFnRows`); each maps
every preset name to its rows or refuses it (`preset_rows`, an exhaustive match) and
binds them to the mesh naming the independent axes its matrices lead with. The chunkwise arches
resolve a chunk-stack census (`implementations/chunkwise/placement.resolve_chunk_census`);
the block-selected arch's rows are the chunkwise rows plus its required expert families.
The global transformer's leaves carry `depth` and `site` rather than a chunk `stack`
(`implementations/global_transformer/placement.py`). The pointwise MLPs read no rows.

`CIFn` parameters own their storage padding, shardings, and Muon staging.
They enumerate independent matrices, each with where Muon stages it, fixed when the fn
is built: a `NamedSharding`, or `StageInPlace` (no reshard) on an unplaced fn. Whether
a sharding staging tiles is a Muon-only claim, checked by
`optimizer.assert_ci_muon_staging_tiles`. The MLP fns do not support Muon and raise
from `matrix_parameters`.
Biases and scales remain elementwise parameters regardless of their array rank.
`prepare` returns the same fn cast to the compute dtype and gathered into its compute
layout: the MLP fns' own class, or a `BackboneCIFn` over the prepared transformer
backbone. Evaluating a prepared fn casts the taps, runs the forward, and squashes
(`CI.from_preactivations`) inside the implementation. Evaluation also receives the
target's prepared components (`model.ComponentActivations`); only a CI conditioned on
component activations reads them. The chunkwise transformers keep
their chunk-stack pads through compute and slice the real chunks off before the scan
(the pad policy and its reason are in `implementations/chunkwise/chunk_stack.py`).
Placement is static, so optimizer state and checkpoints hold exactly the stored
arrays. Preparation remains differentiable back to the parameters.

Component conditioning nests around a backbone the way its network does
(`implementations/transformer/component_conditioning.py`). `ConditionedCIFnArch(inner, site_inputs, output_scale_init)`
wraps any `CIFnBackboneArchitecture` (an architecture that builds its backbone alone,
`initialize_backbone`): its captures, parameter census, and useful FLOPs are the inner
architecture's plus the readouts', and it builds `ComponentConditioned` around the inner
backbone. The readout vectors rest at their own row, which each preset names
(`preset_readout_row`, an exhaustive match): split over `C` as the component activations
are. The LM config builds it around the global transformer only. With `h = x @ V`,
where `x` is the site's clean input requested as an ordinary capture, each site's
preactivation gains `s₊ * σ(a₊ * h + b₊) + s₋ * σ(a₋ * (-h) + b₋) + bias`, `σ` the
`symmetric_leaky_hard_sigmoid`: a positive and a negative arm (each a `ClampedAffineArm`)
and a free bias, seven `[C]` CI vector parameters per site. `initialize` sets every input scale to 1, every input bias and the bias to 0, and
every output scale to `output_scale_init`; `ComponentConditioned.with_calibrated_input_scales` sets every arm's
input scale to `1 / q`, `q` each component's exact 99.9th percentile of `|h|` over every
token of the given taps (which must hold no padding), by a running top-k over
`CALIBRATION_CHUNK_TOKENS`-token chunks, each replicated whole in turn; the chunk size sets
only memory and speed. The arch's `calibration` (`InputScaleCalibration`) names the fewest
tokens a root's calibration batch may hold. Core draws no data: a composition root that
calibrates does so in its CI fn initializer (`init_placed.CIFnInitializer`). V stays the
target's: the readout computes `x @ V` through the target's prepared components
(`component_activations`), in the layout the target's own forwards use, so V's CI
gradient joins its reconstruction gradient in the prepared components before the one
pullback onto the component masters. Every site must be dense.

Runtime evaluation validates that physical target captures match the CI request.
The prepared callable evaluates those captures.

Each placed CI architecture has its own package under `implementations/`: `arch.py`
holds the architecture, its parameters and forward, and the resolved placement types that
enter its compute layout; `placement.py` holds its parameter axes, every preset table and
`preset_rows`, and their binding to the mesh. The pointwise MLPs read no rows and are one
module each.

| Module | Responsibility |
| --- | --- |
| `interface.py` | The CI fn protocol, matrix staging declarations, CI values, and tap specifications |
| `architecture.py` | Implementation-independent architecture lifecycle and cost interface |
| `optimizer.py` | Muon adapter over declared matrices and their staging |
| `squashing.py` | CI clipping and gradient rules |
| `runtime.py` | Evaluation from target captures and compute-weight preparation |
| `implementations/transformer/layers.py` | Shared transformer primitives and initialization; the layers' per-matrix parameter axes, the record of their four parts (`CIFnTransformerStructure`: input projection, attention, FFN, output heads), and their variant-agnostic placement (`CIFnTransformerPlacement` of placed or local projections, built by each architecture from its own rows or local everywhere when unplaced) |
| `implementations/transformer/placement.py` | The transformer arches' shared rows, their binding against a variant's independent axes, and the placed layers and Muon staging they yield |
| `implementations/transformer/backbone.py` | The transformer backbone protocol, the architecture protocol that builds one, and `BackboneCIFn`, which delegates the lifecycle to a backbone, casts taps and squashes its preactivations |
| `implementations/transformer/component_conditioning.py` | The architecture wrapping a backbone architecture, and the backbone wrapping its backbone with each site's two-arm readout of its component activations `x @ V`, read from the target's prepared components; the readouts' calibration to data and their preset row |
| `implementations/chunkwise/arch.py` | Independent transformer backbones for explicit groups of taps and sites |
| `implementations/chunkwise/placement.py` | Chunkwise stacked parameter axes and binding, preset rows (zero1 rows shared with block-selected), chunk-stack census |
| `implementations/chunkwise/chunk_stack.py` | Chunk-storage validation and padding, shared with block-selected |
| `implementations/block_selected/arch.py` | Routed chunkwise transformers, selected expert heads, and the `RoutingConditioning` they consume |
| `implementations/block_selected/placement.py` | The expert families' axes, preset rows (chunkwise rows plus expert families), and binding |
| `implementations/block_selected/routing.py` | Block-selected expert execution, local or sharded |
| `implementations/global_transformer/arch.py` | One transformer backbone over every tap, heads stacked per semantic group |
| `implementations/global_transformer/placement.py` | The global transformer's `depth`/`site` axes, preset rows, and binding |
| `implementations/mlp.py` | Shared pointwise-MLP primitives and initialization |
| `implementations/layerwise_mlp.py` | One pointwise MLP per site |
| `implementations/global_mlp.py` | One pointwise MLP over all named taps |
| `../../attention.py` | Shared attention backend and batch/head sharding |

Import names from their defining modules. Shared code does not import concrete
architectures. Runtime operations delegate through the interfaces; chunk-stack execution belongs
to the chunkwise implementations. Their stored-array traversal still identifies each
leaf's semantic axes and selects its declared placement row. Shared execution
primitives interpret physical layouts. Pointwise MLPs retain a fixed batch-owner
storage policy, implemented in the shared execution layer rather than authored
transformer rows.

Parameter fields and tree paths are independent of this module organization.
Checkpoint restore supplies a destination tree built by the matching architecture;
module paths are not part of the checkpoint schema.
