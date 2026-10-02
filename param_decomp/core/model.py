"""`DecomposedModel` — the interface a vendored target implements for the generic trainer.

The trainer (`train.py`) is abstract over the target model: it sees an ordered set of
decomposed **sites** and a handful of methods on the model `eqx.Module`. The
model carries its FROZEN target weights as fields; the TRAINABLE V/U (`vu`) is passed to
the forward methods explicitly (separate lifecycle). Everything at the boundary is keyed
by site name (flat dicts, torch-module-path style) — except `weight_deltas`, keyed like
`vu.stacks` so its only consumer (`faithfulness_loss`) never slices the stack-sharded
persist layout per site; how a target lays its parameters out internally (e.g. the Llama
target's stacked layer axis) is its own business.

The activation WAIST comes in exactly TWO shapes: positionless `[B, d]` (masks/CI
`[B, C]` — the toys) or with one position axis `[B, P, d]` (masks/CI `[B, P, C]` — an
LM, whose position axis is the token sequence). `has_position_axis` declares which;
`Positionless` / `Positioned` carry the run-scoped extents. Those are the waist's shapes;
a mask's leading axes match only in RANK, and are size 1 wherever the adversary's
`source_shape` says so (`SiteMasks`). Batch is ever-present and
semantics-free (the data/shard axis); CI is always independent over every leading axis.
Masking, routing, source scopes, imp-min, and normalization all operate over the opaque
leading prefix. The three EDGES are generic too — the model INPUT consumed by
`clean_forward` (tokens for an LM, a dict for a bio target), the model
OUTPUT (`ForwardResult[Out, Conditioning].output` — logits, a tuple of heads, coords, or an
LM's factored streamed package; `Out` is the target's declared type, named at every
seam), and the recon comparison (`recon_loss_fn`, `kl_per_position` for an LM). Core never
inspects an output: the two operations it needs on one — comparing two, batch-pinning one
— are the target's own protocol methods on `Out`, and everything else core knows about
outputs (a well-temperedness ablation's damage, say) is derived from those. Activation
identity and capture lowering are target-owned. Core passes immutable canonical names into
the forward and receives a strict one-key-to-one-array capture dictionary back. The
clean forward also returns the target's inputs and pinned decisions (`Conditioning` — `ForwardResult.
conditioning`), which core hands back unread to every masked forward and CI evaluation that
reproduces that clean pass.

The frozen weights ride on the model `eqx.Module` and reach the jitted step as a pytree
ARG (`eqx.filter_jit` traces the array leaves). Never close over the model in a jit: a
frozen 8B target captured as a constant bakes multi-GB weights into the HLO.
"""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Bool, Float, Int

from param_decomp.core.components import ComponentStacks, SiteCI, SiteSpec
from param_decomp.core.placement import (
    PlacementRules,
    batch_axes,
    component_stacks_to_faithfulness_weights,
    constrain_faithfulness_deltas,
)
from param_decomp.core.precision import COMPUTE_DT, cast_floating
from param_decomp.core.pytree import ShardingTree
from param_decomp.core.source_mask import SourceMaskIngredients
from param_decomp.sequence import SequenceLayout


@dataclass(frozen=True)
class Positionless:
    """Waist `[B, d]`; masks/CI `[B, C]`. The toys."""


@dataclass(frozen=True)
class Positioned:
    """Waist `[B, P, d]`; masks/CI `[B, P, C]`. An LM: the position axis is the token
    sequence, so `n_positions` is its training seq_len (run-scoped — from the data
    config, not the model)."""

    n_positions: int


PositionAxis = Positionless | Positioned
"""The run's waist geometry — exactly these two cases, matched exhaustively wherever
shapes are built. Must agree with the model's `has_position_axis`."""


SiteMasks = Mapping[str, SiteCI]
"""Per-site component masks: full `[*leading, C]` arrays for dense sites, `SelectedCI`
bundles (mask values + the block indices that key them, one object) for selected-emitting
block-factored sites. `*leading` always has the WAIST's RANK, but ANY leading axis
may arrive size 1: an adversarial mask is materialized from a source stored per
`source_shape` (`configs.SourceShape`), and every axis that spelling omits is a size-1
broadcast axis — on a positioned target `c` gives `[1, 1, C]`, `bc` `[B, 1, C]`, `sc`
`[1, P, C]`; positionless `c` gives `[1, C]`. Only the stochastic and constant sources
build their masks at the full waist shape (from the CI). A target must therefore BROADCAST the leading axes against its own
waist, never reshape them: a reshape survives every stochastic step and dies on the first
adversarial one, long after the run looks healthy."""

SiteDeltaMasks = Mapping[str, Float[Array, "*leading"]]
"""The weight-delta counterpart of `SiteMasks` — same leading axes and broadcast rule,
with no C axis."""

SiteRoutes = Mapping[str, Bool[Array, "*leading"]]
"""Per-site, per-position routing: True selects the decomposition, False the target
`x @ Wᵀ`. Routes are their own axis of a masked forward, beside its masking: a forward
takes `SiteRoutes | None`, None selecting the decomposition everywhere, and routes cover
every site or none (`validate_routes`)."""


def validate_routes(routes: SiteRoutes | None, sites: tuple[str, ...]) -> None:
    assert routes is None or set(routes) == set(sites), (sorted(routes or ()), sites)


@dataclass(frozen=True, kw_only=True)
class MaterializedMasking:
    """A masked forward driven by concrete per-site mask arrays.

    Adversarial masks use this arm after optimized sources are converted to arrays; their
    provenance does not change target execution. The masks must cover every one of the
    model's sites — the target asserts this when the forward is first traced.
    ``weight_delta_masks=None`` means the frozen-weight correction is disabled; a mapping
    enables it for every site. These constraints make the previous contradictory
    ``zero delta masks + has_delta=False`` state unrepresentable.
    """

    component_masks: SiteMasks
    weight_delta_masks: SiteDeltaMasks | None = None

    def __post_init__(self) -> None:
        sites = set(self.component_masks)
        if self.weight_delta_masks is not None:
            assert set(self.weight_delta_masks) == sites, (
                self.weight_delta_masks.keys(),
                self.component_masks.keys(),
            )


@dataclass(frozen=True)
class ComponentOverrides:
    """Replacement masked activations for chosen components of one site: row `i` sets
    the masked `x@V` at waist index `indices[i]` (the leading coordinates, then the
    component) to `values[i]`. Rows are distinct and index the site's full waist — an
    override has no broadcast axes, whatever its site's mask carries."""

    indices: Int[Array, "n_overrides waist_rank"]
    values: Float[Array, " n_overrides"]

    def __post_init__(self) -> None:
        assert jnp.issubdtype(self.indices.dtype, jnp.integer), self.indices.dtype
        assert self.indices.ndim == 2, self.indices.shape
        assert self.values.shape == self.indices.shape[:1], (self.values.shape, self.indices.shape)


SiteOverrides = Mapping[str, ComponentOverrides]
"""Per-site component activation overrides, orthogonal to the masking: a site absent
from the mapping has none."""


@dataclass(frozen=True, kw_only=True)
class StochasticMasking:
    """A logical per-site CI envelope and its stochastic draw key.

    Targets choose where to sample the masks when preparing this recipe.
    """

    ci: Mapping[str, SiteCI]
    draw_key: Array


@dataclass(frozen=True, kw_only=True)
class SourceMasking:
    """Aligned CI/source pairs consumed under the target's eager or rematerialized policy.

    Each pair carries one selected routing frame, its two component payloads, and
    the token-coordinate delta source. Every decomposed site must be represented.
    """

    ingredients: dict[str, SourceMaskIngredients]


Masking = MaterializedMasking | StochasticMasking | SourceMasking
"""The complete, non-contradictory descriptions of a masked forward."""


type CaptureKeys = frozenset[str]
"""An orderless, immutable request for named activations from a forward."""

EMPTY_CAPTURE_KEYS: CaptureKeys = frozenset()


def select_captures(captures: dict[str, Array], capture_keys: CaptureKeys) -> dict[str, Array]:
    """Project a capture result onto one deterministic requested view."""
    return {key: captures[key] for key in sorted(capture_keys)}


# beartype checks parameters annotated with this protocol by `isinstance`.
@runtime_checkable
class ComponentActivations(Protocol):
    """A target's prepared components applied to one site's input: `x @ V` for `site`,
    computed in the compute layout the target's own forwards use."""

    def component_activations(
        self, site: str, x: Float[Array, "*leading d_in"]
    ) -> Float[Array, "*leading C"]: ...


@partial(
    jax.tree_util.register_dataclass,
    data_fields=("output", "captures", "conditioning", "sequence"),
    meta_fields=("leading_shape",),
)
@dataclass(frozen=True)
class ForwardResult[Out, Conditioning]:
    """A target output, its captured activations (keyed one-to-one), and the conditioning
    decisions the forward ran under — its own for a clean forward, the given ones for a
    masked forward (`masked_forward(prepared, clean.conditioning, ...)` carries them through).
    `sequence` optionally carries document layout to the CI function. Token-only and
    positionless targets use None. `leading_shape` is the batch and optional position
    shape shared by CI values, masks, and sources.

    Core carries `output` without ever looking inside it: the trainer, the recon terms,
    and the eval tiers act on an output only through the target's own `recon_loss_fn`
    and `pin_output_batch` — which is what makes the engine arbitrary over both output
    types and reconstruction metrics."""

    output: Out
    captures: dict[str, Array]
    conditioning: Conditioning
    sequence: SequenceLayout | None
    leading_shape: tuple[int, ...]

    @classmethod
    def from_producer(
        cls,
        *,
        output: Out,
        capture_keys: tuple[str, ...],
        capture_values: tuple[Array, ...],
        conditioning: Conditioning,
        sequence: SequenceLayout | None,
        leading_shape: tuple[int, ...],
    ) -> "ForwardResult[Out, Conditioning]":
        """Label a target's private capture slots and pin their shared device layout.

        A target resolves public activation names into a private slot layout while tracing,
        then produces arrays in that layout's order. This constructor is the single boundary
        that checks the canonical names and produced arrays agree, labels the arrays, and
        fixes their device layout before any consumer uses them.

        Captures always lead with the batch axis: on-mesh they land batch-sharded,
        feature-replicated, so every compiled consumer (cuDNN attention included) sees
        one layout. A capture whose batch does not tile the data axes (a small eval
        micro-batch) keeps its own — already definite — explicit typing instead; the
        reshard would refuse the ragged split.

        This must be an explicit producer constructor rather than ``__post_init__``. JAX
        pytree transformations reconstruct this registered dataclass with abstract or
        non-array leaves; reconstruction must not relabel values or apply device placement
        as a side effect.
        """
        assert len(capture_values) == len(capture_keys), (
            len(capture_values),
            capture_keys,
        )
        captures = dict(zip(capture_keys, capture_values, strict=True))
        mesh = jax.sharding.get_abstract_mesh()
        if captures and not mesh.empty:
            data_axes = batch_axes(mesh)
            data_size = math.prod(mesh.shape[axis] for axis in data_axes)

            def pin(value: Array) -> Array:
                if value.shape[0] % data_size == 0:
                    return jax.sharding.reshard(value, P(data_axes, *((None,) * (value.ndim - 1))))
                # Ragged eval micro-batch: the pin would refuse the split, and the
                # value's explicit typing is already definite — but only on THIS mesh.
                assert jax.typeof(value).sharding.mesh == mesh, (
                    jax.typeof(value).sharding,
                    mesh,
                )
                return value

            captures = {key: pin(value) for key, value in captures.items()}
        return cls(output, captures, conditioning, sequence, leading_shape)


@runtime_checkable
class DecomposedModel[
    TargetIn,
    Out,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](Protocol):
    """The target interface consumed by the generic trainer.

    Core passes an immutable set of canonical activation names into each forward. The target
    validates, orders, and lowers those names into its private capture layout when JAX first
    traces that forward; no plan representation crosses this protocol. An empty set must take
    the target's untouched no-capture computation.

    The target that declares `Out` declares the operations on `Out`, and core touches the
    output only through them: `recon_loss_fn` (every output is scored) and
    `pin_output_batch` (every output is placed). They are `@staticmethod`s — pure,
    array-free — so a step factory may close over them off the static model (the
    HLO-baking rule) while every array reaches them through the traced output value.
    """

    sites: tuple[SiteSpec, ...]
    has_position_axis: bool

    @property
    def site_names(self) -> tuple[str, ...]: ...

    def shardings(self, placement: PlacementRules) -> ShardingTree:
        """One sharding per array leaf, preserving this model's pytree structure."""
        ...

    def recon_loss_fn(self, masked_output: Out, clean_output: Out) -> Float[Array, ""]:
        """The end-to-end reconstruction error between a masked and the clean output,
        reduced to one scalar over the whole batch."""
        ...

    def pin_output_batch(self, output: Out, mesh: Mesh | None) -> Out:
        """Pin the output's batch axis over the data mesh (`sharding.batch_shard_leading`
        on each batch-bearing array of the output); `mesh is None` is the identity."""
        ...

    def site_output_keys(self, sites: tuple[str, ...]) -> tuple[str, ...]:
        """Return each site's canonical linear-output key in request order."""
        ...

    def clean_forward(
        self,
        inputs: TargetIn,
        /,
        capture_keys: CaptureKeys = EMPTY_CAPTURE_KEYS,
        *,
        placement: PlacementRules | None,
    ) -> ForwardResult[Out, Conditioning]:
        """All-frozen forward plus exactly `capture_keys`, returning the decisions it
        pinned for the masked forwards that reproduce it. The same key has the same
        meaning here and in `masked_forward`."""
        ...

    def prepare_compute_weights(
        self, vu: ComponentStacks, placement: PlacementRules | None
    ) -> PreparedT:
        """Relayout compute-dtype components into the target-private per-step view, which
        every masked forward and CI evaluation of the step reads."""
        ...

    def component_activation_forward(
        self,
        prepared_weights: PreparedT,
        inputs: TargetIn,
        /,
        *,
        sites: tuple[str, ...],
        capture_keys: CaptureKeys,
        placement: PlacementRules | None,
    ) -> tuple[ForwardResult[Out, Conditioning], dict[str, SiteCI]]:
        """Run the frozen target once, returning requested captures and each requested
        site's ``x @ V`` — a full `[.., C]` array for a dense site, a `SelectedCI` bundle
        (selected picks' values, pick m keyed by the returned pinned selection's m-th
        block) for a selected-emitting block-factored site.

        Targets that do not support offline component-activation harvest must raise
        ``NotImplementedError`` explicitly.
        """
        ...

    def prepare_masking(self, masking: Masking) -> PreparedMaskingT:
        """Translate a per-site recipe into this target's private execution layout."""
        ...

    def prepare_stochastic_masking(
        self, ci: Mapping[str, SiteCI]
    ) -> Callable[[Array], PreparedMaskingT]:
        """Prepare shared CI once; each invocation prepares one draw's keys."""
        ...

    def masked_forward(
        self,
        prepared_weights: PreparedT,
        conditioning: Conditioning,
        /,
        *,
        masking: PreparedMaskingT,
        routes: SiteRoutes | None,
        placement: PlacementRules | None,
        capture_keys: CaptureKeys = EMPTY_CAPTURE_KEYS,
        remat: bool,
    ) -> ForwardResult[Out, Conditioning]:
        """Masked decomposed forward plus exactly `capture_keys`, reproducing the clean
        forward's `conditioning` decisions (never re-deriving them from its own activations).
        `routes` send each site's unrouted positions through the target `x @ Wᵀ`.

        `masking` is this target's prepared execution layout, returned by its preparation
        methods. Core transports it without inspecting its structure. The target validates
        unsupported capture keys fail-closed when this method is first traced.
        """
        ...

    def target_weight_sq_norms(self) -> dict[str, Float[Array, " g"]]:
        """`‖W_s‖²_F` per stack index of each frozen target stack, fp32 — stack-aligned
        with the `weight_deltas` grouping (`site_stack_indices_for(self.sites)`), read once
        at setup to bind the relative-error scales."""
        ...

    def weight_deltas(self, vu: ComponentStacks) -> dict[str, Float[Array, "g ..."]]:
        """fp32 `W − V@U` per persistence STACK, stack-aligned with `vu.stacks`.
        A dense group's stack is `[g, d_out, d_in]`. A block-factored group's is
        `[g, expert, d_out, d_in]` — `expert` being the rows' spelling of the block
        axis — with per-block dimensions; every entry of the site's weight matrix
        belongs to exactly one block, so summing squared error over the blocks equals
        the site-level Frobenius reduction.

        `vu` arrives on the faithfulness lane and may carry persist-stack PADS
        (`vu.stack_pads` — enumerated, never shape-inferred): a target with a stacked
        frozen weight extends it by `vu.pad_of(group)` all-zero matrices so pad deltas are
        exactly zero, and the returned stacks keep the padded extent.

        Stacked, not per-site: the only trainer consumer is the faithfulness
        loss (per-stack-index reductions of the stacks), and slicing per-site V/U out of a
        stack-sharded persist layout
        redistributes every slice across devices. Per-site access (tests,
        offline consumers) is `site_weight_delta`."""
        ...


class PlacedModel[TargetIn, Out, PreparedT: ComponentActivations, Conditioning, PreparedMaskingT](
    eqx.Module
):
    """A decomposed model paired with ITS placement — resolved exactly once, at run
    assembly (`place_target`, or a literal construction for an unplaced execution), so no
    downstream code ever holds an unresolved (model, rules) combination. `placement is
    None` means the model runs unplaced (the CPU/test execution) — a decided state, not
    an omission. The frozen weights are pytree children (traced — the HLO-baking rule
    holds through the wrapper); the placement is static and rides the treedef, so the
    pair threads through jit/vjp as one value and cannot desync.

    The forwards delegate with this placement supplied. Metadata belongs to `.model`.
    Target-specific surfaces beyond the protocol
    (e.g. attention-pattern probes) narrow via `.model` and receive `.placement`
    explicitly."""

    model: DecomposedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT]
    placement: PlacementRules | None = eqx.field(static=True)

    def recon_loss_fn(self, masked_output: Out, clean_output: Out) -> Float[Array, ""]:
        return self.model.recon_loss_fn(masked_output, clean_output)

    def pin_output_batch(self, output: Out, mesh: Mesh | None) -> Out:
        return self.model.pin_output_batch(output, mesh)

    def site_output_keys(self, sites: tuple[str, ...]) -> tuple[str, ...]:
        return self.model.site_output_keys(sites)

    def prepare_compute_weights(self, components: ComponentStacks) -> PreparedT:
        """Cast fp32 components, then materialize the target-private compute layout."""
        return self.model.prepare_compute_weights(
            cast_floating(components, COMPUTE_DT), self.placement
        )

    def clean_forward(
        self, inputs: TargetIn, /, capture_keys: CaptureKeys = EMPTY_CAPTURE_KEYS
    ) -> ForwardResult[Out, Conditioning]:
        return self.model.clean_forward(inputs, capture_keys, placement=self.placement)

    def component_activation_forward(
        self,
        prepared_weights: PreparedT,
        inputs: TargetIn,
        /,
        *,
        sites: tuple[str, ...],
        capture_keys: CaptureKeys,
    ) -> tuple[ForwardResult[Out, Conditioning], dict[str, SiteCI]]:
        return self.model.component_activation_forward(
            prepared_weights,
            inputs,
            sites=sites,
            capture_keys=capture_keys,
            placement=self.placement,
        )

    def masked_forward(
        self,
        prepared_weights: PreparedT,
        conditioning: Conditioning,
        /,
        *,
        masking: PreparedMaskingT,
        routes: SiteRoutes | None,
        capture_keys: CaptureKeys = EMPTY_CAPTURE_KEYS,
        remat: bool,
    ) -> ForwardResult[Out, Conditioning]:
        return self.model.masked_forward(
            prepared_weights,
            conditioning,
            masking=masking,
            routes=routes,
            placement=self.placement,
            capture_keys=capture_keys,
            remat=remat,
        )


def faithfulness_weight_deltas[
    TargetIn,
    Out,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    placed: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
    components: ComponentStacks,
) -> dict[str, Array]:
    """Build fp32 faithfulness deltas through their complete declared placement lifecycle."""
    if placed.placement is None:
        return placed.model.weight_deltas(components)
    weights = component_stacks_to_faithfulness_weights(components, placed.placement.components)
    return constrain_faithfulness_deltas(
        placed.model.weight_deltas(weights), placed.placement.components
    )


def site_weight_delta(
    deltas: dict[str, Array], vu: ComponentStacks, name: str
) -> Float[Array, "d_out d_in"]:
    """One site's delta out of the stacked `weight_deltas` result."""
    group, index = vu.stack_index_of(name)
    return deltas[group][index]
