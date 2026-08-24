"""`DecomposedModel` — the interface a vendored target implements for the generic trainer.

The trainer (`train.py`) is abstract over the target model: it sees an ordered set of
decomposed **sites** (SPEC §1.2) and a handful of methods on the model `eqx.Module`. The
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
`clean_forward` / `masked_forward` (tokens for an LM, a dict for a bio target), the model
OUTPUT carried by `ForwardResult.output` (`Any` — logits, a tuple of heads, coords), and
the recon comparison (`recon_loss_fn`, `kl_per_position` for an LM). Activation identity
and capture lowering are target-owned. Core passes immutable canonical names into the
forward and receives a strict one-key-to-one-array capture dictionary back.

The frozen weights ride on the model `eqx.Module` and reach the jitted step as a pytree
ARG (`eqx.filter_jit` traces the array leaves). Never close over the model in a jit: a
frozen 8B target captured as a constant bakes multi-GB weights into the HLO.
"""

import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Generic, Protocol, runtime_checkable

import equinox as eqx
import jax
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Bool, Float
from typing_extensions import TypeVar

from param_decomp.core.axes import MeshAxis
from param_decomp.core.components import ComponentStacks, SiteSpec
from param_decomp.core.placement import (
    PlacementRules,
    component_stacks_to_faithfulness_weights,
    constrain_faithfulness_deltas,
)
from param_decomp.core.precision import COMPUTE_DT, cast_floating

BATCH_AXES: tuple[MeshAxis, ...] = ("replicate", "fsdp")
"""Mesh axes that jointly shard the always-leading batch dimension."""


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


SiteMasks = dict[str, Float[Array, "*leading C"]]
"""Per-site component masks. `*leading` always has the WAIST's RANK, but ANY leading axis
may arrive size 1: an adversarial mask is materialized from a source stored per
`source_shape` (`configs.SourceShape`), and every axis that spelling omits is a size-1
broadcast axis — on a positioned target `c` gives `[1, 1, C]`, `bc` `[B, 1, C]`, `sc`
`[1, P, C]`; positionless `c` gives `[1, C]`. Only the stochastic and constant sources
build their masks at the full waist shape (from the CI). A target must therefore BROADCAST the leading axes against its own
waist, never reshape them: a reshape survives every stochastic step and dies on the first
adversarial one, long after the run looks healthy."""

SiteDeltaMasks = dict[str, Float[Array, "*leading"]]
"""The weight-delta counterpart of `SiteMasks` — same leading axes and broadcast rule,
with no C axis."""

SiteRoutes = dict[str, Bool[Array, "*leading"]] | None
"""Per-site per-position routing; `None` routes every position to the decomposition
(SPEC §1.3). Positions routing False take the frozen `x @ W` path."""


@dataclass(frozen=True, kw_only=True)
class MaterializedMasking:
    """A masked forward driven by concrete per-site mask arrays.

    Adversarial masks use this arm after optimized sources are converted to arrays; their
    provenance does not change target execution. The masks must cover every one of the
    model's sites — the target asserts this when the forward is first traced.
    ``weight_delta_masks=None`` means the frozen-weight correction is disabled; a mapping
    enables it for every site. Routes, when present, must cover the same sites. These
    constraints make the previous contradictory ``zero delta masks + has_delta=False``
    state unrepresentable.
    """

    component_masks: SiteMasks
    weight_delta_masks: SiteDeltaMasks | None = None
    routes: SiteRoutes = None

    def __post_init__(self) -> None:
        sites = set(self.component_masks)
        if self.weight_delta_masks is not None:
            assert set(self.weight_delta_masks) == sites, (
                self.weight_delta_masks.keys(),
                self.component_masks.keys(),
            )
        if self.routes is not None:
            assert set(self.routes) == sites, (self.routes.keys(), self.component_masks.keys())


@dataclass(frozen=True, kw_only=True)
class StochasticMasking:
    """A recipe for sampling component and weight-delta masks inside checkpointed blocks.

    Scan targets consume the shared CI activations and key directly, discard each layer's
    masks after its forward block, and deterministically redraw them during backward
    recomputation instead of storing a full layer-by-layer mask stack. Masks are drawn
    for every one of the model's sites; routes, when present, must cover them all —
    the target asserts this when the forward is first traced.
    """

    ci_stacked: Any
    draw_key: Array
    routes: SiteRoutes


Masking = MaterializedMasking | StochasticMasking
"""The two complete, non-contradictory descriptions of a masked forward."""


type CaptureKeys = frozenset[str]
"""An orderless, immutable request for named activations from a forward."""

EMPTY_CAPTURE_KEYS: CaptureKeys = frozenset()


def select_captures(captures: dict[str, Array], capture_keys: CaptureKeys) -> dict[str, Array]:
    """Project a capture result onto one deterministic requested view."""
    return {key: captures[key] for key in sorted(capture_keys)}


PreparedT = TypeVar("PreparedT", default=Any)


@partial(
    jax.tree_util.register_dataclass,
    data_fields=("output", "captures"),
    meta_fields=(),
)
@dataclass(frozen=True)
class ForwardResult:
    """A target output and its captured activations, keyed one-to-one."""

    output: Any
    captures: dict[str, Array]

    @classmethod
    def from_producer(
        cls,
        *,
        output: Any,
        capture_keys: tuple[str, ...],
        capture_values: tuple[Array, ...],
    ) -> "ForwardResult":
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
            data_size = math.prod(mesh.shape[axis] for axis in BATCH_AXES)

            def pin(value: Array) -> Array:
                if value.shape[0] % data_size == 0:
                    return jax.sharding.reshard(value, P(BATCH_AXES, *((None,) * (value.ndim - 1))))
                # Ragged eval micro-batch: the pin would refuse the split, and the
                # value's explicit typing is already definite — but only on THIS mesh.
                assert jax.typeof(value).sharding.mesh == mesh, (
                    jax.typeof(value).sharding,
                    mesh,
                )
                return value

            captures = {key: pin(value) for key, value in captures.items()}
        return cls(output, captures)


@runtime_checkable
class DecomposedModel(Protocol[PreparedT]):
    """The target interface consumed by the generic trainer.

    Core passes an immutable set of canonical activation names into each forward. The target
    validates, orders, and lowers those names into its private capture layout when JAX first
    traces that forward; no plan representation crosses this protocol. An empty set must take
    the target's untouched no-capture computation.
    """

    sites: tuple[SiteSpec, ...]
    has_position_axis: bool

    @property
    def site_names(self) -> tuple[str, ...]: ...

    def shardings(self, placement: PlacementRules) -> "DecomposedModel[PreparedT]": ...

    def recon_loss_fn(self, masked_output: Any, clean_output: Any) -> Float[Array, ""]: ...

    def site_output_keys(self, sites: tuple[str, ...]) -> tuple[str, ...]:
        """Return each site's canonical linear-output key in request order."""
        ...

    def assert_hidden_acts_reconstruction_points(self, keys: tuple[str, ...]) -> None:
        """Refuse capture points that masking can never change."""
        ...

    def clean_forward(
        self,
        inputs: Any,
        /,
        capture_keys: CaptureKeys = EMPTY_CAPTURE_KEYS,
        *,
        placement: PlacementRules | None,
    ) -> ForwardResult:
        """All-frozen forward plus exactly `capture_keys`. The same key has the same meaning
        here and in `masked_forward`."""
        ...

    def prepare_compute_weights(
        self, vu: ComponentStacks, placement: PlacementRules | None
    ) -> PreparedT:
        """Relayout compute-dtype components into the target-private per-step view."""
        ...

    def component_activation_forward(
        self,
        prepared_weights: PreparedT,
        inputs: Any,
        /,
        *,
        capture_keys: CaptureKeys,
        placement: PlacementRules | None,
    ) -> tuple[ForwardResult, dict[str, Array]]:
        """Run the frozen target once, returning requested captures and each site's ``x @ V``.

        Targets that do not support offline component-activation harvest must raise
        ``NotImplementedError`` explicitly.
        """
        ...

    def stack_ci(self, ci_lower: dict[str, Array]) -> Any:
        """Build the target-private CI form shared by stochastic masked forwards."""
        ...

    def masked_forward(
        self,
        prepared_weights: PreparedT,
        inputs: Any,
        /,
        *,
        masking: Masking,
        placement: PlacementRules | None,
        capture_keys: CaptureKeys = EMPTY_CAPTURE_KEYS,
        remat: bool,
    ) -> ForwardResult:
        """Masked decomposed forward plus exactly `capture_keys`.

        `masking` carries the complete masking policy: explicit masks, or shared CI plus
        a draw key for rebuilding stochastic masks inside checkpointed blocks. The target
        validates unsupported capture keys fail-closed when this method is first traced.
        """
        ...

    def target_weight_sq_norms(self) -> dict[str, Float[Array, " g"]]:
        """Per-slot `‖W_s‖²_F` of each frozen target stack, fp32 — slot-aligned with the
        `weight_deltas` grouping (`site_slots_for(self.sites)`), read once at setup to
        bind the S17 relative-error scales."""
        ...

    def weight_deltas(self, vu: ComponentStacks) -> dict[str, Float[Array, "g d_out d_in"]]:
        """fp32 `W − V@U` per persistence STACK, slot-aligned with `vu.stacks` (SPEC N2).

        Stacked, not per-site: the only trainer consumer is the S17 faithfulness
        loss (per-slot reductions of the stacks), and slicing per-site V/U out of a
        stack-sharded persist layout
        redistributes every slice cross-node (task 577). Per-site access (tests,
        offline consumers) is `site_weight_delta`."""
        ...


class PlacedModel(eqx.Module, Generic[PreparedT]):
    """A decomposed model paired with ITS placement — resolved exactly once, at run
    assembly (`place_target`, or a literal construction for an unplaced execution), so no
    downstream code ever holds an unresolved (model, rules) combination. `placement is
    None` means the model runs unplaced (the CPU/test execution) — a decided state, not
    an omission. The frozen weights are pytree children (traced — the HLO-baking rule
    holds through the wrapper); the placement is static and rides the treedef, so the
    pair threads through jit/vjp as one value and cannot desync.

    The forwards delegate with this placement supplied; the static reads (`sites`,
    `site_names`, …) delegate unchanged. Target-specific surfaces beyond the protocol
    (e.g. attention-pattern probes) narrow via `.model` and receive `.placement`
    explicitly."""

    model: DecomposedModel[PreparedT]
    placement: PlacementRules | None = eqx.field(static=True)

    @property
    def sites(self) -> tuple[SiteSpec, ...]:
        return self.model.sites

    @property
    def site_names(self) -> tuple[str, ...]:
        return self.model.site_names

    @property
    def has_position_axis(self) -> bool:
        return self.model.has_position_axis

    def recon_loss_fn(self, masked_output: Any, clean_output: Any) -> Float[Array, ""]:
        return self.model.recon_loss_fn(masked_output, clean_output)

    def site_output_keys(self, sites: tuple[str, ...]) -> tuple[str, ...]:
        return self.model.site_output_keys(sites)

    def assert_hidden_acts_reconstruction_points(self, keys: tuple[str, ...]) -> None:
        self.model.assert_hidden_acts_reconstruction_points(keys)

    def stack_ci(self, ci_lower: dict[str, Array]) -> Any:
        return self.model.stack_ci(ci_lower)

    def clean_forward(
        self, inputs: Any, /, capture_keys: CaptureKeys = EMPTY_CAPTURE_KEYS
    ) -> ForwardResult:
        return self.model.clean_forward(inputs, capture_keys, placement=self.placement)

    def component_activation_forward(
        self, prepared_weights: PreparedT, inputs: Any, /, *, capture_keys: CaptureKeys
    ) -> tuple[ForwardResult, dict[str, Array]]:
        return self.model.component_activation_forward(
            prepared_weights, inputs, capture_keys=capture_keys, placement=self.placement
        )

    def masked_forward(
        self,
        prepared_weights: PreparedT,
        inputs: Any,
        /,
        *,
        masking: Masking,
        capture_keys: CaptureKeys = EMPTY_CAPTURE_KEYS,
        remat: bool,
    ) -> ForwardResult:
        return self.model.masked_forward(
            prepared_weights,
            inputs,
            masking=masking,
            placement=self.placement,
            capture_keys=capture_keys,
            remat=remat,
        )


def prepare_compute_weights[PreparedT](
    placed: PlacedModel[PreparedT], components: ComponentStacks
) -> PreparedT:
    """Cast fp32 master components once, then build the target-private compute layout
    through the bundle's declared placement lifecycle (`None` = the unplaced CPU/test
    execution)."""
    return placed.model.prepare_compute_weights(
        cast_floating(components, COMPUTE_DT), placed.placement
    )


def faithfulness_weight_deltas(
    placed: PlacedModel, components: ComponentStacks
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
    group, slot = vu.slot_of(name)
    return deltas[group][slot]
