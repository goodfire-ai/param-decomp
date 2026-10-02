"""One transformer over every input tap, with output heads for every decomposed site."""

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, PRNGKeyArray

from param_decomp.core.axes import Axes
from param_decomp.core.ci_fn.architecture import sequence_length_for_flops
from param_decomp.core.ci_fn.implementations.global_transformer.placement import (
    ATTN_KV_AXES,
    ATTN_OUT_AXES,
    ATTN_Q_AXES,
    FFN_IN_AXES,
    FFN_OUT_AXES,
    INPUT_AXES,
    OUTPUT_AXES,
    bind_global_rows,
    preset_rows,
)
from param_decomp.core.ci_fn.implementations.transformer.backbone import BackboneCIFn
from param_decomp.core.ci_fn.implementations.transformer.layers import (
    CI_FN_INPUT_AXES,
    CI_FN_OUTPUT_AXES,
    CI_FN_RMS_EPS,
    CI_FN_TRANSFORMER_STAGING_IN_PLACE,
    LOCAL_CI_FN_TRANSFORMER_PLACEMENT,
    CIFnAttention,
    CIFnBlock,
    CIFnFfnKind,
    CIFnTransformerMuonStaging,
    CIFnTransformerPlacement,
    CIFnTransformerStructure,
    LocalProjection,
    PlacedProjection,
    ci_fn_linear,
    dense_transformer_flops,
    dense_transformer_parameters,
    init_transformer_parameters,
    normalized_tap_concatenation,
)
from param_decomp.core.ci_fn.implementations.transformer.placement import PlacedTransformerCIFnRows
from param_decomp.core.ci_fn.interface import CIFnMatrix, SiteDict, TapSpec
from param_decomp.core.components import SiteSpec, vu_groups
from param_decomp.core.flops.types import ForwardBackwardFlops, ParameterCensus
from param_decomp.core.linear_plan import stacked_placed_linear
from param_decomp.core.model import CaptureKeys, ComponentActivations, PositionAxis
from param_decomp.core.placement import (
    CIFnWeightPlacement,
    PlacementRules,
    gather_reduced_weights,
)
from param_decomp.core.precision import COMPUTE_DT, cast_floating
from param_decomp.sequence import SequenceLayout


@dataclass(frozen=True)
class UnplacedGlobalCIFn:
    """Off-mesh execution: plain matmuls, Muon in place."""

    def layers(self) -> CIFnTransformerPlacement:
        return LOCAL_CI_FN_TRANSFORMER_PLACEMENT

    def muon_staging(self) -> CIFnTransformerMuonStaging:
        return CI_FN_TRANSFORMER_STAGING_IN_PLACE

    def constrain_activation(self, x: Array) -> Array:
        return x

    def description_for_log(self) -> str:
        return "ci_fn placement: unplaced"


GlobalCIFnPlacement = PlacedTransformerCIFnRows | UnplacedGlobalCIFn
"""Placed rows tile every leaf exactly, so nothing pads."""


@dataclass(frozen=True)
class GlobalTransformerCIFnArch:
    """One transformer reads every input tap and predicts every decomposed site's CI.

    Each tap is RMS-normalized independently before concatenation. The blocks stack
    along a real `depth` axis and each semantic group's output heads along a `site`
    axis, so placement can own whole blocks and whole heads."""

    input_taps: tuple[TapSpec, ...]
    d_model: int
    n_blocks: int
    attention: CIFnAttention
    ffn_hidden: int
    ffn_kind: CIFnFfnKind
    learned_norm_scale: bool

    @property
    def capture_keys(self) -> CaptureKeys:
        return frozenset(tap.key for tap in self.input_taps)

    @property
    def input_dim(self) -> int:
        return sum(tap.width for tap in self.input_taps)

    def initialize(
        self, sites: tuple[SiteSpec, ...], rules: PlacementRules | None, key: PRNGKeyArray
    ) -> BackboneCIFn:
        return BackboneCIFn(self.initialize_backbone(sites, rules, key))

    def initialize_backbone(
        self, sites: tuple[SiteSpec, ...], rules: PlacementRules | None, key: PRNGKeyArray
    ) -> "GlobalTransformerBackbone":
        match rules:
            case None:
                placement: GlobalCIFnPlacement = UnplacedGlobalCIFn()
            case PlacementRules():
                placement = self.resolve_placement(sites, rules)
        return init_global_transformer_backbone(self, sites, placement, key)

    def resolve_placement(
        self, sites: tuple[SiteSpec, ...], rules: PlacementRules
    ) -> PlacedTransformerCIFnRows:
        """The preset's rows bound to the mesh, each checked against every leaf it places."""
        rows = bind_global_rows(preset_rows(rules.ci_fn), rules.mesh)
        rows.activations.validate_shape(("q_head",), (self.attention.n_heads,))
        rows.activations.validate_shape(("kv_head",), (self.attention.n_kv_heads,))
        leaves = eqx.filter_eval_shape(
            lambda: init_global_transformer_backbone(
                self, sites, UnplacedGlobalCIFn(), jax.random.key(0)
            )
        )
        for leaf, axes, weights in _matrices(leaves, rows.weights):
            for row in (weights.optimizer_state, weights.compute_weights, weights.operands):
                row.validate_shape(axes, leaf.shape)
        for leaf, axes in _vectors(leaves):
            rows.vectors.validate_shape(axes, leaf.shape)
        return rows

    def parameter_census(self, sites: tuple[SiteSpec, ...]) -> ParameterCensus:
        return dense_transformer_parameters(
            self, input_dim=self.input_dim, n_stacks=1, output_widths=tuple(s.C for s in sites)
        )

    def useful_flops(
        self,
        sites: tuple[SiteSpec, ...],
        batch_size: int,
        positions: PositionAxis,
        *,
        n_selected_blocks_per_token: int | None,
    ) -> ForwardBackwardFlops:
        del n_selected_blocks_per_token
        length = sequence_length_for_flops(batch_size, positions, has_position_axis=True)
        return dense_transformer_flops(
            self,
            input_dim=self.input_dim,
            n_stacks=1,
            output_width=sum(site.C for site in sites),
            batch_size=batch_size,
            sequence_length=length,
        )


class CIFnHeadStack(eqx.Module):
    """One semantic group's output heads, stacked along `site` in the group's site order."""

    weights: Float[Array, "site d_model C"]
    biases: Float[Array, "site C"]
    sites: tuple[str, ...] = eqx.field(static=True)


class GlobalTransformerBackbone(eqx.Module):
    """One transformer over independently normalized input taps. Every `blocks` leaf
    leads with `depth`, scanned one block at a time."""

    in_proj_w: Float[Array, "input d_model"]
    in_proj_b: Float[Array, " d_model"]
    blocks: CIFnBlock
    heads: tuple[CIFnHeadStack, ...]
    inv_freq: Array
    input_taps: tuple[TapSpec, ...] = eqx.field(static=True)
    output_names: tuple[str, ...] = eqx.field(static=True)
    placement: GlobalCIFnPlacement = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    @property
    def capture_keys(self) -> CaptureKeys:
        return frozenset(tap.key for tap in self.input_taps)

    def placement_for_log(self) -> str:
        return self.placement.description_for_log()

    def matrix_parameters(self) -> tuple[CIFnMatrix, ...]:
        return tuple(
            CIFnMatrix(value, staging)
            for value, _, staging in _matrices(self, self.placement.muon_staging())
        )

    def shardings(self, mesh: Mesh) -> "GlobalTransformerBackbone":
        placement = self.placement
        match placement:
            case UnplacedGlobalCIFn():
                raise AssertionError("an unplaced CI fn has no mesh to shard its parameters over")
            case PlacedTransformerCIFnRows() as rows:

                def matrix(value: Array, axes: Axes, weights: CIFnWeightPlacement) -> NamedSharding:
                    row = weights.optimizer_state
                    row.validate_shape(axes, value.shape)
                    return row.sharding_for(axes)

                def vector(value: Array, axes: Axes) -> NamedSharding:
                    rows.vectors.validate_shape(axes, value.shape)
                    return rows.vectors.sharding_for(axes)

                return eqx.tree_at(
                    lambda backbone: (
                        *(value for value, _, _ in _matrices(backbone, rows.weights)),
                        *(value for value, _ in _vectors(backbone)),
                        backbone.inv_freq,
                    ),
                    self,
                    (
                        *(matrix(*entry) for entry in _matrices(self, rows.weights)),
                        *(vector(*entry) for entry in _vectors(self)),
                        NamedSharding(mesh, P()),
                    ),
                )

    def prepare(self) -> "GlobalTransformerBackbone":
        """Compute-dtype residents, every matrix gathered to its compute row."""
        compute = cast_floating(self, COMPUTE_DT)
        placement = self.placement
        match placement:
            case UnplacedGlobalCIFn():
                return compute
            case PlacedTransformerCIFnRows() as rows:

                def resident(value: Array, axes: Axes, weights: CIFnWeightPlacement) -> Array:
                    return gather_reduced_weights(
                        value,
                        source=weights.optimizer_state,
                        destination=weights.compute_weights,
                        axes=axes,
                    )

                return eqx.tree_at(
                    lambda backbone: tuple(
                        value for value, _, _ in _matrices(backbone, rows.weights)
                    ),
                    compute,
                    tuple(resident(*entry) for entry in _matrices(compute, rows.weights)),
                )

    def preactivations(
        self,
        taps: dict[str, Array],
        conditioning: object,
        components: ComponentActivations,
        *,
        sequence: SequenceLayout | None,
        remat: bool,
    ) -> SiteDict:
        del conditioning, components
        placement = self.placement
        layers = placement.layers()
        for tap in self.input_taps:
            assert taps[tap.key].shape[-1] == tap.width, (tap, taps[tap.key].shape)
        x = normalized_tap_concatenation(
            [taps[tap.key] for tap in self.input_taps], placement.constrain_activation, self.eps
        )
        if sequence is None:
            sequence = SequenceLayout.unsegmented_sequences_like(x[..., 0])
        inv_freq = jax.lax.stop_gradient(self.inv_freq)
        x = ci_fn_linear(x, self.in_proj_w, layers.input, CI_FN_INPUT_AXES, transposed=False)
        x = x + self.in_proj_b
        block_arrays, block_static = eqx.partition(self.blocks, eqx.is_array)

        def step(carry: Array, arrays: CIFnBlock) -> tuple[Array, None]:
            block = eqx.combine(arrays, block_static)
            return block(carry, inv_freq, placement=layers, sequence=sequence), None

        # `remat` chooses only whether block activations are recomputed; every block
        # re-gathers its weights in the backward either way.
        policy = (
            jax.checkpoint_policies.nothing_saveable
            if remat
            else jax.checkpoint_policies.dots_saveable
        )
        x, _ = jax.lax.scan(jax.checkpoint(step, policy=policy), x, block_arrays)
        preactivations: SiteDict = {}
        for head in self.heads:
            values = _project_heads(x, head, layers)
            for index, site in enumerate(head.sites):
                preactivations[site] = values[index]
        return {name: preactivations[name] for name in self.output_names}


def _matrices[Value](
    backbone: GlobalTransformerBackbone, per_part: CIFnTransformerStructure[Value]
) -> tuple[tuple[Array, Axes, Value], ...]:
    """Every stored matrix with its semantic axes and its part's `per_part` value."""
    blocks = backbone.blocks
    gate = () if blocks.gate is None else ((blocks.gate[0], FFN_IN_AXES, per_part.ffn),)
    return (
        (backbone.in_proj_w, INPUT_AXES, per_part.input),
        (blocks.wq, ATTN_Q_AXES, per_part.attention),
        (blocks.wk, ATTN_KV_AXES, per_part.attention),
        (blocks.wv, ATTN_KV_AXES, per_part.attention),
        (blocks.wo, ATTN_OUT_AXES, per_part.attention),
        (blocks.w1, FFN_IN_AXES, per_part.ffn),
        (blocks.w2, FFN_OUT_AXES, per_part.ffn),
        *gate,
        *((head.weights, OUTPUT_AXES, per_part.output) for head in backbone.heads),
    )


def _vectors(backbone: GlobalTransformerBackbone) -> tuple[tuple[Array, Axes], ...]:
    blocks = backbone.blocks
    gate: tuple[tuple[Array, Axes], ...] = (
        () if blocks.gate is None else ((blocks.gate[1], ("depth", "ffn_hidden")),)
    )
    norm_scales: tuple[tuple[Array, Axes], ...] = (
        ()
        if blocks.norm_scales is None
        else tuple((scale, ("depth", "d_model")) for scale in blocks.norm_scales)
    )
    return (
        (backbone.in_proj_b, ("d_model",)),
        (blocks.b1, ("depth", "ffn_hidden")),
        (blocks.b2, ("depth", "d_model")),
        *gate,
        *norm_scales,
        *((head.biases, ("site", "C")) for head in backbone.heads),
    )


def _project_heads(x: Array, head: CIFnHeadStack, layers: CIFnTransformerPlacement) -> Array:
    """Every head of one stack applied to `x` at once: `[site, *leading, C]`."""
    biases = jnp.expand_dims(head.biases, tuple(range(1, x.ndim)))
    match layers.output:
        case LocalProjection():
            return jnp.einsum("...d,sdc->s...c", x, head.weights) + biases
        case PlacedProjection() as output:
            plan = output.plan(CI_FN_OUTPUT_AXES, x.ndim, transposed=False)
            return stacked_placed_linear(x, head.weights, plan) + biases


def init_global_transformer_backbone(
    arch: GlobalTransformerCIFnArch,
    sites: tuple[SiteSpec, ...],
    placement: GlobalCIFnPlacement,
    key: PRNGKeyArray,
) -> GlobalTransformerBackbone:
    """Heads are drawn in site order, then stacked by semantic group."""
    tap_keys = tuple(tap.key for tap in arch.input_taps)
    assert tap_keys and len(set(tap_keys)) == len(tap_keys), tap_keys
    output_names = tuple(site.name for site in sites)
    assert output_names and len(set(output_names)) == len(output_names), output_names
    assert arch.n_blocks > 0, f"the global transformer needs at least one block: {arch.n_blocks}"
    n_heads = arch.attention.n_heads
    head_dim = arch.d_model // n_heads
    assert arch.d_model % n_heads == 0 and head_dim % 2 == 0, (arch.d_model, n_heads)
    inv_freq = 1.0 / (10000.0 ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    parameters = init_transformer_parameters(
        arch, arch.input_dim, tuple(site.C for site in sites), key
    )
    site_index = {name: index for index, name in enumerate(output_names)}
    heads = tuple(
        CIFnHeadStack(
            weights=jnp.stack([parameters.out_ws[site_index[spec.name]] for spec in group.specs]),
            biases=jnp.stack([parameters.out_bs[site_index[spec.name]] for spec in group.specs]),
            sites=tuple(spec.name for spec in group.specs),
        )
        for group in vu_groups(sites).values()
    )
    return GlobalTransformerBackbone(
        in_proj_w=parameters.in_proj_w,
        in_proj_b=parameters.in_proj_b,
        blocks=jax.tree.map(lambda *leaves: jnp.stack(leaves), *parameters.blocks),
        heads=heads,
        inv_freq=inv_freq,
        input_taps=arch.input_taps,
        output_names=output_names,
        placement=placement,
        eps=CI_FN_RMS_EPS,
    )
