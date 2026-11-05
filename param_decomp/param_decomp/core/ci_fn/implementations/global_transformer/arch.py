"""One transformer over every input tap, with output heads for every decomposed site."""

from dataclasses import dataclass, replace
from enum import StrEnum

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, PRNGKeyArray

from param_decomp.core.axes import Axes
from param_decomp.core.ci_fn.architecture import sequence_length_for_flops
from param_decomp.core.ci_fn.implementations.global_transformer.placement import (
    INPUT_AXES,
    OUTPUT_AXES,
    GlobalCIStacks,
    PlacedGlobalCIFn,
    bind_global_rows,
    preset_rows,
    resolve_global_stacks,
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
    weightless_rms_norm,
)
from param_decomp.core.ci_fn.interface import CIFnMatrix, SiteDict, TapSpec
from param_decomp.core.components import SiteSpec, vu_groups
from param_decomp.core.flops.types import ForwardBackwardFlops, ParameterCensus
from param_decomp.core.linear_plan import stacked_placed_linear
from param_decomp.core.model import CaptureKeys, ComponentActivations, PositionAxis
from param_decomp.core.placement import (
    CIFnWeightPlacement,
    PlacementRules,
    StackCensus,
    gather_reduced_weights,
    pad_stack,
    strip_stack_pad,
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


GlobalCIFnPlacement = PlacedGlobalCIFn | UnplacedGlobalCIFn
"""Only placed rows cut the persist stacks, so only a placed CI pads them
(`PlacedGlobalCIFn.stacks`)."""


class FinalNormalization(StrEnum):
    """What the output heads read: the last block's output as it is, or RMS-normalized.
    The norm is weightless: the heads are linear, so a per-channel gain would only rescale
    their weights."""

    NONE = "none"
    RMS = "rms"


@dataclass(frozen=True)
class GlobalTransformerCIFnArch:
    """One transformer reads every input tap and predicts every decomposed site's CI.

    Each tap is RMS-normalized independently before concatenation. The blocks stack
    along a real `depth` axis and each semantic group's output heads along a `site`
    axis, so placement can own whole blocks and whole heads. `final_norm` selects what the
    heads read (`FinalNormalization`)."""

    input_taps: tuple[TapSpec, ...]
    d_model: int
    n_blocks: int
    attention: CIFnAttention
    ffn_hidden: int
    ffn_kind: CIFnFfnKind
    learned_norm_scale: bool
    final_norm: FinalNormalization = FinalNormalization.NONE

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
    ) -> PlacedGlobalCIFn:
        """The preset's rows bound to the mesh, with each persist stack padded to tile the
        rows that cut it, and every padded leaf checked against each row that places it."""
        rows = bind_global_rows(preset_rows(rules.ci_fn), rules.mesh)
        rows.activations.validate_shape(("q_head",), (self.attention.n_heads,))
        rows.activations.validate_shape(("kv_head",), (self.attention.n_kv_heads,))
        head_lengths = tuple(len(group.specs) for group in vu_groups(sites).values())
        placement = PlacedGlobalCIFn(rows, resolve_global_stacks(rows, self.n_blocks, head_lengths))
        leaves = eqx.filter_eval_shape(
            lambda: init_global_transformer_backbone(self, sites, placement, jax.random.key(0))
        )
        for leaf, axes, weights, _ in _matrices(leaves, rows.weights):
            for row in (weights.optimizer_state, weights.compute_weights, weights.operands):
                row.validate_shape(axes, leaf.shape)
        for leaf, axes, _ in _vectors(leaves):
            rows.vectors.validate_shape(axes, leaf.shape)
        return placement

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
    leads with `depth`, scanned one block at a time. The fp32 masters carry the
    placement's persist pads (`stacks`); the compute residents `prepare` returns carry
    none."""

    in_proj_w: Float[Array, "input d_model"]
    in_proj_b: Float[Array, " d_model"]
    blocks: CIFnBlock
    heads: tuple[CIFnHeadStack, ...]
    inv_freq: Array
    input_taps: tuple[TapSpec, ...] = eqx.field(static=True)
    output_names: tuple[str, ...] = eqx.field(static=True)
    placement: GlobalCIFnPlacement = eqx.field(static=True)
    stacks: GlobalCIStacks = eqx.field(static=True)
    """The stacks these arrays hold: the placement's padded persist stacks on the masters,
    their real lengths on the residents `prepare` returns."""
    eps: float = eqx.field(static=True)
    final_norm: FinalNormalization = eqx.field(static=True)

    @property
    def capture_keys(self) -> CaptureKeys:
        return frozenset(tap.key for tap in self.input_taps)

    def placement_for_log(self) -> str:
        return self.placement.description_for_log()

    def matrix_parameters(self) -> tuple[CIFnMatrix, ...]:
        return tuple(
            CIFnMatrix(value, staging)
            for value, _, staging, _ in _matrices(self, self.placement.muon_staging())
        )

    def shardings(self, mesh: Mesh) -> "GlobalTransformerBackbone":
        placement = self.placement
        match placement:
            case UnplacedGlobalCIFn():
                raise AssertionError("an unplaced CI fn has no mesh to shard its parameters over")
            case PlacedGlobalCIFn(rows=rows):

                def matrix(value: Array, axes: Axes, weights: CIFnWeightPlacement) -> NamedSharding:
                    row = weights.optimizer_state
                    row.validate_shape(axes, value.shape)
                    return row.sharding_for(axes)

                def vector(value: Array, axes: Axes) -> NamedSharding:
                    rows.vectors.validate_shape(axes, value.shape)
                    return rows.vectors.sharding_for(axes)

                return eqx.tree_at(
                    lambda backbone: (
                        *(value for value, *_ in _matrices(backbone, rows.weights)),
                        *(value for value, *_ in _vectors(backbone)),
                        backbone.inv_freq,
                    ),
                    self,
                    (
                        *(
                            matrix(value, axes, weights)
                            for value, axes, weights, _ in _matrices(self, rows.weights)
                        ),
                        *(vector(value, axes) for value, axes, _ in _vectors(self)),
                        NamedSharding(mesh, P()),
                    ),
                )

    def prepare(self) -> "GlobalTransformerBackbone":
        """Compute-dtype residents: every matrix gathered to its compute row, then every
        stacked leaf sliced to its real stack. The pads ride the gather."""
        compute = cast_floating(self, COMPUTE_DT)
        placement = self.placement
        match placement:
            case UnplacedGlobalCIFn():
                return compute
            case PlacedGlobalCIFn(rows=rows):

                def resident(
                    value: Array,
                    axes: Axes,
                    weights: CIFnWeightPlacement,
                    census: StackCensus | None,
                ) -> Array:
                    gathered = gather_reduced_weights(
                        value,
                        source=weights.optimizer_state,
                        destination=weights.compute_weights,
                        axes=axes,
                    )
                    return _real(gathered, census)

                residents = eqx.tree_at(
                    lambda backbone: (
                        *(value for value, *_ in _matrices(backbone, rows.weights)),
                        *(value for value, *_ in _vectors(backbone)),
                    ),
                    compute,
                    (
                        *(resident(*entry) for entry in _matrices(compute, rows.weights)),
                        *(_real(value, census) for value, _, census in _vectors(compute)),
                    ),
                )
                return replace(residents, stacks=self.stacks.real)

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
        assert self.stacks == self.stacks.real, "preactivations reads prepare()'s residents"
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
        match self.final_norm:
            case FinalNormalization.NONE:
                pass
            case FinalNormalization.RMS:
                x = weightless_rms_norm(x, self.eps)
        preactivations: SiteDict = {}
        for head in self.heads:
            values = _project_heads(x, head, layers)
            for index, site in enumerate(head.sites):
                preactivations[site] = values[index]
        return {name: preactivations[name] for name in self.output_names}


def _matrices[Value](
    backbone: GlobalTransformerBackbone, per_part: CIFnTransformerStructure[Value]
) -> tuple[tuple[Array, Axes, Value, StackCensus | None], ...]:
    """Every stored matrix with its semantic axes, its part's `per_part` value, and the
    census of the stack it leads with (`None` for the unstacked input projection)."""
    stacks = backbone.stacks
    return (
        (backbone.in_proj_w, INPUT_AXES, per_part.input, None),
        *(
            (value, ("depth", *axes), part, stacks.depth)
            for value, axes, part in backbone.blocks.matrix_parameters(per_part)
        ),
        *(
            (head.weights, OUTPUT_AXES, per_part.output, census)
            for head, census in zip(backbone.heads, stacks.sites, strict=True)
        ),
    )


def _vectors(
    backbone: GlobalTransformerBackbone,
) -> tuple[tuple[Array, Axes, StackCensus | None], ...]:
    stacks = backbone.stacks
    return (
        (backbone.in_proj_b, ("d_model",), None),
        *(
            (value, ("depth", *axes), stacks.depth)
            for value, axes in backbone.blocks.vector_parameters()
        ),
        *(
            (head.biases, ("site", "C"), census)
            for head, census in zip(backbone.heads, stacks.sites, strict=True)
        ),
    )


def _real(value: Array, census: StackCensus | None) -> Array:
    """A stacked compute or vector leaf without its persist pads. Those rows are scanned
    over the stack and so never cut it (`bind_scanned_ci_fn_row`), which makes the slice
    device-local; its transpose writes exact zeros into the pads' cotangents."""
    return value if census is None else strip_stack_pad(value, census)


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
    """Heads are drawn in site order, then stacked by semantic group. Every stack is then
    padded with the placement's trailing all-zero slots."""
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
    groups = tuple(vu_groups(sites).values())
    head_lengths = tuple(len(group.specs) for group in groups)
    match placement:
        case PlacedGlobalCIFn(stacks=stacks):
            assert stacks.real == GlobalCIStacks.unpadded(arch.n_blocks, head_lengths), (
                f"placement resolved {stacks} for {arch.n_blocks} blocks and heads {head_lengths}"
            )
        case UnplacedGlobalCIFn():
            stacks = GlobalCIStacks.unpadded(arch.n_blocks, head_lengths)
    heads = tuple(
        CIFnHeadStack(
            weights=pad_stack(
                jnp.stack([parameters.out_ws[site_index[spec.name]] for spec in group.specs]),
                census,
            ),
            biases=pad_stack(
                jnp.stack([parameters.out_bs[site_index[spec.name]] for spec in group.specs]),
                census,
            ),
            sites=tuple(spec.name for spec in group.specs),
        )
        for group, census in zip(groups, stacks.sites, strict=True)
    )
    return GlobalTransformerBackbone(
        in_proj_w=parameters.in_proj_w,
        in_proj_b=parameters.in_proj_b,
        blocks=jax.tree.map(
            lambda *leaves: pad_stack(jnp.stack(leaves), stacks.depth), *parameters.blocks
        ),
        heads=heads,
        inv_freq=inv_freq,
        input_taps=arch.input_taps,
        output_names=output_names,
        placement=placement,
        stacks=stacks,
        eps=CI_FN_RMS_EPS,
        final_norm=arch.final_norm,
    )
