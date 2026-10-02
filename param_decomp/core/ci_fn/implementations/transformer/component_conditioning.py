"""CI conditioned on the live components' activations of each site's clean input.
`ConditionedCIFnArch` wraps any backbone architecture and builds `ComponentConditioned`
around that architecture's backbone.

With `h = clean_input @ V` and `σ = symmetric_leaky_hard_sigmoid`, each site's
preactivation gains, per component,

    s₊ * σ(a₊ * h + b₊) + s₋ * σ(a₋ * (-h) + b₋) + bias

a positive and a negative arm, and a free bias. The readout vectors are CI parameters.
V stays owned by the target: evaluation reads `h` from the target's prepared
components, so V's CI gradient joins its reconstruction gradient before the pullback
onto the component masters.
"""

import math
from dataclasses import dataclass
from fractions import Fraction

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import AbstractMesh, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Float32, PRNGKeyArray

from param_decomp.core.axes import Axes
from param_decomp.core.ci_fn.architecture import sequence_length_for_flops
from param_decomp.core.ci_fn.implementations.transformer.backbone import (
    BackboneCIFn,
    CIFnBackbone,
    CIFnBackboneArchitecture,
)
from param_decomp.core.ci_fn.interface import CIFnMatrix, SiteDict
from param_decomp.core.ci_fn.squashing import symmetric_leaky_hard_sigmoid
from param_decomp.core.components import (
    BlockedFactorization,
    DenseFactorization,
    SiteSpec,
    require_full_emission,
)
from param_decomp.core.configs import PlacementPresetName
from param_decomp.core.flops.types import ForwardBackwardFlops, ParameterCensus
from param_decomp.core.model import CaptureKeys, ComponentActivations, PositionAxis
from param_decomp.core.placement import (
    COMPONENT_PARTITION,
    PlacedRule,
    PlacementRules,
    Rule,
    bind_ci_fn_row,
    placed_rule_for_log,
)
from param_decomp.core.precision import COMPUTE_DT, cast_floating
from param_decomp.sequence import SequenceLayout


@dataclass(frozen=True)
class SiteInput:
    """The target capture holding a site's clean input."""

    site: str
    capture_key: str


READOUT_AXES: Axes = ("C",)


def preset_readout_row(name: PlacementPresetName) -> Rule:
    """The readouts scale each site's component activations elementwise, so under every
    preset they split `C` as the component activations do."""
    match name:
        case (
            "owner"
            | "zero1"
            | "zero1-replicated-resident"
            | "owner-replicated-resident"
            | "ddp"
            | "zero1-replicated-resident-moe"
            | "owner-replicated-resident-moe"
            | "zero1-replicated-resident-moe-replicated-ns"
        ):
            return COMPONENT_PARTITION


def bind_readout_row(rule: Rule, mesh: Mesh | AbstractMesh) -> PlacedRule:
    return bind_ci_fn_row("ci_fn/readout_vectors", rule, mesh, frozenset(READOUT_AXES))


@dataclass(frozen=True)
class PlacedConditioning:
    """The row the readout vectors rest at."""

    readout_vectors: PlacedRule


@dataclass(frozen=True)
class UnplacedConditioning:
    """Off-mesh execution: the readout vectors are plain arrays."""


ConditioningPlacement = PlacedConditioning | UnplacedConditioning


@dataclass(frozen=True)
class InputScaleCalibration:
    """The data the arms' input scales are calibrated on: a whole batch of at least
    `min_n_tokens` tokens."""

    min_n_tokens: int

    def __post_init__(self) -> None:
        assert self.min_n_tokens > 0, self


@dataclass(frozen=True)
class ConditionedCIFnArch[Inner: CIFnBackboneArchitecture]:
    """`inner`'s backbone with every decomposed site's CI also conditioned on its
    component activations `x @ V` of its clean input `x`. Every site must be dense and
    named exactly once. `output_scale_init` is every arm's initial output scale;
    `calibration` is the data a composition root calibrates the input scales on."""

    inner: Inner
    site_inputs: tuple[SiteInput, ...]
    output_scale_init: float
    calibration: InputScaleCalibration

    @property
    def capture_keys(self) -> CaptureKeys:
        return self.inner.capture_keys | {value.capture_key for value in self.site_inputs}

    def initialize(
        self, sites: tuple[SiteSpec, ...], rules: PlacementRules | None, key: PRNGKeyArray
    ) -> BackboneCIFn:
        return BackboneCIFn(self.initialize_backbone(sites, rules, key))

    def initialize_backbone(
        self, sites: tuple[SiteSpec, ...], rules: PlacementRules | None, key: PRNGKeyArray
    ) -> "ComponentConditioned":
        """Every arm starts with unit input scale, zero input bias, and `output_scale_init`,
        and the free bias at zero. The readouts draw no randomness, so the inner backbone's
        initialization is unchanged; `ComponentConditioned.with_calibrated_input_scales`
        rescales the arms' inputs to data."""
        capture_key_by_site = self._capture_key_by_site(sites)
        match rules:
            case None:
                placement: ConditioningPlacement = UnplacedConditioning()
            case PlacementRules():
                row = bind_readout_row(preset_readout_row(rules.ci_fn), rules.mesh)
                for site in sites:
                    row.validate_shape(READOUT_AXES, (site.C,))
                placement = PlacedConditioning(row)
        return ComponentConditioned(
            backbone=self.inner.initialize_backbone(sites, rules, key),
            readouts={
                site.name: ComponentActivationReadout(
                    positive=self._uncalibrated_arm(site.C),
                    negative=self._uncalibrated_arm(site.C),
                    bias=jnp.zeros((site.C,), jnp.float32),
                    capture_key=capture_key_by_site[site.name],
                )
                for site in sites
            },
            placement=placement,
        )

    def _uncalibrated_arm(self, C: int) -> "ClampedAffineArm":
        """The unit input scale holds the place of the calibrated one."""
        return ClampedAffineArm(
            input_scale=jnp.ones((C,), jnp.float32),
            input_bias=jnp.zeros((C,), jnp.float32),
            output_scale=jnp.full((C,), self.output_scale_init, jnp.float32),
        )

    def parameter_census(self, sites: tuple[SiteSpec, ...]) -> ParameterCensus:
        """The inner census plus seven readout vectors per site (three per arm and the
        bias); V is a component parameter."""
        inner = self.inner.parameter_census(sites)
        n_readout_parameters = 7 * sum(site.C for site in sites)
        return ParameterCensus(inner.matrices, inner.n_vector_parameters + n_readout_parameters)

    def useful_flops(
        self,
        sites: tuple[SiteSpec, ...],
        batch_size: int,
        positions: PositionAxis,
        *,
        n_selected_blocks_per_token: int | None,
    ) -> ForwardBackwardFlops:
        """The inner FLOPs plus each site's `x @ V`, whose backward reaches V only: the
        clean input is a target capture, not a trainable quantity. The readout itself is
        elementwise, which useful FLOPs do not count."""
        inner = self.inner.useful_flops(
            sites, batch_size, positions, n_selected_blocks_per_token=n_selected_blocks_per_token
        )
        length = sequence_length_for_flops(batch_size, positions, has_position_axis=True)
        projection = sum(2 * batch_size * length * site.d_in * site.C for site in sites)
        return ForwardBackwardFlops(inner.forward + projection, inner.backward + projection)

    def _capture_key_by_site(self, sites: tuple[SiteSpec, ...]) -> dict[str, str]:
        capture_key_by_site = {value.site: value.capture_key for value in self.site_inputs}
        assert len(capture_key_by_site) == len(self.site_inputs), self.site_inputs
        assert capture_key_by_site.keys() == {site.name for site in sites}, (
            f"component conditioning inputs {sorted(capture_key_by_site)} must name exactly "
            f"the decomposed sites {sorted(site.name for site in sites)}"
        )
        for site in sites:
            match site.factorization:
                case DenseFactorization():
                    pass
                case BlockedFactorization():
                    raise ValueError(f"component conditioning requires dense sites: {site.name}")
        return capture_key_by_site


class ClampedAffineArm(eqx.Module):
    """A per-component affine map of the arm's input, clamped by the symmetric leaky
    hard sigmoid and scaled."""

    input_scale: Float[Array, " C"]
    input_bias: Float[Array, " C"]
    output_scale: Float[Array, " C"]

    def __call__(self, arm_input: Float[Array, "*leading C"]) -> Float[Array, "*leading C"]:
        return self.output_scale * symmetric_leaky_hard_sigmoid(
            self.input_scale * arm_input + self.input_bias
        )


class ComponentActivationReadout(eqx.Module):
    """One site's readout of its component activations `h`, reading the site's clean
    input at `capture_key`: the positive arm reads `h` and the negative arm `-h`."""

    positive: ClampedAffineArm
    negative: ClampedAffineArm
    bias: Float[Array, " C"]
    capture_key: str = eqx.field(static=True)

    def __call__(
        self, component_activation: Float[Array, "*leading C"]
    ) -> Float[Array, "*leading C"]:
        return (
            self.positive(component_activation) + self.negative(-component_activation) + self.bias
        )


CALIBRATION_QUANTILE = Fraction(999, 1000)
"""The quantile of `|h|` each arm's input scale maps to 1, exact so that its order-statistic
ranks are."""

CALIBRATION_CHUNK_TOKENS = 2**15
"""Tokens per chunk of the calibration's running top-k: a memory and speed choice alone, since
the quantile is exact at every chunk size. Each chunk is replicated whole in turn."""


class ComponentConditioned(eqx.Module):
    """The backbone's parameters plus every site's readout vectors. V is the target's,
    read at evaluation from its prepared components."""

    backbone: CIFnBackbone
    readouts: dict[str, ComponentActivationReadout]
    placement: ConditioningPlacement = eqx.field(static=True)

    def __check_init__(self) -> None:
        assert self.readouts.keys() == set(self.backbone.output_names), (
            sorted(self.readouts),
            self.backbone.output_names,
        )

    @property
    def capture_keys(self) -> CaptureKeys:
        readout_keys = {readout.capture_key for readout in self.readouts.values()}
        return self.backbone.capture_keys | readout_keys

    @property
    def output_names(self) -> tuple[str, ...]:
        return self.backbone.output_names

    def placement_for_log(self) -> str:
        placement = self.placement
        match placement:
            case UnplacedConditioning():
                return self.backbone.placement_for_log()
            case PlacedConditioning(readout_vectors=row):
                return f"{self.backbone.placement_for_log()}\n{placed_rule_for_log(row)}"

    def matrix_parameters(self) -> tuple[CIFnMatrix, ...]:
        return self.backbone.matrix_parameters()

    def shardings(self, mesh: Mesh) -> "ComponentConditioned":
        placement = self.placement
        match placement:
            case UnplacedConditioning():
                raise AssertionError("an unplaced CI fn has no mesh to shard its parameters over")
            case PlacedConditioning(readout_vectors=row):

                def readout_sharding(value: Array) -> NamedSharding:
                    row.validate_shape(READOUT_AXES, value.shape)
                    return row.sharding_for(READOUT_AXES)

                return eqx.tree_at(
                    lambda conditioned: (conditioned.backbone, conditioned.readouts),
                    self,
                    (self.backbone.shardings(mesh), jax.tree.map(readout_sharding, self.readouts)),
                )

    def prepare(self) -> "ComponentConditioned":
        return ComponentConditioned(
            backbone=self.backbone.prepare(),
            readouts=cast_floating(self.readouts, COMPUTE_DT),
            placement=self.placement,
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
        preactivations = self.backbone.preactivations(
            taps, conditioning, components, sequence=sequence, remat=remat
        )
        # The readout's only dot is `x @ V`: saved, the backward recomputes just the
        # elementwise arms; under remat it recomputes `x @ V` too, from the clean input
        # the backward holds anyway.
        policy = (
            jax.checkpoint_policies.nothing_saveable
            if remat
            else jax.checkpoint_policies.dots_saveable
        )
        return {
            site: require_full_emission(value)
            + eqx.filter_checkpoint(_conditioned_readout, policy=policy)(
                site, self.readouts[site], components, taps[self.readouts[site].capture_key]
            )
            for site, value in preactivations.items()
        }

    def with_calibrated_input_scales(
        self, taps: dict[str, Array], components: ComponentActivations
    ) -> "ComponentConditioned":
        """Every arm's input scale set to `1 / q`, where `q` is each component's exact
        `CALIBRATION_QUANTILE` of `|h|` over every token of the taps, interpolated linearly
        as `np.quantile` does: each arm then maps its side's `[0, q]` onto `[0, 1]`. Every
        token counts, so the taps must hold no padding, and a whole number of
        `CALIBRATION_CHUNK_TOKENS`-token chunks. `h` is read as evaluation reads it, from the
        compute-dtype taps through `components`."""
        compute_taps = cast_floating(taps, COMPUTE_DT)

        def calibrated(
            site: str, readout: ComponentActivationReadout
        ) -> ComponentActivationReadout:
            activation = components.component_activations(site, compute_taps[readout.capture_key])
            q = _magnitude_quantile(activation, self.placement)
            q = eqx.error_if(
                q,
                ~jnp.all(jnp.isfinite(q) & (q > 0)),
                f"component activations at {site} have a zero or non-finite |h| quantile",
            )
            return eqx.tree_at(
                lambda r: (r.positive.input_scale, r.negative.input_scale),
                readout,
                (1 / q, 1 / q),
            )

        return ComponentConditioned(
            backbone=self.backbone,
            readouts={site: calibrated(site, readout) for site, readout in self.readouts.items()},
            placement=self.placement,
        )


def _conditioned_readout(
    site: str,
    readout: ComponentActivationReadout,
    components: ComponentActivations,
    clean_input: Array,
) -> Array:
    return readout(components.component_activations(site, clean_input))


def _magnitude_quantile(
    activation: Float[Array, "*leading C"], placement: ConditioningPlacement
) -> Float32[Array, " C"]:
    """Each component's exact `CALIBRATION_QUANTILE` of `|activation|` over every token,
    interpolated linearly between the order statistics either side of rank
    `(n_tokens - 1) * CALIBRATION_QUANTILE`, as `np.quantile` does. A running top-k over
    contiguous chunks keeps exactly the `k` largest magnitudes, `k` just deep enough to reach
    the lower of those order statistics. Placed, each chunk is replicated whole in turn,
    every component included."""
    *leading, C = activation.shape
    n_tokens = math.prod(leading)
    assert n_tokens % CALIBRATION_CHUNK_TOKENS == 0, (
        f"calibration walks whole {CALIBRATION_CHUNK_TOKENS}-token chunks, "
        f"which do not tile {n_tokens} tokens"
    )
    rank = (n_tokens - 1) * CALIBRATION_QUANTILE
    lower_rank, upper_rank = math.floor(rank), math.ceil(rank)
    k = n_tokens - lower_rank
    magnitudes = jnp.abs(jnp.asarray(activation, jnp.float32)).reshape(n_tokens, C)
    shape = (n_tokens // CALIBRATION_CHUNK_TOKENS, CALIBRATION_CHUNK_TOKENS, C)
    match placement:
        case UnplacedConditioning():
            chunks = magnitudes.reshape(shape)
            gathered = lambda chunk: chunk
        case PlacedConditioning():
            token_spec, component_spec = jax.typeof(magnitudes).sharding.spec
            chunks = jnp.reshape(
                magnitudes, shape, out_sharding=P(None, token_spec, component_spec)
            )
            gathered = lambda chunk: jax.reshard(chunk, P())

    def with_chunk_merged(
        largest: Float32[Array, "C k"], chunk: Float32[Array, "chunk_tokens C"]
    ) -> tuple[Float32[Array, "C k"], None]:
        candidates = jnp.concatenate([largest, gathered(chunk).T], axis=1)
        return jax.lax.top_k(candidates, k)[0], None

    # Every magnitude outranks -inf, and `k <= n_tokens`, so no -inf survives the walk.
    largest, _ = jax.lax.scan(with_chunk_merged, jnp.full((C, k), -jnp.inf), chunks)
    lower = largest[:, n_tokens - 1 - lower_rank]
    upper = largest[:, n_tokens - 1 - upper_rank]
    return lower + (upper - lower) * float(rank - lower_rank)
