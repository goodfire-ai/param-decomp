"""An independent pointwise MLP for each decomposition site."""

from dataclasses import dataclass

import equinox as eqx
import jax
from jax.sharding import Mesh
from jaxtyping import Array, PRNGKeyArray

from param_decomp.core.ci_fn.architecture import sequence_length_for_flops
from param_decomp.core.ci_fn.implementations.mlp import (
    SiteMLP,
    init_mlp_stack,
    mlp_flops,
    mlp_parameters,
)
from param_decomp.core.ci_fn.interface import CI, CIFnMatrix
from param_decomp.core.components import SiteSpec
from param_decomp.core.flops.types import ForwardBackwardFlops, ParameterCensus
from param_decomp.core.model import CaptureKeys, ComponentActivations, PositionAxis
from param_decomp.core.placement import PlacementRules
from param_decomp.core.precision import COMPUTE_DT, cast_floating
from param_decomp.sequence import SequenceLayout


# The MLP arches bind their config to a target at the composition root. Their input taps
# (`input_names` / `input_taps`) are therefore resolved exactly once, like the chunkwise
# architecture's authored tap union, and every downstream consumer reads the same
# authoritative field.
@dataclass(frozen=True)
class LayerwiseMLPCIFnArch:
    """Hidden widths shared by every per-site MLP.

    `has_position_axis` is the TARGET's shape, not a property of the MLP: the stack is
    pointwise over every leading axis, so the same weights serve `[batch, d]` and
    `[batch, position, d]` alike. It is declared here so the CI fn and the model can be
    checked to agree (`core.run_state.init_decomposition`)."""

    hidden_dims: tuple[int, ...]
    has_position_axis: bool
    input_names: tuple[str, ...]

    @property
    def capture_keys(self) -> CaptureKeys:
        return frozenset(self.input_names)

    def initialize(
        self, sites: tuple[SiteSpec, ...], rules: PlacementRules | None, key: PRNGKeyArray
    ) -> "LayerwiseMLPCIFn":
        del rules  # MLP CI reads no table rows
        return init_layerwise_mlp_ci_fn(self, sites, key)

    def parameter_census(self, sites: tuple[SiteSpec, ...]) -> ParameterCensus:
        return layerwise_mlp_ci_fn_parameters(self, sites)

    def useful_flops(
        self,
        sites: tuple[SiteSpec, ...],
        batch_size: int,
        positions: PositionAxis,
        *,
        n_selected_blocks_per_token: int | None,
    ) -> ForwardBackwardFlops:
        del n_selected_blocks_per_token
        length = sequence_length_for_flops(
            batch_size, positions, has_position_axis=self.has_position_axis
        )
        return layerwise_mlp_ci_fn_flops(self, sites, batch_size * length)


def layerwise_mlp_ci_fn_flops(
    arch: LayerwiseMLPCIFnArch, sites: tuple[SiteSpec, ...], n_tokens: int
) -> ForwardBackwardFlops:
    costs = tuple(mlp_flops((site.d_in, *arch.hidden_dims, site.C), n_tokens) for site in sites)
    return ForwardBackwardFlops(
        sum(cost.forward for cost in costs), sum(cost.backward for cost in costs)
    )


def layerwise_mlp_ci_fn_parameters(
    arch: LayerwiseMLPCIFnArch, sites: tuple[SiteSpec, ...]
) -> ParameterCensus:
    networks = tuple(mlp_parameters((site.d_in, *arch.hidden_dims, site.C)) for site in sites)
    return ParameterCensus(
        tuple(matrix for network in networks for matrix in network.matrices),
        sum(network.n_vector_parameters for network in networks),
    )


class LayerwiseMLPCIFn(eqx.Module):
    """One MLP per site, with input taps aligned to output sites by position."""

    site_mlps: dict[str, SiteMLP]
    input_names: tuple[str, ...] = eqx.field(static=True)
    output_names: tuple[str, ...] = eqx.field(static=True)
    has_position_axis: bool = eqx.field(static=True)

    @property
    def capture_keys(self) -> CaptureKeys:
        return frozenset(self.input_names)

    def placement_for_log(self) -> str:
        return "ci_fn placement: MLP matrices shard their outputs over the batch owners"

    def matrix_parameters(self) -> tuple[CIFnMatrix, ...]:
        raise NotImplementedError("MLP CI functions do not support Muon")

    def prepare(self) -> "LayerwiseMLPCIFn":
        return cast_floating(self, COMPUTE_DT)

    def shardings(self, mesh: Mesh) -> "LayerwiseMLPCIFn":
        return eqx.tree_at(
            lambda f: f.site_mlps,
            self,
            {name: mlp.shardings(mesh) for name, mlp in self.site_mlps.items()},
        )

    def site_preactivations(self, taps: dict[str, Array]) -> dict[str, Array]:
        assert set(taps) == set(self.input_names), (
            f"tap keys {sorted(taps)} != CI fn inputs {sorted(self.input_names)}"
        )
        return {
            output_name: self.site_mlps[output_name](taps[input_name])
            for input_name, output_name in zip(self.input_names, self.output_names, strict=True)
        }

    def __call__(
        self,
        taps: dict[str, Array],
        conditioning: object,
        components: ComponentActivations,
        *,
        sequence: SequenceLayout | None,
        remat: bool,
    ) -> CI:
        del conditioning, components
        del sequence  # MLPs operate independently at each position
        del remat  # single-shot (no scan to bound) -> remat is a no-op for the MLP CI fns
        taps = cast_floating(taps, COMPUTE_DT)
        return CI.from_preactivations(self.site_preactivations(taps))


def init_layerwise_mlp_ci_fn(
    arch: LayerwiseMLPCIFnArch,
    sites: tuple[SiteSpec, ...],
    key: PRNGKeyArray,
) -> LayerwiseMLPCIFn:
    """Per-site MLP init: each site's MLP maps `d_in -> hidden_dims... -> C`."""
    assert arch.hidden_dims, "MLP CI fn needs at least one hidden layer"
    site_mlps = {
        spec.name: init_mlp_stack(
            (spec.d_in, *arch.hidden_dims, spec.C), jax.random.fold_in(key, site_idx)
        )
        for site_idx, spec in enumerate(sites)
    }
    output_names = tuple(s.name for s in sites)
    assert len(arch.input_names) == len(output_names), (arch.input_names, output_names)
    assert len(set(arch.input_names)) == len(arch.input_names), arch.input_names
    return LayerwiseMLPCIFn(
        site_mlps=site_mlps,
        input_names=arch.input_names,
        output_names=output_names,
        has_position_axis=arch.has_position_axis,
    )
