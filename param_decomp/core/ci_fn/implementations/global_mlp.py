"""One pointwise MLP over named taps, with outputs partitioned across sites."""

from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
from jax.sharding import Mesh
from jaxtyping import Array, PRNGKeyArray

from param_decomp.core.ci_fn.architecture import sequence_length_for_flops
from param_decomp.core.ci_fn.implementations.mlp import (
    SiteMLP,
    init_mlp_stack,
    mlp_flops,
    mlp_parameters,
)
from param_decomp.core.ci_fn.interface import (
    CI,
    CIFnMatrix,
    TapSpec,
)
from param_decomp.core.components import SiteSpec
from param_decomp.core.flops.types import ForwardBackwardFlops, ParameterCensus
from param_decomp.core.model import CaptureKeys, ComponentActivations, PositionAxis
from param_decomp.core.placement import PlacementRules
from param_decomp.core.precision import COMPUTE_DT, cast_floating
from param_decomp.sequence import SequenceLayout


@dataclass(frozen=True)
class GlobalMLPCIFnArch:
    """Hidden widths of the single global MLP shared across ALL sites, plus the input
    taps it concatenates. The taps are DECOUPLED from the output sites: several sites may
    read one physical tap (an LM block's q/k/v share its attention input), so the taps
    are unique keys with explicit widths, never a per-site alignment
    (`LayerwiseMLPCIFnArch` keeps that alignment — there it is real)."""

    hidden_dims: tuple[int, ...]
    has_position_axis: bool
    input_taps: tuple[TapSpec, ...]

    @property
    def capture_keys(self) -> CaptureKeys:
        return frozenset(tap.key for tap in self.input_taps)

    def initialize(
        self, sites: tuple[SiteSpec, ...], rules: PlacementRules | None, key: PRNGKeyArray
    ) -> "GlobalMLPCIFn":
        del rules  # MLP CI reads no table rows
        return init_global_mlp_ci_fn(self, sites, key)

    def parameter_census(self, sites: tuple[SiteSpec, ...]) -> ParameterCensus:
        return global_mlp_ci_fn_parameters(self, sites)

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
        return global_mlp_ci_fn_flops(self, sites, batch_size * length)


def _global_mlp_dimensions(arch: GlobalMLPCIFnArch, sites: tuple[SiteSpec, ...]) -> tuple[int, ...]:
    return (
        sum(tap.width for tap in arch.input_taps),
        *arch.hidden_dims,
        sum(site.C for site in sites),
    )


def global_mlp_ci_fn_flops(
    arch: GlobalMLPCIFnArch, sites: tuple[SiteSpec, ...], n_tokens: int
) -> ForwardBackwardFlops:
    return mlp_flops(_global_mlp_dimensions(arch, sites), n_tokens)


def global_mlp_ci_fn_parameters(
    arch: GlobalMLPCIFnArch, sites: tuple[SiteSpec, ...]
) -> ParameterCensus:
    return mlp_parameters(_global_mlp_dimensions(arch, sites))


class GlobalMLPCIFn(eqx.Module):
    """ONE shared MLP over all sites behind the `CIFn` protocol. The taps are
    concatenated in `input_taps` order into `[*leading, Σ width]`, mapped to `[*leading,
    Σ C]`, and split back per output site by `c_sizes` in `output_names` order — so every
    site's preactivations depend on every tap."""

    mlp: SiteMLP
    input_taps: tuple[TapSpec, ...] = eqx.field(static=True)
    output_names: tuple[str, ...] = eqx.field(static=True)
    c_sizes: tuple[int, ...] = eqx.field(static=True)
    has_position_axis: bool = eqx.field(static=True)

    @property
    def capture_keys(self) -> CaptureKeys:
        return frozenset(tap.key for tap in self.input_taps)

    def placement_for_log(self) -> str:
        return "ci_fn placement: MLP matrices shard their outputs over the batch owners"

    def matrix_parameters(self) -> tuple[CIFnMatrix, ...]:
        raise NotImplementedError("MLP CI functions do not support Muon")

    def prepare(self) -> "GlobalMLPCIFn":
        return cast_floating(self, COMPUTE_DT)

    def shardings(self, mesh: Mesh) -> "GlobalMLPCIFn":
        return eqx.tree_at(lambda f: f.mlp, self, self.mlp.shardings(mesh))

    def site_preactivations(self, taps: dict[str, Array]) -> dict[str, Array]:
        assert set(taps) == {tap.key for tap in self.input_taps}, (
            f"tap keys {sorted(taps)} != CI fn inputs {sorted(t.key for t in self.input_taps)}"
        )
        for tap in self.input_taps:
            assert taps[tap.key].shape[-1] == tap.width, (
                f"tap {tap.key} width {taps[tap.key].shape[-1]} != expected {tap.width}"
            )
        concatenated = jnp.concatenate([taps[tap.key] for tap in self.input_taps], axis=-1)
        preactivations = self.mlp(concatenated)
        offsets = [0]
        for c in self.c_sizes:
            offsets.append(offsets[-1] + c)
        return {
            name: preactivations[..., offsets[i] : offsets[i + 1]]
            for i, name in enumerate(self.output_names)
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


def init_global_mlp_ci_fn(
    arch: GlobalMLPCIFnArch,
    sites: tuple[SiteSpec, ...],
    key: PRNGKeyArray,
) -> GlobalMLPCIFn:
    """Global MLP init: one stack `Σ tap width -> hidden_dims... -> Σ C`, same Kaiming
    scheme as the per-site MLP."""
    assert arch.hidden_dims, "global MLP CI fn needs at least one hidden layer"
    tap_keys = tuple(tap.key for tap in arch.input_taps)
    assert tap_keys and len(set(tap_keys)) == len(tap_keys), tap_keys
    c_sizes = tuple(s.C for s in sites)
    dims = _global_mlp_dimensions(arch, sites)
    return GlobalMLPCIFn(
        mlp=init_mlp_stack(dims, key),
        input_taps=arch.input_taps,
        output_names=tuple(s.name for s in sites),
        c_sizes=c_sizes,
        has_position_axis=arch.has_position_axis,
    )
