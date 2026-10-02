"""The implementation-independent definition of a causal-importance function."""

from typing import Protocol, runtime_checkable

from jaxtyping import PRNGKeyArray

from param_decomp.core.ci_fn.interface import CIFn
from param_decomp.core.components import SiteSpec
from param_decomp.core.flops.types import ForwardBackwardFlops, ParameterCensus
from param_decomp.core.model import CaptureKeys, PositionAxis, Positioned, Positionless
from param_decomp.core.placement import PlacementRules


class CIFnArchitectureFootprint(Protocol):
    """What a cost model reads from a CI architecture — its capture request, parameter
    count, and useful FLOPs — independent of what it conditions on."""

    @property
    def capture_keys(self) -> CaptureKeys: ...

    def parameter_census(self, sites: tuple[SiteSpec, ...]) -> ParameterCensus: ...

    def useful_flops(
        self,
        sites: tuple[SiteSpec, ...],
        batch_size: int,
        positions: PositionAxis,
        *,
        n_selected_blocks_per_token: int | None,
    ) -> ForwardBackwardFlops: ...


# beartype checks parameters annotated with this protocol by `isinstance`.
@runtime_checkable
class CIFnArchitecture[Conditioning](CIFnArchitectureFootprint, Protocol):
    """Construct placed CI parameters and describe their computational cost.

    Each implementation resolves its own placement from the run's rules (or runs
    unplaced without them) and binds it into the parameters it constructs."""

    def initialize(
        self, sites: tuple[SiteSpec, ...], rules: PlacementRules | None, key: PRNGKeyArray
    ) -> CIFn[Conditioning]: ...


def sequence_length_for_flops(
    batch_size: int, positions: PositionAxis, *, has_position_axis: bool
) -> int:
    if batch_size <= 0:
        raise ValueError("CI FLOPs require a positive batch size")
    match positions:
        case Positionless():
            if has_position_axis:
                raise ValueError("CI and target must agree on the position axis")
            return 1
        case Positioned(n_positions=sequence_length):
            if sequence_length <= 0:
                raise ValueError("CI FLOPs require a positive sequence length")
            if not has_position_axis:
                raise ValueError("CI and target must agree on the position axis")
            return sequence_length
