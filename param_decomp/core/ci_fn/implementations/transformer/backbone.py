"""The network inside a transformer CI fn, and the CI fn that squashes its per-site
outputs. Every transformer backbone reads a position axis."""

from typing import Protocol, Self

import equinox as eqx
from jax.sharding import Mesh
from jaxtyping import Array, PRNGKeyArray

from param_decomp.core.ci_fn.architecture import CIFnArchitectureFootprint
from param_decomp.core.ci_fn.interface import CI, CIFnMatrix, SiteDict
from param_decomp.core.components import SiteSpec
from param_decomp.core.model import CaptureKeys, ComponentActivations
from param_decomp.core.placement import PlacementRules
from param_decomp.core.precision import COMPUTE_DT, cast_floating
from param_decomp.sequence import SequenceLayout


class CIFnBackbone(Protocol):
    """A CI network's parameters, bound to the placement they rest at, and its per-site
    preactivations: the outputs before the leaky hard sigmoids squash them into lower and
    upper CI."""

    @property
    def capture_keys(self) -> CaptureKeys: ...

    @property
    def output_names(self) -> tuple[str, ...]: ...

    def placement_for_log(self) -> str: ...

    def matrix_parameters(self) -> tuple[CIFnMatrix, ...]: ...

    def shardings(self, mesh: Mesh) -> Self: ...

    def prepare(self) -> Self:
        """Compute-dtype parameters in their compute layout."""
        ...

    def preactivations(
        self,
        taps: dict[str, Array],
        conditioning: object,
        components: ComponentActivations,
        *,
        sequence: SequenceLayout | None,
        remat: bool,
    ) -> SiteDict: ...


class CIFnBackboneArchitecture(CIFnArchitectureFootprint, Protocol):
    """An architecture whose CI fn is a backbone under `BackboneCIFn`. It builds the
    backbone alone, so a wrapping architecture can compose around it."""

    def initialize_backbone(
        self, sites: tuple[SiteSpec, ...], rules: PlacementRules | None, key: PRNGKeyArray
    ) -> CIFnBackbone: ...


class BackboneCIFn(eqx.Module):
    """A backbone as a `CIFn`: taps cast to the compute dtype, preactivations squashed
    into CI."""

    backbone: CIFnBackbone

    @property
    def capture_keys(self) -> CaptureKeys:
        return self.backbone.capture_keys

    @property
    def output_names(self) -> tuple[str, ...]:
        return self.backbone.output_names

    @property
    def has_position_axis(self) -> bool:
        return True

    def placement_for_log(self) -> str:
        return self.backbone.placement_for_log()

    def matrix_parameters(self) -> tuple[CIFnMatrix, ...]:
        return self.backbone.matrix_parameters()

    def shardings(self, mesh: Mesh) -> "BackboneCIFn":
        return BackboneCIFn(self.backbone.shardings(mesh))

    def prepare(self) -> "BackboneCIFn":
        return BackboneCIFn(self.backbone.prepare())

    def __call__(
        self,
        taps: dict[str, Array],
        conditioning: object,
        components: ComponentActivations,
        *,
        sequence: SequenceLayout | None,
        remat: bool,
    ) -> CI:
        preactivations = self.backbone.preactivations(
            cast_floating(taps, COMPUTE_DT),
            conditioning,
            components,
            sequence=sequence,
            remat=remat,
        )
        return CI.from_preactivations(preactivations)
