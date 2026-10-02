"""CI values, named input taps, and the common callable interface."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, Self, runtime_checkable

import jax
from jax.sharding import Mesh, NamedSharding
from jaxtyping import Array

from param_decomp.core.ci_fn.squashing import lower_leaky_hard_sigmoid, upper_leaky_hard_sigmoid
from param_decomp.core.components import SiteCI, map_site_ci
from param_decomp.core.model import CaptureKeys, ComponentActivations
from param_decomp.core.muon_stacked import StageInPlace
from param_decomp.core.pytree import ShardingTree
from param_decomp.sequence import SequenceLayout

SiteDict = dict[str, SiteCI]
"""Per-output-site CI value keyed by OUTPUT site name: full `[*leading, C]` arrays for
dense sites, `SelectedCI` bundles for block-factored, selected-emitting sites
(`components.SiteCI` / `components.SelectedCI`)."""


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CI:
    """The CI fn output: raw preactivations + both squashings, all keyed by output site. `preactivations`
    is kept (a consumed view — the histograms / heatmaps plot pre-squash). The squashing
    lives only in `from_preactivations`, so no impl re-triplicates it."""

    preactivations: SiteDict
    lower: SiteDict
    upper: SiteDict

    @staticmethod
    def from_preactivations(preactivations: "Mapping[str, SiteCI]") -> "CI":
        return CI(
            preactivations=dict(preactivations),
            lower={k: map_site_ci(lower_leaky_hard_sigmoid, v) for k, v in preactivations.items()},
            upper={k: map_site_ci(upper_leaky_hard_sigmoid, v) for k, v in preactivations.items()},
        )


class CIFnMetadata(Protocol):
    """The named physical captures a CI fn reads and the decomposition sites it scores."""

    @property
    def capture_keys(self) -> CaptureKeys: ...

    @property
    def output_names(self) -> tuple[str, ...]: ...

    @property
    def has_position_axis(self) -> bool: ...


@dataclass(frozen=True)
class CIFnMatrix:
    """An independently updated matrix leaf, preceded by any independent matrix axes, and
    where stacked Muon stages it: a sharding that splits the independent matrices over its
    leading entry with each matrix whole per device, or in place.

    The value refers to the existing parameter leaf; this description owns no state.
    Biases and normalization scales are not matrices, regardless of their array rank.
    """

    value: Array
    staging: NamedSharding | StageInPlace


@runtime_checkable
class CIFn[Conditioning](CIFnMetadata, Protocol):
    """CI parameters, bound at construction to the placement they rest at, and CI
    evaluation over them.

    Parameters remain runtime pytree arguments; placement is static implementation
    state. `prepare` casts the parameters to the compute dtype and relayouts them into
    their compute layout; the train step evaluates only prepared parameters and pulls
    the CI gradient back through the preparation.

    Evaluation casts its taps to the compute dtype, runs the implementation's forward,
    and squashes the preactivations into `CI`. It also receives the target's prepared
    components, which a CI conditioned on component activations reads and every other
    CI ignores. Physical capture requests are exposed by ``capture_keys``; input
    preparation validates the physical capture request before invoking this operation.
    Arrays remain ordinary pytree leaves: autodiff also uses this structure for
    cotangents, whose dtypes need not equal the forward compute dtype.
    """

    def matrix_parameters(self) -> tuple[CIFnMatrix, ...]:
        """Enumerate independent matrices; other parameter leaves are elementwise."""
        ...

    def shardings(self, mesh: Mesh) -> ShardingTree: ...

    def placement_for_log(self) -> str:
        """The resolved placement as the startup audit prints it."""
        ...

    def prepare(self) -> Self: ...

    def __call__(
        self,
        taps: dict[str, Array],
        conditioning: Conditioning,
        components: ComponentActivations,
        *,
        sequence: SequenceLayout | None,
        remat: bool,
    ) -> CI: ...


@dataclass(frozen=True)
class TapSpec:
    """One input tap: its capture key and feature width. The key is opaque to core (the
    lab authors it, the target resolves it); the width rides alongside so the consumer
    can size and assert its input without deriving it from a site."""

    key: str
    width: int
