"""Mask-info types used by ComponentModel and the decomposed sites.

Lives in its own file (rather than in `components.py` or `decomposed_module.py`)
so the dependency graph stays acyclic: both modules can import from here.
"""

from dataclasses import dataclass
from typing import Literal

from jaxtyping import Bool, Float
from torch import Tensor

WeightDeltaAndMask = tuple[Float[Tensor, "d_out d_in"], Float[Tensor, "..."]]


@dataclass
class ComponentsMaskInfo:
    """Specifies the mask information that will be applied at a decomposition site."""

    component_mask: Float[Tensor, "... C"]
    """When the decomposed path is used, this multiplies the component activations."""

    routing_mask: Bool[Tensor, "..."] | Literal["all"] = "all"
    """Which (batch,) or (batch, seq_len) positions are routed through the decomposed
    path vs. the wrapped target module. If "all", every position uses the decomposed path."""

    weight_delta_and_mask: WeightDeltaAndMask | None = None


RoutingMasks = dict[str, Bool[Tensor, "..."]] | Literal["all"]


def make_mask_infos(
    component_masks: dict[str, Float[Tensor, "... C"]],
    routing_masks: RoutingMasks = "all",
    weight_deltas_and_masks: dict[str, WeightDeltaAndMask] | None = None,
) -> dict[str, ComponentsMaskInfo]:
    """Build a ComponentsMaskInfo dict from per-site masks, routing masks, and weight deltas.

    All input dicts must share the same set of keys.
    """
    if isinstance(routing_masks, dict):
        assert set(routing_masks) == set(component_masks)

    if weight_deltas_and_masks is not None:
        assert set(weight_deltas_and_masks) == set(component_masks)

    result: dict[str, ComponentsMaskInfo] = {}
    for name in component_masks:
        routing_mask = routing_masks[name] if isinstance(routing_masks, dict) else "all"

        weight_delta_and_mask = (
            weight_deltas_and_masks[name] if weight_deltas_and_masks is not None else None
        )

        result[name] = ComponentsMaskInfo(
            component_mask=component_masks[name],
            routing_mask=routing_mask,
            weight_delta_and_mask=weight_delta_and_mask,
        )

    return result
