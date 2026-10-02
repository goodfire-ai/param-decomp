"""Target-relative parameter faithfulness, bound once from frozen model weights."""

import math
from collections.abc import Callable, Mapping

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from param_decomp.core.components import SiteStackIndices, site_stack_indices_for
from param_decomp.core.model import ComponentActivations, PlacedModel

type FaithfulnessLossFn = Callable[[dict[str, Float[Array, "g ..."]]], Float[Array, ""]]


def make_faithfulness_loss(
    site_stack_indices: SiteStackIndices,
    target_sq_norms: dict[str, tuple[float, ...]],
    stack_pads: Mapping[str, int],
) -> FaithfulnessLossFn:
    """Bind validated target scales for mean per-site relative Frobenius error.

    `target_sq_norms` carries one `‖W_s‖²` per stack index of each persistence stack,
    aligned with the `weight_deltas` grouping; the returned loss consumes those stacked
    deltas. `stack_pads` enumerates the persist-layer pads the delta stacks carry beyond
    the real sites (the placement census fact): the faithfulness lane is THE pad's loss
    exit — a pad's delta is exactly zero (zero V·U against a zero frozen pad), so it
    contributes 0 to the sum and the site mean stays over the real sites."""
    stack_names: dict[str, list[str]] = {}
    for name, group, index in site_stack_indices:
        assert index == len(stack_names.setdefault(group, [])), site_stack_indices
        stack_names[group].append(name)
    assert target_sq_norms.keys() == stack_names.keys(), (
        sorted(target_sq_norms),
        sorted(stack_names),
    )
    for group, norms in target_sq_norms.items():
        assert len(norms) == len(stack_names[group]), (group, len(norms), stack_names[group])
    bad = {
        name: value
        for group, norms in target_sq_norms.items()
        for name, value in zip(stack_names[group], norms, strict=True)
        if not math.isfinite(value) or value <= 0.0
    }
    assert not bad, f"faithfulness needs finite positive ‖W_s‖²: {bad}"
    assert stack_pads.keys() <= target_sq_norms.keys(), (sorted(stack_pads), sorted(stack_names))
    n_sites = len(site_stack_indices)
    # Pads divide by 1.0: their zero delta contributes exactly 0 either way, and a
    # real ‖W‖² does not exist for them.
    padded_sq_norms = {
        group: norms + (1.0,) * stack_pads.get(group, 0) for group, norms in target_sq_norms.items()
    }

    def relative_errors(delta_stack: Float[Array, "g ..."], sq_norms: tuple[float, ...]) -> Array:
        """One group's `‖W_s − V_sU_s‖²_F / ‖W_s‖²_F` per stack index, fp32. The squared
        Frobenius norm sums over every axis except the leading stack axis, so a dense
        `[g, d_out, d_in]` stack and a block-factored `[g, expert, d_out, d_in]` stack
        (whose blocks partition the same site matrix) reduce identically."""
        assert delta_stack.shape[0] == len(sq_norms), (delta_stack.shape, len(sq_norms))
        matrix_axes = tuple(range(1, delta_stack.ndim))
        delta_sq_norms = jnp.sum(delta_stack.astype(jnp.float32) ** 2, axis=matrix_axes)
        return delta_sq_norms / jnp.asarray(sq_norms, jnp.float32)

    def faithfulness_loss(weight_deltas: dict[str, Float[Array, "g ..."]]) -> Float[Array, ""]:
        """The mean over REAL sites of each site's relative error."""
        total = sum(
            (
                jnp.sum(relative_errors(weight_deltas[group], sq_norms))
                for group, sq_norms in padded_sq_norms.items()
            ),
            start=jnp.zeros((), jnp.float32),
        )
        return total / n_sites

    return faithfulness_loss


def faithfulness_loss_for[
    TargetIn,
    Out,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    placed: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
) -> FaithfulnessLossFn:
    """Bind each frozen target squared norm into its relative-error loss, with the
    bundle's placement census supplying the persist-stack pads its deltas carry
    (unplaced execution carries none)."""
    target_sq_norms = {
        group: tuple(float(value) for value in values)
        for group, values in jax.device_get(placed.model.target_weight_sq_norms()).items()
    }
    stack_pads = (
        {}
        if placed.placement is None
        else {
            group: entry.stack_pad
            for group, entry in placed.placement.components.group_census.items()
        }
    )
    return make_faithfulness_loss(
        site_stack_indices_for(placed.model.sites), target_sq_norms, stack_pads
    )
