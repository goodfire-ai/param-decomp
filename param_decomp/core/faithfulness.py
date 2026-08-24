"""Target-relative parameter faithfulness, bound once from frozen model weights."""

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from param_decomp.core.components import SiteSlots, site_slots_for
from param_decomp.core.model import DecomposedModel

type FaithfulnessLossFn = Callable[[dict[str, Float[Array, "g _ _"]]], Float[Array, ""]]


def make_faithfulness_loss(
    site_slots: SiteSlots, target_sq_norms: dict[str, tuple[float, ...]]
) -> FaithfulnessLossFn:
    """Bind validated target scales for mean per-site relative Frobenius error (SPEC S17).

    `target_sq_norms` carries one `‖W_s‖²` per slot of each persistence stack, aligned
    with the `weight_deltas` grouping; the returned loss consumes those stacked deltas."""
    slot_names: dict[str, list[str]] = {}
    for name, group, slot in site_slots:
        assert slot == len(slot_names.setdefault(group, [])), site_slots
        slot_names[group].append(name)
    assert target_sq_norms.keys() == slot_names.keys(), (
        sorted(target_sq_norms),
        sorted(slot_names),
    )
    for group, norms in target_sq_norms.items():
        assert len(norms) == len(slot_names[group]), (group, len(norms), slot_names[group])
    bad = {
        name: value
        for group, norms in target_sq_norms.items()
        for name, value in zip(slot_names[group], norms, strict=True)
        if not math.isfinite(value) or value <= 0.0
    }
    assert not bad, f"faithfulness needs finite positive ‖W_s‖²: {bad}"
    n_sites = len(site_slots)

    def faithfulness_loss(weight_deltas: dict[str, Float[Array, "g _ _"]]) -> Float[Array, ""]:
        """`mean_s ‖W_s - V_sU_s‖²_F / ‖W_s‖²_F` over the stacked deltas, fp32."""
        total = sum(
            (
                jnp.sum(
                    jnp.sum(weight_deltas[group].astype(jnp.float32) ** 2, axis=(1, 2))
                    / jnp.asarray(norms, jnp.float32)
                )
                for group, norms in target_sq_norms.items()
            ),
            start=jnp.zeros((), jnp.float32),
        )
        return total / n_sites

    return faithfulness_loss


def faithfulness_loss_for(model: DecomposedModel) -> FaithfulnessLossFn:
    """Bind each frozen target squared norm into its relative-error loss."""
    target_sq_norms = {
        group: tuple(float(value) for value in values)
        for group, values in jax.device_get(model.target_weight_sq_norms()).items()
    }
    return make_faithfulness_loss(site_slots_for(model.sites), target_sq_norms)
