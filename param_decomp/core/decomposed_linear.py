"""The placed decomposed-linear primitive (SPEC §4.1): `((x@V)*m)@U + (x@Δ)*d`.

`site_forward` executes one decomposed site against its frozen linear; `site_out` is its
output-only view. Placement arrives as one of three enumerated shapes: the run's resolved
`PlacementRules` (plans derived here per call), a `PlannedComponentLinear` a target
precompiled once per site, or `None` — the unplaced CPU/test execution.
`constrain_component_activation` pins any `[*leading, C]` tensor (CI squashings, captured
`x@V`) to the same component-waist row `site_forward` places `x@V` on.

This module sits above `placement.py`: it consumes the nominal rules types, while the
representation it executes (`components.py`) stays placement-free."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array

from param_decomp.core.components import activation_axes
from param_decomp.core.linear_plan import LinearPlan, placed_linear
from param_decomp.core.placement import PlacedRule, PlacementRules


@dataclass(frozen=True)
class PlannedComponentLinear:
    """One site's component linear, fully compiled: both plans plus the two rows
    `site_forward` still reshards against (the component waist and the public output)."""

    v: LinearPlan
    u: LinearPlan
    component: PlacedRule
    output: PlacedRule


def constrain_component_activation(x: Array, placement: PlacementRules | None) -> Array:
    if placement is None:
        return x
    axes = activation_axes(x.ndim, "C")
    row = placement.activations.component
    row.validate_shape(axes, x.shape)
    return jax.sharding.reshard(x, row.sharding_for(axes))


@dataclass(frozen=True)
class SiteForward:
    output: Array
    component_activation: Array


def site_forward(
    x: Array,
    V: Array,
    U: Array,
    W: Array,
    mask: Array | None,
    delta_mask: Array | None,
    route: Array | None,
    placement: PlacementRules | PlannedComponentLinear | None,
    frozen_linear: LinearPlan | None,
) -> SiteForward:
    """One decomposed linear (SPEC §4.1): `((x@V)*m)@U + (x@Δ)*d`, routed per position
    against the frozen `x @ W.T`. `mask` may be None (fully on); `route` None routes
    everywhere. `delta_mask` None drops the delta path entirely (constant-source entries
    carry no delta, LOSS_PARITY_DESIGN §4b). `delta_mask`/`route` broadcast over batch;
    trailing dim added here."""
    external_axes = activation_axes(x.ndim, "feature")
    component_axes = activation_axes(x.ndim, "C")
    match placement:
        case None:
            xV = x @ V
            u_linear = None
            component_row = None
            output_row = None
        case PlannedComponentLinear(
            v=v_linear,
            u=u_linear,
            component=component_row,
            output=output_row,
        ):
            xV = placed_linear(x, V, v_linear)
        case PlacementRules():
            operand = placement.components.operands
            input_row = placement.target.component.input
            v_axes = ("d_in", "C")
            u_axes = ("C", "d_out")
            operand.validate_shape(v_axes, V.shape)
            operand.validate_shape(u_axes, U.shape)
            input_row.validate_shape(external_axes, x.shape)
            v_linear = placement.component_linear_plan(v_axes, external_axes, component_axes)
            u_linear = placement.component_linear_plan(u_axes, component_axes, external_axes)
            xV = placed_linear(x, V, v_linear)
            component_row = placement.activations.component
            output_row = placement.target.component.output
    if component_row is not None:
        component_row.validate_shape(component_axes, xV.shape)
        xV = jax.sharding.reshard(xV, component_row.sharding_for(component_axes))
    coefficients = mask
    delta: Array | None = None
    if delta_mask is not None:
        delta = delta_mask[..., None]
        coefficients = 1.0 - delta if coefficients is None else coefficients - delta
    acts = xV * coefficients if coefficients is not None else xV
    match u_linear:
        case None:
            out = acts @ U
        case LinearPlan():
            out = placed_linear(acts, U, u_linear)
    frozen_out: Array | None = None
    if delta_mask is not None or route is not None:
        frozen_out = x @ W.T if frozen_linear is None else placed_linear(x, W.T, frozen_linear)
    if delta_mask is not None:
        assert frozen_out is not None and delta is not None
        out = out + delta * frozen_out
    if route is not None:
        assert frozen_out is not None
        out = jnp.where(route[..., None], out, frozen_out)
    if output_row is not None:
        output_row.validate_shape(external_axes, out.shape)
        out = jax.sharding.reshard(out, output_row.sharding_for(external_axes))
    return SiteForward(output=out, component_activation=xV)


def site_out(
    x: Array,
    V: Array,
    U: Array,
    W: Array,
    mask: Array | None,
    delta_mask: Array | None,
    route: Array | None,
    placement: PlacementRules | PlannedComponentLinear | None,
    frozen_linear: LinearPlan | None,
) -> Array:
    return site_forward(x, V, U, W, mask, delta_mask, route, placement, frozen_linear).output
