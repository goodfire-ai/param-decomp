"""The placed decomposed-linear primitive: `((x@V)*m)@U + (x@Δ)*d`.

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

from param_decomp.core.components import NarrowCI, SiteCI, activation_axes
from param_decomp.core.linear_plan import (
    ExpertBlockLinearPlan,
    ExpertContraction,
    LinearPlan,
    expert_block_einsum,
    expert_block_placed_linear,
    placed_linear,
)
from param_decomp.core.placement import PlacedRule, PlacementRules


@dataclass(frozen=True)
class PlannedComponentLinear:
    """One site's component linear, fully compiled: both plans plus the two rows
    `site_forward` still reshards against (the component waist and the public output)."""

    v: LinearPlan
    u: LinearPlan
    component: PlacedRule
    output: PlacedRule


def constrain_component_activation(x: SiteCI, placement: PlacementRules | None) -> SiteCI:
    if placement is None:
        return x
    row = placement.activations.component
    match x:
        case NarrowCI():
            # A narrow site's routed-slot axes (`routed_c`, and the indices' k) REPLICATE
            # at the waist — slot-major, no expert co-location — so the row's `C: tp` arm
            # structurally cannot apply to a bundle: `routed_c` is unlisted in every rule,
            # and unlisted axes replicate. The batch axes pin exactly as the full arm's.
            values_sharding = row.sharding_for(activation_axes(x.values.ndim, "routed_c"))
            indices_sharding = row.sharding_for(activation_axes(x.router_indices.ndim, "routed_c"))
            return NarrowCI(
                values=jax.sharding.reshard(x.values, values_sharding),
                router_indices=jax.sharding.reshard(x.router_indices, indices_sharding),
                n_experts=x.n_experts,
            )
        case jax.Array():
            axes = activation_axes(x.ndim, "C")
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
    """One decomposed linear: `((x@V)*m)@U + (x@Δ)*d`, routed per position
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


@dataclass(frozen=True)
class ExpertPlannedComponentLinear:
    """One expert-blocked site's component linears, fully compiled: both block plans
    plus the two rows the site forward still reshards against — the flat component
    waist and the public output."""

    v: ExpertBlockLinearPlan
    u: ExpertBlockLinearPlan
    component: PlacedRule
    output: PlacedRule


def expert_block_site_forward(
    x: Array,
    V: Array,
    U: Array,
    W: Array,
    mask: Array | None,
    delta_mask: Array | None,
    route: Array | None,
    contraction: ExpertContraction,
    placement: ExpertPlannedComponentLinear | None,
    frozen_linear: LinearPlan | None,
) -> SiteForward:
    """The expert-blocked sibling of `site_forward` (applied per expert
    block). `V [E, d_in, c]` and `U [E, c, d_out]` hold each expert's factors, `W` is
    the site's fused frozen matrix, and the computation runs densely over every expert.
    `contraction` is the target-declared orientation of the site (`ExpertContraction`);
    it selects both factors' einsums. Inside the site the component activation carries
    the expert axis (`[*leading, E, c]`), but every boundary tensor is flat: `mask`,
    `delta_mask`, and `route` follow `site_forward`'s contract with `C = E·c` in
    expert-major order, and `SiteForward.component_activation` is returned in that same
    flat layout. A routed execution that gathers work items instead of running every
    expert will be a sibling of this function, not a mode of it."""
    n_experts, _, c = V.shape
    assert U.shape[:2] == (n_experts, c), (V.shape, U.shape)
    lead = x.shape[:-1]
    external_axes = activation_axes(x.ndim, "feature")
    component_axes = activation_axes(x.ndim, "C")
    match contraction:
        case "fused_output":
            v_input = x
        case "fused_input":
            assert x.shape[-1] == n_experts * V.shape[1], (x.shape, V.shape)
            v_input = x.reshape(*lead, n_experts, V.shape[1])
    match placement:
        case None:
            xV = jnp.einsum(expert_block_einsum(contraction, "V"), v_input, V)
        case ExpertPlannedComponentLinear(v=v_plan):
            assert v_plan.contraction == contraction, (v_plan.contraction, contraction)
            xV = expert_block_placed_linear(v_input, V, v_plan)
    xV_flat = xV.reshape(*lead, n_experts * c)
    if placement is not None:
        placement.component.validate_shape(component_axes, xV_flat.shape)
        xV_flat = jax.sharding.reshard(xV_flat, placement.component.sharding_for(component_axes))
    coefficients = mask
    delta: Array | None = None
    if delta_mask is not None:
        delta = delta_mask[..., None]
        coefficients = 1.0 - delta if coefficients is None else coefficients - delta
    acts_flat = xV_flat * coefficients if coefficients is not None else xV_flat
    acts = acts_flat.reshape(*lead, n_experts, c)
    match placement:
        case None:
            out = jnp.einsum(expert_block_einsum(contraction, "U"), acts, U)
        case ExpertPlannedComponentLinear(u=u_plan):
            assert u_plan.contraction == contraction, (u_plan.contraction, contraction)
            out = expert_block_placed_linear(acts, U, u_plan)
    if contraction == "fused_output":
        out = out.reshape(*lead, n_experts * U.shape[2])
    frozen_out: Array | None = None
    if delta_mask is not None or route is not None:
        frozen_out = x @ W.T if frozen_linear is None else placed_linear(x, W.T, frozen_linear)
    if delta_mask is not None:
        assert frozen_out is not None and delta is not None
        out = out + delta * frozen_out
    if route is not None:
        assert frozen_out is not None
        out = jnp.where(route[..., None], out, frozen_out)
    if placement is not None:
        placement.output.validate_shape(external_axes, out.shape)
        out = jax.sharding.reshard(out, placement.output.sharding_for(external_axes))
    return SiteForward(output=out, component_activation=xV_flat)
