"""The placed decomposed-linear primitive: `((x@V)*m)@U + (x@Δ)*d`.

`site_forward` executes one decomposed site from its `SiteWeights` and three per-forward
inputs: the component mask, the delta, and the route; `site_out` is its output-only view.
Placement arrives as one of three enumerated shapes: the run's resolved `PlacementRules`
(plans derived here per call), a `PlannedComponentLinear` a target precompiled once per
site, or `None` — the unplaced CPU/test execution. The target output `x @ Wᵀ` is computed
only when the delta or the route reads it (`blend_target_output`).
`constrain_component_activation` pins any `[*leading, C]` tensor (CI squashings, captured
`x@V`) to the same component-waist row `site_forward` places `x@V` on.

This module sits above `placement.py`: it consumes the nominal rules types, while the
representation it executes (`components.py`) stays placement-free."""

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jaxtyping import Array, Bool, Float

from param_decomp.core.components import SelectedCI, SiteCI, activation_axes
from param_decomp.core.linear_plan import (
    BlockContraction,
    BlockedLinearPlan,
    LinearPlan,
    block_einsum,
    blocked_placed_linear,
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


def constrain_component_activation(x: SiteCI, placement: PlacedRule | None) -> SiteCI:
    if placement is None:
        return x
    row = placement
    match x:
        case SelectedCI():
            values_sharding = row.sharding_for(activation_axes(x.values.ndim, "selected_c"))
            indices_sharding = row.sharding_for(activation_axes(x.block_indices.ndim, "selected_c"))
            return SelectedCI(
                values=jax.sharding.reshard(x.values, values_sharding),
                block_indices=jax.sharding.reshard(x.block_indices, indices_sharding),
                n_blocks=x.n_blocks,
            )
        case jax.Array():
            axes = activation_axes(x.ndim, "C")
            row.validate_shape(axes, x.shape)
            return jax.sharding.reshard(x, row.sharding_for(axes))


def component_coefficients(mask: Array, delta: Array | None) -> Array:
    """`m − d`: the component coefficients once the delta's `d·(x@V)@U` share, which
    `blend_target_output` restores through `d·(x@Wᵀ)`, is folded out. `delta` broadcasts
    against `mask`; None is `d = 0`."""
    return mask if delta is None else mask - delta


def blend_target_output(
    decomposed_out: Array,
    delta: Array | None,
    route: Array | None,
    target_out: Callable[[], Array],
) -> Array:
    """`where(route, decomposed + d·(x@Wᵀ), x@Wᵀ)`, with `delta` and `route` broadcasting
    against the output. None is the identity — `d = 0`, every position routed — and
    skips its work; with neither, `target_out` never runs."""
    if delta is None and route is None:
        return decomposed_out
    target = target_out()
    delta_out = decomposed_out if delta is None else decomposed_out + delta * target
    return delta_out if route is None else jnp.where(route, delta_out, target)


def _reshard_output(out: Array, output_row: PlacedRule | None) -> Array:
    if output_row is None:
        return out
    external_axes = activation_axes(out.ndim, "feature")
    output_row.validate_shape(external_axes, out.shape)
    return jax.sharding.reshard(out, output_row.sharding_for(external_axes))


@dataclass(frozen=True)
class SiteWeights:
    """One decomposed site's operands: the target weight `W` (`[d_out, d_in]`) it
    decomposes, the components `V` (`[d_in, C]`) and `U` (`[C, d_out]`) factoring it, and
    the plans its linears run under — `W_plan` for `x @ Wᵀ` (None: unplaced) and
    `placement` for the component linears (`site_forward`'s three shapes)."""

    W: Array
    V: Array
    U: Array
    W_plan: LinearPlan | None
    placement: PlacementRules | PlannedComponentLinear | None


def _target_out(x: Array, W: Array, plan: LinearPlan | None) -> Array:
    match plan:
        case None:
            return x @ W.T
        case LinearPlan():
            return placed_linear(x, W.T, plan)


@dataclass(frozen=True)
class SiteForward:
    output: Array
    component_activation: Array


def _planned_component_linear(
    rules: PlacementRules, x: Array, V: Array, U: Array
) -> PlannedComponentLinear:
    external_axes = activation_axes(x.ndim, "feature")
    component_axes = activation_axes(x.ndim, "C")
    v_axes = ("d_in", "C")
    u_axes = ("C", "d_out")
    rules.components.operands.validate_shape(v_axes, V.shape)
    rules.components.operands.validate_shape(u_axes, U.shape)
    rules.target.component.input.validate_shape(external_axes, x.shape)
    return PlannedComponentLinear(
        v=rules.component_linear_plan(v_axes, external_axes, component_axes),
        u=rules.component_linear_plan(u_axes, component_axes, external_axes),
        component=rules.activations.component,
        output=rules.target.component.output,
    )


def site_forward(
    x: Array, weights: SiteWeights, mask: Array, delta: Array | None, route: Array | None
) -> SiteForward:
    """One decomposed linear: `where(route, ((x@V)·m)@U + d·(x@(W − V@U)ᵀ), x@Wᵀ)`, per
    position. `delta` and `route` carry the leading axes only; None is the identity
    (`blend_target_output`)."""
    V, U, placement = weights.V, weights.U, weights.placement
    delta_c = None if delta is None else delta[..., None]
    coefficients = component_coefficients(mask, delta_c)
    match placement:
        case None:
            planned = None
        case PlannedComponentLinear():
            planned = placement
        case PlacementRules():
            planned = _planned_component_linear(placement, x, V, U)
    match planned:
        case None:
            xV = x @ V
            component_out = (xV * coefficients) @ U
            output_row = None
        case PlannedComponentLinear(v=v_plan, u=u_plan, component=component_row, output=output_row):
            component_axes = activation_axes(x.ndim, "C")
            xV = placed_linear(x, V, v_plan)
            component_row.validate_shape(component_axes, xV.shape)
            xV = jax.sharding.reshard(xV, component_row.sharding_for(component_axes))
            component_out = placed_linear(xV * coefficients, U, u_plan)
    blended = blend_target_output(
        component_out,
        delta_c,
        None if route is None else route[..., None],
        lambda: _target_out(x, weights.W, weights.W_plan),
    )
    return SiteForward(output=_reshard_output(blended, output_row), component_activation=xV)


def site_out(
    x: Array, weights: SiteWeights, mask: Array, delta: Array | None, route: Array | None
) -> Array:
    return site_forward(x, weights, mask, delta, route).output


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DenseComponentOverride:
    """Masked component activations replaced outright: where `overridden`, the masked
    `(x@V)·m` is `values` instead. Both span the site's full waist `[*leading, C]`."""

    overridden: Bool[Array, "*leading C"]
    values: Float[Array, "*leading C"]


def overridden_site_forward(
    x: Array,
    weights: SiteWeights,
    mask: Array,
    override: DenseComponentOverride,
    delta: Array | None,
) -> SiteForward:
    """`site_forward` with the components term's masked activations replaced at the
    override's entries, `((x@V)·m)[overridden ← values] @ U`; the delta term is
    unchanged. Unplaced and unrouted only."""
    if weights.placement is not None or weights.W_plan is not None:
        raise NotImplementedError("component overrides on a placed site")
    xV = x @ weights.V
    masked = xV * mask
    assert override.overridden.shape == override.values.shape == masked.shape, (
        override.overridden.shape,
        override.values.shape,
        masked.shape,
    )
    assert override.values.dtype == masked.dtype, (override.values.dtype, masked.dtype)
    overridden = jnp.where(override.overridden, override.values, masked)
    delta_c = None if delta is None else delta[..., None]
    activation = overridden if delta_c is None else overridden - delta_c * xV
    output = blend_target_output(
        activation @ weights.U, delta_c, None, lambda: _target_out(x, weights.W, None)
    )
    return SiteForward(output=output, component_activation=xV)


@dataclass(frozen=True)
class BlockedPlannedComponentLinear:
    """One block-factored site's component linears, fully compiled: both blocked plans
    plus the two rows the site forward still reshards against — the flat component
    waist and the public output."""

    v: BlockedLinearPlan
    u: BlockedLinearPlan
    component: PlacedRule
    output: PlacedRule


@dataclass(frozen=True)
class BlockedSiteWeights:
    """A block-factored site's operands: the fused target weight `W`, each block's
    factors `V` (`[E, d_in, c]`) and `U` (`[E, c, d_out]`), `W_plan` for `x @ Wᵀ` (None:
    unplaced), and the blocked component plans (None: unplaced)."""

    W: Array
    V: Array
    U: Array
    W_plan: LinearPlan | None
    placement: BlockedPlannedComponentLinear | None


def blocked_site_forward(
    x: Array,
    weights: BlockedSiteWeights,
    mask: Array,
    delta: Array | None,
    route: Array | None,
    contraction: BlockContraction,
) -> SiteForward:
    """The block-factored sibling of `site_forward` (applied per block).
    `weights` holds each block's factors and the site's fused target matrix, and the
    computation runs densely over every block.
    `contraction` is the target-declared orientation of the site (`BlockContraction`);
    it selects both factors' einsums. Inside the site the component activation carries
    the block axis (`[*leading, E, c]`), but every boundary tensor is flat: `mask`,
    `delta` and `route` follow `site_forward`'s contract with `C = E·c` in
    block-major order, and `SiteForward.component_activation` is returned in that same
    flat layout. A selected execution that gathers work items instead of running every
    block will be a sibling of this function, not a mode of it."""
    V, U, placement = weights.V, weights.U, weights.placement
    n_blocks, _, c = V.shape
    assert U.shape[:2] == (n_blocks, c), (V.shape, U.shape)
    lead = x.shape[:-1]
    match contraction:
        case "fused_output":
            v_input = x
        case "fused_input":
            assert x.shape[-1] == n_blocks * V.shape[1], (x.shape, V.shape)
            v_input = x.reshape(*lead, n_blocks, V.shape[1])
    delta_c = None if delta is None else delta[..., None]
    coefficients = component_coefficients(mask, delta_c)
    match placement:
        case None:
            xV = jnp.einsum(block_einsum(contraction, "V"), v_input, V)
            xV_flat = jax.lax.reshape(xV, (*lead, n_blocks * c), out_sharding=None)
            acts = jax.lax.reshape(xV_flat * coefficients, (*lead, n_blocks, c), out_sharding=None)
            component_out = jnp.einsum(block_einsum(contraction, "U"), acts, U)
            output_row = None
        case BlockedPlannedComponentLinear(
            v=v_plan, u=u_plan, component=component_row, output=output_row
        ):
            assert v_plan.contraction == contraction, (v_plan.contraction, contraction)
            assert u_plan.contraction == contraction, (u_plan.contraction, contraction)
            component_axes = activation_axes(x.ndim, "C")
            xV = blocked_placed_linear(v_input, V, v_plan)
            component_row.validate_shape(component_axes, (*lead, n_blocks * c))
            xV_flat = jax.lax.reshape(
                xV, (*lead, n_blocks * c), out_sharding=component_row.sharding_for(component_axes)
            )
            acts = jax.lax.reshape(
                xV_flat * coefficients,
                (*lead, n_blocks, c),
                out_sharding=NamedSharding(u_plan.mesh, u_plan.input),
            )
            component_out = blocked_placed_linear(acts, U, u_plan)
    if contraction == "fused_output":
        component_out = component_out.reshape(*lead, n_blocks * U.shape[2])
    blended = blend_target_output(
        component_out,
        delta_c,
        None if route is None else route[..., None],
        lambda: _target_out(x, weights.W, weights.W_plan),
    )
    return SiteForward(output=_reshard_output(blended, output_row), component_activation=xV_flat)
