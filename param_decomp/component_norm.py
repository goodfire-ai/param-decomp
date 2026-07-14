"""Muon-style per-subcomponent update normalization for the V/U masters (optax).

Muon takes a matrix update, keeps its singular directions, and equalizes the singular
values so no single rank-one mode monopolizes the step. VPD already carries learned
rank-one slices — component `c` of a site is `V[:, c] ⊗ U[c, :]` — so the analogue here
normalizes the update *per learned slice* instead of per discovered SVD slice: each
component's induced first-order weight-space update

    dW_c = gV_c ⊗ U_c + V_c ⊗ gU_c        (rank ≤ 2)

is scaled to unit norm, so every component moves its slice of `W_hat` by the same
`lr`-sized amount per step (first order; the finite update also carries the second-order
`ΔV_c ⊗ ΔU_c` term — `train._component_update_metrics` logs both). `dW_c` is a tangent
update to the rank-one manifold by construction; the deliberate simplification is that
its DIRECTION is the raw-factor gradient, selected in gauge-dependent V/U coordinates.
Because `dW_c` has rank ≤ 2, both its Frobenius norm and its exact spectral norm come in
closed form from the 2×2 Gram cores — no per-component SVD: with
`AᵀA = [[gV_c·gV_c, gV_c·V_c], [·, V_c·V_c]]` and `BᵀB = [[U_c·U_c, U_c·gU_c],
[·, gU_c·gU_c]]`, `‖dW_c‖_F² = tr(AᵀA·BᵀB)` and `‖dW_c‖_2²` is the larger eigenvalue of
that 2×2 product. This normalizes each slice independently — it deliberately does NOT
equalize across components (overlapping components still add coherently; that is the
Candidate-B extension, not this transform).
"""

from collections.abc import Callable
from typing import Literal, NamedTuple, cast

import jax.numpy as jnp
import optax
from jax.typing import ArrayLike
from jaxtyping import Array, Float

from param_decomp.components import DecompVU
from param_decomp.configs import ComponentNormOptimizerConfig


def rank2_pair_frobenius(
    a1: Float[Array, "d_in C"],
    b1: Float[Array, "C d_out"],
    a2: Float[Array, "d_in C"],
    b2: Float[Array, "C d_out"],
) -> Float[Array, " C"]:
    """Per-component `‖a1_c ⊗ b1_c + a2_c ⊗ b2_c‖_F` via the 2×2 Gram trace — the rank-≤2
    matrix is never materialized. All inputs must be fp32 (the master dtype, SPEC N1)."""
    for x in (a1, b1, a2, b2):
        assert x.dtype == jnp.float32, x.dtype
    trace = (
        jnp.sum(a1 * a1, axis=0) * jnp.sum(b1 * b1, axis=1)
        + 2.0 * jnp.sum(a1 * a2, axis=0) * jnp.sum(b1 * b2, axis=1)
        + jnp.sum(a2 * a2, axis=0) * jnp.sum(b2 * b2, axis=1)
    )
    # numerical floor: the trace is a squared norm, ≥ 0 up to float reassociation
    return jnp.sqrt(jnp.maximum(trace, 0.0))


def component_dw_norms(
    V: Float[Array, "d_in C"],
    U: Float[Array, "C d_out"],
    gV: Float[Array, "d_in C"],
    gU: Float[Array, "C d_out"],
    norm: Literal["frobenius", "spectral"],
) -> Float[Array, " C"]:
    """Per-component norm of the induced rank-≤2 update `dW_c = gV_c ⊗ U_c + V_c ⊗ gU_c`."""
    match norm:
        case "frobenius":
            return rank2_pair_frobenius(gV, U, V, gU)
        case "spectral":
            for x in (V, U, gV, gU):
                assert x.dtype == jnp.float32, x.dtype
            p = jnp.sum(gV * gV, axis=0)
            q = jnp.sum(gV * V, axis=0)
            r = jnp.sum(V * V, axis=0)
            a = jnp.sum(U * U, axis=1)
            b = jnp.sum(U * gU, axis=1)
            d = jnp.sum(gU * gU, axis=1)
            trace = p * a + 2.0 * q * b + r * d
            det = (p * r - q * q) * (a * d - b * b)
            lam_max = 0.5 * (trace + jnp.sqrt(jnp.maximum(trace * trace - 4.0 * det, 0.0)))
            return jnp.sqrt(jnp.maximum(lam_max, 0.0))


class ScaleByComponentDWNormState(NamedTuple):
    """`prenorm[site]` is last step's `‖dW_c‖ (C,)` BEFORE division — the norm the divisor
    used. Carried as optimizer state so the train step can log the pre-normalization
    distribution (weak/noisy components get full-sized steps; this is the evidence)."""

    prenorm: dict[str, Array]


def scale_by_component_dw_norm(
    norm: Literal["frobenius", "spectral"], eps: float, norm_floor: float
) -> optax.GradientTransformation:
    """Divide each component's `(gV_c, gU_c)` by `max(‖dW_c‖ + eps, norm_floor)`, per site.
    Operates on the `DecompVU` pytree only (needs `params` to form the induced update).
    `norm_floor > 0` turns the full-step guarantee off below the floor: a component whose
    update norm is under the floor moves proportionally (`lr · ‖dW_c‖ / norm_floor`)
    instead of getting the full `lr` step — the anti-churn variant for rare/noisy
    components. `norm_floor = 0` is the aggressive default (every component steps `lr`)."""

    def init(params: optax.Params) -> ScaleByComponentDWNormState:
        assert isinstance(params, DecompVU)
        return ScaleByComponentDWNormState(
            prenorm={name: jnp.zeros(V.shape[1], jnp.float32) for name, (V, _) in params.vu.items()}
        )

    def update(
        updates: optax.Updates, state: optax.OptState, params: optax.Params | None = None
    ) -> tuple[optax.Updates, optax.OptState]:
        del state
        assert isinstance(updates, DecompVU) and isinstance(params, DecompVU)
        scaled: dict[str, tuple[Array, Array]] = {}
        prenorm: dict[str, Array] = {}
        for name, (gV, gU) in updates.vu.items():
            V, U = params.vu[name]
            n = component_dw_norms(V, U, gV, gU, norm)
            prenorm[name] = n
            denom = jnp.maximum(n + eps, norm_floor)
            scaled[name] = (gV / denom[None, :], gU / denom[:, None])
        # optax types Updates as a builtin-container pytree union; DecompVU is a valid
        # pytree (eqx.Module) it can't express, so hop through object.
        return (
            cast(optax.Updates, cast(object, DecompVU(vu=scaled))),
            ScaleByComponentDWNormState(prenorm=prenorm),
        )

    return optax.GradientTransformation(init, update)


def component_norm_optimizer(
    cfg: ComponentNormOptimizerConfig,
    schedule: Callable[[ArrayLike], Array],
    clip: optax.GradientTransformation | None,
) -> optax.GradientTransformation:
    """Optional global clip → optional Nesterov momentum → per-component induced-dW
    normalization → scheduled LR. The per-component normalization is the trust region;
    a global clip is ~inert through it (per-component scale cancels) except that a
    time-varying clip coefficient reweights the momentum average — `clip` exists to match
    the muon arms' clip→momentum ordering exactly in controlled comparisons."""
    transforms = [] if clip is None else [clip]
    if cfg.momentum is not None:
        transforms.append(optax.trace(decay=cfg.momentum, nesterov=True))
    transforms.append(scale_by_component_dw_norm(cfg.norm, cfg.eps, cfg.norm_floor))
    transforms.append(optax.scale_by_learning_rate(schedule))
    return optax.chain(*transforms)


def component_prenorm(opt_state: optax.OptState) -> dict[str, Array] | None:
    """This step's pre-normalization `‖dW_c‖` per site out of a post-`update` optimizer
    state, or None when the optimizer has no `scale_by_component_dw_norm` stage (adam /
    nesterov_sgd)."""
    if isinstance(opt_state, ScaleByComponentDWNormState):
        return opt_state.prenorm
    if isinstance(opt_state, tuple):
        found = [p for s in opt_state if (p := component_prenorm(s)) is not None]
        assert len(found) <= 1, "multiple scale_by_component_dw_norm stages in one chain"
        if found:
            return found[0]
    return None
