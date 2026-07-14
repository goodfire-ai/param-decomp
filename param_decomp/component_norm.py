"""Muon-style per-subcomponent update normalization for the V/U masters (optax).

Muon takes a matrix update, keeps its singular directions, and equalizes the singular
values so no single rank-one mode monopolizes the step. VPD already carries learned
rank-one slices — component `c` of a site is `V[:, c] ⊗ U[c, :]` — so the analogue here
normalizes the update *per learned slice* instead of per discovered SVD slice: each
component's induced first-order weight-space update

    dW_c = gV_c ⊗ U_c + V_c ⊗ gU_c        (rank ≤ 2)

is scaled to unit norm, so every component moves its slice of `W_hat` by the same
`lr`-sized amount per step. Because `dW_c` has rank ≤ 2, both its Frobenius norm and its
exact spectral norm come in closed form from the 2×2 Gram cores — no per-component SVD:
with `AᵀA = [[gV_c·gV_c, gV_c·V_c], [·, V_c·V_c]]` and `BᵀB = [[U_c·U_c, U_c·gU_c],
[·, gU_c·gU_c]]`, `‖dW_c‖_F² = tr(AᵀA·BᵀB)` and `‖dW_c‖_2²` is the larger eigenvalue of
that 2×2 product. This normalizes each slice independently — it deliberately does NOT
equalize across components (overlapping components still add coherently; that is the
Candidate-B extension, not this transform).
"""

from collections.abc import Callable
from typing import Literal, cast

import jax.numpy as jnp
import optax
from jax.typing import ArrayLike
from jaxtyping import Array, Float

from param_decomp.components import DecompVU
from param_decomp.configs import ComponentNormOptimizerConfig


def component_dw_norms(
    V: Float[Array, "d_in C"],
    U: Float[Array, "C d_out"],
    gV: Float[Array, "d_in C"],
    gU: Float[Array, "C d_out"],
    norm: Literal["frobenius", "spectral"],
) -> Float[Array, " C"]:
    """Per-component norm of the induced rank-≤2 update `dW_c = gV_c ⊗ U_c + V_c ⊗ gU_c`."""
    p = jnp.sum(gV * gV, axis=0)
    q = jnp.sum(gV * V, axis=0)
    r = jnp.sum(V * V, axis=0)
    a = jnp.sum(U * U, axis=1)
    b = jnp.sum(U * gU, axis=1)
    d = jnp.sum(gU * gU, axis=1)
    trace = p * a + 2.0 * q * b + r * d
    match norm:
        case "frobenius":
            # numerical floor: trace is a squared norm, ≥ 0 up to float reassociation
            return jnp.sqrt(jnp.maximum(trace, 0.0))
        case "spectral":
            det = (p * r - q * q) * (a * d - b * b)
            lam_max = 0.5 * (trace + jnp.sqrt(jnp.maximum(trace * trace - 4.0 * det, 0.0)))
            return jnp.sqrt(jnp.maximum(lam_max, 0.0))


def scale_by_component_dw_norm(
    norm: Literal["frobenius", "spectral"], eps: float
) -> optax.GradientTransformation:
    """Divide each component's `(gV_c, gU_c)` by `‖dW_c‖ + eps`, per site. Operates on the
    `DecompVU` pytree only (needs `params` to form the induced update)."""

    def init(params: optax.Params) -> optax.EmptyState:
        del params
        return optax.EmptyState()

    def update(
        updates: optax.Updates, state: optax.OptState, params: optax.Params | None = None
    ) -> tuple[optax.Updates, optax.OptState]:
        assert isinstance(updates, DecompVU) and isinstance(params, DecompVU)
        scaled: dict[str, tuple[Array, Array]] = {}
        for name, (gV, gU) in updates.vu.items():
            V, U = params.vu[name]
            denom = component_dw_norms(V, U, gV, gU, norm) + eps
            scaled[name] = (gV / denom[None, :], gU / denom[:, None])
        # optax types Updates as a builtin-container pytree union; DecompVU is a valid
        # pytree (eqx.Module) it can't express, so hop through object.
        return cast(optax.Updates, cast(object, DecompVU(vu=scaled))), state

    return optax.GradientTransformation(init, update)


def component_norm_optimizer(
    cfg: ComponentNormOptimizerConfig, schedule: Callable[[ArrayLike], Array]
) -> optax.GradientTransformation:
    """Optional Nesterov momentum → per-component induced-dW normalization → scheduled LR.
    No global grad clip: the per-component normalization is the trust region."""
    normalize = scale_by_component_dw_norm(cfg.norm, cfg.eps)
    lr = optax.scale_by_learning_rate(schedule)
    if cfg.momentum is None:
        return optax.chain(normalize, lr)
    return optax.chain(optax.trace(decay=cfg.momentum, nesterov=True), normalize, lr)
