"""Per-subcomponent normalized SGD (`component_norm`, SPEC S20 amendment 2026-07-14).

The closed-form 2×2 norms of the induced rank-≤2 update `dW_c = gV_c ⊗ U_c + V_c ⊗ gU_c`
must match the materialized matrix norms, and the full optimizer must move every
component's weight slice by exactly `lr` (first order, in the chosen norm).
"""

from typing import Literal, cast

import jax
import jax.numpy as jnp
import optax
import pytest
from jaxtyping import Array

from param_decomp.component_norm import (
    component_dw_norms,
    component_norm_optimizer,
    component_prenorm,
    scale_by_component_dw_norm,
)
from param_decomp.components import DecompVU
from param_decomp.configs import ComponentNormOptimizerConfig, PDConfig
from param_decomp.schedule import ScheduleConfig
from param_decomp.train import _component_update_metrics

D_IN, D_OUT, C = 7, 5, 11

Norm = Literal["frobenius", "spectral"]


def _random_site(key: Array) -> tuple[Array, Array, Array, Array]:
    kV, kU, kgV, kgU = jax.random.split(key, 4)
    V = jax.random.normal(kV, (D_IN, C))
    U = jax.random.normal(kU, (C, D_OUT))
    gV = jax.random.normal(kgV, (D_IN, C)) * 0.3
    gU = jax.random.normal(kgU, (C, D_OUT)) * 0.3
    return V, U, gV, gU


def _materialized_dw(V: Array, U: Array, gV: Array, gU: Array) -> Array:
    """All C induced updates as explicit `(C, d_in, d_out)` matrices."""
    return jnp.einsum("ic,co->cio", gV, U) + jnp.einsum("ic,co->cio", V, gU)


def _as_pytree(tree: DecompVU) -> optax.Params:
    """optax types Params/Updates as a builtin-container pytree union that can't express an
    eqx.Module; hop through object."""
    return cast(optax.Params, cast(object, tree))


def _apply(
    transform: optax.GradientTransformation,
    grads: DecompVU,
    state: optax.OptState,
    params: DecompVU,
) -> tuple[DecompVU, optax.OptState]:
    updates, new_state = transform.update(_as_pytree(grads), state, _as_pytree(params))
    assert isinstance(updates, DecompVU)
    return updates, new_state


def _init(transform: optax.GradientTransformation, params: DecompVU) -> optax.OptState:
    return transform.init(_as_pytree(params))


@pytest.mark.parametrize("norm", ["frobenius", "spectral"])
def test_dw_norms_match_materialized(norm: Norm) -> None:
    V, U, gV, gU = _random_site(jax.random.key(0))
    got = component_dw_norms(V, U, gV, gU, norm)
    dw = _materialized_dw(V, U, gV, gU)
    want = {
        "frobenius": jnp.sqrt(jnp.sum(dw**2, axis=(1, 2))),
        "spectral": jnp.linalg.matrix_norm(dw, ord=2),
    }[norm]
    assert jnp.allclose(got, want, rtol=1e-5), jnp.max(jnp.abs(got - want))


@pytest.mark.parametrize("norm", ["frobenius", "spectral"])
def test_scaled_updates_have_unit_dw_norm(norm: Norm) -> None:
    V, U, gV, gU = _random_site(jax.random.key(1))
    params = DecompVU(vu={"site": (V, U)})
    grads = DecompVU(vu={"site": (gV, gU)})
    transform = scale_by_component_dw_norm(norm, eps=0.0, norm_floor=0.0)
    scaled, new_state = _apply(transform, grads, _init(transform, params), params)
    sV, sU = scaled.vu["site"]
    got = component_dw_norms(V, U, sV, sU, norm)
    assert jnp.allclose(got, 1.0, rtol=1e-5), got
    prenorm = component_prenorm(new_state)
    assert prenorm is not None
    assert jnp.allclose(prenorm["site"], component_dw_norms(V, U, gV, gU, norm), rtol=1e-6)


def test_zero_grad_component_stays_zero() -> None:
    V, U, gV, gU = _random_site(jax.random.key(2))
    gV = gV.at[:, 0].set(0.0)
    gU = gU.at[0, :].set(0.0)
    params = DecompVU(vu={"site": (V, U)})
    grads = DecompVU(vu={"site": (gV, gU)})
    transform = scale_by_component_dw_norm("frobenius", eps=1e-8, norm_floor=0.0)
    scaled, _ = _apply(transform, grads, _init(transform, params), params)
    sV, sU = scaled.vu["site"]
    assert jnp.all(sV[:, 0] == 0.0) and jnp.all(sU[0, :] == 0.0)
    assert jnp.all(jnp.isfinite(sV)) and jnp.all(jnp.isfinite(sU))


def test_norm_floor_scales_weak_components_proportionally() -> None:
    V, U, gV, gU = _random_site(jax.random.key(5))
    params = DecompVU(vu={"site": (V, U)})
    grads = DecompVU(vu={"site": (gV, gU)})
    norms = component_dw_norms(V, U, gV, gU, "frobenius")
    floor = float(jnp.median(norms))
    transform = scale_by_component_dw_norm("frobenius", eps=0.0, norm_floor=floor)
    scaled, _ = _apply(transform, grads, _init(transform, params), params)
    sV, sU = scaled.vu["site"]
    got = component_dw_norms(V, U, sV, sU, "frobenius")
    expected = jnp.minimum(norms / floor, 1.0)
    assert jnp.allclose(got, expected, rtol=1e-5), (got, expected)


def _cfg(momentum: float | None) -> ComponentNormOptimizerConfig:
    return ComponentNormOptimizerConfig(
        type="component_norm",
        lr_schedule=ScheduleConfig(fn_type="constant", start_val=1e-3, final_val_frac=1.0),
        momentum=momentum,
        norm="frobenius",
    )


def test_optimizer_step_moves_each_component_by_lr() -> None:
    V, U, gV, gU = _random_site(jax.random.key(3))
    params = DecompVU(vu={"a": (V, U), "b": (2.0 * V, 0.5 * U)})
    grads = DecompVU(vu={"a": (gV, gU), "b": (5.0 * gV, 5.0 * gU)})
    opt = component_norm_optimizer(_cfg(momentum=None), lambda count: jnp.asarray(1e-3), None)
    updates, _ = _apply(opt, grads, _init(opt, params), params)
    for name, (pV, pU) in params.vu.items():
        uV, uU = updates.vu[name]
        dw_norm = component_dw_norms(pV, pU, uV, uU, "frobenius")
        assert jnp.allclose(dw_norm, 1e-3, rtol=1e-4), (name, dw_norm)
        # descent direction: the induced dW opposes the raw gradient's induced dW
        gV_n, gU_n = grads.vu[name]
        inner = jnp.sum(_materialized_dw(pV, pU, uV, uU) * _materialized_dw(pV, pU, gV_n, gU_n))
        assert inner < 0.0


def test_momentum_state_and_step() -> None:
    V, U, gV, gU = _random_site(jax.random.key(4))
    params = DecompVU(vu={"site": (V, U)})
    grads = DecompVU(vu={"site": (gV, gU)})
    opt = component_norm_optimizer(_cfg(momentum=0.95), lambda count: jnp.asarray(1e-3), None)
    state = _init(opt, params)
    updates, state = _apply(opt, grads, state, params)
    updates2, _ = _apply(opt, grads, state, params)
    for u, u2 in zip(jax.tree.leaves(updates), jax.tree.leaves(updates2), strict=True):
        assert u.shape == u2.shape and jnp.all(jnp.isfinite(u2))


def test_component_update_metrics_match_materialized() -> None:
    V, U, gV, gU = _random_site(jax.random.key(6))
    components = DecompVU(vu={"site": (V, U)})
    lr = 1e-2
    updates = DecompVU(vu={"site": (-lr * gV, -lr * gU)})
    metrics = _component_update_metrics(components, updates, prenorm=None)
    dV, dU = -lr * gV, -lr * gU
    fo = _materialized_dw(V, U, dV, dU)
    finite = jnp.einsum("ic,co->cio", V + dV, U + dU) - jnp.einsum("ic,co->cio", V, U)
    fo_norms = jnp.sqrt(jnp.sum(fo**2, axis=(1, 2)))
    fin_norms = jnp.sqrt(jnp.sum(finite**2, axis=(1, 2)))
    so_norms = jnp.linalg.norm(dV, axis=0) * jnp.linalg.norm(dU, axis=1)
    assert jnp.allclose(
        metrics["optim/components/dw_first_order/max"], jnp.max(fo_norms), rtol=1e-5
    )
    assert jnp.allclose(metrics["optim/components/dw_finite/max"], jnp.max(fin_norms), rtol=1e-5)
    assert jnp.allclose(
        metrics["optim/components/dw_second_over_first/median"],
        jnp.median(so_norms / fo_norms),
        rtol=1e-4,
    )
    assert "optim/components/dw_prenorm/median" not in metrics
    with_prenorm = _component_update_metrics(
        components, updates, prenorm={"site": jnp.ones(C, jnp.float32)}
    )
    assert with_prenorm["optim/components/dw_prenorm/median"] == 1.0


def test_pd_config_parses_component_norm_and_defaults_to_adam() -> None:
    base = {
        "ci_config": {
            "type": "layerwise_mlp",
            "hidden_dims": [8],
        },
        "decomposition_targets": [{"module_pattern": "h.*", "C": 4}],
        "loss_metrics": [{"type": "FaithfulnessLoss", "coeff": 1.0}],
        "ci_fn_optimizer": {
            "lr_schedule": {"fn_type": "constant", "start_val": 1e-4, "final_val_frac": 1.0}
        },
        "steps": 10,
        "batch_size": 2,
    }
    comp_norm = PDConfig.model_validate(
        base
        | {
            "components_optimizer": {
                "type": "component_norm",
                "lr_schedule": {"fn_type": "constant", "start_val": 1e-3, "final_val_frac": 1.0},
                "momentum": 0.95,
                "norm": "spectral",
            }
        }
    )
    assert isinstance(comp_norm.components_optimizer, ComponentNormOptimizerConfig)
    # a stored config with no `type` key still parses as the canonical adam optimizer
    adam = PDConfig.model_validate(
        base
        | {
            "components_optimizer": {
                "lr_schedule": {"fn_type": "constant", "start_val": 1e-4, "final_val_frac": 1.0},
                "grad_clip_norm": 0.01,
            }
        }
    )
    assert adam.components_optimizer.type == "adam"
    # the nesterov_sgd control parses and builds (clip -> trace -> lr chain)
    from param_decomp.configs import NesterovSGDOptimizerConfig
    from param_decomp.run_state import build_optimizers

    sgd_pd = PDConfig.model_validate(
        base
        | {
            "components_optimizer": {
                "type": "nesterov_sgd",
                "lr_schedule": {"fn_type": "constant", "start_val": 1e-2, "final_val_frac": 1.0},
                "momentum": 0.95,
                "grad_clip_norm": 0.01,
            }
        }
    )
    assert isinstance(sgd_pd.components_optimizer, NesterovSGDOptimizerConfig)
    opt_vu, _, _ = build_optimizers(sgd_pd)
    V, U, gV, gU = _random_site(jax.random.key(7))
    params = DecompVU(vu={"site": (V, U)})
    updates, _ = _apply(opt_vu, DecompVU(vu={"site": (gV, gU)}), _init(opt_vu, params), params)
    assert component_prenorm(_init(opt_vu, params)) is None
    for u in jax.tree.leaves(updates):
        assert jnp.all(jnp.isfinite(u))
