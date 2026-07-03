"""SRC_STEP `sign` (SPEC §6): stateless `sources += lr·sign(grad)` with the S15 clamp
projection, `opt_state is None` throughout — the moment buffers must not exist."""

import jax.numpy as jnp

from param_decomp.adversary import PersistentAdversary, sources_ascend_project
from param_decomp.configs import SignPGDConfig
from param_decomp.schedule import ScheduleConfig


def _sign_cfg(lr: float) -> SignPGDConfig:
    return SignPGDConfig(lr_schedule=ScheduleConfig(start_val=lr, warmup_pct=0.0))


def test_sign_ascent_moves_by_lr_and_projects():
    sources = {"site": jnp.asarray([0.5, 0.02, 0.99, 0.5])}
    grad = {"site": jnp.asarray([2.7, -0.3, 1.0, 0.0])}
    new_sources, opt_state = sources_ascend_project(
        sources, grad, None, jnp.asarray(0.05), _sign_cfg(0.05)
    )
    assert opt_state is None
    expected = jnp.asarray([0.55, 0.0, 1.0, 0.5])  # +lr, clamped at 0, clamped at 1, sign(0)=0
    assert jnp.allclose(new_sources["site"], expected)


def test_sign_ascent_preserves_source_dtype():
    sources = {"site": jnp.full((3,), 0.5, jnp.bfloat16)}
    grad = {"site": jnp.ones((3,), jnp.bfloat16)}
    new_sources, _ = sources_ascend_project(sources, grad, None, jnp.asarray(0.01), _sign_cfg(0.01))
    assert new_sources["site"].dtype == jnp.bfloat16


def test_sign_adversary_final_ascend_stays_stateless():
    adv = PersistentAdversary(
        sources={"site": jnp.full((2, 3), 0.5)},
        opt_state=None,
        state_key="ppgd",
        coeff=0.5,
        optimizer=_sign_cfg(0.01),
        n_warmup=0,
    )
    scaled_grad = {"site": jnp.full((2, 3), 0.5 * 3.0)}  # coeff-scaled, sign unaffected
    ascended = adv.final_ascend(scaled_grad, jnp.asarray(100.0), 1000)
    assert ascended.opt_state is None
    assert jnp.allclose(ascended.sources["site"], 0.51)
