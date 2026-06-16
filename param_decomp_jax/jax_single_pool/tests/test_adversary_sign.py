"""Sign SRC_STEP variant for the persistent adversary (SPEC §6, S15).

`sign` is the stateless persistent ascent (`sources += lr * sign(grad)` then clamp to
[0,1]); `adam` is the moment-carrying default. These check the per-step math and that
`build_recon_terms` now accepts a sign-optimizer persistent term.
"""

import jax.numpy as jnp

from jax_single_pool.adversary import (
    SourcesSignState,
    init_sources_opt_state,
    sources_ascend_project,
)
from jax_single_pool.recon import build_recon_terms
from param_decomp_config.losses import (
    AdamPGDConfig,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    PersistentPGDReconLossConfig,
    SCScope,
    SignPGDConfig,
    StochasticReconLossConfig,
)
from param_decomp_config.schedule import ScheduleConfig


def _sign_optimizer(lr: float) -> SignPGDConfig:
    return SignPGDConfig(lr_schedule=ScheduleConfig(start_val=lr, warmup_pct=0.0))


def test_sign_state_is_stateless() -> None:
    sources = {"s": jnp.array([0.1, 0.9])}
    state = init_sources_opt_state(sources, _sign_optimizer(0.05))
    assert isinstance(state, SourcesSignState)


def test_sign_ascend_is_lr_times_sign_then_clamp() -> None:
    lr = 0.25
    optimizer = _sign_optimizer(lr)
    sources = {"s": jnp.array([0.5, 0.5, 0.9, 0.1])}
    grad = {"s": jnp.array([3.0, -2.0, 7.0, -7.0])}  # ascent follows the SIGN only
    state = init_sources_opt_state(sources, optimizer)

    new_sources, new_state = sources_ascend_project(
        sources, grad, state, jnp.asarray(lr), optimizer
    )

    # +lr where grad>0, -lr where grad<0, then projected to [0,1].
    expected = jnp.array([0.75, 0.25, 1.0, 0.0])
    assert jnp.allclose(new_sources["s"], expected)
    assert isinstance(new_state, SourcesSignState)


def test_sign_ascend_ignores_grad_magnitude() -> None:
    optimizer = _sign_optimizer(0.1)
    sources = {"s": jnp.array([0.5, 0.5])}
    state = init_sources_opt_state(sources, optimizer)

    small, _ = sources_ascend_project(
        sources, {"s": jnp.array([1e-6, -1e-6])}, state, jnp.asarray(0.1), optimizer
    )
    large, _ = sources_ascend_project(
        sources, {"s": jnp.array([1e6, -1e6])}, state, jnp.asarray(0.1), optimizer
    )
    assert jnp.allclose(small["s"], large["s"])


def test_build_recon_terms_accepts_sign_persistent() -> None:
    site_names = ("a", "b")
    loss_spec = build_recon_terms(
        (
            FaithfulnessLossConfig(coeff=1.0),
            ImportanceMinimalityLossConfig(
                coeff=1e-6,
                pnorm=2.0,
                beta=0.1,
                p_anneal_start_frac=0.0,
                p_anneal_final_p=0.5,
                p_anneal_end_frac=1.0,
            ),
            StochasticReconLossConfig(coeff=0.5),
            PersistentPGDReconLossConfig(
                coeff=0.5,
                scope=SCScope(),
                optimizer=_sign_optimizer(0.01),
                n_warmup_steps=2,
            ),
        ),
        site_names,
        n_mask_samples=1,
        sampling="continuous",
    )
    (cfg,) = loss_spec.persistent.values()
    assert isinstance(cfg.optimizer, SignPGDConfig)


def test_adam_and_sign_diverge_after_one_step() -> None:
    sources = {"s": jnp.array([0.5, 0.5])}
    grad = {"s": jnp.array([0.2, -0.2])}
    lr = jnp.asarray(0.1)

    adam = AdamPGDConfig(lr_schedule=ScheduleConfig(start_val=0.1, warmup_pct=0.0))
    sign = _sign_optimizer(0.1)
    adam_out, _ = sources_ascend_project(
        sources, grad, init_sources_opt_state(sources, adam), lr, adam
    )
    sign_out, _ = sources_ascend_project(
        sources, grad, init_sources_opt_state(sources, sign), lr, sign
    )
    assert not jnp.allclose(adam_out["s"], sign_out["s"])
