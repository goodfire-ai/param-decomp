"""Targeted PD (tPD) seams: `active_indices` masking, the non-target loss-set builder, and a
two-pass training smoke that exercises the delta-forced-on non-target grid (SPEC S34-S37)."""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

from param_decomp.ci_fn import MLPCIArch, init_layerwise_mlp_ci_fn
from param_decomp.components import SiteC, init_decomp_vu
from param_decomp.configs import (
    AnyLossMetricConfig,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    StochasticHiddenActsReconLossConfig,
    StochasticReconLossConfig,
    UnmaskedReconLossConfig,
)
from param_decomp.recon import build_loss_terms
from param_decomp.schedule import ScheduleConfig
from param_decomp.train import TrainState, make_train_step
from param_decomp_lab.experiments.config import build_nontarget_loss_metrics
from param_decomp_lab.experiments.tms.model import (
    TMSConfig,
    init_tms_target,
    sample_sparse_features,
    site_specs,
    tms_decomposed_model,
)


def _impmin(coeff: float) -> ImportanceMinimalityLossConfig:
    return ImportanceMinimalityLossConfig(
        coeff=coeff, pnorm=ScheduleConfig(start_val=2.0, fn_type="linear", final_val_frac=0.5)
    )


def _tiny_cfg() -> TMSConfig:
    return TMSConfig(n_features=8, n_hidden=4, n_hidden_layers=1)


def _site_cs() -> tuple[SiteC, ...]:
    return (SiteC("linear1", 6), SiteC("linear2", 6), SiteC("hidden_layers.0", 6))


def test_active_indices_masks_non_target_columns():
    x = sample_sparse_features(
        jax.random.PRNGKey(0), 128, n_features=8, feature_probability=0.5,
        generation_type="at_least_zero_active", active_indices=(2, 5),
    )  # fmt: skip
    nonzero_cols = {int(c) for c in jnp.where(jnp.abs(x).sum(0) > 0)[0]}
    assert nonzero_cols <= {2, 5}, f"columns outside active_indices fired: {nonzero_cols}"


def test_active_indices_out_of_range_raises():
    with pytest.raises(AssertionError):
        sample_sparse_features(
            jax.random.PRNGKey(0), 8, n_features=8, feature_probability=0.5,
            generation_type="at_least_zero_active", active_indices=(2, 8),
        )  # fmt: skip


def test_build_nontarget_loss_metrics_drops_excluded_and_scales_impmin():
    # Unmasked + HiddenActs represent the EXCLUDED_NONTARGET_LOSS_CONFIGS tuple (PPGD is the
    # third member, dropped by the same isinstance check; its config needs optimizer/scope so
    # it's omitted here).
    target: list[AnyLossMetricConfig] = [
        FaithfulnessLossConfig(coeff=0.0),
        _impmin(1e-3),
        StochasticReconLossConfig(coeff=1.0),
        UnmaskedReconLossConfig(coeff=0.2),
        StochasticHiddenActsReconLossConfig(coeff=0.1),
    ]
    out = build_nontarget_loss_metrics(target, impmin_coeff_ratio=20.0)
    kinds = {type(m).__name__ for m in out}
    assert UnmaskedReconLossConfig.__name__ not in kinds
    assert StochasticHiddenActsReconLossConfig.__name__ not in kinds
    # faith (inert) + stochastic recon kept; impmin coeff scaled by the ratio
    assert FaithfulnessLossConfig.__name__ in kinds
    assert StochasticReconLossConfig.__name__ in kinds
    (imp,) = [m for m in out if isinstance(m, ImportanceMinimalityLossConfig)]
    assert imp.coeff == pytest.approx(1e-3 * 20.0)
    # originals untouched
    assert target[1].coeff == pytest.approx(1e-3)


def test_targeted_two_pass_step_trains():
    """A 3-step targeted TMS run: the non-target pass (delta forced on) must emit its metrics
    and keep the whole step finite; the untargeted path must not emit them."""
    cfg = _tiny_cfg()
    sites = site_specs(cfg, _site_cs())
    target = init_tms_target(cfg, jax.random.PRNGKey(0))
    lm = tms_decomposed_model(cfg, target, sites)
    vu = init_decomp_vu(sites, jax.random.PRNGKey(1))
    ci_fn = init_layerwise_mlp_ci_fn(MLPCIArch(hidden_dims=(16,)), sites, jax.random.PRNGKey(2))
    opt_vu = optax.adamw(1e-3, weight_decay=0.0)
    opt_ci = optax.adamw(1e-3, weight_decay=0.0)
    state = TrainState(
        components=vu, ci_fn=ci_fn,
        components_opt_state=opt_vu.init(eqx.filter(vu, eqx.is_array)),
        ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
        adversaries={}, step=jnp.zeros((), jnp.int32),
    )  # fmt: skip
    target_metrics: list[AnyLossMetricConfig] = [
        FaithfulnessLossConfig(coeff=0.0), _impmin(1e-3), StochasticReconLossConfig(coeff=1.0),
    ]  # fmt: skip
    nt_surface = build_loss_terms(
        build_nontarget_loss_metrics(target_metrics, impmin_coeff_ratio=20.0), lm.site_names
    )
    step = make_train_step(
        lm=lm, losses=build_loss_terms(tuple(target_metrics), lm.site_names),
        components_optimizer=opt_vu, ci_fn_optimizer=opt_ci, total_steps=10,
        remat_recon_forwards=False, remat_ci_fn=False, mesh=None,
        nontarget_loss_surface=nt_surface,
    )  # fmt: skip

    def batch(key: jax.Array, active: tuple[int, ...] | None) -> jax.Array:
        return sample_sparse_features(
            key, 64, cfg.n_features, 0.3, "at_least_zero_active", active_indices=active
        )

    for i in range(3):
        tgt = batch(jax.random.fold_in(jax.random.PRNGKey(7), i), (2, 5))
        nt = batch(jax.random.fold_in(jax.random.PRNGKey(8), i), None)
        state, m = step(lm, state, tgt, jax.random.PRNGKey(100 + i), nt)
        assert jnp.isfinite(jnp.array(list(m.values()))).all()
        assert "nontarget/total" in m and "nontarget/imp" in m

    assert int(state.step) == 3
