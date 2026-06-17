"""CPU tests for the TMS target + layerwise-MLP CI fn over the generic positionless
(`leading_axes=()`) core.

Covers the `DecomposedModel` contract (clean == all-frozen masked forward, masked
identity, MSE recon), the MLP CI fn (`expects_axes=()`, per-site logits), the full SPEC
step trains, and the ground-truth target-CI eval — including an end-to-end
pretrain → decompose → recovers-identity-structure validation on a tiny 5→2 TMS.
"""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

from jax_single_pool.ci_fn import CIValues
from jax_single_pool.ci_fn_mlp import MLPCIArch, init_layerwise_mlp_ci_fn
from jax_single_pool.llama8b import DecompVU, init_decomp_vu
from jax_single_pool.lm import DecomposedModel, SiteC, SiteSpec
from jax_single_pool.recon import build_recon_terms
from jax_single_pool.tms import (
    TMSConfig,
    TMSTarget,
    canonical_site_cs,
    identity_ci_error,
    init_tms_target,
    pretrain_tms_target,
    sample_sparse_features,
    single_feature_ci,
    site_specs,
    tms_decomposed_model,
    tms_mse,
)
from jax_single_pool.train import TrainState, make_faith_warmup_step, make_train_step
from param_decomp_config.losses import (
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    StochasticReconLayerwiseLossConfig,
    StochasticReconLossConfig,
)


def _tiny_cfg() -> TMSConfig:
    return TMSConfig(n_features=5, n_hidden=2)


def _site_cs() -> tuple[SiteC, ...]:
    return (SiteC("linear1", 8), SiteC("linear2", 6))


def test_canonical_order_and_dims():
    cfg = _tiny_cfg()
    # canonical order is linear1 then linear2 regardless of input order
    assert canonical_site_cs((SiteC("linear2", 6), SiteC("linear1", 8))) == (
        SiteC("linear1", 8),
        SiteC("linear2", 6),
    )
    with pytest.raises(AssertionError):
        canonical_site_cs((SiteC("linear1", 8),))  # both sites required

    specs = site_specs(cfg, _site_cs())
    dims = {s.name: (s.d_in, s.d_out, s.C) for s in specs}
    # right-mult orientation: linear1 (n_features -> n_hidden), linear2 (n_hidden -> n_features)
    assert dims["linear1"] == (5, 2, 8)
    assert dims["linear2"] == (2, 5, 6)


def test_leading_axes_empty_and_ci_expects_axes_match():
    cfg = _tiny_cfg()
    sites = site_specs(cfg, _site_cs())
    lm = tms_decomposed_model(cfg, sites)
    ci_fn = init_layerwise_mlp_ci_fn(MLPCIArch(hidden_dims=(16,)), sites, jax.random.PRNGKey(0))
    assert lm.leading_axes == ()  # positionless target
    assert ci_fn.expects_axes == ()  # MLP CI fn declares no position axes
    assert ci_fn.expects_axes == lm.leading_axes


def test_clean_path_and_masked_identity():
    cfg = _tiny_cfg()
    sites = site_specs(cfg, _site_cs())
    lm = tms_decomposed_model(cfg, sites)
    target = init_tms_target(cfg, jax.random.PRNGKey(0))
    vu = init_decomp_vu(sites, jax.random.PRNGKey(1))
    b = 7
    x = sample_sparse_features(
        jax.random.PRNGKey(2), b, cfg.n_features, 0.3, "at_least_zero_active"
    )

    clean = lm.clean_output(target, x)
    assert clean.shape == (b, cfg.n_features)
    assert jnp.all(clean >= 0.0), "TMS output is post-ReLU, non-negative"

    # SPEC S2: live=() is the exact frozen path.
    none_masked = lm.masked_output(target, vu, x, {}, {}, None, (), True)
    assert jnp.array_equal(clean, none_masked), "live=() must be the exact frozen path"

    # All-live, masks=1, delta=1 reconstructs the frozen path up to decomposition rounding.
    names = lm.site_names
    ones_masks = {s.name: jnp.ones((b, s.C)) for s in lm.sites}
    ones_delta = {s: jnp.ones((b,)) for s in names}
    full = lm.masked_output(target, vu, x, ones_masks, ones_delta, None, names, True)
    assert jnp.allclose(clean, full, atol=1e-4), "mask=1 identity drifted"

    # site_inputs: linear1 reads x, linear2 reads frozen linear1(x).
    site_in = lm.site_inputs(target, x)
    assert set(site_in) == set(names)
    assert jnp.array_equal(site_in["linear1"], x)
    assert site_in["linear1"].shape == (b, cfg.n_features)
    assert site_in["linear2"].shape == (b, cfg.n_hidden)
    assert jnp.allclose(site_in["linear2"], x @ target.W1.T, atol=1e-5)

    deltas = lm.weight_deltas(target, vu)
    assert deltas["linear1"].shape == (cfg.n_hidden, cfg.n_features)
    assert deltas["linear2"].shape == (cfg.n_features, cfg.n_hidden)
    assert all(v.dtype == jnp.float32 for v in deltas.values())


def test_zero_masking_one_site_changes_output():
    cfg = _tiny_cfg()
    sites = site_specs(cfg, _site_cs())
    lm = tms_decomposed_model(cfg, sites)
    target = init_tms_target(cfg, jax.random.PRNGKey(0))
    vu = init_decomp_vu(sites, jax.random.PRNGKey(1))
    b = 7
    x = sample_sparse_features(
        jax.random.PRNGKey(2), b, cfg.n_features, 0.5, "at_least_zero_active"
    )
    clean = lm.clean_output(target, x)
    C = {s.name: s.C for s in sites}["linear1"]
    ablated = lm.masked_output(
        target, vu, x, {"linear1": jnp.zeros((b, C))}, {"linear1": jnp.zeros((b,))},
        None, ("linear1",), True,
    )  # fmt: skip
    assert not jnp.allclose(clean, ablated, atol=1e-5), "ablating linear1 did nothing"


def test_mlp_ci_fn_per_site_logits_and_values():
    cfg = _tiny_cfg()
    sites = site_specs(cfg, _site_cs())
    target = init_tms_target(cfg, jax.random.PRNGKey(0))
    lm = tms_decomposed_model(cfg, sites)
    ci_fn = init_layerwise_mlp_ci_fn(MLPCIArch(hidden_dims=(16,)), sites, jax.random.PRNGKey(3))
    b = 7
    x = sample_sparse_features(
        jax.random.PRNGKey(2), b, cfg.n_features, 0.3, "at_least_zero_active"
    )
    inputs = lm.site_inputs(target, x)
    values = ci_fn(inputs)
    assert isinstance(values, CIValues)
    assert values.lower["linear1"].shape == (b, 8)
    assert values.lower["linear2"].shape == (b, 6)
    # lower_leaky is clamped to [0,1]
    for v in values.lower.values():
        assert float(v.min()) >= 0.0 and float(v.max()) <= 1.0


def test_tms_mse_matches_hand_computed():
    a = jax.random.normal(jax.random.PRNGKey(0), (4, 5))
    b = jax.random.normal(jax.random.PRNGKey(1), (4, 5))
    assert jnp.allclose(tms_mse(a, b), jnp.mean((a - b) ** 2))


def test_recon_loss_fn_is_mse_on_the_model():
    cfg = _tiny_cfg()
    lm = tms_decomposed_model(cfg, site_specs(cfg, _site_cs()))
    assert lm.recon_loss_fn is tms_mse


def _loss_metrics():
    return (
        FaithfulnessLossConfig(coeff=1e3),
        ImportanceMinimalityLossConfig(
            coeff=3e-3,
            pnorm=1.0,
            beta=0.0,
            p_anneal_start_frac=0.0,
            p_anneal_final_p=1.0,
            p_anneal_end_frac=1.0,
        ),  # fmt: skip
        StochasticReconLossConfig(coeff=1.0),
        StochasticReconLayerwiseLossConfig(coeff=1.0),
    )


def _make_state_and_step(
    cfg: TMSConfig, sites: tuple[SiteSpec, ...], total_steps: int
) -> tuple[DecomposedModel, TrainState, Callable[..., tuple[TrainState, dict[str, jax.Array]]]]:
    lm = tms_decomposed_model(cfg, sites)
    vu = init_decomp_vu(sites, jax.random.PRNGKey(1))
    ci_fn = init_layerwise_mlp_ci_fn(MLPCIArch(hidden_dims=(16,)), sites, jax.random.PRNGKey(2))
    opt_vu = optax.chain(optax.clip_by_global_norm(0.01), optax.adamw(1e-3, weight_decay=0.0))
    opt_ci = optax.adamw(1e-3, weight_decay=0.0)
    state = TrainState(
        components=vu, ci_fn=ci_fn,
        components_opt_state=opt_vu.init(eqx.filter(vu, eqx.is_array)),
        ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
        sources={}, sources_opt_state={}, step=jnp.zeros((), jnp.int32),
    )  # fmt: skip
    loss_spec = build_recon_terms(
        _loss_metrics(), lm.site_names, n_mask_samples=1, sampling="continuous"
    )
    step = make_train_step(
        lm=lm, loss_spec=loss_spec, components_optimizer=opt_vu, ci_fn_optimizer=opt_ci,
        total_steps=total_steps, remat_recon_forwards=False, mesh=None,
    )  # fmt: skip
    return lm, state, step


def test_step_trains_positionless_no_persistent_sources():
    cfg = _tiny_cfg()
    sites = site_specs(cfg, _site_cs())
    target = init_tms_target(cfg, jax.random.PRNGKey(0))
    lm, state, step = _make_state_and_step(cfg, sites, total_steps=20)
    losses = []
    for i in range(6):
        x = sample_sparse_features(
            jax.random.fold_in(jax.random.PRNGKey(99), i), 64, cfg.n_features, 0.1,
            "at_least_zero_active",
        )  # fmt: skip
        state, m = step(state, target, x, jax.random.PRNGKey(100 + i))
        losses.append({k: float(v) for k, v in m.items()})
    assert all(jnp.isfinite(jnp.array(list(m.values()))).all() for m in losses)
    assert int(state.step) == 6
    # no persistent sources for the TMS stochastic configs
    assert state.sources == {}
    # fp32 masters preserved
    assert isinstance(state.components, DecompVU)
    for V, U in state.components.vu.values():
        assert V.dtype == jnp.float32 and U.dtype == jnp.float32


def test_faith_warmup_decreases_faith():
    cfg = _tiny_cfg()
    sites = site_specs(cfg, _site_cs())
    target = init_tms_target(cfg, jax.random.PRNGKey(0))
    lm = tms_decomposed_model(cfg, sites)
    vu = init_decomp_vu(sites, jax.random.PRNGKey(1))
    opt = optax.adamw(1e-2, weight_decay=0.0)
    wstep = make_faith_warmup_step(lm, opt)
    ostate = opt.init(eqx.filter(vu, eqx.is_array))
    first_loss = None
    loss = None
    for _ in range(40):
        vu, ostate, loss = wstep(vu, ostate, target)
        first_loss = float(loss) if first_loss is None else first_loss
    assert first_loss is not None and loss is not None
    assert float(loss) < first_loss * 0.9, (first_loss, float(loss))


def test_identity_ci_error_perfect_and_imperfect():
    # A clean identity (5 features, 5 cols) -> zero error after Hungarian.
    perfect = jnp.eye(5)
    assert identity_ci_error(perfect, tolerance=0.1) == 0
    # A permuted identity is still zero (Hungarian recovers the assignment).
    permuted = jnp.eye(5)[:, jnp.array([2, 0, 4, 1, 3])]
    assert identity_ci_error(permuted, tolerance=0.1) == 0
    # An all-zeros CI -> every diagonal is missing (5 on-diagonal errors).
    assert identity_ci_error(jnp.zeros((5, 5)), tolerance=0.1) == 5
    # More components than features: extra dead columns don't add error.
    wide = jnp.concatenate([jnp.eye(5), jnp.zeros((5, 3))], axis=1)
    assert identity_ci_error(wide, tolerance=0.1) == 0


def _recovery_loss_metrics():
    """The 5-2/40-10 TMS config losses + a unit-coeff FaithfulnessLoss (faith is pinned
    by the warmup; a small standing coeff keeps V≈W without dominating the CI shaping)."""
    return (
        FaithfulnessLossConfig(coeff=1.0),
        ImportanceMinimalityLossConfig(
            coeff=3e-3,
            pnorm=1.0,
            beta=0.0,
            p_anneal_start_frac=0.0,
            p_anneal_final_p=1.0,
            p_anneal_end_frac=1.0,
        ),  # fmt: skip
        StochasticReconLossConfig(coeff=1.0),
        StochasticReconLayerwiseLossConfig(coeff=1.0),
    )


def _faith_warmed_state(
    lm: DecomposedModel,
    sites: tuple[SiteSpec, ...],
    target: TMSTarget,
    total_steps: int,
    warmup_steps: int,
) -> tuple[TrainState, Callable[..., tuple[TrainState, dict[str, jax.Array]]]]:
    """Build a train state, run faith warmup (TMS needs it — the from-scratch V/U start
    far from `W`), then return state + step factory."""
    vu = init_decomp_vu(sites, jax.random.PRNGKey(1))
    ci_fn = init_layerwise_mlp_ci_fn(MLPCIArch(hidden_dims=(16,)), sites, jax.random.PRNGKey(2))
    warm_opt = optax.adamw(1e-2, weight_decay=0.0)
    wstep = make_faith_warmup_step(lm, warm_opt)
    warm_state = warm_opt.init(eqx.filter(vu, eqx.is_array))
    for _ in range(warmup_steps):
        vu, warm_state, _ = wstep(vu, warm_state, target)
    opt_vu = optax.chain(optax.clip_by_global_norm(0.01), optax.adamw(1e-3, weight_decay=0.0))
    opt_ci = optax.adamw(1e-3, weight_decay=0.0)
    state = TrainState(
        components=vu, ci_fn=ci_fn,
        components_opt_state=opt_vu.init(eqx.filter(vu, eqx.is_array)),
        ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
        sources={}, sources_opt_state={}, step=jnp.zeros((), jnp.int32),
    )  # fmt: skip
    loss_spec = build_recon_terms(_recovery_loss_metrics(), lm.site_names, 1, "continuous")
    step = make_train_step(
        lm=lm, loss_spec=loss_spec, components_optimizer=opt_vu, ci_fn_optimizer=opt_ci,
        total_steps=total_steps, remat_recon_forwards=False, mesh=None,
    )  # fmt: skip
    return state, step


@pytest.mark.slow
def test_end_to_end_pretrain_decompose_recovers_identity():
    """The proof the port is correct end-to-end: pretrain a 5->5 TMS from scratch (the
    non-superposed regime, where the ground truth is a clean per-feature decomposition),
    run the full PD decomposition over the unified core, and show the recovered CI is the
    IDENTITY up to permutation — zero `IdentityCIError`. This is the recovers-structure
    gate; it exercises pretrain + faith warmup + the generic step + MSE recon + the MLP
    CI fn + the target-CI eval.

    (n_hidden < n_features — true superposition, e.g. the 5->2 wrapper config — trains and
    drives recon down the same way, but the 2-D bottleneck genuinely superposes features
    so the per-site identity is only partially recoverable from the vector-input MLP; the
    n_hidden==n_features case is the unambiguous correctness proof.)"""
    cfg = TMSConfig(n_features=5, n_hidden=5)
    sites = site_specs(cfg, (SiteC("linear1", 5), SiteC("linear2", 5)))
    target = pretrain_tms_target(
        cfg, feature_probability=0.05, generation_type="at_least_zero_active",
        steps=5000, batch_size=2048, lr=1e-2, seed=0,
    )  # fmt: skip
    x = sample_sparse_features(jax.random.PRNGKey(7), 1024, 5, 0.05, "at_least_zero_active")
    lm = tms_decomposed_model(cfg, sites)
    recon = jnp.mean((jnp.abs(x) - lm.clean_output(target, x)) ** 2)
    assert float(recon) < 0.05, f"pretrained TMS recon too high: {recon}"

    state, step = _faith_warmed_state(lm, sites, target, total_steps=8000, warmup_steps=200)
    data_key = jax.random.PRNGKey(123)
    totals: list[float] = []
    for i in range(8000):
        x = sample_sparse_features(
            jax.random.fold_in(data_key, i), 2048, 5, 0.05, "at_least_zero_active"
        )
        state, m = step(state, target, x, jax.random.fold_in(jax.random.PRNGKey(321), i))
        totals.append(float(m["total"]))
    assert totals[-1] < totals[0], (totals[0], totals[-1])

    ci_lower = single_feature_ci(lm, target, state.ci_fn, n_features=5)
    err1 = identity_ci_error(ci_lower["linear1"], tolerance=0.2)
    err2 = identity_ci_error(ci_lower["linear2"], tolerance=0.2)
    assert err1 == 0, (
        f"linear1 did not recover identity (err={err1}):\n{jnp.round(ci_lower['linear1'], 2)}"
    )
    assert err2 == 0, (
        f"linear2 did not recover identity (err={err2}):\n{jnp.round(ci_lower['linear2'], 2)}"
    )
