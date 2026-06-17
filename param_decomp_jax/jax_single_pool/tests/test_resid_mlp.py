"""CPU tests for the ResidualMLP target + layerwise-MLP CI fn over the generic
positionless (`leading_axes=()`) core.

Covers the `DecomposedModel` contract (clean == all-frozen masked forward, masked
identity, MSE recon, residual accumulation), the reused MLP CI fn, the full SPEC step
trains, and the ground-truth target-CI eval — including an end-to-end pretrain →
decompose → recovers-identity-structure validation on a tiny single-layer ResidMLP.
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
from jax_single_pool.resid_mlp import (
    ResidMLPConfig,
    ResidMLPTarget,
    canonical_site_cs,
    identity_ci_error,
    init_resid_mlp_target,
    pretrain_resid_mlp_target,
    readoff_labels,
    resid_mlp_decomposed_model,
    resid_mlp_mse,
    sample_sparse_features,
    single_feature_ci,
    site_specs,
)
from jax_single_pool.train import TrainState, make_faith_warmup_step, make_train_step
from param_decomp_config.losses import (
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    StochasticReconLayerwiseLossConfig,
    StochasticReconLossConfig,
)


def _tiny_cfg(n_layers: int = 1) -> ResidMLPConfig:
    return ResidMLPConfig(
        n_features=5,
        d_embed=5,
        d_mlp=8,
        n_layers=n_layers,
        act_fn_name="relu",
        in_bias=False,
        out_bias=False,
        fixed_identity_embedding=False,
    )


def _site_cs(n_layers: int = 1) -> tuple[SiteC, ...]:
    return tuple(
        SiteC(f"layers.{i}.{kind}", C)
        for i in range(n_layers)
        for kind, C in (("mlp_in", 6), ("mlp_out", 7))
    )


def test_canonical_order_and_dims():
    cfg = _tiny_cfg(n_layers=2)
    scrambled = (
        SiteC("layers.1.mlp_out", 7),
        SiteC("layers.0.mlp_out", 7),
        SiteC("layers.1.mlp_in", 6),
        SiteC("layers.0.mlp_in", 6),
    )
    ordered = canonical_site_cs(scrambled)
    assert [s.name for s in ordered] == [
        "layers.0.mlp_in",
        "layers.0.mlp_out",
        "layers.1.mlp_in",
        "layers.1.mlp_out",
    ]

    specs = site_specs(cfg, _site_cs(n_layers=2))
    dims = {s.name: (s.d_in, s.d_out, s.C) for s in specs}
    # right-mult orientation: mlp_in (d_embed -> d_mlp), mlp_out (d_mlp -> d_embed)
    assert dims["layers.0.mlp_in"] == (5, 8, 6)
    assert dims["layers.0.mlp_out"] == (8, 5, 7)


def test_site_specs_requires_both_sites_per_layer():
    cfg = _tiny_cfg(n_layers=1)
    with pytest.raises(AssertionError):
        site_specs(cfg, (SiteC("layers.0.mlp_in", 6),))  # missing mlp_out


def test_leading_axes_empty_and_ci_expects_axes_match():
    cfg = _tiny_cfg()
    sites = site_specs(cfg, _site_cs())
    lm = resid_mlp_decomposed_model(cfg, sites)
    ci_fn = init_layerwise_mlp_ci_fn(MLPCIArch(hidden_dims=(16,)), sites, jax.random.PRNGKey(0))
    assert lm.leading_axes == ()  # positionless target
    assert ci_fn.expects_axes == ()
    assert ci_fn.expects_axes == lm.leading_axes


def test_clean_path_and_masked_identity():
    cfg = _tiny_cfg(n_layers=2)
    sites = site_specs(cfg, _site_cs(n_layers=2))
    lm = resid_mlp_decomposed_model(cfg, sites)
    target = init_resid_mlp_target(cfg, jax.random.PRNGKey(0))
    vu = init_decomp_vu(sites, jax.random.PRNGKey(1))
    b = 7
    x = sample_sparse_features(
        jax.random.PRNGKey(2), b, cfg.n_features, 0.5, "at_least_zero_active"
    )
    resid = x @ target.W_E

    clean = lm.clean_output(target, resid)
    assert clean.shape == (b, cfg.n_features)

    # SPEC S2: live=() is the exact frozen path.
    none_masked = lm.masked_output(target, vu, resid, {}, {}, None, (), True)
    assert jnp.array_equal(clean, none_masked), "live=() must be the exact frozen path"

    # All-live, masks=1, delta=1 reconstructs the frozen path up to decomposition rounding.
    names = lm.site_names
    ones_masks = {s.name: jnp.ones((b, s.C)) for s in lm.sites}
    ones_delta = {s: jnp.ones((b,)) for s in names}
    full = lm.masked_output(target, vu, resid, ones_masks, ones_delta, None, names, True)
    assert jnp.allclose(clean, full, atol=1e-4), "mask=1 identity drifted"

    # site_inputs: mlp_in reads the residual entering its layer, mlp_out the post-act hidden.
    site_in = lm.site_inputs(target, resid)
    assert set(site_in) == set(names)
    assert jnp.array_equal(site_in["layers.0.mlp_in"], resid)
    assert site_in["layers.0.mlp_in"].shape == (b, cfg.d_embed)
    assert site_in["layers.0.mlp_out"].shape == (b, cfg.d_mlp)
    expected_hidden0 = jax.nn.relu(resid @ target.layers[0].W_in.T)
    assert jnp.allclose(site_in["layers.0.mlp_out"], expected_hidden0, atol=1e-5)

    deltas = lm.weight_deltas(target, vu)
    assert deltas["layers.0.mlp_in"].shape == (cfg.d_mlp, cfg.d_embed)
    assert deltas["layers.0.mlp_out"].shape == (cfg.d_embed, cfg.d_mlp)
    assert all(v.dtype == jnp.float32 for v in deltas.values())


def test_zero_masking_one_site_changes_output():
    cfg = _tiny_cfg()
    sites = site_specs(cfg, _site_cs())
    lm = resid_mlp_decomposed_model(cfg, sites)
    target = init_resid_mlp_target(cfg, jax.random.PRNGKey(0))
    vu = init_decomp_vu(sites, jax.random.PRNGKey(1))
    b = 7
    x = sample_sparse_features(
        jax.random.PRNGKey(2), b, cfg.n_features, 0.8, "at_least_zero_active"
    )
    resid = x @ target.W_E
    clean = lm.clean_output(target, resid)
    C = {s.name: s.C for s in sites}["layers.0.mlp_out"]
    ablated = lm.masked_output(
        target, vu, resid, {"layers.0.mlp_out": jnp.zeros((b, C))},
        {"layers.0.mlp_out": jnp.zeros((b,))}, None, ("layers.0.mlp_out",), True,
    )  # fmt: skip
    assert not jnp.allclose(clean, ablated, atol=1e-5), "ablating mlp_out did nothing"


def test_residual_accumulation_across_layers():
    # Two layers must accumulate into the residual: the 2-layer clean output differs from a
    # 1-layer one with the same first layer.
    cfg2 = _tiny_cfg(n_layers=2)
    target2 = init_resid_mlp_target(cfg2, jax.random.PRNGKey(0))
    lm2 = resid_mlp_decomposed_model(cfg2, site_specs(cfg2, _site_cs(n_layers=2)))
    x = sample_sparse_features(
        jax.random.PRNGKey(2), 4, cfg2.n_features, 1.0, "at_least_zero_active"
    )
    resid = x @ target2.W_E
    one_layer_target = eqx.tree_at(lambda t: t.layers, target2, target2.layers[:1])
    cfg1 = _tiny_cfg(n_layers=1)
    lm1 = resid_mlp_decomposed_model(cfg1, site_specs(cfg1, _site_cs(n_layers=1)))
    assert not jnp.allclose(
        lm2.clean_output(target2, resid), lm1.clean_output(one_layer_target, resid), atol=1e-4
    )


def test_mlp_ci_fn_per_site_logits_and_values():
    cfg = _tiny_cfg()
    sites = site_specs(cfg, _site_cs())
    target = init_resid_mlp_target(cfg, jax.random.PRNGKey(0))
    lm = resid_mlp_decomposed_model(cfg, sites)
    ci_fn = init_layerwise_mlp_ci_fn(MLPCIArch(hidden_dims=(16,)), sites, jax.random.PRNGKey(3))
    b = 7
    x = sample_sparse_features(
        jax.random.PRNGKey(2), b, cfg.n_features, 0.3, "at_least_zero_active"
    )
    inputs = lm.site_inputs(target, x @ target.W_E)
    values = ci_fn(inputs)
    assert isinstance(values, CIValues)
    assert values.lower["layers.0.mlp_in"].shape == (b, 6)
    assert values.lower["layers.0.mlp_out"].shape == (b, 7)
    for v in values.lower.values():
        assert float(v.min()) >= 0.0 and float(v.max()) <= 1.0


def test_resid_mlp_mse_matches_hand_computed():
    a = jax.random.normal(jax.random.PRNGKey(0), (4, 5))
    b = jax.random.normal(jax.random.PRNGKey(1), (4, 5))
    assert jnp.allclose(resid_mlp_mse(a, b), jnp.mean((a - b) ** 2))


def test_recon_loss_fn_is_mse_on_the_model():
    cfg = _tiny_cfg()
    lm = resid_mlp_decomposed_model(cfg, site_specs(cfg, _site_cs()))
    assert lm.recon_loss_fn is resid_mlp_mse


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
    cfg: ResidMLPConfig, sites: tuple[SiteSpec, ...], total_steps: int
) -> tuple[DecomposedModel, TrainState, Callable[..., tuple[TrainState, dict[str, jax.Array]]]]:
    lm = resid_mlp_decomposed_model(cfg, sites)
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
    target = init_resid_mlp_target(cfg, jax.random.PRNGKey(0))
    lm, state, step = _make_state_and_step(cfg, sites, total_steps=20)
    losses = []
    for i in range(6):
        x = sample_sparse_features(
            jax.random.fold_in(jax.random.PRNGKey(99), i), 64, cfg.n_features, 0.1,
            "at_least_zero_active",
        )  # fmt: skip
        state, m = step(state, target, x @ target.W_E, jax.random.PRNGKey(100 + i))
        losses.append({k: float(v) for k, v in m.items()})
    assert all(jnp.isfinite(jnp.array(list(m.values()))).all() for m in losses)
    assert int(state.step) == 6
    assert state.sources == {}  # no persistent sources for the stochastic configs
    assert isinstance(state.components, DecompVU)
    for V, U in state.components.vu.values():
        assert V.dtype == jnp.float32 and U.dtype == jnp.float32


def test_faith_warmup_decreases_faith():
    cfg = _tiny_cfg()
    sites = site_specs(cfg, _site_cs())
    target = init_resid_mlp_target(cfg, jax.random.PRNGKey(0))
    lm = resid_mlp_decomposed_model(cfg, sites)
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
    perfect = jnp.eye(5)
    assert identity_ci_error(perfect, tolerance=0.1) == 0
    permuted = jnp.eye(5)[:, jnp.array([2, 0, 4, 1, 3])]
    assert identity_ci_error(permuted, tolerance=0.1) == 0
    assert identity_ci_error(jnp.zeros((5, 5)), tolerance=0.1) == 5
    wide = jnp.concatenate([jnp.eye(5), jnp.zeros((5, 3))], axis=1)
    assert identity_ci_error(wide, tolerance=0.1) == 0


def test_pretrain_drives_readoff_recon_down():
    cfg = _tiny_cfg(n_layers=1)
    target = pretrain_resid_mlp_target(
        cfg, feature_probability=0.1, generation_type="at_least_zero_active",
        steps=400, batch_size=512, lr=1e-2, seed=0,
    )  # fmt: skip
    lm = resid_mlp_decomposed_model(cfg, site_specs(cfg, _site_cs()))
    x = sample_sparse_features(
        jax.random.PRNGKey(7), 512, cfg.n_features, 0.1, "at_least_zero_active"
    )
    out = lm.clean_output(target, x @ target.W_E)
    recon = jnp.mean((out - readoff_labels(target, x)) ** 2)
    assert float(recon) < 0.05, f"pretrained ResidMLP read-off recon too high: {recon}"


def _recovery_loss_metrics():
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
    target: ResidMLPTarget,
    total_steps: int,
    warmup_steps: int,
) -> tuple[TrainState, Callable[..., tuple[TrainState, dict[str, jax.Array]]]]:
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
    """The end-to-end correctness proof: pretrain a single-layer ResidMLP (d_mlp == n_features
    == d_embed, so the ground truth is a clean per-feature MLP-in decomposition) on the
    read-off objective, run the full PD decomposition over the unified core, and show the
    recovered `mlp_in` CI is the IDENTITY up to permutation — zero `IdentityCIError`.

    `mlp_out` is a dense read-back (each neuron writes a feature direction into the residual)
    so its CI is not identity-patterned; the `mlp_in` site is the unambiguous structure
    gate. Exercises pretrain + faith warmup + the generic step + MSE recon + the MLP CI fn +
    the target-CI eval, end to end."""
    cfg = ResidMLPConfig(
        n_features=5, d_embed=5, d_mlp=5, n_layers=1, act_fn_name="relu",
        in_bias=False, out_bias=False, fixed_identity_embedding=True,
    )  # fmt: skip
    sites = site_specs(cfg, (SiteC("layers.0.mlp_in", 5), SiteC("layers.0.mlp_out", 5)))
    target = pretrain_resid_mlp_target(
        cfg, feature_probability=0.05, generation_type="at_least_zero_active",
        steps=5000, batch_size=2048, lr=1e-2, seed=0,
    )  # fmt: skip
    lm = resid_mlp_decomposed_model(cfg, sites)
    x = sample_sparse_features(jax.random.PRNGKey(7), 1024, 5, 0.05, "at_least_zero_active")
    recon = jnp.mean((lm.clean_output(target, x @ target.W_E) - readoff_labels(target, x)) ** 2)
    assert float(recon) < 0.05, f"pretrained ResidMLP recon too high: {recon}"

    state, step = _faith_warmed_state(lm, sites, target, total_steps=8000, warmup_steps=200)
    data_key = jax.random.PRNGKey(123)
    totals: list[float] = []
    for i in range(8000):
        x = sample_sparse_features(
            jax.random.fold_in(data_key, i), 2048, 5, 0.05, "at_least_zero_active"
        )
        state, m = step(
            state, target, x @ target.W_E, jax.random.fold_in(jax.random.PRNGKey(321), i)
        )
        totals.append(float(m["total"]))
    assert totals[-1] < totals[0], (totals[0], totals[-1])

    ci_lower = single_feature_ci(lm, target, state.ci_fn, n_features=5)
    err = identity_ci_error(ci_lower["layers.0.mlp_in"], tolerance=0.2)
    assert err == 0, (
        f"mlp_in did not recover identity (err={err}):\n{jnp.round(ci_lower['layers.0.mlp_in'], 2)}"
    )
