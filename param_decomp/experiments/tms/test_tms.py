"""CPU tests for the TMS target + layerwise-MLP CI fn over the generic positionless core.

Covers the `DecomposedModel` contract (mask=1 identity reconstructs the clean forward,
MSE recon), the MLP CI fn (positionless, per-site preactivations), the full SPEC
step trains, and the ground-truth target-CI eval — including an end-to-end
pretrain → decompose → recovers-identity-structure validation on a tiny 5→2 TMS.
"""

import dataclasses
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

from param_decomp.core.ci_fn import CI, LayerwiseMLPCIArch, PlacedCIFn, init_layerwise_mlp_ci_fn
from param_decomp.core.components import ComponentStacks, SiteC, SiteSpec, init_component_stacks
from param_decomp.core.configs import (
    AnyImportanceMinimalityLossConfig,
    AnyLossMetricConfig,
    FaithfulnessLossConfig,
    FrequencyMinimalityConfig,
    ImportanceMinimalityLossConfig,
    NonlinearityLocalityLossConfig,
    StochasticReconLossConfig,
    StochasticReconSubsetLossConfig,
)
from param_decomp.core.faithfulness import faithfulness_loss_for
from param_decomp.core.losses import EmaFrequency, resolve_frequency
from param_decomp.core.model import PlacedModel, site_weight_delta
from param_decomp.core.nonlinearity import (
    Neurons,
)
from param_decomp.core.objective import build_objective
from param_decomp.core.recon import OutputAndHiddenActsReconstruction
from param_decomp.core.recon_eval import FreshPGDReconEval, make_fresh_pgd_eval_step
from param_decomp.core.schedule import Knot, ScheduleConfig
from param_decomp.core.train import (
    Decomposition,
    ForwardSubstrate,
    TrainingItem,
    TrainState,
    make_faith_warmup_step,
    make_train_step,
)
from param_decomp.targets.testing import capture_clean, run_clean, run_masked
from param_decomp.targets.tms import (
    HiddenLayerInit,
    TMSConfig,
    TMSGenerationType,
    TMSTarget,
    canonical_site_cs,
    dense_ci_error,
    hidden_layer_name,
    identity_ci_error,
    init_tms_target,
    pretrain_tms_target,
    sample_sparse_features,
    single_feature_ci,
    site_input_tap_keys,
    site_inputs,
    site_specs,
    tms_decomposed_model,
    tms_mse,
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
    partitions = {site.name: site.nonlinearity_partition for site in specs}
    assert partitions == {
        "linear1": None,
        "linear2": Neurons(),
    }


def test_positionless_and_ci_fn_position_kind_match():
    cfg = _tiny_cfg()
    sites = site_specs(cfg, _site_cs())
    target = init_tms_target(cfg, jax.random.PRNGKey(0))
    model = tms_decomposed_model(cfg, target, sites)
    ci_fn = init_layerwise_mlp_ci_fn(
        LayerwiseMLPCIArch(
            hidden_dims=(16,),
            has_position_axis=False,
            input_names=site_input_tap_keys(tuple(s.name for s in sites)),
        ),
        sites,
        jax.random.PRNGKey(0),
    )
    assert not model.has_position_axis
    assert not ci_fn.has_position_axis


def test_clean_path_and_masked_identity():
    cfg = _tiny_cfg()
    sites = site_specs(cfg, _site_cs())
    target = init_tms_target(cfg, jax.random.PRNGKey(0))
    model = tms_decomposed_model(cfg, target, sites)
    vu = init_component_stacks(sites, jax.random.PRNGKey(1))
    b = 7
    x = sample_sparse_features(
        jax.random.PRNGKey(2), b, cfg.n_features, 0.3, "at_least_zero_active"
    )

    clean = run_clean(model, x)
    assert clean.shape == (b, cfg.n_features)
    assert jnp.all(clean >= 0.0), "TMS output is post-ReLU, non-negative"

    # Masks=1, delta=1 reconstructs the frozen path up to decomposition rounding.
    names = model.site_names
    ones_masks = {s.name: jnp.ones((b, s.C)) for s in model.sites}
    ones_delta = {s: jnp.ones((b,)) for s in names}
    full = run_masked(model, vu, x, ones_masks, ones_delta, None, True, remat=False)
    assert jnp.allclose(clean, full, atol=1e-4), "mask=1 identity drifted"

    # site_inputs: linear1 reads x, linear2 reads frozen linear1(x).
    site_in = site_inputs(target, x)
    assert set(site_in) == set(names)
    assert jnp.array_equal(site_in["linear1"], x)
    assert site_in["linear1"].shape == (b, cfg.n_features)
    assert site_in["linear2"].shape == (b, cfg.n_hidden)
    assert jnp.allclose(site_in["linear2"], x @ target.W1.T, atol=1e-5)

    deltas = model.weight_deltas(vu)
    assert site_weight_delta(deltas, vu, "linear1").shape == (cfg.n_hidden, cfg.n_features)
    assert site_weight_delta(deltas, vu, "linear2").shape == (cfg.n_features, cfg.n_hidden)
    assert all(v.dtype == jnp.float32 for v in deltas.values())
    target_sq_norms = model.target_weight_sq_norms()
    for name, group, slot in vu.site_slots:
        site = vu.site(name)
        delta = site_weight_delta(deltas, vu, name)
        target_weight = delta + (site.V.astype(jnp.float32) @ site.U.astype(jnp.float32)).T
        assert jnp.allclose(target_sq_norms[group][slot], jnp.sum(target_weight**2))


def test_zero_masking_one_site_changes_output():
    cfg = _tiny_cfg()
    sites = site_specs(cfg, _site_cs())
    target = init_tms_target(cfg, jax.random.PRNGKey(0))
    model = tms_decomposed_model(cfg, target, sites)
    vu = init_component_stacks(sites, jax.random.PRNGKey(1))
    b = 7
    x = sample_sparse_features(
        jax.random.PRNGKey(2), b, cfg.n_features, 0.5, "at_least_zero_active"
    )
    clean = run_clean(model, x)
    # linear1 gets mask=0 + delta=0 (fully ablated); linear2 mask=1 + delta=1 (exactly the
    # frozen W), so the change is attributable to linear1 alone.
    fill = {s.name: 0.0 if s.name == "linear1" else 1.0 for s in sites}
    masks = {s.name: jnp.full((b, s.C), fill[s.name]) for s in sites}
    delta_masks = {s.name: jnp.full((b,), fill[s.name]) for s in sites}
    ablated = run_masked(model, vu, x, masks, delta_masks, None, True, remat=False)
    assert not jnp.allclose(clean, ablated, atol=1e-5), "ablating linear1 did nothing"


def test_mlp_ci_fn_per_site_preactivations_and_values():
    cfg = _tiny_cfg()
    sites = site_specs(cfg, _site_cs())
    target = init_tms_target(cfg, jax.random.PRNGKey(0))
    model = tms_decomposed_model(cfg, target, sites)
    ci_fn = init_layerwise_mlp_ci_fn(
        LayerwiseMLPCIArch(
            hidden_dims=(16,),
            has_position_axis=False,
            input_names=site_input_tap_keys(tuple(s.name for s in sites)),
        ),
        sites,
        jax.random.PRNGKey(3),
    )
    b = 7
    x = sample_sparse_features(
        jax.random.PRNGKey(2), b, cfg.n_features, 0.3, "at_least_zero_active"
    )
    inputs = capture_clean(model, x, ci_fn.capture_keys)
    values = ci_fn(inputs, remat=False, placement=None)
    assert isinstance(values, CI)
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
    target = init_tms_target(cfg, jax.random.PRNGKey(0))
    model = tms_decomposed_model(cfg, target, site_specs(cfg, _site_cs()))
    a = jax.random.normal(jax.random.PRNGKey(1), (4, cfg.n_features))
    b = jax.random.normal(jax.random.PRNGKey(2), (4, cfg.n_features))
    assert jnp.array_equal(model.recon_loss_fn(a, b), tms_mse(a, b))


def _loss_metrics():
    return (
        FaithfulnessLossConfig(coeff=1e3),
        ImportanceMinimalityLossConfig(
            coeff=3e-3,
            pnorm=ScheduleConfig.constant(1.0),
        ),
        StochasticReconLossConfig(coeff=1.0),
        StochasticReconSubsetLossConfig(coeff=1.0),
    )


def _make_state_and_step(
    cfg: TMSConfig,
    target: TMSTarget,
    sites: tuple[SiteSpec, ...],
    total_steps: int,
    loss_metrics: tuple[AnyLossMetricConfig, ...],
) -> tuple[PlacedModel, TrainState, Callable[..., tuple[TrainState, dict[str, jax.Array]]]]:
    model = PlacedModel(model=tms_decomposed_model(cfg, target, sites), placement=None)
    vu = init_component_stacks(sites, jax.random.PRNGKey(1))
    ci_fn = init_layerwise_mlp_ci_fn(
        LayerwiseMLPCIArch(
            hidden_dims=(16,),
            has_position_axis=False,
            input_names=site_input_tap_keys(tuple(s.name for s in sites)),
        ),
        sites,
        jax.random.PRNGKey(2),
    )
    opt_vu = optax.chain(optax.clip_by_global_norm(0.01), optax.adamw(1e-3, weight_decay=0.0))
    opt_ci = optax.adamw(1e-3, weight_decay=0.0)
    imp = next(m for m in loss_metrics if isinstance(m, AnyImportanceMinimalityLossConfig))
    freq_role = resolve_frequency(imp.frequency)
    state = TrainState(
        decomposition=Decomposition(components=vu, ci_fn=ci_fn),
        training=TrainingItem(
            components_opt_state=opt_vu.init(eqx.filter(vu, eqx.is_array)),
            ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
            adversaries={},
            freq_ema=freq_role.initial_state(sites)
            if isinstance(freq_role, EmaFrequency)
            else None,
            step=jnp.zeros((), jnp.int32),
        ),
    )
    loss_terms = build_objective(loss_metrics, model.site_names)
    step = make_train_step(
        model_static=model,
        substrate=ForwardSubstrate.of(
            model,
            remat_recon_forwards=False,
            remat_ci_fn=False,
            ci_capture_keys=ci_fn.capture_keys,
            ci_placement=None,
        ),
        objective=loss_terms,
        components_optimizer=opt_vu,
        ci_fn_optimizer=opt_ci,
        total_steps=total_steps,
        faithfulness=faithfulness_loss_for(model.model),
    )
    return model, state, step


def test_step_with_ema_frequency_penalty():
    """The EMA frequency penalty (SPEC S8'') through the real jitted step: the state
    threads, the smoothed penalty starts at the un-smoothed value (debias), and the
    per-site EMA buffers fill in."""
    cfg = _tiny_cfg()
    sites = site_specs(cfg, _site_cs())
    target = init_tms_target(cfg, jax.random.PRNGKey(0))
    metrics_cfg = (
        FaithfulnessLossConfig(coeff=1e3),
        ImportanceMinimalityLossConfig(
            coeff=3e-3,
            pnorm=ScheduleConfig.constant(1.0),
            frequency=FrequencyMinimalityConfig(
                coeff=1e-3, reference_datapoint_count=64, ema_halflife_steps=4.0
            ),
        ),
        StochasticReconLossConfig(coeff=1.0),
    )
    model, state, step = _make_state_and_step(cfg, target, sites, 20, metrics_cfg)
    assert state.training.freq_ema is not None

    def batch(i: int) -> jax.Array:
        return sample_sparse_features(
            jax.random.fold_in(jax.random.PRNGKey(99), i),
            64,
            cfg.n_features,
            0.1,
            "at_least_zero_active",
        )

    state, first = step(model, state, batch(0), jax.random.PRNGKey(100))
    assert jnp.allclose(first["freq"], first["freq_batch"], rtol=1e-4)  # step-0 debias identity
    for i in range(1, 4):
        state, m = step(model, state, batch(i), jax.random.PRNGKey(100 + i))
        assert jnp.isfinite(m["freq"]) and jnp.isfinite(m["freq_batch"])
    ema = state.training.freq_ema
    assert ema is not None
    assert set(ema) == {s.name for s in sites}
    for s_ in sites:
        assert ema[s_.name].shape == (s_.C,)
        assert float(jnp.max(ema[s_.name])) > 0.0


def test_step_with_ema_and_scheduled_frequency_coeff():
    """The merge seam of S8'' with schedulable coefficients: a ramping `frequency.coeff`
    resolves per step via `coeff_at` while the EMA buffers still thread."""
    cfg = _tiny_cfg()
    sites = site_specs(cfg, _site_cs())
    target = init_tms_target(cfg, jax.random.PRNGKey(0))
    ramp = ScheduleConfig(max_val=1e-3, points=(Knot(at=0.0, frac=0.0), Knot(at=1.0, frac=1.0)))
    metrics_cfg = (
        FaithfulnessLossConfig(coeff=1e3),
        ImportanceMinimalityLossConfig(
            coeff=3e-3,
            pnorm=ScheduleConfig.constant(1.0),
            frequency=FrequencyMinimalityConfig(
                coeff=ramp, reference_datapoint_count=64, ema_halflife_steps=4.0
            ),
        ),
        StochasticReconLossConfig(coeff=1.0),
    )
    model, state, step = _make_state_and_step(cfg, target, sites, 20, metrics_cfg)
    assert state.training.freq_ema is not None

    coeff_key = "schedules/coeff/ImportanceMinimalityLoss/frequency"
    seen_coeffs = []
    for i in range(3):
        x = sample_sparse_features(
            jax.random.fold_in(jax.random.PRNGKey(99), i),
            64,
            cfg.n_features,
            0.1,
            "at_least_zero_active",
        )
        state, m = step(model, state, x, jax.random.PRNGKey(100 + i))
        assert jnp.isfinite(m["freq"]) and jnp.isfinite(m["freq_batch"])
        seen_coeffs.append(float(m[coeff_key]))
    assert seen_coeffs[0] == 0.0  # the ramp starts at frac 0
    assert seen_coeffs[2] > seen_coeffs[1] > 0.0
    ema = state.training.freq_ema
    assert ema is not None
    assert all(float(jnp.max(v)) > 0.0 for v in ema.values())


def test_step_refuses_forged_freq_ema_without_frequency_config():
    # Fail-closed (S8''): a freq_ema buffer alongside a config with no frequency term is
    # a state/config mismatch, refused at trace time rather than silently dropped.
    cfg = _tiny_cfg()
    sites = site_specs(cfg, _site_cs())
    target = init_tms_target(cfg, jax.random.PRNGKey(0))
    metrics_cfg = (
        FaithfulnessLossConfig(coeff=1e3),
        ImportanceMinimalityLossConfig(coeff=3e-3, pnorm=ScheduleConfig.constant(1.0)),
        StochasticReconLossConfig(coeff=1.0),
    )
    model, state, step = _make_state_and_step(cfg, target, sites, 20, metrics_cfg)
    forged = TrainState(
        decomposition=state.decomposition,
        training=dataclasses.replace(
            state.training,
            freq_ema={s.name: jnp.zeros((s.C,), jnp.float32) for s in sites},
        ),
    )
    x = sample_sparse_features(
        jax.random.PRNGKey(99), 64, cfg.n_features, 0.1, "at_least_zero_active"
    )
    with pytest.raises(AssertionError, match="S8''"):
        step(model, forged, x, jax.random.PRNGKey(100))


def test_fresh_pgd_eval_runs_on_positionless_tms() -> None:
    cfg = _tiny_cfg()
    sites = site_specs(cfg, _site_cs())
    target = init_tms_target(cfg, jax.random.PRNGKey(0))
    model, state, _ = _make_state_and_step(cfg, target, sites, 20, _loss_metrics())
    batch = sample_sparse_features(
        jax.random.PRNGKey(3), 16, cfg.n_features, 0.3, "at_least_zero_active"
    )
    eval_step = make_fresh_pgd_eval_step(
        model,
        FreshPGDReconEval(
            n_steps=2,
            step_size=0.1,
            reconstruction=OutputAndHiddenActsReconstruction(
                coeff=1.0, points=(f"{model.site_names[0]}.out",)
            ),
        ),
        state.decomposition.ci_fn.capture_keys,
    )

    value = eval_step(
        model,
        state.decomposition.components,
        PlacedCIFn(fn=state.decomposition.ci_fn, placement=None),
        batch,
        jax.random.PRNGKey(4),
    )

    assert value.shape == ()
    assert jnp.isfinite(value)
    assert value >= 0.0


def test_step_trains_positionless_no_persistent_sources():
    cfg = _tiny_cfg()
    sites = site_specs(cfg, _site_cs())
    target = init_tms_target(cfg, jax.random.PRNGKey(0))
    model, state, step = _make_state_and_step(cfg, target, sites, 20, _loss_metrics())
    losses = []
    for i in range(6):
        x = sample_sparse_features(
            jax.random.fold_in(jax.random.PRNGKey(99), i),
            64,
            cfg.n_features,
            0.1,
            "at_least_zero_active",
        )
        state, m = step(model, state, x, jax.random.PRNGKey(100 + i))
        losses.append({k: float(v) for k, v in m.items()})
    assert all(jnp.isfinite(jnp.array(list(m.values()))).all() for m in losses)
    assert int(state.training.step) == 6
    # no persistent sources for the TMS stochastic configs
    assert state.training.adversaries == {}
    # fp32 masters preserved
    assert isinstance(state.decomposition.components, ComponentStacks)
    for _, site_components in state.decomposition.components.sites_items():
        assert site_components.V.dtype == jnp.float32
        assert site_components.U.dtype == jnp.float32


def test_step_trains_with_nonlinearity_loss():
    cfg = _tiny_cfg()
    sites = site_specs(cfg, _site_cs())
    target = init_tms_target(cfg, jax.random.PRNGKey(0))
    relative_threshold = ScheduleConfig(
        max_val=8.0,
        points=(Knot(at=0.0, frac=1.0), Knot(at=1.0, frac=0.25)),
    )
    # Make the term dominate Adam's first sign-normalized update so a detached loss cannot
    # hide behind an identical post-step component value.
    nonlinearity = NonlinearityLocalityLossConfig(
        name="nonlinearity_probe",
        coeff=1_000.0,
        relative_threshold=relative_threshold,
        unit_kind_coefficients={"neuron": 1.0},
    )
    with pytest.raises(AssertionError, match="unit_kind_coefficients"):
        # the closed check: authored kinds must be exactly the target's declared kinds
        _make_state_and_step(
            cfg,
            target,
            sites,
            total_steps=20,
            loss_metrics=(
                *_loss_metrics(),
                nonlinearity.model_copy(
                    update={"unit_kind_coefficients": {"neuron": 1.0, "attention_head": 1.0}}
                ),
            ),
        )
    model, state, step = _make_state_and_step(
        cfg,
        target,
        sites,
        total_steps=20,
        loss_metrics=(*_loss_metrics(), nonlinearity),
    )
    batch = sample_sparse_features(
        jax.random.PRNGKey(99), 64, cfg.n_features, 0.1, "at_least_zero_active"
    )
    trained_state, metrics = step(model, state, batch, jax.random.PRNGKey(100))
    value = float(metrics["loss/nonlinearity_probe"])
    assert value > 0.0
    assert float(metrics["loss/nonlinearity_probe_neuron"]) == pytest.approx(value)

    _, baseline_state, baseline_step = _make_state_and_step(
        cfg, target, sites, total_steps=20, loss_metrics=_loss_metrics()
    )
    baseline_trained_state, baseline = baseline_step(
        model, baseline_state, batch, jax.random.PRNGKey(100)
    )
    assert float(metrics["total"]) == pytest.approx(
        float(baseline["total"]) + 1_000.0 * value, rel=1e-5
    )

    trained_arrays = jax.tree.leaves(
        eqx.filter(trained_state.decomposition.components, eqx.is_array)
    )
    baseline_arrays = jax.tree.leaves(
        eqx.filter(baseline_trained_state.decomposition.components, eqx.is_array)
    )
    assert any(
        float(jnp.max(jnp.abs(actual - without_nonlinearity))) > 0.0
        for actual, without_nonlinearity in zip(trained_arrays, baseline_arrays, strict=True)
    )

    _, second_metrics = step(model, trained_state, batch, jax.random.PRNGKey(101))
    expected_threshold = 8.0 * (1.0 - 0.75 / (20 - 1))
    assert float(second_metrics["nonlinearity_relative_threshold"]) == pytest.approx(
        expected_threshold
    )


def test_faith_warmup_decreases_faith():
    cfg = _tiny_cfg()
    sites = site_specs(cfg, _site_cs())
    target = init_tms_target(cfg, jax.random.PRNGKey(0))
    model = PlacedModel(model=tms_decomposed_model(cfg, target, sites), placement=None)
    vu = init_component_stacks(sites, jax.random.PRNGKey(1))
    opt = optax.adamw(1e-2, weight_decay=0.0)
    wstep = make_faith_warmup_step(opt, faithfulness_loss_for(model.model))
    ostate = opt.init(eqx.filter(vu, eqx.is_array))
    first_loss = None
    loss = None
    for _ in range(40):
        vu, ostate, loss = wstep(model, vu, ostate)
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
            pnorm=ScheduleConfig.constant(1.0),
        ),
        StochasticReconLossConfig(coeff=1.0),
        StochasticReconSubsetLossConfig(coeff=1.0),
    )


def _faith_warmed_state(
    model: PlacedModel,
    sites: tuple[SiteSpec, ...],
    total_steps: int,
    warmup_steps: int,
) -> tuple[TrainState, Callable[..., tuple[TrainState, dict[str, jax.Array]]]]:
    """Build a train state, run faith warmup (TMS needs it — the from-scratch V/U start
    far from `W`), then return state + step factory."""
    vu = init_component_stacks(sites, jax.random.PRNGKey(1))
    ci_fn = init_layerwise_mlp_ci_fn(
        LayerwiseMLPCIArch(
            hidden_dims=(16,),
            has_position_axis=False,
            input_names=site_input_tap_keys(tuple(s.name for s in sites)),
        ),
        sites,
        jax.random.PRNGKey(2),
    )
    warm_opt = optax.adamw(1e-2, weight_decay=0.0)
    wstep = make_faith_warmup_step(warm_opt, faithfulness_loss_for(model.model))
    warm_state = warm_opt.init(eqx.filter(vu, eqx.is_array))
    for _ in range(warmup_steps):
        vu, warm_state, _ = wstep(model, vu, warm_state)
    opt_vu = optax.chain(optax.clip_by_global_norm(0.01), optax.adamw(1e-3, weight_decay=0.0))
    opt_ci = optax.adamw(1e-3, weight_decay=0.0)
    state = TrainState(
        decomposition=Decomposition(components=vu, ci_fn=ci_fn),
        training=TrainingItem(
            components_opt_state=opt_vu.init(eqx.filter(vu, eqx.is_array)),
            ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
            adversaries={},
            freq_ema=None,
            step=jnp.zeros((), jnp.int32),
        ),
    )
    loss_terms = build_objective(_recovery_loss_metrics(), model.site_names)
    step = make_train_step(
        model_static=model,
        substrate=ForwardSubstrate.of(
            model,
            remat_recon_forwards=False,
            remat_ci_fn=False,
            ci_capture_keys=ci_fn.capture_keys,
            ci_placement=None,
        ),
        objective=loss_terms,
        components_optimizer=opt_vu,
        ci_fn_optimizer=opt_ci,
        total_steps=total_steps,
        faithfulness=faithfulness_loss_for(model.model),
    )
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
        cfg,
        feature_probability=0.05,
        generation_type="at_least_zero_active",
        steps=2000,
        batch_size=2048,
        lr=1e-2,
        seed=0,
    )
    x = sample_sparse_features(jax.random.PRNGKey(7), 1024, 5, 0.05, "at_least_zero_active")
    model = tms_decomposed_model(cfg, target, sites)
    recon = jnp.mean((jnp.abs(x) - run_clean(model, x)) ** 2)
    assert float(recon) < 0.05, f"pretrained TMS recon too high: {recon}"

    placed = PlacedModel(model=model, placement=None)
    state, step = _faith_warmed_state(placed, sites, total_steps=3000, warmup_steps=150)
    data_key = jax.random.PRNGKey(123)
    totals: list[float] = []
    for i in range(3000):
        x = sample_sparse_features(
            jax.random.fold_in(data_key, i), 2048, 5, 0.05, "at_least_zero_active"
        )
        state, m = step(placed, state, x, jax.random.fold_in(jax.random.PRNGKey(321), i))
        totals.append(float(m["total"]))
    assert totals[-1] < totals[0], (totals[0], totals[-1])

    ci_lower = single_feature_ci(model, state.decomposition.ci_fn, n_features=5)
    err1 = identity_ci_error(ci_lower["linear1"], tolerance=0.2)
    err2 = identity_ci_error(ci_lower["linear2"], tolerance=0.2)
    assert err1 == 0, (
        f"linear1 did not recover identity (err={err1}):\n{jnp.round(ci_lower['linear1'], 2)}"
    )
    assert err2 == 0, (
        f"linear2 did not recover identity (err={err2}):\n{jnp.round(ci_lower['linear2'], 2)}"
    )


# ----------------------------- deeper (-id) variant -----------------------------


def _deeper_cfg(hidden_layer_init: HiddenLayerInit = "identity") -> TMSConfig:
    return TMSConfig(
        n_features=5, n_hidden=3, n_hidden_layers=2, hidden_layer_init=hidden_layer_init
    )


def _deeper_site_cs():
    return (
        SiteC("linear1", 8),
        SiteC(hidden_layer_name(0), 7),
        SiteC(hidden_layer_name(1), 6),
        SiteC("linear2", 5),
    )


def test_deeper_canonical_order_and_dims():
    cfg = _deeper_cfg()
    # canonical order: linear1, hidden_layers.0, hidden_layers.1, linear2 — regardless of input order.
    shuffled = (
        _deeper_site_cs()[3],
        _deeper_site_cs()[1],
        _deeper_site_cs()[0],
        _deeper_site_cs()[2],
    )
    assert tuple(s.name for s in canonical_site_cs(shuffled)) == (
        "linear1",
        "hidden_layers.0",
        "hidden_layers.1",
        "linear2",
    )
    specs = site_specs(cfg, _deeper_site_cs())
    dims = {s.name: (s.d_in, s.d_out, s.C) for s in specs}
    assert dims["linear1"] == (5, 3, 8)
    assert dims["hidden_layers.0"] == (3, 3, 7)  # n_hidden -> n_hidden
    assert dims["hidden_layers.1"] == (3, 3, 6)
    assert dims["linear2"] == (3, 5, 5)


def test_capture_uses_one_key_per_chain_activation():
    cfg = _tiny_cfg()
    model = tms_decomposed_model(
        cfg,
        init_tms_target(cfg, jax.random.PRNGKey(0)),
        site_specs(cfg, _site_cs()),
    )
    assert site_input_tap_keys(model.site_names) == ("linear1", "linear1.out")
    with pytest.raises(AssertionError, match="unknown TMS activation"):
        model.clean_forward(jnp.ones((1, cfg.n_features)), frozenset(("linear2",)), placement=None)

    cfg = _deeper_cfg()
    model = tms_decomposed_model(
        cfg,
        init_tms_target(cfg, jax.random.PRNGKey(0)),
        site_specs(cfg, _deeper_site_cs()),
    )
    assert site_input_tap_keys(model.site_names) == (
        "linear1",
        "linear1.out",
        "hidden_layers.0.out",
        "hidden_layers.1.out",
    )
    with pytest.raises(AssertionError, match="unknown TMS activation"):
        model.clean_forward(
            jnp.ones((1, cfg.n_features)),
            frozenset(("hidden_layers.0",)),
            placement=None,
        )


def test_deeper_clean_and_masked_forward_with_identity_hidden_layers():
    cfg = _deeper_cfg("identity")
    sites = site_specs(cfg, _deeper_site_cs())
    target = init_tms_target(cfg, jax.random.PRNGKey(0))
    model = tms_decomposed_model(cfg, target, sites)
    vu = init_component_stacks(sites, jax.random.PRNGKey(1))
    b = 7
    x = sample_sparse_features(
        jax.random.PRNGKey(2), b, cfg.n_features, 0.3, "at_least_zero_active"
    )

    clean = run_clean(model, x)
    assert clean.shape == (b, cfg.n_features)
    # Identity hidden layers must be no-ops in the frozen path: 5->2... here 5->3 with
    # n_hidden==n_features=5? no — n_hidden=3, so just check it equals a 2-layer reference.
    hidden = x @ target.W1.T
    ref = jax.nn.relu(hidden @ target.W2.T + target.b2)
    assert jnp.allclose(clean, ref, atol=1e-5), "identity hidden layers changed the frozen path"

    # Masks=1, delta=1 reconstructs the frozen path up to decomposition rounding.
    ones_masks = {s.name: jnp.ones((b, s.C)) for s in model.sites}
    ones_delta = {s.name: jnp.ones((b,)) for s in model.sites}
    full = run_masked(model, vu, x, ones_masks, ones_delta, None, True, remat=False)
    assert jnp.allclose(clean, full, atol=1e-4), "mask=1 identity drifted"

    # site_inputs threads through the hidden layers: each hidden-layer site reads the chain
    # output up to it; with identity layers those equal the linear1 output.
    site_in = site_inputs(target, x)
    assert set(site_in) == {"linear1", "hidden_layers.0", "hidden_layers.1", "linear2"}
    assert jnp.allclose(site_in["hidden_layers.0"], hidden, atol=1e-5)
    assert jnp.allclose(site_in["hidden_layers.1"], hidden, atol=1e-5)  # identity passthrough
    assert jnp.allclose(site_in["linear2"], hidden, atol=1e-5)


def test_random_hidden_layers_are_frozen_and_non_identity():
    cfg = _deeper_cfg("random")
    target = init_tms_target(cfg, jax.random.PRNGKey(0))
    assert len(target.hidden) == 2
    for W in target.hidden:
        assert W.shape == (cfg.n_hidden, cfg.n_hidden)
        assert not jnp.allclose(W, jnp.eye(cfg.n_hidden)), "random hidden layer is the identity"
    # Pretrain leaves the frozen hidden layers untouched (only W1, b2 train). Pretrain
    # derives its init from `split(PRNGKey(seed))[0]`, so reproduce that exact init key.
    trained = pretrain_tms_target(
        cfg,
        feature_probability=0.05,
        generation_type="at_least_zero_active",
        steps=3,
        batch_size=64,
        lr=1e-2,
        seed=0,
    )
    init_key, _ = jax.random.split(jax.random.PRNGKey(0))
    init_hidden = init_tms_target(cfg, init_key).hidden
    for trained_W, init_W in zip(trained.hidden, init_hidden, strict=True):
        assert jnp.array_equal(trained_W, init_W), "pretrain mutated a frozen hidden layer"


def test_init_bias_to_zero_toggle():
    cfg_zero = TMSConfig(n_features=5, n_hidden=2, init_bias_to_zero=True)
    cfg_rand = TMSConfig(n_features=5, n_hidden=2, init_bias_to_zero=False)
    assert jnp.array_equal(init_tms_target(cfg_zero, jax.random.PRNGKey(0)).b2, jnp.zeros(5))
    assert not jnp.allclose(init_tms_target(cfg_rand, jax.random.PRNGKey(0)).b2, jnp.zeros(5))


# ----------------------------- dense-CI eval -----------------------------


def test_dense_ci_error_perfect_and_imperfect():
    # k=2 active columns, both fully strong, the rest fully inactive -> zero error.
    perfect = jnp.concatenate([jnp.ones((4, 2)), jnp.zeros((4, 3))], axis=1)
    assert dense_ci_error(perfect, k=2, tolerance=0.1) == 0
    # Column order doesn't matter (sorted by mass first).
    shuffled = perfect[:, jnp.array([2, 0, 3, 1, 4])]
    assert dense_ci_error(shuffled, k=2, tolerance=0.1) == 0
    # A missing strong activation in a should-be-dense column -> one first-k error.
    weak_dense = jnp.concatenate([jnp.ones((4, 1)), jnp.zeros((4, 4))], axis=1)
    assert dense_ci_error(weak_dense, k=2, tolerance=0.1) == 1  # second dense column empty
    # A leak in a should-be-inactive column -> one inactive-column error per leaked entry.
    leak = perfect.at[0, 4].set(0.9)
    assert dense_ci_error(leak, k=2, tolerance=0.1) == 1


# ----------------------------- exactly_n_active dataset -----------------------------


@pytest.mark.parametrize(
    ("gen_type", "n"),
    [
        ("exactly_one_active", 1),
        ("exactly_two_active", 2),
        ("exactly_three_active", 3),
        ("exactly_five_active", 5),
    ],
)
def test_exactly_n_active_shape_and_count(gen_type: TMSGenerationType, n: int):
    b, n_features = 64, 8
    x = sample_sparse_features(jax.random.PRNGKey(0), b, n_features, 0.5, gen_type)
    assert x.shape == (b, n_features)
    assert jnp.all((x >= 0.0) & (x <= 1.0)), "values out of [0,1]"
    active_per_row = (x > 0.0).sum(axis=1)
    # Exactly n active per row (active values are U[0,1], a.s. nonzero).
    assert jnp.all(active_per_row == n), active_per_row
