"""The tPD engine (SPEC §11) exercised over the TMS target as a TEST FIXTURE: the
two-pass step trains, adversaries ride the target pass, the factory boundary holds, and
the whole engine path runs end-to-end as a library. There is no shipped toy tPD run
shape — the LM is the only targeted product surface."""

import dataclasses
from pathlib import Path
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.sharding import AxisType

from param_decomp.core.ci_fn import LayerwiseMLPCIArch, init_layerwise_mlp_ci_fn
from param_decomp.core.components import SiteC, init_component_stacks
from param_decomp.core.configs import (
    AdamPGDConfig,
    ImportanceMinimalityLossConfig,
    NontargetConfig,
    PersistentPGDReconLossConfig,
    StochasticReconLossConfig,
    TargetedLossMetricConfig,
    UnmaskedNoDeltaReconLossConfig,
)
from param_decomp.core.model import MaterializedMasking, PlacedModel, prepare_compute_weights
from param_decomp.core.objective import (
    NontargetPass,
    TargetedObjective,
    build_targeted_objective,
)
from param_decomp.core.recon import (
    ConstantSources,
    ReconLossTerm,
    StochasticSources,
    UnmaskedNoDeltaSources,
)
from param_decomp.core.schedule import ScheduleConfig
from param_decomp.core.train import (
    CIScaledWeightDecay,
    Decomposition,
    ForwardSubstrate,
    TrainingItem,
    TrainState,
    make_targeted_train_step,
)
from param_decomp.targets.tms import (
    TMSConfig,
    init_tms_target,
    sample_sparse_features,
    scatter_features,
    site_input_tap_keys,
    site_specs,
    tms_decomposed_model,
)


def _stochastic_nontarget() -> NontargetConfig:
    return NontargetConfig(
        batch_size=32, impmin_coeff=6e-3, recon=[StochasticReconLossConfig(coeff=1.0)]
    )


def _untyped[T](tree: T) -> T:
    """Host round-trip (value-identical): these targeted steps run UNPLACED, and
    explicitly-typed sources meeting the unplaced step's bare matmuls is the
    unsupported mixed mode — the mesh above exists only to exercise the placed init."""
    return jax.tree.map(lambda a: jnp.asarray(np.asarray(a)), tree)


def _tiny_setup(
    loss_metrics: tuple[TargetedLossMetricConfig, ...],
    nontarget: NontargetConfig,
    total_steps: int = 20,
    ci_scaled_weight_decay: CIScaledWeightDecay | None = None,
):
    cfg = TMSConfig(n_features=5, n_hidden=2)
    sites = site_specs(cfg, (SiteC("linear1", 8), SiteC("linear2", 6)))
    target = init_tms_target(cfg, jax.random.PRNGKey(0))
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
    opt_vu = optax.adamw(1e-3, weight_decay=0.0)
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
    objective = build_targeted_objective(loss_metrics, nontarget, model.site_names)
    step = make_targeted_train_step(
        model_static=model,
        substrate=ForwardSubstrate.of(
            model,
            remat_recon_forwards=False,
            remat_ci_fn=False,
            ci_capture_keys=ci_fn.capture_keys,
            ci_placement=None,
        ),
        objective=objective,
        ci_scaled_weight_decay=ci_scaled_weight_decay,
        components_optimizer=opt_vu,
        ci_fn_optimizer=opt_ci,
        total_steps=total_steps,
    )
    return cfg, model, state, step


def _loss_metrics():
    return (
        ImportanceMinimalityLossConfig(coeff=3e-3, gamma=ScheduleConfig.constant(1.0)),
        StochasticReconLossConfig(coeff=1.0),
    )


def test_targeted_two_pass_step_trains():
    """T1/T2: the two-pass step consumes both streams and advances finite training."""
    cfg, model, state, step = _tiny_setup(_loss_metrics(), _stochastic_nontarget())
    active = (0, 1)
    tkey, ntkey = jax.random.PRNGKey(10), jax.random.PRNGKey(11)
    metrics = {}
    for i in range(6):
        narrow = sample_sparse_features(
            jax.random.fold_in(tkey, i), 16, len(active), 0.3, "exactly_one_active"
        )
        target_batch = scatter_features(narrow, active, cfg.n_features)
        nontarget_batch = sample_sparse_features(
            jax.random.fold_in(ntkey, i), 32, cfg.n_features, 0.3, "at_least_zero_active"
        )
        state, metrics = step(
            model, state, target_batch, nontarget_batch, jax.random.PRNGKey(100 + i)
        )
    assert int(state.training.step) == 6
    assert all(bool(jnp.isfinite(jnp.asarray(v)).all()) for v in metrics.values())
    # T3: no faithfulness anywhere in the record.
    assert "faith" not in metrics
    # Both passes' losses are reported.
    assert "loss/StochasticReconLoss" in metrics
    assert "loss/nontarget/StochasticReconLoss" in metrics
    assert "loss/nontarget/total" in metrics


def test_targeted_step_trains_with_persistent_adversary():
    """T7: a persistent-PGD term rides the TARGET pass; sources size at pd.batch_size."""
    ppgd = PersistentPGDReconLossConfig.model_validate(
        {
            "type": "PersistentPGDReconLoss",
            "coeff": 1.0,
            "n_warmup_steps": 1,
            "source_shape": "bc",
            "optimizer": {
                "type": "adam",
                "beta1": 0.01,
                "beta2": 0.99,
                "eps": 1.0e-8,
                "lr_schedule": 0.01,
            },
        }
    )
    cfg, model, state, step = _tiny_setup((*_loss_metrics(), ppgd), _stochastic_nontarget())
    import numpy as np
    from jax.sharding import Mesh

    from param_decomp.core.adversary import PersistentAdversary, init_sources_adam_state
    from param_decomp.core.init_placed import init_sources_sharded
    from param_decomp.core.model import Positionless

    target_batch_size = 16  # == the target stream's batch below (pd.batch_size's role)
    # A degenerate (1, 1, 1) mesh built directly: this is a step unit test, not a toy
    # root, so it must not assert the whole process is single-device
    # (`single_device_mesh` does — the simulated-multidevice pass runs this too).
    mesh = Mesh(
        np.array(jax.devices()[:1]).reshape(1, 1, 1),
        axis_names=("replicate", "fsdp", "tp"),
        axis_types=(AxisType.Explicit,) * 3,
    )
    sources = _untyped(
        init_sources_sharded(
            model.site_names,
            tuple(s.C for s in model.sites),
            Positionless(),
            "bc",
            target_batch_size,
            jnp.dtype(jnp.float32),
            jax.random.PRNGKey(7),
            mesh,
        )
    )
    adversary = PersistentAdversary(
        sources=sources,
        opt_state=init_sources_adam_state(sources),
        state_key="PersistentPGDReconLoss",
        adam=ppgd.optimizer,
        n_warmup=1,
    )
    state = TrainState(
        decomposition=state.decomposition,
        training=TrainingItem(
            components_opt_state=state.training.components_opt_state,
            ci_fn_opt_state=state.training.ci_fn_opt_state,
            adversaries={"PersistentPGDReconLoss": adversary},
            freq_ema=None,
            step=state.training.step,
        ),
    )
    active = (0, 1)
    metrics: dict[str, jax.Array] = {}
    for i in range(3):
        narrow = sample_sparse_features(
            jax.random.fold_in(jax.random.PRNGKey(20), i),
            target_batch_size,
            len(active),
            0.3,
            "exactly_one_active",
        )
        target_batch = scatter_features(narrow, active, cfg.n_features)
        nontarget_batch = sample_sparse_features(
            jax.random.fold_in(jax.random.PRNGKey(21), i),
            32,
            cfg.n_features,
            0.3,
            "at_least_zero_active",
        )
        state, metrics = step(
            model, state, target_batch, nontarget_batch, jax.random.PRNGKey(200 + i)
        )
    assert all(bool(jnp.isfinite(jnp.asarray(v)).all()) for v in metrics.values())
    assert "src_lr" in metrics


def test_targeted_step_refuses_forged_freq_ema():
    # S8'': a targeted config cannot carry the EMA knob, so a freq_ema buffer on the
    # state is a forgery — refused at trace time rather than silently overwritten.
    cfg, model, state, step = _tiny_setup(_loss_metrics(), _stochastic_nontarget())
    forged = TrainState(
        decomposition=state.decomposition,
        training=dataclasses.replace(
            state.training,
            freq_ema={
                "linear1": jnp.zeros((8,), jnp.float32),
                "linear2": jnp.zeros((6,), jnp.float32),
            },
        ),
    )
    narrow = sample_sparse_features(jax.random.PRNGKey(10), 16, 2, 0.3, "exactly_one_active")
    target_batch = scatter_features(narrow, (0, 1), cfg.n_features)
    nontarget_batch = sample_sparse_features(
        jax.random.PRNGKey(11), 32, cfg.n_features, 0.3, "at_least_zero_active"
    )
    with pytest.raises(AssertionError, match="S8''"):
        step(model, forged, target_batch, nontarget_batch, jax.random.PRNGKey(100))


def test_targeted_step_unmasked_no_delta_scores_components_only():
    """T4's one exception: the UnmaskedNoDeltaReconLoss non-target term scores the FULL
    component sum with the weight delta actually OFF — the step's reported loss matches
    a delta-zero masked forward and not the delta-pinned-on one (which reproduces the
    frozen output near-exactly, collapsing the loss toward zero)."""
    nontarget = NontargetConfig(
        batch_size=32, impmin_coeff=6e-3, recon=[UnmaskedNoDeltaReconLossConfig(coeff=1.0)]
    )
    cfg, model, state, step = _tiny_setup(_loss_metrics(), nontarget)
    vu = state.decomposition.components
    active = (0, 1)
    narrow = sample_sparse_features(
        jax.random.PRNGKey(30), 16, len(active), 0.3, "exactly_one_active"
    )
    target_batch = scatter_features(narrow, active, cfg.n_features)
    nontarget_batch = sample_sparse_features(
        jax.random.PRNGKey(31), 32, cfg.n_features, 0.3, "at_least_zero_active"
    )

    # Expected values computed BEFORE the step — the jitted step donates its inputs.
    prepared = prepare_compute_weights(model, vu)
    clean_output = model.clean_forward(nontarget_batch).output
    n = nontarget_batch.shape[0]

    def all_ones_recon_at_delta(delta_value: float) -> jax.Array:
        masking = MaterializedMasking(
            component_masks={s.name: jnp.ones((n, s.C)) for s in model.sites},
            weight_delta_masks={s.name: jnp.full((n,), delta_value) for s in model.sites},
        )
        masked = model.masked_forward(prepared, nontarget_batch, masking=masking, remat=False)
        return model.recon_loss_fn(masked.output, clean_output)

    delta_off = all_ones_recon_at_delta(0.0)
    delta_on = all_ones_recon_at_delta(1.0)

    _, metrics = step(model, state, target_batch, nontarget_batch, jax.random.PRNGKey(300))
    reported = metrics["loss/nontarget/UnmaskedNoDeltaReconLoss"]
    assert jnp.allclose(reported, delta_off, rtol=1e-4, atol=1e-7)
    # Delta ON would make the same all-ones forward reproduce the frozen output, so its
    # loss collapses; a material gap pins that the reported arm really ran delta-off.
    assert float(delta_off) > 10 * float(delta_on)
    assert not jnp.allclose(reported, delta_on, rtol=1e-2, atol=1e-8)


def _gated_ppgd():
    """A PPGD term whose coeff is 0 for the first half of the run (an activation gate)."""
    return PersistentPGDReconLossConfig.model_validate(
        {
            "type": "PersistentPGDReconLoss",
            "coeff": {
                "max_val": 0.5,
                "points": [
                    {"at": 0.0, "frac": 0.0},
                    {"at": 0.5, "frac": 0.0},
                    {"at": 1.0, "frac": 1.0},
                ],
            },
            "n_warmup_steps": 2,
            "source_shape": "bc",
            "optimizer": {
                "type": "adam",
                "beta1": 0.5,
                "beta2": 0.99,
                "eps": 1.0e-8,
                "lr_schedule": 0.01,
            },
        }
    )


def _with_adversary_sources(
    state: TrainState, cfg_optimizer: AdamPGDConfig, source_key: int
) -> TrainState:
    import numpy as np
    from jax.sharding import Mesh

    from param_decomp.core.adversary import PersistentAdversary, init_sources_adam_state
    from param_decomp.core.init_placed import init_sources_sharded
    from param_decomp.core.model import Positionless

    mesh = Mesh(
        np.array(jax.devices()[:1]).reshape(1, 1, 1),
        axis_names=("replicate", "fsdp", "tp"),
        axis_types=(AxisType.Explicit,) * 3,
    )
    sources = _untyped(
        init_sources_sharded(
            ("linear1", "linear2"),
            (8, 6),
            Positionless(),
            "bc",
            16,
            jnp.dtype(jnp.float32),
            jax.random.PRNGKey(source_key),
            mesh,
        )
    )
    adversary = PersistentAdversary(
        sources=sources,
        opt_state=init_sources_adam_state(sources),
        state_key="PersistentPGDReconLoss",
        adam=cfg_optimizer,
        n_warmup=2,
    )
    return TrainState(
        decomposition=state.decomposition,
        training=TrainingItem(
            components_opt_state=state.training.components_opt_state,
            ci_fn_opt_state=state.training.ci_fn_opt_state,
            adversaries={"PersistentPGDReconLoss": adversary},
            freq_ema=None,
            step=state.training.step,
        ),
    )


def test_tpd_three_step_golden():
    """REFACTOR PIN: a 3-step tPD trajectory (stochastic + persistent-PGD target grid,
    stochastic non-target grid, constant coeffs) against literals generated on the
    remove-Lp branch (the imp-min seat is smooth-L0, gamma constant 1.0 — the one
    deliberate math change of that PR; previous literals: 81c510cee, Lp p=1). A pure
    restructuring of the step
    machinery must reproduce these
    values — needing to regenerate them means the MATH changed, which is a different PR.
    Tolerance absorbs the D4 float-reassociation class (SPEC D4): the per-step-metric
    rel=5e-3 is sized from the site_forward regrouping `((x@V)*m)@U + d*(x@W - (x@V)@U)`
    -> `((x@V)*(m-d))@U + d*(x@W)` (9fe2b6246, algebraically identical — verified by
    reverting only that hunk, which reproduces the literals bit-for-bit), whose drift on
    these near-cancelling recon residuals is rel <= 1.04e-3; x5 headroom covers
    cross-platform BLAS low bits on the same class. The final V/U and source sums stay
    at rel=1e-4 (observed drift <= 8e-6 and <= 6.8e-5) — a tolerance is widened only
    when a D4-class event fires it, so each family keeps its maximum catching power."""
    import numpy as np
    from jax.sharding import Mesh

    from param_decomp.core.adversary import PersistentAdversary, init_sources_adam_state
    from param_decomp.core.init_placed import init_sources_sharded
    from param_decomp.core.model import Positionless

    ppgd = PersistentPGDReconLossConfig.model_validate(
        {
            "type": "PersistentPGDReconLoss",
            "coeff": 0.5,
            "n_warmup_steps": 2,
            "source_shape": "bc",
            "optimizer": {
                "type": "adam",
                "beta1": 0.5,
                "beta2": 0.99,
                "eps": 1.0e-8,
                "lr_schedule": 0.01,
            },
        }
    )
    cfg, model, state, step = _tiny_setup((*_loss_metrics(), ppgd), _stochastic_nontarget())
    mesh = Mesh(
        np.array(jax.devices()[:1]).reshape(1, 1, 1),
        axis_names=("replicate", "fsdp", "tp"),
        axis_types=(AxisType.Explicit,) * 3,
    )
    sources = _untyped(
        init_sources_sharded(
            model.site_names,
            tuple(s.C for s in model.sites),
            Positionless(),
            "bc",
            16,
            jnp.dtype(jnp.float32),
            jax.random.PRNGKey(7),
            mesh,
        )
    )
    adversary = PersistentAdversary(
        sources=sources,
        opt_state=init_sources_adam_state(sources),
        state_key="PersistentPGDReconLoss",
        adam=ppgd.optimizer,
        n_warmup=2,
    )
    state = TrainState(
        decomposition=state.decomposition,
        training=TrainingItem(
            components_opt_state=state.training.components_opt_state,
            ci_fn_opt_state=state.training.ci_fn_opt_state,
            adversaries={"PersistentPGDReconLoss": adversary},
            freq_ema=None,
            step=state.training.step,
        ),
    )

    expected_per_step = (
        {
            "total": 1.427467726171e-02,
            "loss/StochasticReconLoss": 2.224520547315e-03,
            "loss/PersistentPGDReconLoss": 2.428050385788e-03,
            "loss/nontarget/total": 9.804574772716e-03,
        },
        {
            "total": 1.091865170747e-02,
            "loss/StochasticReconLoss": 1.462063402869e-03,
            "loss/PersistentPGDReconLoss": 1.721132895909e-03,
            "loss/nontarget/total": 7.962030358613e-03,
        },
        {
            "total": 9.593502618372e-03,
            "loss/StochasticReconLoss": 1.872145221569e-03,
            "loss/PersistentPGDReconLoss": 3.473097924143e-03,
            "loss/nontarget/total": 5.056168884039e-03,
        },
    )
    for i, expected in enumerate(expected_per_step):
        narrow = sample_sparse_features(
            jax.random.fold_in(jax.random.PRNGKey(20), i), 16, 2, 0.3, "exactly_one_active"
        )
        tb = scatter_features(narrow, (0, 1), cfg.n_features)
        ntb = sample_sparse_features(
            jax.random.fold_in(jax.random.PRNGKey(21), i),
            32,
            cfg.n_features,
            0.3,
            "at_least_zero_active",
        )
        state, metrics = step(model, state, tb, ntb, jax.random.PRNGKey(100 + i))
        for key, want in expected.items():
            got = float(metrics[key])
            assert got == pytest.approx(want, rel=5e-3), (i, key, got, want)

    expected_sums = {
        ("V", "linear2"): -1.364225268364e00,
        ("U", "linear2"): -2.958668231964e00,
        ("V", "linear1"): 8.874316215515e-01,
        ("U", "linear1"): 1.312882065773e00,
    }
    for shape, (v_stack, u_stack) in state.decomposition.components.stacks.items():
        assert float(jnp.sum(v_stack)) == pytest.approx(expected_sums[("V", shape)], rel=1e-4)
        assert float(jnp.sum(u_stack)) == pytest.approx(expected_sums[("U", shape)], rel=1e-4)
    final = state.training.adversaries["PersistentPGDReconLoss"]
    expected_sources = {"linear1": 7.531780433655e01, "linear2": 5.923494911194e01}
    for site, want in expected_sources.items():
        total = sum(float(jnp.sum(leaf)) for leaf in jax.tree.leaves(final.sources[site]))
        assert total == pytest.approx(want, rel=1e-4)


def test_gated_ppgd_shapes_nothing_but_the_adversary_still_ascends():
    """S14′ with an activation gate: while the PPGD coeff is 0 the term must not shape
    the decomposition — the step's new components and CI fn are bit-equal across
    DIFFERENT persistent-source values — while the adversary itself still takes its
    warmup AND final ascents (the source path is never coeff-scaled)."""
    import numpy as np

    ppgd = _gated_ppgd()
    outcomes = {}
    for source_key in (7, 8):
        cfg, model, state, step = _tiny_setup((*_loss_metrics(), ppgd), _stochastic_nontarget())
        state = _with_adversary_sources(state, ppgd.optimizer, source_key)
        # host copies: the jitted step donates its input state's buffers
        before = jax.tree.map(
            np.asarray, state.training.adversaries["PersistentPGDReconLoss"].sources
        )
        narrow = sample_sparse_features(jax.random.PRNGKey(20), 16, 2, 0.3, "exactly_one_active")
        tb = scatter_features(narrow, (0, 1), cfg.n_features)
        ntb = sample_sparse_features(
            jax.random.PRNGKey(21), 32, cfg.n_features, 0.3, "at_least_zero_active"
        )
        new_state, _ = step(model, state, tb, ntb, jax.random.PRNGKey(100))
        after = new_state.training.adversaries["PersistentPGDReconLoss"].sources
        assert any(
            not jnp.array_equal(old, new)
            for old, new in zip(jax.tree.leaves(before), jax.tree.leaves(after), strict=True)
        ), "the gated adversary must still ascend"
        outcomes[source_key] = new_state.decomposition
    a, b = outcomes[7], outcomes[8]
    for group in a.components.stacks:
        for got, want in zip(a.components.stacks[group], b.components.stacks[group], strict=True):
            assert jnp.array_equal(got, want), "gated PPGD leaked into the components"
    for got, want in zip(
        jax.tree.leaves(eqx.filter(a.ci_fn, eqx.is_array)),
        jax.tree.leaves(eqx.filter(b.ci_fn, eqx.is_array)),
        strict=True,
    ):
        assert jnp.array_equal(got, want), "gated PPGD leaked into the CI fn"


def test_targeted_factory_refuses_adversarial_nontarget_surface():
    """T5's library boundary: a programmatically-built non-target term with adversarial
    sources refuses at factory build even though the config schema can't spell one."""
    _, model, state, _ = _tiny_setup(_loss_metrics(), _stochastic_nontarget())
    nontarget = NontargetConfig(
        batch_size=32, impmin_coeff=6e-3, recon=[StochasticReconLossConfig(coeff=1.0)]
    )
    objective = build_targeted_objective(_loss_metrics(), nontarget, model.site_names)
    from param_decomp.core.configs import PGDReconLossConfig
    from param_decomp.core.objective import build_recon_terms

    (adversarial_term,) = build_recon_terms(
        (
            PGDReconLossConfig.model_validate(
                {
                    "type": "PGDReconLoss",
                    "coeff": 1.0,
                    "init": "random",
                    "n_steps": 1,
                    "step_size": 0.1,
                    "source_shape": "bc",
                }
            ),
        ),
        model.site_names,
    )
    # Deliberately breaching NontargetPass's narrow type (hence the cast) to exercise
    # the factory's boundary assert behind it.
    forged = TargetedObjective(
        target=objective.target,
        nontarget=NontargetPass(
            recon=cast(
                "tuple[ReconLossTerm[StochasticSources | ConstantSources | UnmaskedNoDeltaSources], ...]",
                (adversarial_term,),
            ),
            impmin_coeff=objective.nontarget.impmin_coeff,
        ),
    )
    import optax

    with pytest.raises(AssertionError, match="PGDReconLoss"):
        make_targeted_train_step(
            model_static=model,
            substrate=ForwardSubstrate.of(
                model,
                remat_recon_forwards=False,
                remat_ci_fn=False,
                ci_capture_keys=state.decomposition.ci_fn.capture_keys,
                ci_placement=None,
            ),
            objective=forged,
            ci_scaled_weight_decay=None,
            components_optimizer=optax.adamw(1e-3),
            ci_fn_optimizer=optax.adamw(1e-3),
            total_steps=10,
        )


_ACTIVE = (0, 1)


def _target_batch(cfg: TMSConfig, i: int) -> jax.Array:
    narrow = sample_sparse_features(
        jax.random.fold_in(jax.random.PRNGKey(10), i), 16, len(_ACTIVE), 0.3, "exactly_one_active"
    )
    return scatter_features(narrow, _ACTIVE, cfg.n_features)


def _nontarget_batch(cfg: TMSConfig, i: int) -> jax.Array:
    return sample_sparse_features(
        jax.random.fold_in(jax.random.PRNGKey(11), i),
        32,
        cfg.n_features,
        0.3,
        "at_least_zero_active",
    )


def _pin_ci_fn(state: TrainState) -> TrainState:
    """Freeze the CI landscape so T11's statistic is known exactly: every CI-fn weight
    zeroed, head biases pinned to saturation — linear1 component 0 at CI 1 everywhere,
    every other component at CI 0 — EXCEPT linear1 component 1, which reads feature 3
    through a large pass-through weight: CI 1 whenever feature 3 fires. Feature 3 lies
    outside the target stream's `_ACTIVE` set, so component 1 is alive on the NON-TARGET
    stream only — the probe that the decay's max spans both streams."""
    ci_fn = jax.tree.map(jnp.zeros_like, state.decomposition.ci_fn)
    linear1 = ci_fn.site_mlps["linear1"]
    linear2 = ci_fn.site_mlps["linear2"]
    pinned = eqx.tree_at(
        lambda f: (
            f.site_mlps["linear1"].weights[0],
            f.site_mlps["linear1"].weights[1],
            f.site_mlps["linear1"].biases[-1],
            f.site_mlps["linear2"].biases[-1],
        ),
        ci_fn,
        (
            linear1.weights[0].at[3, 0].set(100.0),
            linear1.weights[1].at[0, 1].set(1.0),
            jnp.full_like(linear1.biases[-1], -3.0).at[0].set(3.0),
            jnp.full_like(linear2.biases[-1], -3.0),
        ),
    )
    return TrainState(
        decomposition=Decomposition(components=state.decomposition.components, ci_fn=pinned),
        training=state.training,
    )


def _component_norms(state: TrainState, site: str) -> jax.Array:
    vu = state.decomposition.components.site(site)
    return jnp.sqrt(jnp.sum(vu.V**2, axis=0) + jnp.sum(vu.U**2, axis=1))


def test_ci_scaled_weight_decay_decays_exactly_the_dead_components():
    """T11, one step from one state with the decay and without: the decay-run differs
    from the None-run by EXACTLY the post-optimizer V/U scaling. Dead components (CI 0 on
    both streams) shrink by the full `lr·wd` rate; the component saturated on the target
    stream and the one alive only on the NON-TARGET stream are both untouched
    bit-for-bit, and so is everything that is not a component master."""
    wd = CIScaledWeightDecay(coeff=0.2, components_lr=ScheduleConfig.constant(1.0))
    cfg, model, state_a, step_none = _tiny_setup(_loss_metrics(), _stochastic_nontarget())
    _, _, state_b, step_decay = _tiny_setup(
        _loss_metrics(), _stochastic_nontarget(), ci_scaled_weight_decay=wd
    )
    state_a, state_b = _pin_ci_fn(state_a), _pin_ci_fn(state_b)  # identical values

    target_batch, nontarget_batch = _target_batch(cfg, 0), _nontarget_batch(cfg, 0)
    assert float(target_batch[:, 3].max()) == 0.0  # feature 3 never fires on-target...
    assert float(nontarget_batch[:, 3].max()) > 0.05  # ...and does fire off-target
    key = jax.random.PRNGKey(0)
    state_n, metrics_n = step_none(model, state_a, target_batch, nontarget_batch, key)
    state_d, metrics_d = step_decay(model, state_b, target_batch, nontarget_batch, key)

    protected = {"linear1": (0, 1), "linear2": ()}
    for site in ("linear1", "linear2"):
        vu_n = state_n.decomposition.components.site(site)
        vu_d = state_d.decomposition.components.site(site)
        for c in range(vu_n.V.shape[1]):
            if c in protected[site]:
                assert jnp.array_equal(vu_d.V[:, c], vu_n.V[:, c]), (site, c)
                assert jnp.array_equal(vu_d.U[c], vu_n.U[c]), (site, c)
            else:
                assert jnp.allclose(vu_d.V[:, c], 0.8 * vu_n.V[:, c], rtol=1e-6), (site, c)
                assert jnp.allclose(vu_d.U[c], 0.8 * vu_n.U[c], rtol=1e-6), (site, c)
    # Nothing but the component masters moved.
    assert eqx.tree_equal(state_d.decomposition.ci_fn, state_n.decomposition.ci_fn)
    assert eqx.tree_equal(state_d.training, state_n.training)
    # The mechanism is observable: dead components decay at the full rate...
    assert float(metrics_d["ci_scaled_weight_decay/max"]) == pytest.approx(0.2)
    assert 0.0 < float(metrics_d["ci_scaled_weight_decay/mean"]) < 0.2
    # ...and the None run carries no trace of it.
    assert not any(k.startswith("ci_scaled_weight_decay") for k in metrics_n)


def test_ci_scaled_weight_decay_drags_dead_norms_down_across_steps():
    """T11's cleanup force over a short run: never-important components' V/U norms
    strictly shrink every step (nothing else in the objective shrinks them), while the
    always-important component's norm sees only ordinary optimizer drift."""
    wd = CIScaledWeightDecay(coeff=0.2, components_lr=ScheduleConfig.constant(1.0))
    cfg, model, state, step = _tiny_setup(
        _loss_metrics(), _stochastic_nontarget(), ci_scaled_weight_decay=wd
    )
    state = _pin_ci_fn(state)
    dead_norms = [_component_norms(state, "linear2")]
    live_norm_before = float(_component_norms(state, "linear1")[0])
    for i in range(5):
        state, _ = step(
            model,
            state,
            _target_batch(cfg, i),
            _nontarget_batch(cfg, i),
            jax.random.PRNGKey(100 + i),
        )
        dead_norms.append(_component_norms(state, "linear2"))
    for before, after in zip(dead_norms[:-1], dead_norms[1:], strict=True):
        assert bool(jnp.all(after < before))  # every dead component, every step
    assert bool(jnp.all(dead_norms[-1] < 0.6 * dead_norms[0]))  # ≈0.8^5 + gradient noise
    live_norm_after = float(_component_norms(state, "linear1")[0])
    assert abs(live_norm_after - live_norm_before) < 0.05


def test_scatter_features_places_and_zeroes():
    x = jnp.asarray([[0.5, 0.7], [0.0, 0.9]])
    out = scatter_features(x, (1, 3), 5)
    assert out.shape == (2, 5)
    assert jnp.array_equal(out[:, 1], x[:, 0]) and jnp.array_equal(out[:, 3], x[:, 1])
    assert jnp.array_equal(out[:, jnp.asarray([0, 2, 4])], jnp.zeros((2, 3)))
    with pytest.raises(AssertionError):
        scatter_features(x, (1, 7), 5)  # out of range
    with pytest.raises(AssertionError):
        scatter_features(x, (3, 1), 5)  # unsorted


def test_targeted_engine_end_to_end(tmp_path: Path):
    """The whole targeted engine path driven as a LIBRARY — TMS is the test fixture, not
    a shipped tPD surface (the LM is the only targeted run shape): prepare → two-pass
    train → checkpoint → metrics, via run_targeted_decomposition_training directly."""
    import numpy as np
    import yaml
    from jax.sharding import Mesh

    from param_decomp.core import placement
    from param_decomp.core.built_run import RunInstance
    from param_decomp.core.configs import Cadence, TargetedPDConfig
    from param_decomp.core.model import Positionless
    from param_decomp.core.run import MetricsSink, run_targeted_decomposition_training

    pd = TargetedPDConfig.model_validate(
        yaml.safe_load(
            """
            seed: 0
            steps: 4
            batch_size: 16
            loss_metrics:
              - {type: ImportanceMinimalityLoss, coeff: 3.0e-03, gamma: 1.0}
              - {type: StochasticReconLoss, coeff: 1.0}
            components_optimizer: {lr_schedule: 1.0e-03}
            ci_fn_optimizer: {lr_schedule: 1.0e-03}
            """
        )
    )
    nontarget = NontargetConfig(
        batch_size=32, impmin_coeff=6e-3, recon=[StochasticReconLossConfig(coeff=1.0)]
    )
    cfg = TMSConfig(n_features=5, n_hidden=2)
    sites = site_specs(cfg, (SiteC("linear1", 8), SiteC("linear2", 6)))
    model = tms_decomposed_model(cfg, init_tms_target(cfg, jax.random.PRNGKey(0)), sites)
    ci_arch = LayerwiseMLPCIArch(
        hidden_dims=(16,),
        has_position_axis=False,
        input_names=site_input_tap_keys(tuple(s.name for s in sites)),
    )
    run = RunInstance(
        run_name="tms-targeted-e2e",
        run_id="p-00000000",
        out_dir=tmp_path / "runs",
        wandb=None,
        resume_provenance=None,
    )
    run.run_dir.mkdir(parents=True)  # the roots mkdir before building the sink
    mesh = Mesh(
        np.array(jax.devices()[:1]).reshape(1, 1, 1),
        axis_names=("replicate", "fsdp", "tp"),
        axis_types=(AxisType.Explicit,) * 3,
    )
    active = (0, 1)
    tkey, ntkey = jax.random.PRNGKey(10), jax.random.PRNGKey(11)

    def sample_target_batch(step: int) -> jax.Array:
        narrow = sample_sparse_features(
            jax.random.fold_in(tkey, step), 16, len(active), 0.3, "exactly_one_active"
        )
        return scatter_features(narrow, active, cfg.n_features)

    def sample_nontarget_batch(step: int) -> jax.Array:
        return sample_sparse_features(
            jax.random.fold_in(ntkey, step), 32, cfg.n_features, 0.3, "at_least_zero_active"
        )

    run_targeted_decomposition_training(
        pd=pd,
        nontarget=nontarget,
        cadence=Cadence.model_validate(
            {
                "train_log_every": 2,
                "checkpointing": {
                    "kind": "periodic",
                    "save_every": 4,
                    "retention": {"kind": "keep_last", "n": 1},
                },
            }
        ),
        run=run,
        model=PlacedModel(model=model, placement=placement.from_config("ddp", mesh, model.sites)),
        ci_fn=ci_arch,
        positions=Positionless(),
        remat_recon_forwards=False,
        remat_ci_fn=False,
        compiler_options={},
        sample_target_batch=sample_target_batch,
        sample_nontarget_batch=sample_nontarget_batch,
        evaluation=None,
        profiling=None,
        sink=MetricsSink.for_run(run, is_main=True),
    )

    run_dir = tmp_path / "runs" / "p-00000000"
    assert (run_dir / "ckpts" / "4" / "decomposition").exists()
    import json

    lines = (run_dir / "metrics.jsonl").read_text().strip().splitlines()
    assert lines, "the targeted run logged no metrics"
    last = json.loads(lines[-1])
    assert "train/loss/nontarget/total" in last
    assert not any("Faithfulness" in k for k in last)
