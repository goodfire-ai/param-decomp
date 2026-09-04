"""Target-relative faithfulness: formula, validation, train step, and warmup."""

import equinox as eqx
import jax.numpy as jnp
import optax
import pytest
from jax import random

from param_decomp.core.configs import (
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    StochasticReconLossConfig,
)
from param_decomp.core.faithfulness import make_faithfulness_loss
from param_decomp.core.model import DecomposedModel, PlacedModel
from param_decomp.core.objective import build_objective
from param_decomp.core.schedule import ScheduleConfig
from param_decomp.core.train import ForwardSubstrate, make_faith_warmup_step, make_train_step
from param_decomp.tests.core.test_generic_model_io import (
    SITE,
    _initial_state,
    _synthetic_ci_arch,
    _synthetic_inputs,
    _synthetic_lm,
    _synthetic_vu,
)


def _objective(model: DecomposedModel):
    return build_objective(
        (
            FaithfulnessLossConfig(coeff=1.0),
            ImportanceMinimalityLossConfig(coeff=1e-4, gamma=ScheduleConfig.constant(1.0)),
            StochasticReconLossConfig(coeff=1.0),
        ),
        model.site_names,
    )


def test_faithfulness_is_mean_of_per_site_relative_errors():
    site_slots = (("a", "g", 0), ("b", "g", 1), ("c", "h", 0))
    loss = make_faithfulness_loss(site_slots, {"g": (8.0, 16.0), "h": (2.0,)})
    deltas = {
        "g": jnp.stack([jnp.full((2, 2), 2.0), jnp.full((2, 2), 1.0)]),
        "h": jnp.full((1, 1, 4), 1.0),
    }
    assert float(loss(deltas)) == pytest.approx((16 / 8.0 + 4 / 16.0 + 4 / 2.0) / 3)


def test_faithfulness_weights_sites_equally_not_by_parameter_count():
    site_slots = (("a", "g", 0), ("b", "h", 0))
    loss = make_faithfulness_loss(site_slots, {"g": (4.0,), "h": (1.0,)})
    deltas = {"g": jnp.ones((1, 2, 2)), "h": jnp.zeros((1, 100, 100))}
    assert float(loss(deltas)) == pytest.approx(0.5)


def test_make_faithfulness_loss_validates_norm_keys_and_values():
    site_slots = ((SITE, SITE, 0),)
    with pytest.raises(AssertionError):
        make_faithfulness_loss(site_slots, {"not.a.group": (1.0,)})
    with pytest.raises(AssertionError):
        make_faithfulness_loss(site_slots, {SITE: (1.0, 1.0)})
    for value in (0.0, float("nan"), float("inf")):
        with pytest.raises(AssertionError, match="finite positive"):
            make_faithfulness_loss(site_slots, {SITE: (value,)})


def test_train_step_uses_target_relative_faithfulness():
    key = random.PRNGKey(3)
    model = _synthetic_lm(key)
    placed = PlacedModel(model=model, placement=None)
    state, opt_vu, opt_ci = _initial_state(model, _synthetic_vu(key), _synthetic_ci_arch())
    faithfulness = make_faithfulness_loss(((SITE, SITE, 0),), {SITE: (2.5,)})
    expected = float(faithfulness(model.weight_deltas(state.decomposition.components)))
    step = make_train_step(
        model_static=placed,
        substrate=ForwardSubstrate.of(
            placed,
            remat_recon_forwards=False,
            remat_ci_fn=False,
            ci_capture_keys=state.decomposition.ci_fn.capture_keys,
            ci_placement=None,
        ),
        objective=_objective(model),
        components_optimizer=opt_vu,
        ci_fn_optimizer=opt_ci,
        total_steps=10,
        faithfulness=faithfulness,
    )
    _, metrics = step(placed, state, _synthetic_inputs(key), random.PRNGKey(7))
    assert float(metrics["faith"]) == pytest.approx(expected, rel=1e-5)


def test_faith_warmup_uses_target_relative_faithfulness():
    key = random.PRNGKey(5)
    model = _synthetic_lm(key)
    placed = PlacedModel(model=model, placement=None)
    components = _synthetic_vu(key)
    faithfulness = make_faithfulness_loss(((SITE, SITE, 0),), {SITE: (2.5,)})
    expected = float(faithfulness(model.weight_deltas(components)))
    opt = optax.adamw(1e-2, weight_decay=0.0)
    warmup_step = make_faith_warmup_step(opt, faithfulness)
    _, _, loss = warmup_step(placed, components, opt.init(eqx.filter(components, eqx.is_array)))
    assert float(loss) == pytest.approx(expected, rel=1e-5)
