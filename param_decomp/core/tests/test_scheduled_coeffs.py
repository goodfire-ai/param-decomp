"""Schedulable loss coefficients (`configs.LossCoeff`): the tPD paper's two shapes —
linear warmup→decay and 0-until-activation — the preserved bare-float / eval-None arms,
and the model-side cotangent scaling that keeps the S14′ final ascent coeff-blind."""

import jax
import jax.numpy as jnp
import pytest

from param_decomp.core.adversary import (
    PersistentAdversary,
    SiteSource,
    init_persistent_sources,
    init_sources_adam_state,
)
from param_decomp.core.configs import (
    AdamPGDConfig,
    FaithfulnessLossConfig,
    HiddenActsReconstruction,
    ImportanceMinimalityLossConfig,
    NontargetConfig,
    StochasticReconLossConfig,
)
from param_decomp.core.losses import coeff_at, reconstruction_spec_at, scheduled_value_traced
from param_decomp.core.objective import build_objective
from param_decomp.core.recon import (
    OutputAndHiddenActsReconstruction,
    resolve_reconstruction_spec,
)
from param_decomp.core.schedule import Knot, ScheduleConfig, get_scheduled_value
from param_decomp.core.train import model_cotangents_scaled

# The tPD paper's importance-minimality coefficient: linear 0 → 4e-3 over the first 20%
# of training, then linear decay to 1e-3 over the remaining 80%.
WARMUP_THEN_DECAY = ScheduleConfig(
    max_val=4e-3,
    points=(Knot(at=0.0, frac=0.0), Knot(at=0.2, frac=1.0), Knot(at=1.0, frac=0.25)),
)

# The tPD paper's persistent-PGD activation gate: 0 until 20% of training, then the
# value — `hold` keeps the previous knot's frac (0) and jumps AT its knot.
GATED = ScheduleConfig(
    max_val=0.5,
    points=(
        Knot(at=0.0, frac=0.0),
        Knot(at=0.2, frac=1.0, interp="hold"),
        Knot(at=1.0, frac=1.0),
    ),
)

TOTAL = 1001  # t = step / 1000, so 20% of the run lands exactly on step 200


def _frac(step: float) -> jax.Array:
    return jnp.asarray(step / (TOTAL - 1), jnp.float32)


def _traced(step: int, config: ScheduleConfig) -> float:
    return float(scheduled_value_traced(jnp.asarray(float(step)), TOTAL, config))


class TestWarmupThenDecay:
    def test_boundary_and_midpoint_values(self):
        expected = {
            0: 0.0,  # warmup start
            100: 2e-3,  # mid-warmup, linear
            200: 4e-3,  # the 20% peak
            600: 2.5e-3,  # mid-decay, linear
            1000: 1e-3,  # final step
        }
        for step, value in expected.items():
            assert get_scheduled_value(step, TOTAL, WARMUP_THEN_DECAY) == pytest.approx(value)
            assert _traced(step, WARMUP_THEN_DECAY) == pytest.approx(value, rel=1e-5)


class TestActivationGate:
    def test_exactly_zero_before_activation_then_the_value(self):
        for step in (0, 1, 100, 199):
            assert get_scheduled_value(step, TOTAL, GATED) == 0.0
            assert _traced(step, GATED) == 0.0
        for step in (200, 201, 600, 1000):
            assert get_scheduled_value(step, TOTAL, GATED) == pytest.approx(0.5)
            assert _traced(step, GATED) == pytest.approx(0.5, rel=1e-5)

    def test_coeff_at_evaluates_the_gate(self):
        # `total = coeff·loss` per term, so coeff == 0.0 IS zero contribution before X.
        assert float(jnp.asarray(coeff_at(_frac(0.0), GATED))) == 0.0
        assert float(jnp.asarray(coeff_at(_frac(600.0), GATED))) == pytest.approx(0.5)


class TestCoeffParsing:
    def test_bare_float_coeff_stays_the_constant(self):
        for raw in (0.5, 0.0):  # 0.0: the plain-off constant a schedule's positive peak can't spell
            cfg = StochasticReconLossConfig.model_validate(
                {"type": "StochasticReconLoss", "coeff": raw}
            )
            assert isinstance(cfg.coeff, float) and cfg.coeff == raw
            assert coeff_at(_frac(123.0), cfg.coeff) == raw

    def test_eval_only_none_coeff_is_preserved(self):
        cfg = StochasticReconLossConfig.model_validate({"type": "StochasticReconLoss"})
        assert cfg.coeff is None

    def test_scheduled_coeff_parses_and_flows_into_the_objective(self):
        cfg = StochasticReconLossConfig.model_validate(
            {"type": "StochasticReconLoss", "coeff": WARMUP_THEN_DECAY.model_dump()}
        )
        assert isinstance(cfg.coeff, ScheduleConfig)
        losses = build_objective(
            (
                FaithfulnessLossConfig(coeff=1.0),
                ImportanceMinimalityLossConfig(coeff=3e-3, pnorm=ScheduleConfig.constant(1.0)),
                cfg,
            ),
            ("a", "b"),
        )
        (term,) = losses.recon
        assert term.coeff == WARMUP_THEN_DECAY

    def test_nontarget_impmin_coeff_accepts_a_schedule(self):
        nt = NontargetConfig.model_validate(
            {
                "batch_size": 32,
                "impmin_coeff": WARMUP_THEN_DECAY.model_dump(),
                "recon": [{"type": "StochasticReconLoss", "coeff": 1.0}],
            }
        )
        assert nt.impmin_coeff == WARMUP_THEN_DECAY


class TestScheduledHiddenActsReconstruction:
    def test_coeff_resolves_per_step(self):
        hidden_acts_reconstruction = HiddenActsReconstruction.model_validate(
            {"coeff": GATED.model_dump(), "points": ["resid.1"]}
        )
        assert isinstance(hidden_acts_reconstruction.coeff, ScheduleConfig)
        before = reconstruction_spec_at(hidden_acts_reconstruction, _frac(0.0))
        after = reconstruction_spec_at(hidden_acts_reconstruction, _frac(1000.0))
        assert isinstance(before, OutputAndHiddenActsReconstruction)
        assert isinstance(after, OutputAndHiddenActsReconstruction)
        assert float(jnp.asarray(before.coeff)) == 0.0
        assert float(jnp.asarray(after.coeff)) == pytest.approx(0.5, rel=1e-5)

    def test_eval_probe_refuses_a_scheduled_coeff(self):
        hidden_acts_reconstruction = HiddenActsReconstruction.model_validate(
            {"coeff": GATED.model_dump(), "points": ["resid.1"]}
        )
        with pytest.raises(AssertionError, match="constant float"):
            resolve_reconstruction_spec(hidden_acts_reconstruction)


class TestModelCotangentsScaled:
    """S14′ post-refactor: `model_cotangents_scaled` applies a persistent term's
    coefficient only to its model-side cotangents. The source path is never scaled, so
    the final ascent consumes `dL/ds` directly and stays live through an activation gate."""

    def test_forward_is_bit_identical(self):
        x = jax.random.normal(jax.random.PRNGKey(0), (4, 3), jnp.bfloat16)
        for c in (0.0, 0.5, 1.0):
            wrapped = model_cotangents_scaled({"w": x}, jnp.asarray(c, jnp.float32))
            assert wrapped["w"].dtype == x.dtype
            assert jnp.array_equal(wrapped["w"], x)

    def test_backward_scales_by_the_coeff_exactly(self):
        x = jax.random.normal(jax.random.PRNGKey(1), (5,))

        def loss(x_in: jax.Array, c: jax.Array) -> jax.Array:
            return jnp.sum(model_cotangents_scaled({"w": x_in}, c)["w"] ** 2)

        base = jax.grad(loss)(x, jnp.asarray(1.0))
        for c in (0.0, 0.25, 2.0):
            scaled = jax.grad(loss)(x, jnp.asarray(c))
            assert jnp.allclose(scaled, c * base)
        assert jnp.array_equal(jax.grad(loss)(x, jnp.asarray(0.0)), jnp.zeros_like(x))

    def test_final_ascend_is_one_plain_adam_ascent(self):
        sources = init_persistent_sources(("a",), (4,), (1, 1), jnp.float32, jax.random.PRNGKey(0))
        adv = PersistentAdversary(
            sources=sources,
            opt_state=init_sources_adam_state(sources),
            state_key="k",
            adam=AdamPGDConfig(lr_schedule=ScheduleConfig.constant(0.1)),
            n_warmup=0,
        )
        grad_array = jax.random.normal(jax.random.PRNGKey(2), (1, 1, 5))
        grad = {"a": SiteSource(grad_array[..., :-1], grad_array[..., -1])}
        out = adv.final_ascend(grad, _frac(0.0))
        via_helper = adv.after_one_adam_ascent(grad, _frac(0.0))
        assert all(
            jnp.array_equal(a, b)
            for a, b in zip(
                jax.tree.leaves(out.sources["a"]),
                jax.tree.leaves(via_helper.sources["a"]),
                strict=True,
            )
        )
        assert any(
            not jnp.array_equal(a, b)
            for a, b in zip(
                jax.tree.leaves(out.sources["a"]),
                jax.tree.leaves(adv.sources["a"]),
                strict=True,
            )
        )
