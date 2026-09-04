"""Core tPD semantics (SPEC §11, T1–T10): the targeted objective builders' closure
rules, the delta-pinned mask constructions, and the two-pass step factory's boundary
refusals. The full two-pass training run is pinned by the TMS seat's tests
(`param_decomp/tests/experiments/tms/test_targeted_tms.py`); this module needs no target."""

from typing import cast

import jax
import jax.numpy as jnp
import pytest

from param_decomp.core.configs import (
    FaithfulnessLossConfig,
    FrequencyMinimalityConfig,
    ImportanceMinimalityLossConfig,
    NontargetConfig,
    PDConfig,
    StochasticReconLossConfig,
    StochasticReconSubsetLossConfig,
    TargetedLossMetricConfig,
    TargetedPDConfig,
    UniformKSubsetRoutingConfig,
    UnmaskedNoDeltaReconLossConfig,
)
from param_decomp.core.masking import (
    constant_delta_pinned_masks,
    stochastic_delta_pinned_masks,
    unmasked_no_delta_masks,
)
from param_decomp.core.objective import build_targeted_objective
from param_decomp.core.recon import ConstantSources, StochasticSources, UnmaskedNoDeltaSources
from param_decomp.core.schedule import ScheduleConfig


def _target_metrics():
    return (
        ImportanceMinimalityLossConfig(coeff=3e-3, gamma=ScheduleConfig.constant(1.0)),
        StochasticReconLossConfig(coeff=1.0),
    )


def _nontarget():
    return NontargetConfig(
        batch_size=32,
        impmin_coeff=6e-3,
        recon=[StochasticReconSubsetLossConfig(coeff=1.0, routing=UniformKSubsetRoutingConfig())],
    )


def test_targeted_objective_shape_and_shared_imp_config():
    objective = build_targeted_objective(_target_metrics(), _nontarget(), ("a", "b"))
    # T6: the non-target pass carries only a coefficient — it structurally CANNOT hold
    # its own penalty config; the step reads the target pass's.
    assert objective.target.imp.coeff == 3e-3
    assert objective.nontarget.impmin_coeff == 6e-3
    # T5: the non-target surface is stochastic/constant-source only, output-only scored.
    for term in objective.nontarget.recon:
        assert isinstance(term.sources, StochasticSources | ConstantSources)


def test_targeted_pd_config_cannot_spell_faithfulness():
    # T3: a faithfulness term — even at coeff 0 — is a different algorithm, and the
    # targeted shape's loss union has no member for it: refusal is a parse error, not a
    # runtime check.
    with pytest.raises(Exception, match="FaithfulnessLoss"):
        TargetedPDConfig.model_validate(
            {
                "loss_metrics": [
                    {"type": "FaithfulnessLoss", "coeff": 0.0},
                    {"type": "StochasticReconLoss", "coeff": 1.0},
                ],
                "components_optimizer": {"lr_schedule": 1e-3},
                "ci_fn_optimizer": {"lr_schedule": 1e-3},
                "steps": 10,
                "batch_size": 8,
            }
        )
    # ...and no warmup fields exist to set.
    with pytest.raises(Exception, match="faithfulness_warmup_steps"):
        TargetedPDConfig.model_validate(
            {
                "loss_metrics": [{"type": "StochasticReconLoss", "coeff": 1.0}],
                "components_optimizer": {"lr_schedule": 1e-3},
                "ci_fn_optimizer": {"lr_schedule": 1e-3},
                "steps": 10,
                "batch_size": 8,
                "faithfulness_warmup_steps": 0,
            }
        )


def test_targeted_pd_config_refuses_frequency_ema():
    # The EMA carries one frequency stream per site, while the tPD step takes the penalty
    # on two independent streams — tPD EMA is not implemented (S8''), so an authored
    # config with the knob fails loudly at parse rather than half-applying it.
    with pytest.raises(Exception, match="not implemented for the targeted"):
        TargetedPDConfig.model_validate(
            {
                "loss_metrics": [
                    {
                        "type": "ImportanceMinimalityLoss",
                        "coeff": 1e-4,
                        "gamma": 1.0,
                        "frequency": {
                            "coeff": 1e-4,
                            "reference_datapoint_count": 128,
                            "ema_halflife_steps": 500,
                        },
                    },
                    {"type": "StochasticReconLoss", "coeff": 1.0},
                ],
                "components_optimizer": {"lr_schedule": 1e-3},
                "ci_fn_optimizer": {"lr_schedule": 1e-3},
                "steps": 10,
                "batch_size": 8,
            }
        )


def test_targeted_pd_config_ci_scaled_weight_decay_parses():
    # T11: absent is the real, intended state (None — no decay); a set value must be a
    # positive float.
    base = {
        "loss_metrics": [
            {"type": "ImportanceMinimalityLoss", "coeff": 3e-3, "gamma": 1.0},
            {"type": "StochasticReconLoss", "coeff": 1.0},
        ],
        "components_optimizer": {"lr_schedule": 1e-3},
        "ci_fn_optimizer": {"lr_schedule": 1e-3},
        "steps": 10,
        "batch_size": 8,
    }
    assert TargetedPDConfig.model_validate(base).ci_scaled_weight_decay is None
    on = TargetedPDConfig.model_validate({**base, "ci_scaled_weight_decay": 0.1})
    assert on.ci_scaled_weight_decay == 0.1
    for not_positive in (0.0, -0.1):
        with pytest.raises(Exception, match="ci_scaled_weight_decay"):
            TargetedPDConfig.model_validate({**base, "ci_scaled_weight_decay": not_positive})


def test_plain_pd_config_cannot_spell_ci_scaled_weight_decay():
    # T11 is targeted-only: in plain PD faithfulness penalizes the residual delta, so
    # decaying component vectors would fight it head-on — the field does not exist on
    # the plain shape, and refusal is a parse error.
    with pytest.raises(Exception, match="ci_scaled_weight_decay"):
        PDConfig.model_validate(
            {
                "loss_metrics": [
                    {"type": "FaithfulnessLoss", "coeff": 1.0},
                    {"type": "ImportanceMinimalityLoss", "coeff": 3e-3, "gamma": 1.0},
                    {"type": "StochasticReconLoss", "coeff": 1.0},
                ],
                "components_optimizer": {"lr_schedule": 1e-3},
                "ci_fn_optimizer": {"lr_schedule": 1e-3},
                "steps": 10,
                "batch_size": 8,
                "ci_scaled_weight_decay": 0.1,
            }
        )


def test_targeted_objective_boundary_refuses_programmatic_faithfulness():
    # The library boundary behind the schema: a loss list built outside pydantic (hence
    # the cast) still cannot smuggle a faithfulness role into the objective.
    forged = cast(
        "list[TargetedLossMetricConfig]", [FaithfulnessLossConfig(coeff=0.0), *_target_metrics()]
    )
    with pytest.raises(AssertionError, match="FaithfulnessLossConfig"):
        build_targeted_objective(forged, _nontarget(), ("a", "b"))


def test_targeted_objective_boundary_refuses_programmatic_frequency_ema():
    # The library boundary behind the schema: a loss list built outside pydantic cannot
    # smuggle the plain-PD-only EMA into the two-pass objective (S8'').
    with_ema = ImportanceMinimalityLossConfig(
        coeff=1e-4,
        gamma=ScheduleConfig.constant(1.0),
        frequency=FrequencyMinimalityConfig(
            coeff=1e-4, reference_datapoint_count=128, ema_halflife_steps=500
        ),
    )
    forged = cast(
        "list[TargetedLossMetricConfig]",
        [
            with_ema,
            *[m for m in _target_metrics() if not isinstance(m, ImportanceMinimalityLossConfig)],
        ],
    )
    with pytest.raises(AssertionError, match="not implemented for the targeted"):
        build_targeted_objective(forged, _nontarget(), ("a", "b"))


def test_nontarget_schema_refuses_hidden_acts_at_parse():
    # T5: S35 hidden-activation reconstruction is target-pass-only — refused when the
    # seat parses, not at objective build on the GPUs.
    with pytest.raises(Exception, match="hidden_acts_reconstruction"):
        NontargetConfig.model_validate(
            {
                "batch_size": 32,
                "impmin_coeff": 6e-3,
                "recon": [
                    {
                        "type": "StochasticReconLoss",
                        "coeff": 1.0,
                        "hidden_acts_reconstruction": {"coeff": 0.1, "points": ["resid.1"]},
                    }
                ],
            }
        )


def test_nontarget_schema_rejects_adversarial_sources():
    # T5: adversarial/unmasked sources are unrepresentable, not filtered.
    with pytest.raises(Exception, match="PGDReconLoss"):
        NontargetConfig.model_validate(
            {
                "batch_size": 32,
                "impmin_coeff": 0.0,
                "recon": [
                    {
                        "type": "PGDReconLoss",
                        "coeff": 1.0,
                        "init": "random",
                        "n_steps": 2,
                        "step_size": 0.1,
                        "source_shape": "bc",
                    }
                ],
            }
        )


def test_unmasked_no_delta_config_is_fully_determined():
    # The term is fully determined by its type: routing, sampling, and optional
    # hidden-activation reconstruction fields are structurally absent, refused at parse (`extra="forbid"`), not validated away.
    cfg = UnmaskedNoDeltaReconLossConfig.model_validate(
        {"type": "UnmaskedNoDeltaReconLoss", "coeff": 1.0}
    )
    assert cfg.coeff == 1.0
    for extra in (
        {"routing": {"type": "UniformKSubsetRouting"}},
        {"n_mask_samples": 2},
        {"hidden_acts_reconstruction": {"coeff": 0.1, "points": ["resid.1"]}},
    ):
        with pytest.raises(Exception, match="[Ee]xtra"):
            UnmaskedNoDeltaReconLossConfig.model_validate(
                {"type": "UnmaskedNoDeltaReconLoss", "coeff": 1.0, **extra}
            )


def test_unmasked_no_delta_masks_are_ones_and_zeros():
    # T4's one exception: all-ones component masks, all-zeros delta masks, deterministic.
    ci_lower = {
        "a": jax.random.uniform(jax.random.PRNGKey(0), (4, 6)),
        "b": jax.random.uniform(jax.random.PRNGKey(1), (4, 3)),
    }
    masks, deltas = unmasked_no_delta_masks(ci_lower)
    for site in ("a", "b"):
        assert jnp.array_equal(masks[site], jnp.ones_like(ci_lower[site]))
        assert jnp.array_equal(deltas[site], jnp.zeros((4,)))


def test_targeted_objective_admits_unmasked_no_delta_for_nontarget():
    nontarget = NontargetConfig(
        batch_size=32,
        impmin_coeff=6e-3,
        recon=[
            UnmaskedNoDeltaReconLossConfig(coeff=0.5),
            StochasticReconSubsetLossConfig(coeff=1.0, routing=UniformKSubsetRoutingConfig()),
        ],
    )
    objective = build_targeted_objective(_target_metrics(), nontarget, ("a", "b"))
    unmasked_term = next(
        t for t in objective.nontarget.recon if t.name == "UnmaskedNoDeltaReconLoss"
    )
    # A single all-routed draw — the term is fully determined.
    assert isinstance(unmasked_term.sources, UnmaskedNoDeltaSources)
    assert unmasked_term.sample_routing(jax.random.PRNGKey(0), (4,)) == (None,)


def test_target_pass_and_plain_unions_refuse_unmasked_no_delta():
    # The term is NON-TARGET-ONLY vocabulary: neither the targeted TARGET-pass union nor
    # the plain-PD union has a member for it — refusal is a parse error.
    base = {
        "components_optimizer": {"lr_schedule": 1e-3},
        "ci_fn_optimizer": {"lr_schedule": 1e-3},
        "steps": 10,
        "batch_size": 8,
    }
    loss_metrics = [
        {"type": "ImportanceMinimalityLoss", "coeff": 3e-3, "gamma": 1.0},
        {"type": "StochasticReconLoss", "coeff": 1.0},
        {"type": "UnmaskedNoDeltaReconLoss", "coeff": 1.0},
    ]
    with pytest.raises(Exception, match="UnmaskedNoDeltaReconLoss"):
        TargetedPDConfig.model_validate({**base, "loss_metrics": loss_metrics})
    with pytest.raises(Exception, match="UnmaskedNoDeltaReconLoss"):
        PDConfig.model_validate(
            {**base, "loss_metrics": [{"type": "FaithfulnessLoss", "coeff": 1.0}, *loss_metrics]}
        )


def test_delta_pinned_masks_pin_every_delta_to_one():
    # T4: both non-target mask constructions carry an all-ones delta mask per site.
    ci_lower = {
        "a": jax.random.uniform(jax.random.PRNGKey(0), (4, 6)),
        "b": jax.random.uniform(jax.random.PRNGKey(1), (4, 3)),
    }
    stoch_masks, stoch_deltas = stochastic_delta_pinned_masks(ci_lower, jax.random.PRNGKey(2))
    const_masks, const_deltas = constant_delta_pinned_masks(0.0, ci_lower)
    for site in ("a", "b"):
        assert jnp.array_equal(stoch_deltas[site], jnp.ones((4,)))
        assert jnp.array_equal(const_deltas[site], jnp.ones((4,)))
        # S1 interpolation: masks lie in [ci, 1] for stochastic, equal ci at value 0.
        assert bool(jnp.all(stoch_masks[site] >= ci_lower[site]))
        assert jnp.array_equal(const_masks[site], ci_lower[site])
