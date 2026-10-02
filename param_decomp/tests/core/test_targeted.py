"""Core tPD semantics: the targeted objective builders' closure
rules, the delta-pinned mask constructions, and the two-pass step factory's boundary
refusals. The full two-pass training run is pinned by the TMS seat's tests
(`param_decomp/tests/experiments/tms/test_targeted_tms.py`); this module needs no target."""

from typing import cast

import jax
import jax.numpy as jnp
import pytest

from param_decomp.core.components import DenseFactorization, SiteSpec, require_full_emission
from param_decomp.core.configs import (
    FaithfulnessLossConfig,
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
from param_decomp.core.losses import BatchFrequency, EmaFrequency, init_frequency_estimator
from param_decomp.core.masking import (
    constant_delta_pinned_masking,
    stochastic_delta_pinned_masking,
    unmasked_no_delta_masking,
)
from param_decomp.core.objective import build_targeted_objective
from param_decomp.core.recon import ConstantSources, StochasticSources, UnmaskedNoDeltaSources
from param_decomp.core.schedule import ScheduleConfig

SITES = tuple(
    SiteSpec(name=name, factorization=DenseFactorization(d_in=4, d_out=4, C=4), group="g")
    for name in ("a", "b")
)


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
    objective = build_targeted_objective(_target_metrics(), _nontarget(), SITES)
    assert float(
        objective.target.minimality.activity_coeff.at(jnp.asarray(0.0, jnp.float32))
    ) == pytest.approx(3e-3)
    assert float(
        objective.nontarget.minimality.activity_coeff.at(jnp.asarray(0.0, jnp.float32))
    ) == pytest.approx(6e-3)
    # The non-target surface is stochastic/constant-source only, output-only scored.
    for term in objective.nontarget.recon:
        assert isinstance(term.sources, StochasticSources | ConstantSources)


def test_targeted_pd_config_cannot_spell_faithfulness():
    # A faithfulness term — even at coeff 0 — is a different algorithm, and the
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


@pytest.mark.parametrize("halflife", [None, 500.0])
def test_targeted_pd_config_accepts_both_frequency_modes(halflife: float | None):
    pd = TargetedPDConfig.model_validate(
        {
            "loss_metrics": [
                {
                    "type": "ImportanceMinimalityLoss",
                    "coeff": 1e-4,
                    "gamma": 1.0,
                    "frequency": {
                        "coeff": 1e-4,
                        "reference_datapoint_count": 128,
                        "ema_halflife_steps": halflife,
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
    objective = build_targeted_objective(pd.loss_metrics, _nontarget(), SITES)
    assert objective.target.minimality.frequency is not None
    importance = next(
        term for term in pd.loss_metrics if isinstance(term, ImportanceMinimalityLossConfig)
    )
    frequency = init_frequency_estimator(importance.frequency, SITES)
    match halflife:
        case None:
            assert isinstance(frequency, BatchFrequency)
        case int() | float():
            assert isinstance(frequency, EmaFrequency)
            assert frequency.halflife_steps == halflife


def test_targeted_pd_config_ci_scaled_weight_decay_parses():
    # Absent is the real, intended state (None — no decay); a set value must be a
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
    # Decaying components would oppose plain PD's faithfulness objective, so the
    # plain schema excludes CI-scaled weight decay.
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
        build_targeted_objective(
            forged,
            _nontarget(),
            tuple(
                SiteSpec(
                    name=name, factorization=DenseFactorization(d_in=4, d_out=4, C=4), group="g"
                )
                for name in ("a", "b")
            ),
        )


def test_nontarget_schema_refuses_hidden_acts_at_parse():
    # Hidden-activation reconstruction is target-pass-only — refused when the
    # seat parses, not at objective build on the GPUs.
    with pytest.raises(Exception, match="auxiliar"):
        NontargetConfig.model_validate(
            {
                "batch_size": 32,
                "impmin_coeff": 6e-3,
                "recon": [
                    {
                        "type": "StochasticReconLoss",
                        "coeff": 1.0,
                        "auxiliaries": [
                            {
                                "name": "hidden_acts_reconstruction",
                                "coeff": 0.1,
                                "comparisons": [
                                    {"capture": "resid.1", "distance": "relative_squared_error"}
                                ],
                            }
                        ],
                    }
                ],
            }
        )


def test_nontarget_schema_rejects_adversarial_sources():
    # Adversarial sources are excluded by the non-target schema.
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
    # The term is fully determined by its type: routing and optional hidden-activation
    # reconstruction fields are structurally absent, refused at parse (`extra="forbid"`), not validated away.
    cfg = UnmaskedNoDeltaReconLossConfig.model_validate(
        {"type": "UnmaskedNoDeltaReconLoss", "coeff": 1.0}
    )
    assert cfg.coeff == 1.0
    for extra in (
        {"routing": {"type": "UniformKSubsetRouting"}},
        {
            "auxiliaries": [
                {
                    "name": "hidden_acts_reconstruction",
                    "coeff": 0.1,
                    "comparisons": [{"capture": "resid.1", "distance": "relative_squared_error"}],
                }
            ]
        },
    ):
        with pytest.raises(Exception, match="[Ee]xtra"):
            UnmaskedNoDeltaReconLossConfig.model_validate(
                {"type": "UnmaskedNoDeltaReconLoss", "coeff": 1.0, **extra}
            )


def test_unmasked_no_delta_masking_is_ones_without_delta():
    ci_lower = {
        "a": jax.random.uniform(jax.random.PRNGKey(0), (4, 6)),
        "b": jax.random.uniform(jax.random.PRNGKey(1), (4, 3)),
    }
    masking = unmasked_no_delta_masking(ci_lower)
    masks = masking.component_masks
    assert masking.weight_delta_masks is None
    for site in ("a", "b"):
        assert jnp.array_equal(require_full_emission(masks[site]), jnp.ones_like(ci_lower[site]))


def test_targeted_objective_admits_unmasked_no_delta_for_nontarget():
    nontarget = NontargetConfig(
        batch_size=32,
        impmin_coeff=6e-3,
        recon=[
            UnmaskedNoDeltaReconLossConfig(coeff=0.5),
            StochasticReconSubsetLossConfig(coeff=1.0, routing=UniformKSubsetRoutingConfig()),
        ],
    )
    objective = build_targeted_objective(_target_metrics(), nontarget, SITES)
    unmasked_term = next(
        t for t in objective.nontarget.recon if t.name == "UnmaskedNoDeltaReconLoss"
    )
    # A single all-routed draw — the term is fully determined.
    assert isinstance(unmasked_term.sources, UnmaskedNoDeltaSources)
    assert unmasked_term.sample_routing(jax.random.PRNGKey(0), (4,)) is None


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
    # Both non-target mask constructions carry an all-ones delta mask per site.
    ci_lower = {
        "a": jax.random.uniform(jax.random.PRNGKey(0), (4, 6)),
        "b": jax.random.uniform(jax.random.PRNGKey(1), (4, 3)),
    }
    stoch_masking = stochastic_delta_pinned_masking(ci_lower, jax.random.PRNGKey(2))
    stoch_masks = stoch_masking.component_masks
    assert stoch_masking.weight_delta_masks is not None
    stoch_deltas = stoch_masking.weight_delta_masks
    const_masking = constant_delta_pinned_masking(0.0, ci_lower)
    const_masks = const_masking.component_masks
    assert const_masking.weight_delta_masks is not None
    const_deltas = const_masking.weight_delta_masks
    for site in ("a", "b"):
        assert jnp.array_equal(stoch_deltas[site], jnp.ones((4,)))
        assert jnp.array_equal(const_deltas[site], jnp.ones((4,)))
        # Interpolation: masks lie in [ci, 1] for stochastic, equal ci at value 0.
        assert bool(jnp.all(require_full_emission(stoch_masks[site]) >= ci_lower[site]))
        assert jnp.array_equal(require_full_emission(const_masks[site]), ci_lower[site])


def test_ci_scaled_weight_decay_scales_expert_blocked_stacks_expert_major():
    """CI-scaled decay maps flat component `(e, k)` to expert `e`'s factor block.

    Blocked stacks use expert-major `C = E * c` ordering; dense stacks retain their
    `[g, d_in, C]` broadcast."""
    from param_decomp.core.components import (
        BlockedFactorization,
        ComponentStacks,
        DenseFactorization,
    )
    from param_decomp.core.train import _scale_subcomponents

    g, E, d_in, d_out, c = 2, 3, 4, 5, 2
    expert = BlockedFactorization(n_blocks=E, d_in=d_in, d_out=d_out, c_per_block=c)
    dense = DenseFactorization(d_in=d_in, d_out=d_out, C=4)
    stacks = ComponentStacks(
        stacks={
            "experts": (jnp.ones((g, E, d_in, c)), jnp.ones((g, E, c, d_out))),
            "shared": (jnp.ones((g, d_in, dense.C)), jnp.ones((g, dense.C, d_out))),
        },
        site_stack_indices=(
            ("experts.0", "experts", 0),
            ("experts.1", "experts", 1),
            ("shared.0", "shared", 0),
            ("shared.1", "shared", 1),
        ),
    )
    # Zero exactly component (e=1, k=0) of site experts.0: flat index e*c + k = 2.
    expert_scale = jnp.ones((E * c,)).at[2].set(0.0)
    scale = {
        "experts.0": expert_scale,
        "experts.1": jnp.ones((E * c,)),
        "shared.0": jnp.full((dense.C,), 0.5),
        "shared.1": jnp.ones((dense.C,)),
    }
    scaled = _scale_subcomponents(stacks, scale, {"experts": expert, "shared": dense})

    vs, us = scaled.stacks["experts"]
    assert vs.shape == (g, E, d_in, c) and us.shape == (g, E, c, d_out)
    assert jnp.array_equal(vs[0, 1, :, 0], jnp.zeros((d_in,)))
    assert jnp.array_equal(us[0, 1, 0, :], jnp.zeros((d_out,)))
    # Everything else in the expert group is untouched (slot 1 entirely).
    assert bool(jnp.all(vs[1] == 1.0)) and bool(jnp.all(us[1] == 1.0))
    assert bool(jnp.all(vs[0, 1, :, 1] == 1.0)) and bool(jnp.all(vs[0, 0] == 1.0))

    dvs, dus = scaled.stacks["shared"]
    assert bool(jnp.all(dvs[0] == 0.5)) and bool(jnp.all(dus[0] == 0.5))
    assert bool(jnp.all(dvs[1] == 1.0)) and bool(jnp.all(dus[1] == 1.0))
