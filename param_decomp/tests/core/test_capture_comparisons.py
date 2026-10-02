"""Named capture groups compare paired arrays without imposing target semantics."""

from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array
from pydantic import ValidationError

from param_decomp.core.configs import (
    AuxiliaryReconstructionConfig,
    CaptureReconstruction,
    NontargetConfig,
    StochasticReconLossConfig,
)
from param_decomp.core.losses import (
    ReconstructionLoss,
    categorical_kl_from_logits,
    reconstruction_loss,
    reconstruction_loss_metrics,
)
from param_decomp.core.recon import (
    AuxiliaryReconstruction,
    ForwardObservations,
    auxiliary_capture_keys,
    resolve_auxiliary_reconstruction,
)
from param_decomp.core.schedule import Knot, ScheduleConfig


def test_categorical_kl_direction_and_gradients_match_categorical_oracle():
    p = jnp.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    q = jnp.array([[0.2, 0.3, 0.5], [0.6, 0.2, 0.2]])
    masked, clean = jnp.log(q), jnp.log(p)
    expected = np.mean(np.sum(np.asarray(p) * np.log(np.asarray(p / q)), axis=-1))
    assert float(categorical_kl_from_logits(masked, clean)) == pytest.approx(expected, rel=1e-6)
    grad_masked, grad_clean = jax.grad(categorical_kl_from_logits, argnums=(0, 1))(masked, clean)
    np.testing.assert_allclose(grad_masked, (q - p) / 2, atol=3e-8)
    np.testing.assert_array_equal(grad_clean, jnp.zeros_like(clean))
    assert float(categorical_kl_from_logits(clean, clean)) == 0.0


def test_categorical_kl_does_not_clip_underflowing_probabilities():
    clean = jnp.array([[1000.0, -1000.0]])
    masked = -clean
    value, gradient = jax.value_and_grad(categorical_kl_from_logits)(masked, clean)
    assert float(value) == 2000.0
    np.testing.assert_array_equal(gradient, jnp.array([[-1.0, 1.0]]))


@pytest.mark.parametrize("position_shape", [(), (3,), (2, 3)])
def test_valid_rows_count_tokens_without_physical_padding(position_shape: tuple[int, ...]):
    clean = jax.random.normal(jax.random.key(1), (4, *position_shape, 5))
    masked = jax.random.normal(jax.random.key(2), clean.shape)
    valid = jnp.array([1.0, 0.0, 1.0, 0.0])
    expected = categorical_kl_from_logits(masked[jnp.array([0, 2])], clean[jnp.array([0, 2])])
    value, gradient = jax.value_and_grad(categorical_kl_from_logits)(
        masked, clean, valid_row_mask=valid
    )
    np.testing.assert_allclose(value, expected, rtol=2e-6)
    np.testing.assert_array_equal(gradient[jnp.array([1, 3])], 0.0)
    assert float(categorical_kl_from_logits(masked, clean, valid_row_mask=jnp.zeros(4))) == 0.0


def test_router_and_hidden_comparisons_compose_and_average_layers():
    clean = ForwardObservations(
        jnp.array(7.0),
        {
            "hidden": jnp.array([[2.0]]),
            "router.0": jnp.log(jnp.array([[0.8, 0.2]])),
            "router.1": jnp.log(jnp.array([[0.3, 0.3, 0.4]])),
        },
    )
    masked = ForwardObservations(
        jnp.array(10.0),
        {
            "hidden": jnp.array([[1.0]]),
            "router.0": jnp.log(jnp.array([[0.2, 0.8]])),
            "router.1": jnp.log(jnp.array([[0.5, 0.2, 0.3]])),
        },
    )
    spec = (
        AuxiliaryReconstruction(
            "hidden_acts_reconstruction",
            2.0,
            (CaptureReconstruction(capture="hidden", distance="relative_squared_error"),),
        ),
        AuxiliaryReconstruction(
            "router_kl",
            3.0,
            tuple(
                CaptureReconstruction(capture=point, distance="categorical_kl_from_logits")
                for point in ("router.0", "router.1")
            ),
        ),
    )
    result = reconstruction_loss(
        lambda m, c: m - c, masked=masked, clean=clean, reconstruction=spec
    )
    router_mean = (
        sum(
            float(categorical_kl_from_logits(masked.captures[k], clean.captures[k]))
            for k in ("router.0", "router.1")
        )
        / 2
    )
    assert float(result.total) == pytest.approx(3 + 2 * 0.25 + 3 * router_mean, rel=1e-6)
    metrics = reconstruction_loss_metrics(result)
    assert float(metrics["router_kl"]) == pytest.approx(router_mean, rel=1e-6)
    assert float(metrics["hidden_acts_reconstruction"]) == 0.25
    assert auxiliary_capture_keys(spec) == frozenset(clean.captures)
    assert tuple(result.auxiliaries) == ("hidden_acts_reconstruction", "router_kl")


def test_static_eval_rejects_scheduled_auxiliary_strength():
    schedule = ScheduleConfig(max_val=2.0, points=(Knot(at=0.0, frac=0.0), Knot(at=1.0, frac=1.0)))
    comparisons = (
        CaptureReconstruction(
            capture="categorical_prediction", distance="categorical_kl_from_logits"
        ),
    )
    config = AuxiliaryReconstructionConfig(
        name="prediction", coeff=schedule, comparisons=comparisons
    )
    with pytest.raises(AssertionError, match="constant float"):
        resolve_auxiliary_reconstruction((config,))
    measurement = resolve_auxiliary_reconstruction(
        (AuxiliaryReconstructionConfig(name="prediction", coeff=0.0, comparisons=comparisons),)
    )
    assert measurement[0].coeff == 0.0
    assert auxiliary_capture_keys(measurement) == frozenset({"categorical_prediction"})


def test_auxiliary_config_requires_explicit_nonnegative_strength():
    comparison = {"capture": "prediction", "distance": "categorical_kl_from_logits"}
    for invalid in ({}, {"coeff": -1.0}):
        with pytest.raises(ValidationError):
            AuxiliaryReconstructionConfig.model_validate(
                {"name": "prediction", "comparisons": [comparison], **invalid}
            )
    config = StochasticReconLossConfig.model_validate(
        {
            "coeff": 1.0,
            "auxiliaries": [{"name": "prediction", "coeff": 0.2, "comparisons": [comparison]}],
        }
    )
    assert len(config.auxiliaries) == 1
    assert config.auxiliaries[0].coeff == 0.2
    with pytest.raises(ValidationError, match="target-pass-only"):
        NontargetConfig.model_validate(
            {"batch_size": 4, "impmin_coeff": 1.0, "recon": [config.model_dump()]}
        )


def test_heterogeneous_comparisons_share_group_mean_under_jit():
    clean = ForwardObservations(
        jnp.array(2.0),
        {"features": jnp.array([[2.0, 4.0]]), "prediction": jnp.log(jnp.array([[0.2, 0.3, 0.5]]))},
    )
    masked = ForwardObservations(
        jnp.array(5.0),
        {"features": jnp.array([[1.0, 2.0]]), "prediction": jnp.log(jnp.array([[0.5, 0.2, 0.3]]))},
    )
    spec = (
        AuxiliaryReconstruction(
            "readouts",
            3.0,
            (
                CaptureReconstruction(capture="features", distance="relative_squared_error"),
                CaptureReconstruction(capture="prediction", distance="categorical_kl_from_logits"),
            ),
        ),
    )

    @jax.jit
    def compare(
        masked: ForwardObservations[Array], clean: ForwardObservations[Array]
    ) -> ReconstructionLoss:
        return reconstruction_loss(
            lambda m, c: m - c, masked=masked, clean=clean, reconstruction=spec
        )

    result = compare(masked, clean)
    kl = categorical_kl_from_logits(masked.captures["prediction"], clean.captures["prediction"])
    assert float(result.total) == pytest.approx(3.0 + 3.0 * (0.25 + float(kl)) / 2, rel=1e-6)
    np.testing.assert_allclose(result.auxiliaries["readouts"]["features"], 0.25)


@pytest.mark.parametrize("distance", ["relative_squared_error", "categorical_kl_from_logits"])
def test_comparisons_reject_broadcasting_at_the_array_pair_boundary(
    distance: Literal["relative_squared_error", "categorical_kl_from_logits"],
):
    spec = (
        AuxiliaryReconstruction(
            "readout", 1.0, (CaptureReconstruction(capture="point", distance=distance),)
        ),
    )
    from jaxtyping import TypeCheckError

    with pytest.raises(TypeCheckError):
        reconstruction_loss(
            lambda m, c: m - c,
            masked=ForwardObservations(jnp.array(0.0), {"point": jnp.ones((2, 1))}),
            clean=ForwardObservations(jnp.array(0.0), {"point": jnp.ones((2, 3))}),
            reconstruction=spec,
        )


@pytest.mark.parametrize("distance", ["relative_squared_error", "categorical_kl_from_logits"])
def test_comparison_row_mask_must_match_its_capture_batch(
    distance: Literal["relative_squared_error", "categorical_kl_from_logits"],
):
    from jaxtyping import TypeCheckError

    spec = (
        AuxiliaryReconstruction(
            "readout", 1.0, (CaptureReconstruction(capture="point", distance=distance),)
        ),
    )
    observations = ForwardObservations(jnp.array(0.0), {"point": jnp.ones((2, 3, 4))})
    with pytest.raises(TypeCheckError):
        reconstruction_loss(
            lambda m, c: m - c,
            masked=observations,
            clean=observations,
            reconstruction=spec,
            valid_row_mask=jnp.ones(1),
        )
