"""StepControls threading: protect_ci pins protected slots to 1 in both squashings and
zeroes their gradient into the CI logits; unprotected slots are untouched."""

import jax
import jax.numpy as jnp

from param_decomp.core.ci_fn import CI, protect_ci


def test_protect_ci_pins_values_and_zeroes_gradient() -> None:
    logits = {"s1": jnp.array([[-2.0, 0.5, 3.0]]), "s2": jnp.array([[0.25, 0.75]])}
    protected = {"s1": jnp.array([True, False, False])}

    def summed_lower(raw: dict[str, jax.Array]) -> jax.Array:
        ci = protect_ci(CI.from_logits(raw), protected)
        assert ci.lower["s1"][0, 0] == 1.0 and ci.upper["s1"][0, 0] == 1.0
        total = jnp.zeros(())
        for v in list(ci.lower.values()) + list(ci.upper.values()):
            total = total + v.sum()
        return total

    grads = jax.grad(summed_lower)(logits)
    assert grads["s1"][0, 0] == 0.0  # protected: where() cuts the path to the logit
    assert grads["s1"][0, 1] != 0.0  # unprotected slot in a protected site still trains
    assert jnp.all(grads["s2"] != 0.0)  # unlisted site untouched


def test_protect_ci_none_is_identity() -> None:
    ci = CI.from_logits({"s1": jnp.array([[0.5]])})
    assert protect_ci(ci, None) is ci
