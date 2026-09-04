import jax
import jax.numpy as jnp
import pytest

from param_decomp.core.components import SiteSpec
from param_decomp.targets.glu_transformer import _neuron_aligned_site_factors


@pytest.mark.parametrize("units_on_input", [True, False])
@pytest.mark.parametrize("component_count", [2, 4, 6, 9])
def test_neuron_aligned_factors_select_or_exactly_partition(
    units_on_input: bool, component_count: int
) -> None:
    spec = (
        SiteSpec(name="site", d_in=4, d_out=5, C=component_count, group="site")
        if units_on_input
        else SiteSpec(name="site", d_in=5, d_out=4, C=component_count, group="site")
    )
    weight = 1 + jnp.arange(spec.d_out * spec.d_in, dtype=jnp.float32).reshape(
        spec.d_out, spec.d_in
    )

    V, U = _neuron_aligned_site_factors(weight, spec, units_on_input, jax.random.key(0))
    reconstructed = (V @ U).T

    assert V.shape == (spec.d_in, component_count)
    assert U.shape == (component_count, spec.d_out)
    assert jnp.all(jnp.linalg.norm(V, axis=0) > 0)
    assert jnp.all(jnp.linalg.norm(U, axis=1) > 0)
    if component_count >= 4:
        assert jnp.array_equal(reconstructed, weight)
    else:
        retained = jnp.any(reconstructed != 0, axis=0 if units_on_input else 1)
        assert int(retained.sum()) == component_count


@pytest.mark.parametrize("units_on_input", [True, False])
def test_neuron_aligned_equal_width_keeps_canonical_factorization(
    units_on_input: bool,
) -> None:
    spec = (
        SiteSpec(name="site", d_in=4, d_out=5, C=4, group="site")
        if units_on_input
        else SiteSpec(name="site", d_in=5, d_out=4, C=4, group="site")
    )
    weight = 1 + jnp.arange(spec.d_out * spec.d_in, dtype=jnp.float32).reshape(
        spec.d_out, spec.d_in
    )

    V, U = _neuron_aligned_site_factors(weight, spec, units_on_input, jax.random.key(0))

    if units_on_input:
        assert jnp.array_equal(V, jnp.eye(spec.d_in))
        assert jnp.array_equal(U, weight.T)
    else:
        assert jnp.array_equal(V, weight.T)
        assert jnp.array_equal(U, jnp.eye(spec.d_out))
