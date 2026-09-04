import jax.numpy as jnp
import pytest

from param_decomp.core.components import component_stacks_from_sites
from param_decomp.core.train import uv_norm_ratio_metrics


def test_uv_norm_ratio_uses_each_sites_frobenius_norms():
    components = component_stacks_from_sites(
        {
            "a": (jnp.full((3, 2), 1.0), jnp.full((2, 3), 2.0)),
            "b": (jnp.full((3, 2), 2.0), jnp.full((2, 3), 1.0)),
        }
    )

    metrics = uv_norm_ratio_metrics(components)

    assert float(metrics["uv_norm_ratio['a']"]) == pytest.approx(2.0)
    assert float(metrics["uv_norm_ratio['b']"]) == pytest.approx(0.5)
    assert float(metrics["uv_norm_ratio_mean"]) == pytest.approx(1.25)
    assert float(metrics["uv_norm_ratio_max"]) == pytest.approx(2.0)
