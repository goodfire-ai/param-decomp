"""`_grad_norm_metrics` reports components grad norms per SITE (the semantic unit),
not per shape-group stack (the storage unit), under the historical per-site key format
`grad_norms/components.vu['<site>'][0|1]`."""

import jax
import jax.numpy as jnp
import numpy as np

from param_decomp.core.components import component_stacks_from_sites
from param_decomp.core.train import _grad_norm_metrics

SITE_VU_SHAPES = {
    "layers.0.mlp.gate_proj": (4, 8, 3),
    "layers.1.mlp.gate_proj": (4, 8, 3),
    "layers.0.self_attn.q_proj": (4, 6, 2),
}


def test_components_grad_norms_are_per_site() -> None:
    keys = iter(jax.random.split(jax.random.key(0), 2 * len(SITE_VU_SHAPES) + 1))
    vu = {
        name: (
            jax.random.normal(next(keys), (d_in, c)),
            jax.random.normal(next(keys), (c, d_out)),
        )
        for name, (d_in, d_out, c) in SITE_VU_SHAPES.items()
    }
    components_grad = component_stacks_from_sites(vu)
    assert len(components_grad.stacks) == 2  # two shape groups, three sites
    ci_fn_grad = {"w": jax.random.normal(next(keys), (5,))}

    metrics = _grad_norm_metrics(components_grad, ci_fn_grad)

    for name, (v, u) in vu.items():
        for factor, grad in enumerate((v, u)):
            got = metrics[f"grad_norms/components.vu['{name}'][{factor}]"]
            assert np.isfinite(got) and got > 0
            np.testing.assert_allclose(got, jnp.linalg.norm(grad), rtol=1e-6)

    per_site_keys = {k for k in metrics if k.startswith("grad_norms/components")}
    assert per_site_keys == {
        f"grad_norms/components.vu['{name}'][{factor}]"
        for name in SITE_VU_SHAPES
        for factor in (0, 1)
    }

    all_grads = [g for pair in vu.values() for g in pair] + [ci_fn_grad["w"]]
    total = jnp.sqrt(sum(jnp.sum(g**2) for g in all_grads))
    np.testing.assert_allclose(metrics["grad_norms/summary/total"], total, rtol=1e-6)
    assert metrics["grad_norms/ci_fns['w']"] == jnp.linalg.norm(ci_fn_grad["w"])
