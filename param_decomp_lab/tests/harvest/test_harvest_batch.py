"""The JAX `HarvestBatch` bridge: lower-leaky CI as `causal_importance`, ‖U‖·(x@V) as
`component_activation`, firing = CI > threshold, int64 tokens."""

import jax.numpy as jnp
import numpy as np

from param_decomp_lab.experiments.lm.load_run import HarvestForward
from param_decomp_lab.harvest.scripts.run_worker import harvest_batch_from_forward


def test_harvest_batch_from_forward_semantics() -> None:
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 100, size=(2, 5)).astype(np.int32)
    ci = rng.uniform(0.0, 1.0, size=(2, 5, 3)).astype(np.float32)
    acts = rng.normal(size=(2, 5, 3)).astype(np.float32)
    probs = rng.uniform(size=(2, 5, 100)).astype(np.float32)
    fwd = HarvestForward(
        lower_leaky_ci={"h.0.mlp.c_fc": jnp.asarray(ci)},
        component_acts={"h.0.mlp.c_fc": jnp.asarray(acts)},
        output_probs=jnp.asarray(probs),
    )

    hb = harvest_batch_from_forward(tokens, fwd, activation_threshold=0.3)

    assert hb.tokens.dtype == np.int64  # harvest path keys on int64 tokens
    np.testing.assert_array_equal(hb.tokens, tokens.astype(np.int64))
    site = "h.0.mlp.c_fc"
    np.testing.assert_allclose(hb.activations[site]["causal_importance"], ci)
    np.testing.assert_allclose(hb.activations[site]["component_activation"], acts)
    np.testing.assert_array_equal(hb.firings[site], ci > 0.3)
    np.testing.assert_allclose(hb.output_probs, probs)
