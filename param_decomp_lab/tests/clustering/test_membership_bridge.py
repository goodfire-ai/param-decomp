"""The JAX clustering bridge turns per-site lower-leaky CI `(B, T, C)` into the
`(samples, C)` numpy dict `MembershipBuilder` consumes, sampling token positions
via `flatten_lm_activations`."""

import jax.numpy as jnp
import numpy as np

from param_decomp_lab.clustering.memberships import flatten_lm_activations
from param_decomp_lab.clustering.scripts.run_worker import sampled_ci_from_forward


def test_sampled_ci_all_positions_matches_flatten() -> None:
    rng = np.random.default_rng(0)
    ci = rng.uniform(0.0, 1.0, size=(2, 5, 3)).astype(np.float32)
    site = "h.0.mlp.c_fc"

    sampled = sampled_ci_from_forward(
        {site: jnp.asarray(ci)},
        n_tokens_per_seq=None,
        use_all_tokens_per_seq=True,
        rng=np.random.default_rng(0),
    )

    assert sampled[site].shape == (2 * 5, 3)
    np.testing.assert_allclose(sampled[site], ci.reshape(2 * 5, 3))


def test_sampled_ci_random_positions_matches_flatten_under_same_rng() -> None:
    rng = np.random.default_rng(1)
    ci = rng.uniform(0.0, 1.0, size=(4, 7, 6)).astype(np.float32)
    site = "h.0.mlp.c_fc"

    sampled = sampled_ci_from_forward(
        {site: jnp.asarray(ci)},
        n_tokens_per_seq=3,
        use_all_tokens_per_seq=False,
        rng=np.random.default_rng(123),
    )

    expected = flatten_lm_activations(
        ci,
        batch_size=4,
        n_ctx=7,
        n_tokens_per_seq=3,
        use_all_tokens_per_seq=False,
        rng=np.random.default_rng(123),
    )

    assert sampled[site].shape == (4 * 3, 6)
    np.testing.assert_allclose(sampled[site], expected)
