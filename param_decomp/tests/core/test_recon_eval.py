"""A fresh-PGD attack's read-outs come from one ascent."""

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from param_decomp.core.components import DenseFactorization, SiteSpec, require_full_emission
from param_decomp.core.model import MaterializedMasking
from param_decomp.core.recon_eval import FreshPGDAttack, fresh_pgd_masking, fresh_pgd_read_outs

SITES = (
    SiteSpec("a", DenseFactorization(d_in=4, d_out=4, C=6), "dense"),
    SiteSpec("b", DenseFactorization(d_in=4, d_out=4, C=3), "dense"),
)
LEADING = (2, 5)


def test_each_read_out_is_the_plain_ascent_of_that_depth():
    ci_key, weight_key, source_key = random.split(random.PRNGKey(0), 3)
    ci_lower = {
        site.name: random.uniform(key, (*LEADING, site.C))
        for site, key in zip(SITES, random.split(ci_key, len(SITES)), strict=True)
    }
    weights = {
        site.name: random.normal(key, (site.C,))
        for site, key in zip(SITES, random.split(weight_key, len(SITES)), strict=True)
    }

    def loss(masking: MaterializedMasking) -> jax.Array:
        # Oscillating in the masks, so the ascent's sign pattern changes as it climbs.
        return sum(
            (
                jnp.mean(jnp.sin(4.0 * require_full_emission(mask)) * weights[name])
                for name, mask in masking.component_masks.items()
            ),
            start=jnp.zeros(()),
        )

    attack = FreshPGDAttack(step_size=0.3, read_out_steps=(0, 1, 4, 9))
    read_outs = jax.jit(
        lambda: fresh_pgd_read_outs(SITES, ci_lower, LEADING, source_key, attack, loss, loss)
    )()
    plain = jax.jit(
        lambda n_steps: loss(
            fresh_pgd_masking(SITES, ci_lower, LEADING, source_key, attack.step_size, n_steps, loss)
        ),
        static_argnums=0,
    )

    np.testing.assert_allclose(
        read_outs, [plain(n_steps) for n_steps in attack.read_out_steps], rtol=1e-6
    )
    assert len(set(np.asarray(read_outs).tolist())) == len(read_outs), read_outs
