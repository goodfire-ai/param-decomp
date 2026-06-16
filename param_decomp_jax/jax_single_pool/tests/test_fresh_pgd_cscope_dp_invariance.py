"""Multi-device invariance of the fresh-PGD c-scope sign-ascent (SPEC S24, S12', S15, D1).

The fresh-PGD eval probe (`PGDReconLoss`, fresh sign-PGD, c-scope, 20 step;
`eval.py`) and the training-loss path (`train.py` `sign_ascend_body`) ascend a
`c`-scope source — shape `(1, 1, C+1)`, shared across the whole batch and sequence —
by `step_size * sign(grad)` with a clamp to [0,1], where the grad comes from a
batch-reduced KL loss.

Torch AVG-reduces that source grad across data-parallel ranks BEFORE `sign()`; JAX
lets GSPMD SUM-reduce the per-shard grads and takes `sign(global grad)`. The property
under test is `sign(avg(g)) == sign(sum(g))`: both pick the same sign per source
entry, so the ascended source — and therefore the materialized mask — is BIT-identical
across device layouts. Sign is an exact decision, so no float tolerance is needed.

This pins the missing multi-device invariance for fresh-PGD c-scope (PARITY_MATRIX
§7/§1; Part 2 open question #7). It exercises the SAME ascent body as production:
a genuine batch-reduced `kl_per_position` loss whose source grad, for a c-scope
`(1, 1, C+1)` source, must be reduced across the sharded batch axis.

Run at the default device count AND under simulated multi-device CPU:

  XLA_FLAGS="--xla_force_host_platform_device_count=4" \
    python -m pytest jax_single_pool/tests/test_fresh_pgd_cscope_dp_invariance.py
"""

import jax
import jax.numpy as jnp
from jax import random

from jax_single_pool.adversary import init_fresh_pgd_sources, source_masks
from jax_single_pool.llama8b import (
    init_decomp_vu,
    llama_decomposed_lm,
    llama_site_specs,
    mlp_family_site_cs,
)
from jax_single_pool.losses import kl_per_position
from jax_single_pool.sharding import dp_mesh, shard_batch
from jax_single_pool.tests.test_llama8b import _tiny_cfg, _tiny_target


def _ascend_cscope_source(
    sharded: bool, n_steps: int, step_size: float
) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
    """Run the fresh-PGD c-scope sign-ascent on a fixed batch+seed and return the
    ascended sources plus their materialized masks (`source_masks`).

    Mirrors `train.py` `sign_ascend_body`: a batch-reduced KL ascent loss, grad w.r.t.
    a `(1, 1, C+1)` c-scope source, `step_size * sign(grad)`, clamp to [0,1]. When
    `sharded`, the residual is GSPMD-sharded over all visible devices, so the c-scope
    source grad is born from a cross-shard reduction."""
    cfg = _tiny_cfg()
    first_layer = 3
    frozen = _tiny_target(cfg, first_layer, random.PRNGKey(0))
    C, seq, gbatch = 8, 16, 8
    sites = llama_site_specs(cfg, mlp_family_site_cs(first_layer, first_layer + 2, C))
    lm = llama_decomposed_lm(cfg, sites)
    components = jax.tree.map(
        lambda x: jax.lax.stop_gradient(x), init_decomp_vu(sites, random.PRNGKey(1))
    )

    residual = random.normal(random.PRNGKey(4), (gbatch, seq, cfg.n_embd)) * 0.5
    mesh = dp_mesh() if sharded else None
    if mesh is not None:
        residual = shard_batch(residual, mesh, batch_axis=0)

    clean_output = jax.lax.stop_gradient(lm.clean_output(frozen, residual))
    # ci_lower = 0 so the mask is just the c-scope source — the cleanest probe of the
    # sign-ascent. Shapes match the masked forward's per-site (B, T, C) expectation.
    ci_lower = {s.name: jnp.zeros((gbatch, seq, s.C), jnp.float32) for s in sites}

    init = init_fresh_pgd_sources(sites, "random", "c", gbatch, seq, random.PRNGKey(5))

    def ascent_loss(sources: dict[str, jax.Array]) -> jax.Array:
        masks, delta_masks = source_masks(ci_lower, sources, lm.site_names)
        masked = lm.masked_output(
            frozen, components, residual, masks, delta_masks, None, lm.site_names, True
        )
        return kl_per_position(masked, clean_output)

    def sign_ascend_body(
        sources: dict[str, jax.Array], _: None
    ) -> tuple[dict[str, jax.Array], None]:
        sources_grad = jax.grad(ascent_loss)(sources)
        return {
            site: jnp.clip(sources[site] + step_size * jnp.sign(sources_grad[site]), 0.0, 1.0)
            for site in sources
        }, None

    ascended, _ = jax.lax.scan(sign_ascend_body, init, None, length=n_steps)
    masks, _ = source_masks(ci_lower, ascended, lm.site_names)
    return ascended, masks


def test_fresh_pgd_cscope_sign_ascent_is_device_count_invariant():
    """The c-scope ascended source AND its mask are bit-identical at 1 layout vs N
    GSPMD shards. `sign(avg)==sign(sum)`, so the sign decision is exact — assert with
    NO float tolerance. Guards fresh-PGD c-scope DP equivalence (SPEC S24, S12', S15, D1)."""
    n_dev = len(jax.devices())
    n_steps, step_size = 20, 0.05

    src_single, mask_single = _ascend_cscope_source(False, n_steps, step_size)
    src_sharded, mask_sharded = _ascend_cscope_source(True, n_steps, step_size)

    for name in src_single:
        a, b = jnp.asarray(src_single[name]), jnp.asarray(src_sharded[name])
        assert a.shape == (1, 1, 8 + 1), (name, a.shape)  # c-scope: (1, 1, C+1)
        assert jnp.array_equal(a, b), (
            f"fresh-PGD c-scope source diverged at {name} across 1 vs {n_dev} shards "
            f"— sign(avg)!=sign(sum)? (SPEC D1)"
        )
        assert jnp.array_equal(jnp.asarray(mask_single[name]), jnp.asarray(mask_sharded[name])), (
            f"materialized mask diverged at {name} across 1 vs {n_dev} shards"
        )
