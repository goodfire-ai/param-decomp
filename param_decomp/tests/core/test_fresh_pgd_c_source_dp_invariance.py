"""Multi-device invariance of the fresh-PGD `c`-source sign-ascent.

The fresh-PGD eval probe (`PGDReconLoss`, fresh sign-PGD, `c`-source, 20 step;
`eval.py`) and the training-loss path (`train.py` `sign_ascend_body`) ascend a
`c`-scope component source — shape `(1, 1, C)`, shared across the whole batch and sequence —
by `step_size * sign(grad)` with a clamp to [0,1], where the grad comes from a
batch-reduced KL loss.

Torch AVG-reduces that source grad across data-parallel ranks BEFORE `sign()`; JAX
lets GSPMD SUM-reduce the per-shard grads and takes `sign(global grad)`. The property
under test is `sign(avg(g)) == sign(sum(g))`: both pick the same sign per source
entry, so the ascended source — and therefore the materialized mask — is BIT-identical
across device layouts. Sign is an exact decision, so no float tolerance is needed.

This pins the missing multi-device invariance for fresh-PGD `c`-source. It exercises
the SAME ascent body as production:
a genuine batch-reduced `kl_per_position` loss whose source grad, for a `c`-source
`(1, 1, C)` source, must be reduced across the sharded batch axis.

Run at the default device count AND under simulated multi-device CPU:

  XLA_FLAGS="--xla_force_host_platform_device_count=4" \
    python -m pytest param_decomp/tests/core/test_fresh_pgd_c_source_dp_invariance.py
"""

import contextlib

import jax
import jax.numpy as jnp
from jax import random

from param_decomp.core.adversary import Sources, full_source_components, init_fresh_pgd_sources
from param_decomp.core.components import (
    ComponentStacks,
    SiteSpec,
    init_component_stacks,
    require_full_emission,
)
from param_decomp.core.masking import materialize_masking, source_masking
from param_decomp.core.model import DecomposedModel
from param_decomp.core.sharding import hsdp_mesh, shard_batch
from param_decomp.lm.batch import LMBatchWithDocuments
from param_decomp.targets.lm_output import LMOutput
from param_decomp.targets.losses import kl_per_position
from param_decomp.targets.testing import run_clean, run_masked, tiny_glu_cfg, tiny_glu_decomposed_lm
from param_decomp.targets.transformer import (
    TransformerPreparedMasking,
    TransformerPreparedWeights,
    glu_site_specs,
    mlp_family_site_cs,
)


def _ascend_c_source(
    sharded: bool, n_steps: int, step_size: float
) -> tuple[Sources, dict[str, jax.Array]]:
    """Run the fresh-PGD `c`-source sign-ascent on a fixed batch+seed and return the
    ascended sources plus their materialized masks (`materialize_masking`).

    Mirrors `train.py` `sign_ascend_body`: a batch-reduced KL ascent loss, grad w.r.t.
    a `(1, 1, C)` `c` source, `step_size * sign(grad)`, clamp to [0,1]. When
    `sharded`, the residual is GSPMD-sharded over all visible devices, so the `c`-source
    source grad is born from a cross-shard reduction."""
    cfg = tiny_glu_cfg()
    first_layer = 3
    C, seq, gbatch = 8, 16, 8
    sites = glu_site_specs(cfg, mlp_family_site_cs(first_layer, first_layer + 2, C))
    model = tiny_glu_decomposed_lm(cfg, sites, random.PRNGKey(0))
    components = jax.tree.map(
        lambda x: jax.lax.stop_gradient(x), init_component_stacks(sites, random.PRNGKey(1))
    )

    residual = random.randint(random.PRNGKey(4), (gbatch, seq), 0, cfg.vocab_size)
    mesh = hsdp_mesh(1, jax.device_count(), 1) if sharded else None
    if mesh is not None:
        residual = shard_batch(residual, mesh, batch_axis=0)
        # Explicit mode refuses ops mixing mesh-committed and off-mesh operands;
        # everything the forward touches lives on the mesh (frozen weights replicated).
        from jax.sharding import NamedSharding, PartitionSpec

        replicated = NamedSharding(mesh, PartitionSpec())
        model = jax.tree.map(lambda a: jax.device_put(a, replicated), model)
        components = jax.tree.map(lambda a: jax.device_put(a, replicated), components)

    mesh_context = jax.set_mesh(mesh) if mesh is not None else contextlib.nullcontext()
    with mesh_context:
        return _ascend(sites, model, components, residual, n_steps, step_size)


def _ascend(
    sites: tuple[SiteSpec, ...],
    model: DecomposedModel[
        LMBatchWithDocuments,
        LMOutput,
        TransformerPreparedWeights,
        LMBatchWithDocuments,
        TransformerPreparedMasking,
    ],
    components: ComponentStacks,
    residual: jax.Array,
    n_steps: int,
    step_size: float,
) -> tuple[Sources, dict[str, jax.Array]]:
    gbatch, seq = residual.shape
    clean_output = jax.lax.stop_gradient(
        run_clean(model, LMBatchWithDocuments.from_unsegmented_sequences(residual))
    )
    assert isinstance(clean_output, jax.Array)
    # ci_lower = 0 so the mask is just the `c` source — the cleanest probe of the
    # sign-ascent. Shapes match the masked forward's per-site (B, T, C) expectation.
    ci_lower = {s.name: jnp.zeros((gbatch, seq, s.C), jnp.float32) for s in sites}

    init = init_fresh_pgd_sources(sites, "random", "c", (gbatch, seq), random.PRNGKey(5))

    def ascent_loss(sources: Sources) -> jax.Array:
        masking = materialize_masking(source_masking(ci_lower, sources))
        masked = run_masked(
            model,
            model.prepare_compute_weights(components, None),
            LMBatchWithDocuments.from_unsegmented_sequences(residual),
            masking,
            remat=False,
            routes=None,
        )
        assert isinstance(masked, jax.Array)
        return kl_per_position(masked, clean_output)

    def sign_ascend_body(sources: Sources, _: None) -> tuple[Sources, None]:
        sources_grad = jax.grad(ascent_loss)(sources)
        return jax.tree.map(
            lambda source, gradient: jnp.clip(source + step_size * jnp.sign(gradient), 0.0, 1.0),
            sources,
            sources_grad,
        ), None

    ascended, _ = jax.lax.scan(sign_ascend_body, init, None, length=n_steps)
    masking = materialize_masking(source_masking(ci_lower, ascended))
    masks = masking.component_masks
    return ascended, {site: require_full_emission(mask) for site, mask in masks.items()}


def test_fresh_pgd_c_source_sign_ascent_is_device_count_invariant():
    """The `c`-source ascended source AND its mask are bit-identical at 1 layout vs N
    GSPMD shards. `sign(avg)==sign(sum)`, so the sign decision is exact — assert with
    NO float tolerance. Guards fresh-PGD `c`-source DP equivalence."""
    n_dev = len(jax.devices())
    n_steps, step_size = 20, 0.05

    src_single, mask_single = _ascend_c_source(False, n_steps, step_size)
    src_sharded, mask_sharded = _ascend_c_source(True, n_steps, step_size)

    for name in src_single:
        assert full_source_components(src_single[name].components).shape == (1, 1, 8)
        assert src_single[name].delta.shape == (1, 1)
        assert all(
            jnp.array_equal(a, b)
            for a, b in zip(
                jax.tree.leaves(src_single[name]),
                jax.tree.leaves(src_sharded[name]),
                strict=True,
            )
        ), f"fresh-PGD source diverged at {name} across 1 vs {n_dev} shards"
        assert jnp.array_equal(jnp.asarray(mask_single[name]), jnp.asarray(mask_sharded[name])), (
            f"materialized mask diverged at {name} across 1 vs {n_dev} shards"
        )
