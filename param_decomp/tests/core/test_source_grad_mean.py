"""Batch-indexed training source gradients retain global-mean normalization.

The same fixed batch and independent source rows run through unplaced and placed
target forwards. Every sharded row must receive the same gradient as its global
reference, without multiplying or dividing by the number of data replicas.
"""

from contextlib import nullcontext

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from param_decomp.core.adversary import SourceStacks, init_persistent_sources
from param_decomp.core.ci_fn.implementations.chunkwise.arch import (
    Chunk,
    ChunkwiseTransformerCIFnArch,
)
from param_decomp.core.ci_fn.implementations.transformer.layers import MHACIFnAttention
from param_decomp.core.ci_fn.interface import CIFn
from param_decomp.core.components import ComponentStacks, init_component_stacks
from param_decomp.core.init_placed import (
    init_component_stacks_placed,
    init_sources_sharded,
)
from param_decomp.core.masking import source_masking
from param_decomp.core.model import PlacedModel, Positioned
from param_decomp.core.placement import from_config
from param_decomp.core.sharding import hsdp_mesh, place_target, shard_batch
from param_decomp.core.train import ForwardSubstrate
from param_decomp.lm.batch import LMBatchWithDocuments
from param_decomp.targets.lm_output import LMOutput
from param_decomp.targets.testing import (
    tiny_glu_cfg,
    tiny_glu_decomposed_lm,
)
from param_decomp.targets.transformer import (
    TransformerPreparedMasking,
    TransformerPreparedWeights,
    glu_site_specs,
    mlp_family_site_cs,
)
from param_decomp.tests.placed_ci_fn import placed_ci_fn


def _source_grad(sharded: bool) -> SourceStacks:
    """Score independent training source rows with frozen components and CI."""
    cfg = tiny_glu_cfg()
    C, seq, gbatch = 8, 16, 8
    sites = glu_site_specs(cfg, mlp_family_site_cs(3, 6, C))
    model = tiny_glu_decomposed_lm(cfg, sites, random.PRNGKey(0))
    first_block = min(int(name.split(".")[1]) for name in model.site_names)
    ci_fn_arch = ChunkwiseTransformerCIFnArch(
        chunks=(Chunk(input_taps=(f"resid.{first_block}",), output_sites=model.site_names),),
        input_dim=cfg.n_embd,
        d_model=16,
        n_blocks=2,
        attention=MHACIFnAttention(mask="bidirectional", implementation="xla", n_heads=2),
        ffn_hidden=32,
        ffn_kind="gelu",
        learned_norm_scale=False,
    )
    src = init_persistent_sources(model.sites, (gbatch, seq), jnp.float32, random.PRNGKey(3))
    resid = random.randint(random.PRNGKey(4), (gbatch, seq), 0, cfg.vocab_size)
    mesh = hsdp_mesh(jax.device_count(), 1, 1) if sharded else None

    with nullcontext() if mesh is None else jax.set_mesh(mesh):
        if mesh is None:
            placed = PlacedModel(model=model, placement=None)
            vu = init_component_stacks(sites, random.PRNGKey(1))
            ci_fn = ci_fn_arch.initialize(sites, None, random.PRNGKey(2))
        else:
            rules = from_config("zero1", mesh, sites)
            placed = place_target(model, rules)
            vu = init_component_stacks_placed(sites, random.PRNGKey(1), rules)
            ci_fn = placed_ci_fn(ci_fn_arch, sites, random.PRNGKey(2), mesh, rules)
            resid = shard_batch(resid, mesh, batch_axis=0)
            src = init_sources_sharded(
                sites, Positioned(seq), "bsc", gbatch, jnp.float32, random.PRNGKey(3), mesh
            )

        substrate = ForwardSubstrate.of(
            placed,
            remat_recon_forwards=False,
            remat_ci_fn=False,
            ci_capture_keys=ci_fn_arch.capture_keys,
        )

        @eqx.filter_jit
        def source_grad(
            target: PlacedModel[
                LMBatchWithDocuments,
                LMOutput,
                TransformerPreparedWeights,
                LMBatchWithDocuments,
                TransformerPreparedMasking,
            ],
            components: ComponentStacks,
            ci_fn: CIFn[LMBatchWithDocuments],
            sources: SourceStacks,
            batch: LMBatchWithDocuments,
        ) -> SourceStacks:
            stream = substrate.prep_stream(target, batch, frozenset())
            prepared = target.prepare_compute_weights(components)
            compute_ci_fn, _ = substrate.ci_fn_prepare_vjp(ci_fn)
            ci, _ = substrate.ci_fn_forward_vjp(compute_ci_fn, prepared, stream)

            def loss(source: SourceStacks) -> jax.Array:
                return substrate.masked_recon(
                    target,
                    prepared_weights=prepared,
                    stream=stream,
                    masking=target.model.prepare_masking(
                        source_masking(ci.lower, source.per_site())
                    ),
                    routes=None,
                    reconstruction=(),
                ).output

            return jax.grad(loss)(sources)

        return source_grad(
            placed, vu, ci_fn, src, LMBatchWithDocuments.from_unsegmented_sequences(resid)
        )


def test_source_leaf_grad_is_global_mean_not_sum():
    n_dev = len(jax.devices())
    single = _source_grad(sharded=False)
    for path, g in jax.tree.flatten_with_path(single)[0]:
        assert jnp.all(jnp.isfinite(g)), jax.tree_util.keystr(path)

    if n_dev == 1:
        return  # SUM vs MEAN (N× the mean) is only observable with >1 device

    sharded = _source_grad(sharded=True)
    # Placed reductions may reassociate, but a replica-count scaling error is much larger.
    REL, ABS = 1e-4, 1e-6
    single_paths, single_leaves = jax.tree.flatten_with_path(single)[0], jax.tree.leaves(single)
    sharded_leaves = jax.tree.leaves(sharded)
    for (path, _), a, b in zip(single_paths, single_leaves, sharded_leaves, strict=True):
        name = jax.tree_util.keystr(path)
        a, b = np.asarray(a), np.asarray(b)
        max_abs_err = float(np.max(np.abs(a - b)))
        max_allowed = REL * float(np.max(np.abs(a))) + ABS
        assert max_abs_err <= max_allowed, (
            f"site {name}: source grad differs across 1 vs {n_dev} devices "
            f"(max abs err {max_abs_err:.2e} > {max_allowed:.2e}); per-device grad is "
            f"not normalized by the global batch"
        )
        ratio = float(np.sum(np.abs(b)) / np.sum(np.abs(a)))
        assert abs(ratio - 1.0) < 1e-3, (
            f"site {name}: sharded/single grad magnitude ratio {ratio:.4f} (expected ~1.0); "
            f"the same global source row must receive the same gradient under either layout"
        )
