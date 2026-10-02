"""Placed narrow MoE CI: value/gradient parity, optimizer staging and evaluation."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array

from param_decomp.core.components import (
    ComponentStacks,
    aligned_component_vectors,
)
from param_decomp.core.init_placed import init_component_stacks_placed
from param_decomp.core.model import PlacedModel
from param_decomp.core.placement import from_config
from param_decomp.core.sharding import place_target
from param_decomp.core.tools.hlo_census import collective_census
from param_decomp.lm.batch import LMBatch
from param_decomp.targets.qwen36_moe import (
    full_site_cs,
    qwen36_moe_site_specs,
)
from param_decomp.targets.testing import tiny_qwen36_cfg
from param_decomp.tests.targets.qwen36_placement_helpers import (
    BATCH,
    CENSUS_CS,
    DATA,
    SEQ,
    TP,
    assert_no_weight_gather_in_any_loop,
    batch_placed,
    model_and_components,
    moe_ci_fn_arch,
    multidevice,
    placement_mesh,
)


@multidevice
@pytest.mark.multidevice
@pytest.mark.parametrize("tensor_parallel_size", (1, TP))
def test_placed_moe_ci_fn_census_and_parity(tensor_parallel_size: int):
    """The MoE chunkwise CI fn placed at the moe resident preset: the placed CI values
    (narrow bundles included) and CI-weight gradients match the unplaced run, and the
    compiled forward+backward census pins ZERO in-loop cross-data collectives through
    the CI MoE blocks — the narrow combine's all-reduce rides tp, an activation
    collective; the masters→resident entry gather is the only data crossing."""
    import equinox as eqx

    from param_decomp.core.ci_fn.implementations.block_selected.arch import (
        BlockSelectedChunkwiseTransformerCIFn,
        BlockSelectedTransformerCIFnPlacement,
    )
    from param_decomp.core.components import SelectedCI, site_ci_values
    from param_decomp.tests.placed_ci_fn import placed_ci_fn

    mesh = Mesh(
        np.asarray(jax.devices()[: DATA * tensor_parallel_size]).reshape(
            DATA, tensor_parallel_size
        ),
        ("data", "tp"),
        axis_types=(AxisType.Explicit,) * 2,
    )
    model, components = model_and_components(CENSUS_CS)
    arch = moe_ci_fn_arch(model.cfg)
    rules = from_config("zero1-replicated-resident-moe", mesh, model.sites)
    seed = jax.random.PRNGKey(5)
    unplaced_fn = arch.initialize(model.sites, None, seed)
    assert isinstance(unplaced_fn, BlockSelectedChunkwiseTransformerCIFn)
    tokens = jax.random.randint(jax.random.PRNGKey(6), (BATCH, SEQ), 0, model.cfg.vocab_size)
    unplaced_model = PlacedModel(model=model, placement=None)
    clean = unplaced_model.clean_forward(LMBatch(tokens), unplaced_fn.capture_keys)
    taps, conditioning = dict(clean.captures), clean.conditioning

    expected_ci = unplaced_fn.prepare()(taps, conditioning, components, sequence=None, remat=False)

    def unplaced_total(fn: BlockSelectedChunkwiseTransformerCIFn) -> Array:
        ci = fn.prepare()(taps, conditioning, components, sequence=None, remat=True)
        # raw preactivations, not a squashing: the squashings' piecewise-derivative
        # regions flip under bf16 divergence, amplifying reassociation into O(1)
        # gradient noise at these tiny widths
        return sum(
            (
                jnp.sum(site_ci_values(v).astype(jnp.float32) ** 2)
                for v in ci.preactivations.values()
            ),
            start=jnp.zeros((), jnp.float32),
        )

    expected_total, expected_grads = eqx.filter_value_and_grad(unplaced_total)(unplaced_fn)

    with jax.set_mesh(mesh):
        placed_fn = placed_ci_fn(arch, model.sites, seed, mesh, rules)
        assert isinstance(placed_fn, BlockSelectedChunkwiseTransformerCIFn)
        assert isinstance(placed_fn.placement, BlockSelectedTransformerCIFnPlacement)
        placed_model = place_target(model, rules)
        if tensor_parallel_size == 1:
            compute_fn = placed_fn.prepare()
            for leaf in jax.tree.leaves(eqx.filter((placed_model, compute_fn), eqx.is_array)):
                assert leaf.sharding.is_fully_replicated, leaf.sharding
        placed_tokens = batch_placed(tokens, mesh)
        placed_clean = placed_model.clean_forward(LMBatch(placed_tokens), placed_fn.capture_keys)
        placed_taps, placed_pinned = dict(placed_clean.captures), placed_clean.conditioning

        def placed_total(fn: BlockSelectedChunkwiseTransformerCIFn) -> Array:
            ci = fn.prepare()(
                placed_taps,
                placed_pinned,
                components,
                sequence=None,
                remat=True,
            )
            return sum(
                (
                    jnp.sum(site_ci_values(v).astype(jnp.float32) ** 2)
                    for v in ci.preactivations.values()
                ),
                start=jnp.zeros((), jnp.float32),
            )

        placed_ci = jax.jit(
            lambda fn, t, r: fn.prepare()(t, r, components, sequence=None, remat=False)
        )(placed_fn, placed_taps, placed_pinned)
        grad_fn = eqx.filter_jit(eqx.filter_value_and_grad(placed_total))
        got_total, got_grads = grad_fn(placed_fn)
        hlo = (
            jax.jit(lambda arrays: eqx.filter_value_and_grad(placed_total)(arrays))
            .lower(placed_fn)
            .compile()
            .as_text()
        )
    assert hlo is not None

    def assert_close_frobenius(got: np.ndarray, expected: np.ndarray, name_for_err: str) -> None:
        # bf16 compute at tiny widths: the tp-split matmuls, the EP job spelling, and
        # the auto-axes attention arm all reassociate, and the divergence compounds
        # through the blocks — pointwise ulp bounds don't hold, the norm-level one does
        # (the grad comparison below uses the same criterion). A 12-wide residual puts
        # the shared-expert values' reassociation floor at ~2e-2; 3e-2 keeps headroom.
        denom = np.linalg.norm(expected)
        error = np.linalg.norm(got - expected) / (denom if denom > 0 else 1.0)
        assert error < 3e-2, (name_for_err, error)

    for name in unplaced_fn.output_names:
        expected_value, got_value = expected_ci.lower[name], placed_ci.lower[name]
        match expected_value:
            case SelectedCI():
                assert isinstance(got_value, SelectedCI)
                np.testing.assert_array_equal(
                    np.asarray(got_value.block_indices), np.asarray(expected_value.block_indices)
                )
                assert_close_frobenius(
                    np.asarray(got_value.values), np.asarray(expected_value.values), name
                )
            case jax.Array():
                assert_close_frobenius(np.asarray(got_value), np.asarray(expected_value), name)
    np.testing.assert_allclose(np.asarray(got_total), np.asarray(expected_total), rtol=5e-3)
    for got_leaf, expected_leaf in zip(
        jax.tree.leaves(eqx.filter(got_grads, eqx.is_array)),
        jax.tree.leaves(eqx.filter(expected_grads, eqx.is_array)),
        strict=True,
    ):
        got_np, expected_np = np.asarray(got_leaf), np.asarray(expected_leaf)
        denom = np.linalg.norm(expected_np)
        error = np.linalg.norm(got_np - expected_np) / (denom if denom > 0 else 1.0)
        # bf16 silu·up products at width 8 compound reassociation a little past the
        # masked-forward suite's 2e-2; the bound keeps headroom over the observed 2.1e-2
        assert error < 4e-2, error

    census = collective_census(
        hlo, replica_stride=tensor_parallel_size, n_devices=DATA * tensor_parallel_size
    )
    # The one sanctioned in-loop cross-data collective: the replicated-persisted
    # bias/norm-vector grads' whole-batch sums (the dense chunkwise residency test's
    # carve-out, test_tp_boundary_topology). The byte bound keeps a matrix grad from
    # hiding behind it; the expert banks, heads, and every matrix defer to the entry
    # reductions, and the narrow combine's all-reduce rides tp, not data.
    assert census.in_loop_cross_replicate <= 1, census.counts
    assert all(size <= 2**12 for size in census.in_loop_cross_replicate_bytes), (
        census.in_loop_cross_replicate_bytes
    )
    assert census.counts.get("entry:all-gather[xrep]", 0) > 0, census.counts
    assert_no_weight_gather_in_any_loop(hlo, BATCH // DATA)


def test_stacked_muon_moe_ci_staging_claims():
    """The MoE CI staging claim over the built fn's matrices: dense families stage
    `n_chunks` matrices, expert families the canonical fold `n_chunks·E` — a non-tiling
    chunk count refuses with the remedy."""
    from jax.sharding import AbstractMesh

    from param_decomp.core.ci_fn.implementations.block_selected.arch import (
        BlockSelectedChunkwiseTransformerCIFn,
    )
    from param_decomp.core.ci_fn.optimizer import assert_ci_fn_muon_staging_tiles

    mesh = AbstractMesh((2, 2), ("data", "tp"))

    def abstract_ci_fn(n_layer: int) -> BlockSelectedChunkwiseTransformerCIFn:
        # one chunk per two-layer stage; the moe CI rows rest intra-matrix, so no
        # chunk-stack pad resolves under them
        cfg = dataclasses.replace(tiny_qwen36_cfg(), n_layer=n_layer)
        sites = qwen36_moe_site_specs(cfg, full_site_cs(cfg, CENSUS_CS))
        rules = from_config("zero1-replicated-resident-moe", mesh, sites)
        return eqx.filter_eval_shape(
            lambda: moe_ci_fn_arch(cfg).initialize(sites, rules, jax.random.PRNGKey(0))
        )

    assert_ci_fn_muon_staging_tiles(abstract_ci_fn(4))
    with pytest.raises(AssertionError, match="does not tile"):
        assert_ci_fn_muon_staging_tiles(abstract_ci_fn(6))


def test_replicated_moe_compute_with_sharded_masters_and_replicated_muon():
    from jax.sharding import AbstractMesh

    from param_decomp.core.ci_fn.implementations.block_selected.arch import (
        BlockSelectedTransformerCIFnPlacement,
    )
    from param_decomp.core.ci_fn.implementations.transformer.layers import MHACIFnAttention
    from param_decomp.core.ci_fn.optimizer import assert_ci_fn_muon_staging_tiles
    from param_decomp.core.configs import PlacementTableConfig
    from param_decomp.core.placement import PRESETS, assert_stacked_muon_component_staging

    table = dataclasses.asdict(PRESETS["zero1-replicated-resident-moe"])
    table["components"]["ns_compute"] = {}
    authored = PlacementTableConfig.model_validate(
        {**table, "ci_fn": "zero1-replicated-resident-moe-replicated-ns"}
    )
    cfg = dataclasses.replace(
        tiny_qwen36_cfg(),
        full_attention_interval=4,
        n_embd=512,
        n_experts=42,
        n_experts_per_token=8,
        moe_intermediate=128,
        shared_expert_intermediate=128,
    )
    cs = {kind: 5376 if kind.startswith("experts_") else 128 for kind in CENSUS_CS}
    sites = qwen36_moe_site_specs(cfg, full_site_cs(cfg, cs))
    mesh = AbstractMesh((32, 1), ("data", "tp"))
    rules = from_config(authored, mesh, sites)
    arch = dataclasses.replace(
        moe_ci_fn_arch(cfg),
        input_dim=4 * cfg.n_embd,
        d_model=1344,
        n_blocks=8,
        attention=MHACIFnAttention(mask="bidirectional", implementation="xla", n_heads=14),
        selected_ffn_hidden=128,
        shared_ffn_hidden=128,
    )
    ci_fn = eqx.filter_eval_shape(lambda: arch.initialize(sites, rules, jax.random.PRNGKey(0)))
    placement = ci_fn.placement
    assert isinstance(placement, BlockSelectedTransformerCIFnPlacement)
    assert placement.chunks.stack_len == 1
    assert placement.chunks.stack_pad == 0
    assert_stacked_muon_component_staging(rules)
    assert_ci_fn_muon_staging_tiles(ci_fn)
    assert all(entry.stack_pad == 0 for entry in rules.components.group_census.values())
    assert rules.components.optimizer_state.shard_count("C_block") == 32
    assert rules.components.compute_weights.shard_count("expert") == 1
    assert placement.rows.expert_ffn.operands.shard_count("expert") == 1
    assert placement.rows.expert_head.operands.shard_count("expert") == 1


@multidevice
@pytest.mark.multidevice
def test_placed_narrow_slow_eval_runs_at_data_gt_1():
    """Regression, at tiny C: the STEP-0 SLOW-EVAL tier
    (ComponentActivationDensity counts, mean-CI sums, CIHistograms bins, CI_L0) on the
    placed seat at data>1 feeds dp-sharded NARROW values into the per-component
    reductions — the trace gate and fit check never run this tier, so it must lower and
    execute here. Density/sums are also pinned exactly against the same reductions on
    host-replicated copies of the step's own CI values."""

    from param_decomp.core.ci_fn.implementations.block_selected.arch import (
        BlockSelectedChunkwiseTransformerCIFn,
    )
    from param_decomp.core.ci_l0_eval import make_ci_l0_eval_step
    from param_decomp.core.components import SelectedCI
    from param_decomp.core.slow_eval import make_ci_reduction_step
    from param_decomp.tests.core.test_selected_ci import _scatter_to_full

    mesh = placement_mesh()
    model, components = model_and_components(CENSUS_CS)
    arch = moe_ci_fn_arch(model.cfg)
    rules = from_config("zero1-replicated-resident-moe", mesh, model.sites)
    tokens = jax.random.randint(jax.random.PRNGKey(31), (BATCH, SEQ), 0, model.cfg.vocab_size)
    ci_fn = arch.initialize(model.sites, rules, jax.random.PRNGKey(32))
    assert isinstance(ci_fn, BlockSelectedChunkwiseTransformerCIFn)
    with jax.set_mesh(mesh):
        placed_model = place_target(model, rules)
        placed_fn = jax.device_put(ci_fn, ci_fn.shardings(mesh))
        placed_tokens = batch_placed(tokens, mesh)
        # The shared-context shape of the production tier: one CI envelope, then the
        # jitted reduction over its compute-precision preactivations.
        clean = placed_model.clean_forward(LMBatch(placed_tokens), placed_fn.capture_keys)
        context_ci = placed_fn.prepare()(
            clean.captures, clean.conditioning, components, sequence=None, remat=False
        )
        reduction_step = jax.jit(make_ci_reduction_step(0.0, None, 4))
        density, ci_sums, n_positions, binned_lower, binned_pre, density_hist = reduction_step(
            context_ci.preactivations
        )
        l0_step = jax.jit(
            make_ci_l0_eval_step(placed_model, placed_fn.capture_keys, 0.0, groups=None, mesh=mesh)
        )
        l0 = l0_step(
            placed_model,
            components,
            placed_fn,
            LMBatch(placed_tokens),
            jax.random.PRNGKey(0),
        )
        jax.block_until_ready((density, ci_sums, binned_lower, binned_pre, l0))
        # The oracle: the SAME envelope's CI values, replicated to host, reduced full-width.
        lower = context_ci.lower

    assert int(n_positions) == BATCH * SEQ
    assert not density_hist
    for spec in model.sites:
        value = lower[spec.name]
        if isinstance(value, SelectedCI):
            host = SelectedCI(
                jnp.asarray(np.asarray(value.values)),
                jnp.asarray(np.asarray(value.block_indices)),
                value.n_blocks,
            )
            full = np.asarray(_scatter_to_full(host), dtype=np.float32)
        else:
            full = np.asarray(value, dtype=np.float32)
        flat = full.reshape(-1, spec.C)
        np.testing.assert_array_equal(np.asarray(density[spec.name]), (flat > 0.0).sum(0))
        np.testing.assert_allclose(
            np.asarray(ci_sums[spec.name]), flat.sum(0), rtol=1e-5, atol=1e-5
        )
        counts, lo, hi = binned_lower[spec.name]
        assert np.isfinite(np.asarray(counts)).all() and float(lo) <= float(hi)
        np.testing.assert_allclose(
            np.asarray(l0[f"l0/0.0_{spec.name}"]), (flat > 0.0).sum(-1).mean(), rtol=1e-5
        )


@multidevice
@pytest.mark.multidevice
def test_placed_owner_nonlinearity_eval_reads_slots_of_the_stack_sharded_masters():
    """The standing nonlinearity eval over the owner-placed fp32 masters, whose persist
    stack axis is sharded over `data` (one slot per device here): the device step reduces
    each group's whole stack and the per-site read happens on the host, so no program
    slices the sharded stack axis. Pinned site by site against the same statistics of
    the unplaced masters."""
    from param_decomp.core.components import nonlinearity_alignments
    from param_decomp.core.losses import nonlinearity_loss
    from param_decomp.core.nonlinearity_eval import (
        component_nonlinearity_stats,
        make_nonlinearity_eval_step,
        site_nonlinearity_stats,
    )

    mesh = placement_mesh()
    model, components = model_and_components(
        {**CENSUS_CS, "gdn_v": 8, "gdn_out": 8, "gdn_a": 8, "gdn_b": 8}
    )
    rules = from_config("owner-replicated-resident-moe", mesh, model.sites)

    def loss(value: ComponentStacks) -> Array:
        return nonlinearity_loss(
            value,
            nonlinearity_alignments(model.sites),
            jnp.asarray(4.0),
            {"neuron": 1.0, "deltanet_head": 1.0},
            normalize_at_one=False,
        )[0]

    expected_loss, expected_grad = eqx.filter_value_and_grad(loss)(components)
    with jax.set_mesh(mesh):
        placed = init_component_stacks_placed(model.sites, jax.random.PRNGKey(1), rules)
        actual_loss, actual_grad = eqx.filter_jit(eqx.filter_value_and_grad(loss))(placed)
        assert all(jax.typeof(us).sharding.spec[0] == "data" for _, us in placed.stacks.values()), {
            group: jax.typeof(us).sharding.spec for group, (_, us) in placed.stacks.items()
        }
        stats = site_nonlinearity_stats(
            jax.jit(make_nonlinearity_eval_step(model.sites, NamedSharding(mesh, P())))(placed),
            model.sites,
        )

    np.testing.assert_allclose(actual_loss, expected_loss, rtol=1e-5)
    for group, expected_factors in expected_grad.stacks.items():
        for actual, expected in zip(actual_grad.stacks[group], expected_factors, strict=True):
            np.testing.assert_allclose(
                np.asarray(actual)[: expected.shape[0]], expected, rtol=1e-5, atol=1e-7
            )
            np.testing.assert_array_equal(np.asarray(actual)[expected.shape[0] :], 0)

    partitioned = [site for site in model.sites if site.alignment is not None]
    assert partitioned and set(stats) == {site.name for site in partitioned}
    for site in partitioned:
        assert site.alignment is not None
        factors = components.site(site.name)
        vectors = aligned_component_vectors((factors.V, factors.U), site.alignment.side)
        expected = component_nonlinearity_stats(
            vectors.reshape(-1, vectors.shape[-1]),
            site.alignment.partition,
        )
        assert stats[site.name].soft_use_count.shape == (site.C,)
        np.testing.assert_allclose(
            stats[site.name].soft_use_count, expected.soft_use_count, rtol=1e-6, err_msg=site.name
        )
        np.testing.assert_allclose(
            stats[site.name].effective_use_count_per_subcomponent,
            expected.effective_use_count_per_subcomponent,
            rtol=1e-6,
            err_msg=site.name,
        )
