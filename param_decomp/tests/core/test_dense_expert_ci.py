"""Selected CI interfaces are independent of dense or routed expert computation."""

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from param_decomp.core.ci_fn.implementations.block_selected.arch import (
    BlockSelectedChunk,
    BlockSelectedChunkwiseTransformerCIFn,
    BlockSelectedChunkwiseTransformerCIFnArch,
    FullSlot,
    SelectedSlot,
)
from param_decomp.core.ci_fn.implementations.transformer.layers import MHACIFnAttention
from param_decomp.core.components import (
    BlockedFactorization,
    BlockSelection,
    DenseFactorization,
    SelectedCI,
    SiteSpec,
    init_component_stacks,
)
from param_decomp.core.placement import from_config
from param_decomp.lm.batch import LMBatch, LMBatchWithRouting
from param_decomp.tests.placed_ci_fn import placed_ci_fn


@pytest.mark.parametrize("tp", [None, 1, 2])
@pytest.mark.parametrize("remat", [False, True])
@pytest.mark.multidevice
def test_routed_ci_values_and_gradients_match_dense_with_selected_outputs(
    tp: int | None, remat: bool
) -> None:
    if jax.device_count() < 4:
        pytest.skip("requires four devices")
    sites = (
        SiteSpec("dense", DenseFactorization(d_in=8, d_out=8, C=8), "dense"),
        SiteSpec(
            "expert", BlockedFactorization(n_blocks=4, d_in=8, d_out=8, c_per_block=4), "expert"
        ),
    )
    arch = BlockSelectedChunkwiseTransformerCIFnArch(
        chunks=(
            BlockSelectedChunk(("x",), (0, 1), (FullSlot("dense"), SelectedSlot("expert", 1))),
        ),
        input_dim=8,
        d_model=8,
        n_blocks=2,
        attention=MHACIFnAttention(mask="bidirectional", implementation="xla", n_heads=2),
        table_size=4,
        selected_ffn_hidden=8,
        shared_ffn_hidden=8,
        learned_norm_scale=False,
        expert_implementation="dense_masked",
    )
    key = jax.random.PRNGKey(33)
    taps = {"x": jax.random.normal(key, (4, 3, 8))}
    ids = jnp.asarray(
        [
            [[[0, 2], [1, 2], [2, 0]]] * 4,
            [[[1, 0], [2, 1], [0, 1]]] * 4,
        ]
    )
    conditioning = BlockSelection(
        ids, jax.nn.softmax(jax.random.normal(jax.random.key(34), ids.shape), axis=-1)
    )

    components = init_component_stacks(sites, jax.random.PRNGKey(9))

    def compare(mesh: Mesh | None) -> None:
        rules = None if mesh is None else from_config("zero1-replicated-resident-moe", mesh, sites)
        inputs = (
            taps
            if mesh is None
            else jax.device_put(taps, NamedSharding(mesh, P("data", None, None)))
        )
        selection = (
            conditioning
            if mesh is None
            else jax.device_put(conditioning, NamedSharding(mesh, P(None, "data", None, None)))
        )

        routed = LMBatchWithRouting(LMBatch(jnp.zeros((4, 3), dtype=jnp.int32)), selection)

        def evaluate(fn: BlockSelectedChunkwiseTransformerCIFn):
            ci = (
                fn(inputs, routed, components, sequence=None, remat=remat)
                if mesh is None
                else fn.prepare()(inputs, routed, components, sequence=None, remat=remat)
            )
            dense = ci.preactivations["dense"]
            selected = ci.preactivations["expert"]
            assert isinstance(dense, jax.Array)
            assert isinstance(selected, SelectedCI)
            if mesh is not None:
                assert jax.typeof(selected.values).sharding.spec == P("data", None, None)
            return jnp.mean(dense**2) + jnp.mean(selected.values**2), (dense, selected.values)

        results = []
        for implementation in ("dense_masked", "ragged_dot"):
            selected_arch = replace(arch, expert_implementation=implementation)
            if rules is None:
                fn = selected_arch.initialize(sites, None, key)
            else:
                assert mesh is not None
                fn = placed_ci_fn(selected_arch, sites, key, mesh, rules)
            assert isinstance(fn, BlockSelectedChunkwiseTransformerCIFn)
            results.append(eqx.filter_jit(eqx.filter_value_and_grad(evaluate, has_aux=True))(fn))
        expected, actual = results
        for reference, candidate in zip(
            jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True
        ):
            if mesh is None:
                np.testing.assert_allclose(candidate, reference, atol=2e-5, rtol=2e-5)
            else:
                expected_array = np.asarray(reference, dtype=np.float32)
                difference = np.asarray(candidate, dtype=np.float32) - expected_array
                assert np.linalg.norm(difference) <= 0.03 * np.linalg.norm(expected_array) + 1e-5

    if tp is None:
        compare(None)
    else:
        mesh = Mesh(
            np.asarray(jax.devices()[:4]).reshape(4 // tp, tp),
            ("data", "tp"),
            axis_types=(AxisType.Explicit,) * 2,
        )
        with jax.set_mesh(mesh):
            compare(mesh)
