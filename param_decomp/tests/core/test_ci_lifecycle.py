from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh

from param_decomp.core.ci_fn.implementations.chunkwise.arch import (
    Chunk,
    ChunkwiseTransformerBackbone,
    ChunkwiseTransformerCIFnArch,
    ChunkwiseTransformerCIFnPlacement,
    UnplacedChunkwiseCIFn,
)
from param_decomp.core.ci_fn.implementations.chunkwise.chunk_stack import pad_chunk_stack
from param_decomp.core.ci_fn.implementations.global_mlp import GlobalMLPCIFn, GlobalMLPCIFnArch
from param_decomp.core.ci_fn.implementations.global_transformer.arch import (
    GlobalTransformerCIFnArch,
)
from param_decomp.core.ci_fn.implementations.layerwise_mlp import (
    LayerwiseMLPCIFn,
    LayerwiseMLPCIFnArch,
)
from param_decomp.core.ci_fn.implementations.transformer.backbone import BackboneCIFn
from param_decomp.core.ci_fn.implementations.transformer.layers import MHACIFnAttention
from param_decomp.core.ci_fn.interface import CIFn, TapSpec
from param_decomp.core.ci_fn.runtime import evaluate_ci_from_captures
from param_decomp.core.components import (
    ComponentStacks,
    DenseFactorization,
    SiteSpec,
    init_component_stacks,
    require_full_emission,
)
from param_decomp.core.placement import StackCensus, from_config
from param_decomp.targets.testing import chunkwise_transformer_backbone

SITES = (SiteSpec("site", DenseFactorization(d_in=8, d_out=8, C=4), "site"),)
CHUNKWISE = ChunkwiseTransformerCIFnArch(
    chunks=(Chunk(("input",), ("site",)),),
    input_dim=8,
    d_model=8,
    n_blocks=1,
    attention=MHACIFnAttention(n_heads=2, implementation="xla", mask="bidirectional"),
    ffn_hidden=16,
    ffn_kind="gelu",
    learned_norm_scale=False,
)
GLOBAL_MLP = GlobalMLPCIFnArch(
    hidden_dims=(8,), has_position_axis=True, input_taps=(TapSpec("input", 8),)
)
LAYERWISE_MLP = LayerwiseMLPCIFnArch(
    hidden_dims=(8,), has_position_axis=True, input_names=("input",)
)
GLOBAL_TRANSFORMER = GlobalTransformerCIFnArch(
    input_taps=(TapSpec("input", 8),),
    d_model=8,
    n_blocks=1,
    attention=CHUNKWISE.attention,
    ffn_hidden=16,
    ffn_kind="gelu",
    learned_norm_scale=False,
)
Arch = (
    ChunkwiseTransformerCIFnArch
    | GlobalMLPCIFnArch
    | LayerwiseMLPCIFnArch
    | GlobalTransformerCIFnArch
)


def _components() -> ComponentStacks:
    return init_component_stacks(SITES, jax.random.key(9))


@pytest.mark.parametrize(
    ("arch", "prepared_type"),
    [
        (CHUNKWISE, BackboneCIFn),
        (GLOBAL_MLP, GlobalMLPCIFn),
        (LAYERWISE_MLP, LayerwiseMLPCIFn),
        (GLOBAL_TRANSFORMER, BackboneCIFn),
    ],
)
def test_preparation_preserves_master_parameters_and_gradient_paths(
    arch: Arch, prepared_type: type
):
    masters = arch.initialize(SITES, None, jax.random.key(7))
    original = [np.asarray(value).copy() for value in jax.tree.leaves(masters)]
    components = _components()
    prepared = eqx.filter_jit(lambda value: value.prepare())(masters)
    assert type(prepared) is prepared_type
    assert all(value.dtype == jnp.bfloat16 for value in jax.tree.leaves(prepared))
    taps = {"input": jax.random.normal(jax.random.key(8), (2, 3, 8))}

    def loss(parameters: CIFn[object]):
        compute = parameters.prepare()
        ci = compute(taps, None, components, sequence=None, remat=False)
        return jnp.mean(require_full_emission(ci.preactivations["site"]).astype(jnp.float32) ** 2)

    value, gradient = eqx.filter_jit(eqx.filter_value_and_grad(loss))(masters)
    assert jnp.isfinite(value)
    assert jax.tree.structure(gradient) == jax.tree.structure(masters)
    gradient_leaves = jax.tree.leaves(gradient)
    assert all(leaf.dtype == jnp.float32 for leaf in gradient_leaves)
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in gradient_leaves)
    assert any(jnp.any(leaf != 0) for leaf in gradient_leaves)
    for current, expected in zip(jax.tree.leaves(masters), original, strict=True):
        np.testing.assert_array_equal(current, expected)


@pytest.mark.parametrize("extra", [False, True])
def test_physical_capture_boundary_rejects_a_different_capture_request(extra: bool):
    fn = CHUNKWISE.initialize(SITES, None, jax.random.key(7))
    prepared = fn.prepare()
    captures = {name: jnp.zeros((2, 3, 8)) for name in ("input", "unrequested")} if extra else {}
    with pytest.raises(ValueError, match="CI captures must match its request"):
        evaluate_ci_from_captures(
            prepared,
            captures,
            None,
            _components(),
            sequence=None,
            remat=False,
        )


def test_stored_chunk_preparation_rejects_mismatched_census_and_leaf_extents():
    mesh = Mesh(
        np.asarray(jax.devices()[:1]).reshape((1, 1, 1)),
        ("data", "fsdp", "tp"),
        axis_types=(AxisType.Explicit,) * 3,
    )
    rules = from_config("zero1-replicated-resident", mesh, SITES)
    parameters = CHUNKWISE.initialize(SITES, rules, jax.random.key(7))
    backbone = chunkwise_transformer_backbone(parameters)
    placement = backbone.placement
    assert isinstance(placement, ChunkwiseTransformerCIFnPlacement)
    parameters.prepare()

    def with_backbone(**changes: object) -> BackboneCIFn:
        return BackboneCIFn(replace(backbone, **changes))

    wrong_count = replace(placement, chunks=StackCensus(stack_len=2, stack_pad=0))
    with pytest.raises(AssertionError, match="this fn routes 1 chunks"):
        with_backbone(placement=wrong_count).prepare()

    padded_placement = replace(placement, chunks=StackCensus(stack_len=1, stack_pad=1))
    with pytest.raises(AssertionError, match="chunk leaf disagrees"):
        with_backbone(placement=padded_placement).prepare()

    chunks = pad_chunk_stack(backbone.chunks, padded_placement.chunks)
    padded = with_backbone(chunks=chunks, placement=padded_placement)
    prepared = padded.prepare()
    assert isinstance(prepared.backbone, ChunkwiseTransformerBackbone)
    assert prepared.backbone.census.stack_pad == 1
    assert all(leaf.shape[0] == 2 for leaf in jax.tree.leaves(prepared.backbone.chunks))
    with pytest.raises(AssertionError, match="chunk leaf disagrees"):
        with_backbone(chunks=chunks, placement=UnplacedChunkwiseCIFn()).prepare()
    malformed = with_backbone(
        chunks=eqx.tree_at(lambda value: value.in_proj_b, chunks, backbone.chunks.in_proj_b),
        placement=padded_placement,
    )
    with pytest.raises(AssertionError, match="chunk leaf disagrees"):
        malformed.prepare()
