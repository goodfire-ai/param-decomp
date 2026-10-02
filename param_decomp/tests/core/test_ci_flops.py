"""Analytical CI costs agree with differentiated MLP contractions and selected routing."""

import math
from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax.core import ShapedArray
from jax.extend import core

from param_decomp.core.ci_fn.implementations.block_selected.arch import (
    BlockSelectedChunk,
    BlockSelectedChunkwiseTransformerCIFnArch,
    FullSlot,
    RoutingConditioning,
    SelectedSlot,
)
from param_decomp.core.ci_fn.implementations.chunkwise.arch import (
    Chunk,
    ChunkwiseTransformerCIFnArch,
)
from param_decomp.core.ci_fn.implementations.global_mlp import GlobalMLPCIFnArch
from param_decomp.core.ci_fn.implementations.layerwise_mlp import LayerwiseMLPCIFnArch
from param_decomp.core.ci_fn.implementations.transformer.layers import (
    GQACIFnAttention,
    MHACIFnAttention,
)
from param_decomp.core.ci_fn.interface import CIFn, TapSpec
from param_decomp.core.components import (
    BlockedFactorization,
    DenseFactorization,
    SiteSpec,
    init_component_stacks,
    require_full_emission,
)
from param_decomp.core.flops.ci import ci_fn_flops
from param_decomp.core.flops.types import ForwardBackwardFlops
from param_decomp.core.model import Positioned, Positionless

SITES = (
    SiteSpec("first", DenseFactorization(d_in=3, d_out=4, C=2), "first"),
    SiteSpec("second", DenseFactorization(d_in=4, d_out=3, C=3), "second"),
)


def _contraction_flops(graph: core.ClosedJaxpr) -> int:
    total = 0
    for equation in graph.jaxpr.eqns:
        if equation.primitive.name == "dot_general":
            (contracted_axes, _), _ = equation.params["dimension_numbers"]
            input_value = equation.invars[0].aval
            output_value = equation.outvars[0].aval
            assert isinstance(input_value, ShapedArray)
            assert isinstance(output_value, ShapedArray)
            input_shape = input_value.shape
            output_shape = output_value.shape
            total += (
                2 * math.prod(output_shape) * math.prod(input_shape[a] for a in contracted_axes)
            )
    return total


@pytest.mark.parametrize("global_mlp", [False, True])
@pytest.mark.parametrize("positioned", [False, True])
def test_mlp_matches_the_actual_forward_and_parameter_gradient(global_mlp: bool, positioned: bool):
    arch = (
        GlobalMLPCIFnArch(
            (5, 7), positioned, (TapSpec("first_input", 3), TapSpec("second_input", 4))
        )
        if global_mlp
        else LayerwiseMLPCIFnArch((5, 7), positioned, ("first_input", "second_input"))
    )
    positions = Positioned(6) if positioned else Positionless()
    leading = (2, 6) if positioned else (2,)
    taps = {
        "first_input": jnp.ones((*leading, 3)),
        "second_input": jnp.ones((*leading, 4)),
    }
    predictor = arch.initialize(SITES, None, jax.random.key(0))
    components = init_component_stacks(SITES, jax.random.key(3))

    def objective(weights: CIFn[object]) -> jax.Array:
        ci = weights.prepare()(taps, None, components, sequence=None, remat=False)
        return sum(
            (jnp.sum(require_full_emission(value)) for value in ci.preactivations.values()),
            start=jnp.array(0.0),
        )

    forward_graph, _, _ = eqx.filter_make_jaxpr(objective)(predictor)
    training_graph, _, _ = eqx.filter_make_jaxpr(eqx.filter_value_and_grad(objective))(predictor)
    forward = _contraction_flops(forward_graph)
    backward = _contraction_flops(training_graph) - forward
    assert forward > 0
    assert ci_fn_flops(
        arch, SITES, 2, positions, n_selected_blocks_per_token=None
    ) == ForwardBackwardFlops(forward, backward)


def _transformer() -> ChunkwiseTransformerCIFnArch:
    return ChunkwiseTransformerCIFnArch(
        chunks=(Chunk(("input",), ("first", "second")),),
        input_dim=3,
        d_model=8,
        n_blocks=1,
        attention=MHACIFnAttention(2, "xla", "bidirectional"),
        ffn_hidden=12,
        ffn_kind="gelu",
        learned_norm_scale=False,
    )


def test_zero_block_transformer_has_only_an_input_projection_and_output_heads():
    cost = ci_fn_flops(
        replace(_transformer(), n_blocks=0),
        SITES,
        2,
        Positioned(3),
        n_selected_blocks_per_token=None,
    )
    # Six inputs pass through 3 -> 8 -> (2 + 3); only the first input is frozen.
    assert cost == ForwardBackwardFlops(768, 1248)


def test_causal_gqa_and_swiglu_change_only_their_own_contractions():
    baseline = _transformer()
    gqa = replace(baseline, attention=GQACIFnAttention(2, 1, "xla", "bidirectional"))
    causal = replace(baseline, attention=MHACIFnAttention(2, "xla", "causal"))
    swiglu = replace(baseline, ffn_kind="swiglu")
    costs = [
        ci_fn_flops(arch, SITES, 2, Positioned(3), n_selected_blocks_per_token=None)
        for arch in (baseline, gqa, causal, swiglu)
    ]
    assert costs[0] == ForwardBackwardFlops(6720, 13152)
    assert costs[0].forward - costs[1].forward == 768  # Half-width K and V projections.
    assert costs[0].forward - costs[2].forward == 192  # Three masked pairs per sequence.
    assert costs[3].forward - costs[0].forward == 1152  # One extra 8 -> 12 gate.


def _selected_transformer() -> tuple[
    BlockSelectedChunkwiseTransformerCIFnArch, tuple[SiteSpec, ...]
]:
    sites = (
        SITES[0],
        SiteSpec(
            "experts", BlockedFactorization(n_blocks=8, d_in=4, d_out=4, c_per_block=3), "experts"
        ),
    )
    arch = BlockSelectedChunkwiseTransformerCIFnArch(
        chunks=(
            BlockSelectedChunk(("input",), (0, 1), (FullSlot("first"), SelectedSlot("experts", 0))),
        ),
        input_dim=3,
        d_model=8,
        n_blocks=2,
        attention=MHACIFnAttention(2, "xla", "bidirectional"),
        table_size=8,
        selected_ffn_hidden=5,
        shared_ffn_hidden=6,
        learned_norm_scale=False,
        expert_implementation="ragged_dot",
    )
    return arch, sites


def test_selected_ci_counts_active_banks_and_heads_independently_of_backend():
    arch, sites = _selected_transformer()
    one = ci_fn_flops(arch, sites, 1, Positioned(1), n_selected_blocks_per_token=1)
    two = ci_fn_flops(arch, sites, 1, Positioned(1), n_selected_blocks_per_token=2)
    dense = ci_fn_flops(
        replace(arch, expert_implementation="dense_masked"),
        sites,
        1,
        Positioned(1),
        n_selected_blocks_per_token=2,
    )
    # One additional expert in each of two banks and two blocks, plus one 5 -> 3 head.
    assert two.forward - one.forward == 990
    assert two.backward - one.backward == 1980
    assert dense == two


def test_selected_only_outputs_do_not_compute_an_unused_final_residual():
    arch, sites = _selected_transformer()
    selected_only = replace(
        arch, chunks=(BlockSelectedChunk(("input",), (0, 1), (SelectedSlot("experts", 0),)),)
    )
    full = ci_fn_flops(arch, sites, 1, Positioned(1), n_selected_blocks_per_token=2)
    selected = ci_fn_flops(
        selected_only, (sites[1],), 1, Positioned(1), n_selected_blocks_per_token=2
    )
    # The final block loses its dense head, shared FFN, both expert down projections,
    # and the gate/up pair for the second bank, whose hiddens feed no selected head.
    assert full.forward - selected.forward == 960
    assert full.backward - selected.backward == 1920


def test_selection_and_position_requirements_fail_at_the_boundary():
    arch, sites = _selected_transformer()
    with pytest.raises(ValueError, match="top-k"):
        ci_fn_flops(arch, sites, 1, Positioned(4), n_selected_blocks_per_token=None)
    with pytest.raises(ValueError, match="position axis"):
        ci_fn_flops(_transformer(), SITES, 1, Positionless(), n_selected_blocks_per_token=None)
    with pytest.raises(ValueError, match="positive"):
        ci_fn_flops(_transformer(), SITES, 1, Positioned(0), n_selected_blocks_per_token=None)


def test_dense_ci_cost_is_independent_of_target_routing():
    arch = _transformer()
    assert ci_fn_flops(arch, SITES, 1, Positioned(4), n_selected_blocks_per_token=2) == ci_fn_flops(
        arch, SITES, 1, Positioned(4), n_selected_blocks_per_token=None
    )


def _useful_transformer_contractions(graph: core.Jaxpr) -> int:
    from jax._src.interpreters.partial_eval import dce_jaxpr

    live, _ = dce_jaxpr(graph, True)
    total = 0
    for equation in live.eqns:
        if equation.primitive.name == "dot_general":
            scope = str(equation.source_info.name_stack)
            # These dense one-hot contractions implement gather/scatter routing.
            if "...ke,...k->...e" in scope or "...ec,...ke->...kc" in scope:
                continue
            (contracted_axes, _), _ = equation.params["dimension_numbers"]
            input_value = equation.invars[0].aval
            output_value = equation.outvars[0].aval
            assert isinstance(input_value, ShapedArray)
            assert isinstance(output_value, ShapedArray)
            total += (
                2
                * math.prod(output_value.shape)
                * math.prod(input_value.shape[axis] for axis in contracted_axes)
            )
        else:
            for nested in equation.params.values():
                if isinstance(nested, core.ClosedJaxpr):
                    nested = nested.jaxpr
                if isinstance(nested, core.Jaxpr):
                    total += equation.params.get("length", 1) * _useful_transformer_contractions(
                        nested
                    )
    return total


@pytest.mark.parametrize("learned_norm_scale", [False, True])
@pytest.mark.parametrize("selected_only", [False, True])
def test_selected_transformer_matches_production_autodiff(
    monkeypatch: pytest.MonkeyPatch, learned_norm_scale: bool, selected_only: bool
):
    from dataclasses import dataclass

    from param_decomp.core.components import BlockSelection, site_ci_values

    # Disable checkpointing to inspect the retained-intermediate mathematical graph.
    monkeypatch.setattr(jax, "checkpoint", lambda function, **_: function)
    monkeypatch.setattr(eqx, "filter_checkpoint", lambda function, **_: function)
    arch, sites = _selected_transformer()
    arch = replace(
        arch,
        expert_implementation="dense_masked",
        learned_norm_scale=learned_norm_scale,
        attention=GQACIFnAttention(2, 1, "xla", "bidirectional"),
    )
    if selected_only:
        arch = replace(
            arch, chunks=(BlockSelectedChunk(("input",), (0, 1), (SelectedSlot("experts", 0),)),)
        )
        sites = (sites[1],)
    predictor = arch.initialize(sites, None, jax.random.key(1))
    components = init_component_stacks(sites, jax.random.key(3))
    indices = jnp.broadcast_to(jnp.arange(arch.table_size), (2, 1, 3, arch.table_size))

    @dataclass(frozen=True)
    class Conditioning:
        selection: BlockSelection

    conditioning = Conditioning(
        BlockSelection(indices, jnp.ones_like(indices, dtype=jnp.float32) / arch.table_size)
    )

    def objective(weights: CIFn[RoutingConditioning]) -> jax.Array:
        ci = weights.prepare()(
            {"input": jnp.ones((1, 3, 3))}, conditioning, components, sequence=None, remat=False
        )
        return sum(
            (jnp.sum(site_ci_values(value)) for value in ci.preactivations.values()),
            start=jnp.array(0.0),
        )

    forward_graph, _, _ = eqx.filter_make_jaxpr(objective)(predictor)
    training_graph, _, _ = eqx.filter_make_jaxpr(eqx.filter_value_and_grad(objective))(predictor)
    forward = _useful_transformer_contractions(forward_graph.jaxpr)
    backward = _useful_transformer_contractions(training_graph.jaxpr) - forward
    assert ci_fn_flops(
        arch, sites, 1, Positioned(3), n_selected_blocks_per_token=arch.table_size
    ) == ForwardBackwardFlops(forward, backward)


@pytest.mark.parametrize("learned_norm_scale", [False, True])
def test_dense_transformer_matches_production_autodiff(
    monkeypatch: pytest.MonkeyPatch, learned_norm_scale: bool
):
    monkeypatch.setattr(jax, "checkpoint", lambda function, **_: function)
    monkeypatch.setattr(eqx, "filter_checkpoint", lambda function, **_: function)
    arch = replace(
        _transformer(),
        learned_norm_scale=learned_norm_scale,
        ffn_kind="swiglu",
        attention=GQACIFnAttention(2, 1, "xla", "bidirectional"),
    )
    predictor = arch.initialize(SITES, None, jax.random.key(2))
    components = init_component_stacks(SITES, jax.random.key(3))

    def objective(weights: CIFn[object]) -> jax.Array:
        ci = weights.prepare()(
            {"input": jnp.ones((1, 3, 3))}, None, components, sequence=None, remat=False
        )
        return sum(
            (jnp.sum(require_full_emission(value)) for value in ci.preactivations.values()),
            start=jnp.array(0.0),
        )

    forward_graph, _, _ = eqx.filter_make_jaxpr(objective)(predictor)
    training_graph, _, _ = eqx.filter_make_jaxpr(eqx.filter_value_and_grad(objective))(predictor)
    forward = _useful_transformer_contractions(forward_graph.jaxpr)
    backward = _useful_transformer_contractions(training_graph.jaxpr) - forward
    assert ci_fn_flops(
        arch, SITES, 1, Positioned(3), n_selected_blocks_per_token=None
    ) == ForwardBackwardFlops(forward, backward)
