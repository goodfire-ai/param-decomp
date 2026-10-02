"""Initialized EMA state already has its recurrent layout."""

from dataclasses import asdict
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array

from param_decomp.core.ci_fn.implementations.block_selected.arch import (
    BlockSelectedChunk,
    BlockSelectedChunkwiseTransformerCIFnArch,
    FullSlot,
    SelectedSlot,
)
from param_decomp.core.ci_fn.implementations.chunkwise.arch import (
    Chunk,
    ChunkwiseTransformerCIFnArch,
)
from param_decomp.core.ci_fn.implementations.global_mlp import GlobalMLPCIFnArch
from param_decomp.core.ci_fn.implementations.layerwise_mlp import LayerwiseMLPCIFnArch
from param_decomp.core.ci_fn.implementations.transformer.layers import MHACIFnAttention
from param_decomp.core.ci_fn.interface import CI, CIFn, TapSpec
from param_decomp.core.components import (
    BlockedFactorization,
    BlockSelection,
    DenseFactorization,
    SelectedCI,
    SiteSpec,
    init_component_stacks,
)
from param_decomp.core.configs import FrequencyMinimalityConfig, PlacementTableConfig
from param_decomp.core.init_placed import init_frequency_estimator_placed
from param_decomp.core.losses import EmaFrequency, per_component_frequencies
from param_decomp.core.placement import PRESETS, batch_axes, from_config
from param_decomp.core.train import ForwardSubstrate
from param_decomp.lm.batch import LMBatch, LMBatchWithRouting
from param_decomp.targets.tms import TMSDecomposedModel
from param_decomp.tests.placed_ci_fn import placed_ci_fn
from param_decomp.tests.sequence import unsegmented_sequence_layout


@pytest.mark.parametrize("architecture", ["transformer", "ddp", "layerwise", "global", "selected"])
@pytest.mark.parametrize("component_tp", [True, False])
@pytest.mark.parametrize("world", [1, pytest.param(4, marks=pytest.mark.multidevice)])
def test_frequency_state_keeps_its_layout_and_compiles_once(
    architecture: Literal["transformer", "ddp", "layerwise", "global", "selected"],
    world: int,
    component_tp: bool,
):
    if jax.device_count() < world:
        pytest.skip(f"requires {world} local devices")
    data, tp = (1, 1) if world == 1 else (2, 2)
    if architecture == "ddp":
        mesh = Mesh(
            np.asarray(jax.devices()[:world]).reshape(1, data, tp),
            ("replicate", "fsdp", "tp"),
            axis_types=(AxisType.Explicit,) * 3,
        )
    else:
        mesh = Mesh(
            np.asarray(jax.devices()[:world]).reshape(data, tp),
            ("data", "tp"),
            axis_types=(AxisType.Explicit,) * 2,
        )
    dense = SiteSpec("dense", DenseFactorization(d_in=8, d_out=8, C=8), "dense")
    match architecture:
        case "transformer" | "ddp":
            sites = (dense,)
            arch = ChunkwiseTransformerCIFnArch(
                chunks=(Chunk(input_taps=("input",), output_sites=("dense",)),),
                input_dim=8,
                d_model=8,
                n_blocks=0,
                attention=MHACIFnAttention(mask="bidirectional", implementation="xla", n_heads=2),
                ffn_hidden=16,
                ffn_kind="gelu",
                learned_norm_scale=False,
            )
            preset = "ddp" if architecture == "ddp" else "zero1-replicated-resident"
        case "layerwise":
            sites = (dense,)
            arch = LayerwiseMLPCIFnArch(
                hidden_dims=(8,), has_position_axis=True, input_names=("input",)
            )
            preset = "zero1-replicated-resident"
        case "global":
            sites = (dense,)
            arch = GlobalMLPCIFnArch(
                hidden_dims=(8,), has_position_axis=True, input_taps=(TapSpec("input", 8),)
            )
            preset = "zero1-replicated-resident"
        case "selected":
            sites = (
                dense,
                SiteSpec(
                    "selected",
                    BlockedFactorization(n_blocks=4, d_in=8, d_out=8, c_per_block=4),
                    "selected",
                ),
            )
            arch = BlockSelectedChunkwiseTransformerCIFnArch(
                chunks=(
                    BlockSelectedChunk(
                        input_taps=("input",),
                        layers=(0,),
                        slots=(FullSlot("dense"), SelectedSlot("selected", 0)),
                    ),
                ),
                input_dim=12,
                d_model=8,
                n_blocks=1,
                attention=MHACIFnAttention(mask="bidirectional", implementation="xla", n_heads=2),
                table_size=4,
                selected_ffn_hidden=8,
                shared_ffn_hidden=8,
                learned_norm_scale=False,
                expert_implementation="ragged_dot",
            )
            preset = "zero1-replicated-resident-moe"
    authored: dict[str, Any] = {**asdict(PRESETS[preset]), "ci_fn": preset}
    if not component_tp:
        authored["activations"]["component"].pop("C", None)
        authored["activations"]["component"].pop("expert", None)
    table = PlacementTableConfig.model_validate(authored)
    if not component_tp and architecture == "ddp" and tp > 1:
        with pytest.raises(AssertionError, match="shards nothing over mesh axes"):
            from_config(table, mesh, sites)
        return
    rules = from_config(table, mesh, sites)
    substrate = ForwardSubstrate(
        remat_recon_forwards=False,
        remat_ci_fn=False,
        placement_rules=rules,
        ci_capture_keys=frozenset({"input"}),
        recon_loss_fn=TMSDecomposedModel.recon_loss_fn,
        pin_output_batch=TMSDecomposedModel.pin_output_batch,
    )

    with jax.set_mesh(mesh):
        fn = placed_ci_fn(arch, sites, jax.random.PRNGKey(0), mesh, rules)
        frequency = FrequencyMinimalityConfig(
            coeff=1.0, reference_datapoint_count=16, ema_halflife_steps=8
        )
        state = init_frequency_estimator_placed(frequency, sites, rules.frequency_sharding)
        abstract_fn = eqx.filter_eval_shape(
            placed_ci_fn, arch, sites, jax.random.PRNGKey(0), mesh, rules
        )
        abstract_state = eqx.filter_eval_shape(
            init_frequency_estimator_placed, frequency, sites, rules.frequency_sharding
        )
        assert isinstance(state, EmaFrequency)
        assert isinstance(abstract_state, EmaFrequency)
        for actual, abstract in zip(
            jax.tree.leaves((fn, state.estimate)),
            jax.tree.leaves((abstract_fn, abstract_state.estimate)),
            strict=True,
        ):
            assert actual.shape == abstract.shape
            assert actual.dtype == abstract.dtype
            assert isinstance(abstract.sharding, NamedSharding)
            assert actual.sharding.is_equivalent_to(
                NamedSharding(mesh, abstract.sharding.spec), ndim=actual.ndim
            )
        expected = NamedSharding(mesh, P("tp") if component_tp else P(None))
        width = 12 if architecture == "selected" else 8
        taps = {
            "input": jax.device_put(
                np.linspace(-1, 1, 4 * 4 * width, dtype=np.float32).reshape(4, 4, width),
                NamedSharding(mesh, P(batch_axes(mesh), None, None)),
            )
        }
        conditioning = None
        if architecture == "selected":
            conditioning = BlockSelection(
                indices=jax.device_put(
                    np.broadcast_to(np.array([0, 2], np.int32), (1, 4, 4, 2)),
                    NamedSharding(mesh, P(None, batch_axes(mesh), None, None)),
                ),
                weights=jax.device_put(
                    np.full((1, 4, 4, 2), 0.5, np.float32),
                    NamedSharding(mesh, P(None, batch_axes(mesh), None, None)),
                ),
            )

            conditioning = LMBatchWithRouting(
                LMBatch(jnp.zeros((4, 4), dtype=jnp.int32)), conditioning
            )

        components = init_component_stacks(sites, jax.random.PRNGKey(1))

        @eqx.filter_jit
        def public_ci(fn: CIFn[Any], taps: dict[str, Array], conditioning: object) -> CI:
            raw = fn.prepare()(
                taps,
                conditioning,
                components,
                sequence=unsegmented_sequence_layout(taps),
                remat=False,
            )
            if architecture in ("layerwise", "global"):
                for value in raw.preactivations.values():
                    assert isinstance(value, jax.Array)
                    assert jax.typeof(value).sharding.spec[-1] is None
            return substrate.shard_ci(raw)

        ci = public_ci(fn, taps, conditioning)
        if architecture == "selected":
            selected = ci.upper["selected"]
            assert isinstance(selected, SelectedCI)
            assert selected.values.shape[:-1] == (4, 4)
            assert selected.values.sharding.is_equivalent_to(
                NamedSharding(mesh, P(batch_axes(mesh), None, None)), ndim=3
            )
        traces: list[None] = []

        natural = per_component_frequencies(ci.upper, jnp.asarray(1.0), normalize_at_one=False)
        if architecture == "selected":
            assert natural["selected"].sharding.is_equivalent_to(NamedSharding(mesh, P()), ndim=1)

        @jax.jit
        def update(estimator: EmaFrequency):
            traces.append(None)
            frequencies = substrate.component_frequencies(
                ci, jnp.asarray(1.0), normalize_at_one=False
            )
            for frequency in frequencies.values():
                actual = jax.typeof(frequency).sharding
                assert actual.is_equivalent_to(NamedSharding(actual.mesh, expected.spec), ndim=1)
            return estimator.evaluate(frequencies, 16)[1]

        for _ in range(3):
            for leaf in state.estimate.values():
                assert leaf.sharding.is_equivalent_to(expected, ndim=1)
            state = update(state)
            jax.block_until_ready(state)
        assert len(traces) == 1
        for leaf in state.estimate.values():
            assert leaf.sharding.is_equivalent_to(expected, ndim=1)
        frequencies = per_component_frequencies(ci.upper, jnp.asarray(1.0), normalize_at_one=False)
        for name, leaf in state.estimate.items():
            np.testing.assert_allclose(
                np.asarray(leaf), np.asarray(frequencies[name]) * (1 - 2 ** (-3 / 8)), rtol=1e-6
            )
