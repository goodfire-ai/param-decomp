"""The global transformer CI: one-chunk numerics, depth/site placement, and Muon staging."""

from dataclasses import replace
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AbstractMesh, AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from param_decomp.core.ci_fn.implementations.chunkwise.arch import (
    Chunk,
    ChunkwiseTransformerBackbone,
    ChunkwiseTransformerCIFnArch,
)
from param_decomp.core.ci_fn.implementations.global_transformer.arch import (
    GlobalTransformerBackbone,
    GlobalTransformerCIFnArch,
)
from param_decomp.core.ci_fn.implementations.global_transformer.placement import (
    bind_global_rows,
    preset_rows,
)
from param_decomp.core.ci_fn.implementations.transformer.backbone import BackboneCIFn
from param_decomp.core.ci_fn.implementations.transformer.layers import (
    GQACIFnAttention,
    MHACIFnAttention,
)
from param_decomp.core.ci_fn.interface import TapSpec
from param_decomp.core.ci_fn.optimizer import ci_fn_muon_waypoints
from param_decomp.core.components import (
    DenseFactorization,
    SiteSpec,
    init_component_stacks,
    require_full_emission,
)
from param_decomp.core.configs import AdamWOptimizerConfig, MuonOptimizerConfig, PDConfigBase
from param_decomp.core.optimizer import ScheduledOptimizerState
from param_decomp.core.placement import from_config
from param_decomp.core.precision import cast_floating
from param_decomp.core.run_state import build_optimizers
from param_decomp.core.schedule import ScheduleConfig
from param_decomp.sequence import SequenceLayout
from param_decomp.targets.testing import chunkwise_transformer_backbone
from param_decomp.tests.placed_ci_fn import placed_ci_fn

SITES = tuple(
    SiteSpec(f"site_{i}", DenseFactorization(d_in=8, d_out=8, C=c), f"width_{c}")
    for i, c in enumerate((4, 8) * 4)
)
TAPS = (TapSpec("early", 3), TapSpec("late", 5))
COMPONENTS = init_component_stacks(SITES, jax.random.key(3))
THREE_AXES = ("replicate", "fsdp", "tp")


def _arch(
    mask: Literal["bidirectional", "causal"], ffn: Literal["gelu", "swiglu"]
) -> GlobalTransformerCIFnArch:
    match ffn:
        case "gelu":
            attention = MHACIFnAttention(n_heads=2, implementation="xla", mask=mask)
        case "swiglu":
            attention = GQACIFnAttention(n_heads=2, n_kv_heads=1, implementation="xla", mask=mask)
    return GlobalTransformerCIFnArch(
        input_taps=TAPS,
        d_model=8,
        n_blocks=2,
        attention=attention,
        ffn_hidden=16,
        ffn_kind=ffn,
        learned_norm_scale=ffn == "swiglu",
    )


def _taps(batch: int) -> dict[str, jax.Array]:
    return {
        tap.key: jax.random.normal(jax.random.key(i + 1), (batch, 5, tap.width))
        for i, tap in enumerate(TAPS)
    }


UnplacedBackbone = GlobalTransformerBackbone | ChunkwiseTransformerBackbone


def _global_backbone(fn: object) -> GlobalTransformerBackbone:
    assert isinstance(fn, BackboneCIFn), type(fn)
    backbone = fn.backbone
    assert isinstance(backbone, GlobalTransformerBackbone), type(backbone)
    return backbone


def _preactivations(
    backbone: UnplacedBackbone,
    taps: dict[str, jax.Array],
    sequence: SequenceLayout,
    remat: bool,
) -> dict[str, jax.Array]:
    """The float32 evaluation of unplaced parameters, without the compute-dtype cast."""
    preactivations = backbone.preactivations(taps, None, COMPONENTS, sequence=sequence, remat=remat)
    return {name: require_full_emission(value) for name, value in preactivations.items()}


def _objective(
    backbone: UnplacedBackbone, taps: dict[str, jax.Array], sequence: SequenceLayout, remat: bool
) -> jax.Array:
    values = _preactivations(backbone, taps, sequence, remat).values()
    return jnp.stack([jnp.mean(value**2) for value in values]).sum()


def _resident_objective(
    fn: BackboneCIFn, taps: dict[str, jax.Array], sequence: SequenceLayout
) -> jax.Array:
    """Evaluated in float32 on the compute-dtype residents, so placements differ only by
    float32 reassociation."""
    residents = cast_floating(fn.prepare().backbone, jnp.float32)
    preactivations = residents.preactivations(
        taps, None, COMPONENTS, sequence=sequence, remat=False
    )
    values = (require_full_emission(value) for value in preactivations.values())
    return jnp.stack([jnp.mean(value**2) for value in values]).sum()


def _assert_close(actual: object, expected: object, *, tolerance: float) -> None:
    for x, y in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(
            np.asarray(x, np.float32), np.asarray(y, np.float32), atol=tolerance, rtol=tolerance
        )


def _global_from_chunk(
    backbone: GlobalTransformerBackbone, chunk: ChunkwiseTransformerBackbone
) -> GlobalTransformerBackbone:
    """The one chunk's parameters in the global layout: blocks stacked along depth, heads
    stacked by semantic group."""
    one = jax.tree.map(lambda x: x[0], chunk.chunks)
    index = {name: i for i, name in enumerate(chunk.output_names)}
    return replace(
        backbone,
        in_proj_w=one.in_proj_w,
        in_proj_b=one.in_proj_b,
        blocks=jax.tree.map(lambda *leaves: jnp.stack(leaves), *one.blocks),
        heads=tuple(
            replace(
                head,
                weights=jnp.stack([one.out_ws[index[site]] for site in head.sites]),
                biases=jnp.stack([one.out_bs[index[site]] for site in head.sites]),
            )
            for head in backbone.heads
        ),
    )


@pytest.mark.parametrize("remat", [False, True])
@pytest.mark.parametrize("mask", ["bidirectional", "causal"])
@pytest.mark.parametrize("ffn", ["gelu", "swiglu"])
def test_global_matches_one_chunk_values_and_gradients(
    remat: bool, mask: Literal["bidirectional", "causal"], ffn: Literal["gelu", "swiglu"]
):
    arch = _arch(mask, ffn)
    chunk_arch = ChunkwiseTransformerCIFnArch(
        chunks=(Chunk(tuple(tap.key for tap in TAPS), tuple(site.name for site in SITES)),),
        input_dim=arch.input_dim,
        d_model=arch.d_model,
        n_blocks=arch.n_blocks,
        attention=arch.attention,
        ffn_hidden=arch.ffn_hidden,
        ffn_kind=arch.ffn_kind,
        learned_norm_scale=arch.learned_norm_scale,
    )
    chunk = chunkwise_transformer_backbone(chunk_arch.initialize(SITES, None, jax.random.key(0)))
    converted = _global_from_chunk(
        _global_backbone(arch.initialize(SITES, None, jax.random.key(1))), chunk
    )

    taps = _taps(2)
    sequence = SequenceLayout(jnp.array([[0, 0, 1, 1, 1], [0, 0, 0, 1, 1]], jnp.int32))
    _assert_close(
        _preactivations(converted, taps, sequence, remat),
        _preactivations(chunk, taps, sequence, remat),
        tolerance=2e-5,
    )
    global_gradient = eqx.filter_grad(_objective)(converted, taps, sequence, remat)
    chunk_gradient = eqx.filter_grad(_objective)(chunk, taps, sequence, remat)
    _assert_close(
        global_gradient, _global_from_chunk(global_gradient, chunk_gradient), tolerance=2e-5
    )


@pytest.mark.parametrize("mask", ["bidirectional", "causal"])
def test_global_attention_preserves_documents_and_reads_every_tap(
    mask: Literal["bidirectional", "causal"],
):
    fn = _global_backbone(_arch(mask, "gelu").initialize(SITES, None, jax.random.key(0)))
    taps = _taps(2)
    sequence = SequenceLayout(jnp.array([[0, 0, 1, 1, 1]] * 2, jnp.int32))
    original = _preactivations(fn, taps, sequence, remat=True)
    for key in taps:
        changed = _preactivations(
            fn, {**taps, key: taps[key].at[:, 2:, :].set(0.1)}, sequence, remat=True
        )
        for name in original:
            np.testing.assert_array_equal(original[name][:, :2], changed[name][:, :2])
            assert not np.allclose(original[name][:, 2:], changed[name][:, 2:])
    if mask == "causal":
        later = {key: value.at[:, 1, :].set(0.1) for key, value in taps.items()}
        for name, value in _preactivations(fn, later, sequence, remat=True).items():
            np.testing.assert_array_equal(value[:, 0], original[name][:, 0])
    rescaled = {key: value * (i + 2) for i, (key, value) in enumerate(taps.items())}
    _assert_close(_preactivations(fn, rescaled, sequence, remat=True), original, tolerance=2e-5)


def test_global_rows_refuse_presets_they_do_not_name():
    mesh = AbstractMesh((2, 2, 2), THREE_AXES, axis_types=(AxisType.Explicit,) * 3)
    with pytest.raises(NotImplementedError, match="no rows for placement preset 'zero1'"):
        _arch("causal", "gelu").resolve_placement(SITES, from_config("zero1", mesh, SITES))


def test_global_row_refusals_name_each_leaf_independent_axis():
    mesh = AbstractMesh((2, 2, 2), THREE_AXES, axis_types=(AxisType.Explicit,) * 3)
    owner = preset_rows("owner")
    weights = owner.weights
    scanned_depth = replace(
        owner,
        weights=replace(
            weights,
            attention=replace(
                weights.attention,
                compute_weights={**weights.attention.compute_weights, "depth": ("replicate",)},
            ),
        ),
    )
    with pytest.raises(AssertionError, match=r"'ci_fn/attention.compute_weights' assigns `depth`"):
        bind_global_rows(scanned_depth, mesh)
    staged_stack = replace(
        owner,
        weights=replace(
            weights, output=replace(weights.output, ns_compute={"stack": ("replicate",)})
        ),
    )
    with pytest.raises(AssertionError, match=r"ci_fn/output.ns_compute may assign only `site`"):
        bind_global_rows(staged_stack, mesh)
    staged_input = replace(
        owner,
        weights=replace(
            weights, input=replace(weights.input, ns_compute={"input": ("replicate",)})
        ),
    )
    with pytest.raises(AssertionError, match=r"ci_fn/input.ns_compute may assign no axis"):
        bind_global_rows(staged_input, mesh)


def test_global_owners_must_tile_real_depth_and_site_counts():
    mesh = AbstractMesh((2, 2, 2), THREE_AXES, axis_types=(AxisType.Explicit,) * 3)
    rules = from_config("owner", mesh, SITES)
    arch = _arch("causal", "gelu")
    arch.resolve_placement(SITES, rules)
    with pytest.raises(AssertionError, match=r"'depth' \(dim 3\) does not tile"):
        replace(arch, n_blocks=3).resolve_placement(SITES, rules)
    with pytest.raises(AssertionError, match=r"'site' \(dim 3\) does not tile"):
        arch.resolve_placement(SITES[:6], rules)


def _mesh(shape: tuple[int, int, int]) -> Mesh:
    return Mesh(
        np.asarray(jax.devices()[: np.prod(shape)]).reshape(shape),
        THREE_AXES,
        axis_types=(AxisType.Explicit,) * 3,
    )


def _spec(value: jax.Array) -> P:
    sharding = value.sharding
    assert isinstance(sharding, NamedSharding), sharding
    return sharding.spec


type MuonSteps = tuple[jax.Array, BackboneCIFn, jax.Array, BackboneCIFn]


def _muon_steps(mesh: Mesh) -> tuple[BackboneCIFn, MuonSteps]:
    """Two placed Muon steps from one seed on `mesh`, with the loss and gradient of each."""
    arch = replace(
        _arch("causal", "swiglu"),
        attention=MHACIFnAttention(n_heads=2, implementation="xla", mask="causal"),
    )
    rules = from_config("owner", mesh, SITES)
    pd = PDConfigBase(
        steps=10,
        batch_size=8,
        components_optimizer=AdamWOptimizerConfig(lr_schedule=ScheduleConfig.constant(1e-3)),
        ci_fn_optimizer=MuonOptimizerConfig(
            type="muon", lr_schedule=ScheduleConfig.constant(1e-2), ns_dtype="float32"
        ),
    )
    _, optimizer = build_optimizers(pd, rules, SITES)
    sequence = SequenceLayout(jnp.zeros((8, 5), jnp.int32))

    @eqx.filter_jit
    def step(
        fn: BackboneCIFn, state: ScheduledOptimizerState, taps: dict[str, jax.Array]
    ) -> tuple[jax.Array, BackboneCIFn, BackboneCIFn, ScheduledOptimizerState]:
        loss, gradient = eqx.filter_value_and_grad(_resident_objective)(fn, taps, sequence)
        updates, state = optimizer.update(gradient, state, eqx.filter(fn, eqx.is_array))
        return loss, gradient, eqx.apply_updates(fn, updates), state

    activations = bind_global_rows(preset_rows("owner"), mesh).activations
    with jax.set_mesh(mesh):
        fn = placed_ci_fn(arch, SITES, jax.random.key(7), mesh, rules)
        assert isinstance(fn, BackboneCIFn), type(fn)
        taps = {
            key: jax.device_put(value, activations.sharding_for(("batch", "position", "feature")))
            for key, value in _taps(8).items()
        }
        state = optimizer.init(eqx.filter(fn, eqx.is_array))
        first_loss, first_gradient, updated, state = step(fn, state, taps)
        second_loss, second_gradient, _, _ = step(updated, state, taps)
    return fn, (first_loss, first_gradient, second_loss, second_gradient)


@pytest.mark.multidevice
def test_global_depth_and_site_owners_match_one_device_muon_steps():
    if jax.device_count() < 8:
        pytest.skip("requires eight local devices")
    _, reference = _muon_steps(_mesh((1, 1, 1)))
    fn, placed = _muon_steps(_mesh((2, 2, 2)))
    backbone = _global_backbone(fn)
    first_loss, gradient, second_loss, second_gradient = placed
    _assert_close(first_loss, reference[0], tolerance=1e-5)
    # The residents' cotangents are bfloat16, and the placed run reduces them across
    # replicas in a different order: each gradient leaf agrees to its bfloat16 rounding
    # (8 significand bits at the leaf's scale), and the second step inherits that
    # difference through the Muon update.
    for grads, reference_grads in ((gradient, reference[1]), (second_gradient, reference[3])):
        for x, y in zip(jax.tree.leaves(grads), jax.tree.leaves(reference_grads), strict=True):
            bfloat16_rounding = 2.0**-7 * float(jnp.max(jnp.abs(y)))
            np.testing.assert_allclose(x, y, atol=bfloat16_rounding, rtol=0)
    _assert_close(second_loss, reference[2], tolerance=1e-3)

    for parameter, grad in zip(jax.tree.leaves(fn), jax.tree.leaves(gradient), strict=True):
        assert grad.sharding.is_equivalent_to(parameter.sharding, parameter.ndim)
    assert _spec(backbone.blocks.wq) == P(("replicate",), ("tp",), ("fsdp",))
    assert _spec(backbone.blocks.w1) == P(("replicate",), None, ("tp", "fsdp"))
    assert _spec(backbone.heads[0].weights) == P(("replicate",), ("fsdp",), ("tp",))
    assert _spec(backbone.in_proj_w) == P(("tp",), ("fsdp", "replicate"))

    staging = ci_fn_muon_waypoints()(eqx.filter(fn, eqx.is_array))
    staged = _global_backbone(staging)
    for sharding in (staged.blocks.wq, staged.blocks.w2, staged.heads[1].weights):
        assert isinstance(sharding, NamedSharding)
        assert sharding.spec == P(("replicate",), None, None)
    assert isinstance(staged.in_proj_w, NamedSharding)
    assert staged.in_proj_w.spec == P(None, None, None)
