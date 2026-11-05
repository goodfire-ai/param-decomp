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
    FinalNormalization,
    GlobalTransformerBackbone,
    GlobalTransformerCIFnArch,
)
from param_decomp.core.ci_fn.implementations.global_transformer.placement import (
    GlobalCIStacks,
    bind_global_rows,
    preset_rows,
)
from param_decomp.core.ci_fn.implementations.transformer.backbone import BackboneCIFn
from param_decomp.core.ci_fn.implementations.transformer.layers import (
    AttentionSink,
    CIFnSelfAttentionWithSink,
    GQACIFnAttention,
    MHACIFnAttention,
    QueryKeyNormalization,
    RMSNormalizedQueryKey,
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
from param_decomp.core.placement import StackCensus, from_config
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
    """The `gelu` arm is the plain architecture; the `swiglu` arm also turns on every
    optional block feature: GQA, learned norm scales, and QK-norm."""
    match ffn:
        case "gelu":
            attention = MHACIFnAttention(n_heads=2, implementation="xla", mask=mask)
        case "swiglu":
            attention = GQACIFnAttention(
                n_heads=2,
                n_kv_heads=1,
                implementation="xla",
                mask=mask,
                qk_norm=QueryKeyNormalization.RMS,
            )
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


@pytest.mark.parametrize("qk_norm", [QueryKeyNormalization.NONE, QueryKeyNormalization.RMS])
def test_qk_norm_makes_ci_invariant_to_query_and_key_scale(qk_norm: QueryKeyNormalization):
    arch = _arch("causal", "gelu")
    arch = replace(arch, attention=replace(arch.attention, qk_norm=qk_norm))
    fn = _global_backbone(arch.initialize(SITES, None, jax.random.key(0)))
    taps = _taps(2)
    sequence = SequenceLayout(jnp.array([[0, 0, 0, 1, 1]] * 2, jnp.int32))
    original = _preactivations(fn, taps, sequence, remat=False)
    blocks = replace(
        fn.blocks,
        attention=replace(
            fn.blocks.attention, wq=fn.blocks.attention.wq * 7.0, wk=fn.blocks.attention.wk * 3.0
        ),
    )
    rescaled = _preactivations(replace(fn, blocks=blocks), taps, sequence, remat=False)
    if qk_norm is QueryKeyNormalization.RMS:
        _assert_close(rescaled, original, tolerance=2e-5)
    else:
        assert not all(np.allclose(rescaled[name], original[name]) for name in original)


@pytest.mark.parametrize("ffn", ["gelu", "swiglu"])
def test_logit_softcap_reaches_every_block_attention(ffn: Literal["gelu", "swiglu"]):
    """The cap holds no parameters, so a capped and an uncapped CI fn initialize identically:
    a cap far above every logit leaves the CI unchanged, and a small one changes it."""
    arch = _arch("causal", ffn)
    taps = _taps(2)
    sequence = SequenceLayout(jnp.array([[0, 0, 0, 1, 1]] * 2, jnp.int32))

    def capped(cap: float | None) -> dict[str, jax.Array]:
        capped_arch = replace(arch, attention=replace(arch.attention, logit_softcap=cap))
        fn = _global_backbone(capped_arch.initialize(SITES, None, jax.random.key(0)))
        assert fn.blocks.attention.logit_softcap == cap
        return _preactivations(fn, taps, sequence, remat=False)

    original = capped(None)
    _assert_close(capped(1e6), original, tolerance=2e-5)
    small = capped(1e-2)
    assert not all(np.allclose(small[name], original[name]) for name in original)


@pytest.mark.parametrize("which", ["q", "k"])
def test_qk_norm_gains_scale_each_heads_logits(which: Literal["q", "k"]):
    """The gains init to ones, one per q/k channel; sharpening one head's queries or
    keys changes the CI."""
    arch = _arch("bidirectional", "gelu")
    arch = replace(
        arch, n_blocks=1, attention=replace(arch.attention, qk_norm=QueryKeyNormalization.RMS)
    )
    fn = _global_backbone(arch.initialize(SITES, None, jax.random.key(0)))
    normalization = fn.blocks.attention.query_key_norm
    assert isinstance(normalization, RMSNormalizedQueryKey)
    q_scale, k_scale = normalization.q_gain, normalization.k_gain
    assert q_scale.shape == k_scale.shape == (1, arch.d_model)
    np.testing.assert_array_equal(q_scale, 1.0)
    np.testing.assert_array_equal(k_scale, 1.0)
    taps = _taps(2)
    sequence = SequenceLayout.unsegmented_sequences_like(taps["early"][..., 0])
    original = _preactivations(fn, taps, sequence, remat=False)
    head_dim = arch.d_model // arch.attention.n_heads
    match which:
        case "q":
            gains = (q_scale.at[:, :head_dim].multiply(4.0), k_scale)
        case "k":
            gains = (q_scale, k_scale.at[:, :head_dim].multiply(4.0))
    blocks = replace(
        fn.blocks,
        attention=replace(fn.blocks.attention, query_key_norm=RMSNormalizedQueryKey(*gains)),
    )
    changed = _preactivations(replace(fn, blocks=blocks), taps, sequence, remat=False)
    assert not all(np.allclose(changed[name], original[name]) for name in original)


@pytest.mark.parametrize("final_norm", list(FinalNormalization))
@pytest.mark.parametrize("ffn", ["gelu", "swiglu"])
def test_final_norm_makes_ci_invariant_to_the_residual_streams_scale(
    ffn: Literal["gelu", "swiglu"], final_norm: FinalNormalization
):
    """Every block reads its residual through a norm, so scaling the input projection and
    every write into the residual stream scales the whole stream. Only the final norm keeps
    that scale from the heads."""
    arch = replace(_arch("causal", ffn), final_norm=final_norm)
    fn = _global_backbone(arch.initialize(SITES, None, jax.random.key(0)))
    assert fn.final_norm is final_norm
    taps = _taps(2)
    sequence = SequenceLayout(jnp.array([[0, 0, 0, 1, 1]] * 2, jnp.int32))
    original = _preactivations(fn, taps, sequence, remat=False)
    blocks = fn.blocks
    louder = replace(
        fn,
        in_proj_w=fn.in_proj_w * 5.0,
        in_proj_b=fn.in_proj_b * 5.0,
        blocks=replace(
            blocks,
            attention=replace(blocks.attention, wo=blocks.attention.wo * 5.0),
            w2=blocks.w2 * 5.0,
            b2=blocks.b2 * 5.0,
        ),
    )
    rescaled = _preactivations(louder, taps, sequence, remat=False)
    if final_norm is FinalNormalization.RMS:
        _assert_close(rescaled, original, tolerance=2e-5)
    else:
        assert not all(np.allclose(rescaled[name], original[name]) for name in original)


def _with_sink_logits(fn: GlobalTransformerBackbone, value: float) -> GlobalTransformerBackbone:
    attention = fn.blocks.attention
    assert isinstance(attention, CIFnSelfAttentionWithSink)
    attention = replace(attention, sink_logits=jnp.full_like(attention.sink_logits, value))
    return replace(fn, blocks=replace(fn.blocks, attention=attention))


@pytest.mark.parametrize("logit_softcap", [None, 5.0])
@pytest.mark.parametrize("mask", ["bidirectional", "causal"])
@pytest.mark.parametrize("ffn", ["gelu", "swiglu"])
def test_a_sink_lets_every_head_attend_to_nothing(
    ffn: Literal["gelu", "swiglu"],
    mask: Literal["bidirectional", "causal"],
    logit_softcap: float | None,
):
    """The sink holds no random draws, so a sunk and a plain CI fn initialize alike. Its
    logits start at zero, one per block and query head. Far below every logit it leaves the
    CI as it was; far above, every head attends to nothing, as if attention wrote nothing."""
    plain_arch = _arch(mask, ffn)
    plain_arch = replace(
        plain_arch, attention=replace(plain_arch.attention, logit_softcap=logit_softcap)
    )
    sunk_arch = replace(
        plain_arch, attention=replace(plain_arch.attention, sink=AttentionSink.LEARNED)
    )
    plain = _global_backbone(plain_arch.initialize(SITES, None, jax.random.key(0)))
    sunk = _global_backbone(sunk_arch.initialize(SITES, None, jax.random.key(0)))
    assert type(plain.blocks.attention) is not CIFnSelfAttentionWithSink
    attention = sunk.blocks.attention
    assert isinstance(attention, CIFnSelfAttentionWithSink)
    assert attention.sink_logits.shape == (plain_arch.n_blocks, plain_arch.attention.n_heads)
    np.testing.assert_array_equal(attention.sink_logits, 0.0)
    taps = _taps(2)
    sequence = SequenceLayout(jnp.array([[0, 0, 0, 1, 1]] * 2, jnp.int32))
    original = _preactivations(plain, taps, sequence, remat=False)
    at_zero = _preactivations(sunk, taps, sequence, remat=False)
    assert not all(np.allclose(at_zero[name], original[name]) for name in original)
    _assert_close(
        _preactivations(_with_sink_logits(sunk, -1e4), taps, sequence, remat=False),
        original,
        tolerance=2e-5,
    )
    silent = replace(
        plain,
        blocks=replace(
            plain.blocks,
            attention=replace(plain.blocks.attention, wo=plain.blocks.attention.wo * 0),
        ),
    )
    _assert_close(
        _preactivations(_with_sink_logits(sunk, 1e4), taps, sequence, remat=False),
        _preactivations(silent, taps, sequence, remat=False),
        tolerance=2e-5,
    )
    gradient = eqx.filter_grad(_objective)(sunk, taps, sequence, False)
    gradient_attention = gradient.blocks.attention
    assert isinstance(gradient_attention, CIFnSelfAttentionWithSink)
    assert np.all(np.asarray(gradient_attention.sink_logits) != 0.0)


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


def test_global_owners_pad_depth_and_site_stacks_that_do_not_tile():
    mesh = AbstractMesh((2, 2, 2), THREE_AXES, axis_types=(AxisType.Explicit,) * 3)
    arch = _arch("causal", "gelu")
    tiled = arch.resolve_placement(SITES, from_config("owner", mesh, SITES))
    assert tiled.stacks == GlobalCIStacks.unpadded(2, (4, 4))
    padded = replace(arch, n_blocks=3).resolve_placement(
        SITES[:6], from_config("owner", mesh, SITES[:6])
    )
    one_pad = StackCensus(stack_len=3, stack_pad=1)
    assert padded.stacks == GlobalCIStacks(depth=one_pad, sites=(one_pad, one_pad))


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


def _muon_steps(
    mesh: Mesh,
    sites: tuple[SiteSpec, ...] = SITES,
    n_blocks: int = 2,
    sink: AttentionSink = AttentionSink.NONE,
    final_norm: FinalNormalization = FinalNormalization.NONE,
) -> tuple[BackboneCIFn, BackboneCIFn, MuonSteps]:
    """Two placed Muon steps from one seed on `mesh`: the initial and once-updated CI fns,
    and the loss and gradient of each step."""
    arch = replace(
        _arch("causal", "swiglu"),
        n_blocks=n_blocks,
        attention=MHACIFnAttention(
            n_heads=2,
            implementation="xla",
            mask="causal",
            qk_norm=QueryKeyNormalization.RMS,
            sink=sink,
        ),
        final_norm=final_norm,
    )
    rules = from_config("owner", mesh, sites)
    pd = PDConfigBase(
        steps=10,
        batch_size=8,
        components_optimizer=AdamWOptimizerConfig(lr_schedule=ScheduleConfig.constant(1e-3)),
        ci_fn_optimizer=MuonOptimizerConfig(
            type="muon", lr_schedule=ScheduleConfig.constant(1e-2), ns_dtype="float32"
        ),
    )
    _, optimizer = build_optimizers(pd, rules, sites)
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
        fn = placed_ci_fn(arch, sites, jax.random.key(7), mesh, rules)
        assert isinstance(fn, BackboneCIFn), type(fn)
        taps = {
            key: jax.device_put(value, activations.sharding_for(("batch", "position", "feature")))
            for key, value in _taps(8).items()
        }
        state = optimizer.init(eqx.filter(fn, eqx.is_array))
        first_loss, first_gradient, updated, state = step(fn, state, taps)
        second_loss, second_gradient, _, _ = step(updated, state, taps)
    return fn, updated, (first_loss, first_gradient, second_loss, second_gradient)


def _assert_bfloat16_close(gradients: object, reference: object, roundings: float = 1.0) -> None:
    """The residents' cotangents are bfloat16, and a placed run reduces them across
    replicas in a different order: each gradient leaf agrees to `roundings` of its bfloat16
    rounding (8 significand bits at the leaf's scale). A padded leaf's real slice agrees and
    its pad slice is exactly zero."""
    for x, y in zip(jax.tree.leaves(gradients), jax.tree.leaves(reference), strict=True):
        x, y = np.asarray(x, np.float32), np.asarray(y, np.float32)
        bfloat16_rounding = 2.0**-7 * float(np.max(np.abs(y)))
        np.testing.assert_allclose(x[: y.shape[0]], y, atol=roundings * bfloat16_rounding, rtol=0)
        np.testing.assert_array_equal(x[y.shape[0] :], 0.0)


@pytest.mark.multidevice
def test_global_depth_and_site_owners_match_one_device_muon_steps():
    if jax.device_count() < 8:
        pytest.skip("requires eight local devices")
    _, _, reference = _muon_steps(_mesh((1, 1, 1)))
    fn, _, placed = _muon_steps(_mesh((2, 2, 2)))
    backbone = _global_backbone(fn)
    assert backbone.stacks == GlobalCIStacks.unpadded(2, (4, 4))
    first_loss, gradient, second_loss, second_gradient = placed
    _assert_close(first_loss, reference[0], tolerance=1e-5)
    # The second step inherits the first gradient's bfloat16 difference through the
    # Muon update.
    _assert_bfloat16_close(gradient, reference[1])
    _assert_bfloat16_close(second_gradient, reference[3])
    _assert_close(second_loss, reference[2], tolerance=1e-3)

    for parameter, grad in zip(jax.tree.leaves(fn), jax.tree.leaves(gradient), strict=True):
        assert grad.sharding.is_equivalent_to(parameter.sharding, parameter.ndim)
    assert _spec(backbone.blocks.attention.wq) == P(("replicate",), ("tp",), ("fsdp",))
    assert _spec(backbone.blocks.w1) == P(("replicate",), None, ("tp", "fsdp"))
    assert _spec(backbone.heads[0].weights) == P(("replicate",), ("fsdp",), ("tp",))
    assert _spec(backbone.in_proj_w) == P(("tp",), ("fsdp", "replicate"))

    staging = ci_fn_muon_waypoints()(eqx.filter(fn, eqx.is_array))
    staged = _global_backbone(staging)
    for sharding in (staged.blocks.attention.wq, staged.blocks.w2, staged.heads[1].weights):
        assert isinstance(sharding, NamedSharding)
        assert sharding.spec == P(("replicate",), None, None)
    assert isinstance(staged.in_proj_w, NamedSharding)
    assert staged.in_proj_w.spec == P(None, None, None)


@pytest.mark.multidevice
@pytest.mark.parametrize("sink_and_final_norm", [False, True])
def test_padded_depth_and_site_owners_match_one_device_and_keep_pads_zero(
    sink_and_final_norm: bool,
):
    """Three blocks and three heads per group under `replicate: 2`: every persist stack
    carries one zero pad, which the forward never sees, which receives exactly zero
    gradient, and which a Muon step leaves at zero. With learned sinks, too, whose
    per-block logits pad along `depth` with the rest of the block."""
    if jax.device_count() < 8:
        pytest.skip("requires eight local devices")
    sites = SITES[:6]
    sink = AttentionSink.LEARNED if sink_and_final_norm else AttentionSink.NONE
    final_norm = FinalNormalization.RMS if sink_and_final_norm else FinalNormalization.NONE
    _, _, reference = _muon_steps(_mesh((1, 1, 1)), sites, 3, sink, final_norm)
    fn, updated, placed = _muon_steps(_mesh((2, 2, 2)), sites, 3, sink, final_norm)
    one_pad = StackCensus(stack_len=3, stack_pad=1)
    assert _global_backbone(fn).stacks == GlobalCIStacks(depth=one_pad, sites=(one_pad, one_pad))
    first_loss, gradient, second_loss, second_gradient = placed
    _assert_close(first_loss, reference[0], tolerance=1e-5)
    _assert_bfloat16_close(gradient, reference[1])
    # The second step inherits the first gradient's rounding through the Muon update, and
    # the final norm amplifies it: unpadded layouts drift 1.4-1.5 roundings there too.
    _assert_bfloat16_close(
        second_gradient, reference[3], roundings=3.0 if sink_and_final_norm else 1.0
    )
    _assert_close(second_loss, reference[2], tolerance=1e-3)

    stepped = _global_backbone(updated)
    assert stepped.blocks.attention.wq.shape[0] == 4
    sink_logits = stepped.blocks.attention.sink()
    if sink_and_final_norm:
        assert isinstance(stepped.blocks.attention, CIFnSelfAttentionWithSink)
        assert sink_logits is not None and sink_logits.shape == (4, 2)
    else:
        assert sink_logits is None
    for leaf in jax.tree.leaves(eqx.filter(stepped.blocks, eqx.is_array)):
        np.testing.assert_array_equal(np.asarray(leaf)[3:], 0.0)
    for head in stepped.heads:
        assert head.weights.shape[0] == 4 and len(head.sites) == 3
        np.testing.assert_array_equal(np.asarray(head.weights)[3:], 0.0)
        np.testing.assert_array_equal(np.asarray(head.biases)[3:], 0.0)
    residents = _global_backbone(updated.prepare())
    assert residents.stacks == GlobalCIStacks.unpadded(3, (3, 3))
    assert residents.blocks.attention.wq.shape[0] == 3
    assert all(head.weights.shape[0] == 3 for head in residents.heads)
