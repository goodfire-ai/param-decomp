"""Block-dim adversarial sources for block-factored sites.

A block-factored site's component sources store `[.., E, c]` (`BlockedSourceComponents`) — value
`(e, j)` sources component `(e, j)` with no flat-offset arithmetic — while dense sites
keep the bare flat spelling; the delta source is an explicit tensor for every site.
The U[0,1] draws happen directly in those honest shapes (components and delta off the
split site key — no packed intermediate); persistent sources land in their semantic
stacks (`SourceStacks`) and are read through the `per_site()` view."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from param_decomp.core.adversary import (
    BlockedSourceComponents,
    SiteSource,
    Sources,
    SourcesSgdState,
    SourceStack,
    SourceStacks,
    init_fresh_pgd_sources,
    init_persistent_sources,
    init_sources_adam_state,
    init_sources_opt_state,
    source_values_to_float,
    sources_ascend_project,
)
from param_decomp.core.components import (
    BlockedFactorization,
    DenseFactorization,
    SelectedCI,
    SiteCI,
    SiteSpec,
    require_full_emission,
)
from param_decomp.core.configs import PGDInitStrategy, SgdPGDConfig
from param_decomp.core.masking import materialize_masking, source_mask_ingredients, source_masking
from param_decomp.core.schedule import ScheduleConfig

N_BLOCKS = 4
C_PER_BLOCK = 3
DENSE_C = 6


def _sites() -> tuple[SiteSpec, ...]:
    blocked = BlockedFactorization(n_blocks=N_BLOCKS, d_in=8, d_out=4, c_per_block=C_PER_BLOCK)
    return (
        SiteSpec(name="blocked_a", factorization=blocked, group="blocked"),
        SiteSpec(
            name="dense",
            factorization=DenseFactorization(d_in=8, d_out=8, C=DENSE_C),
            group="shared",
        ),
        SiteSpec(name="blocked_b", factorization=blocked, group="blocked"),
    )


def test_init_arms_draw_in_honest_shapes():
    sites = _sites()
    leading = (1, 5)
    sources = init_persistent_sources(sites, leading, jnp.float32, jax.random.PRNGKey(0)).per_site()
    blocked_source = sources["blocked_a"].components
    assert isinstance(blocked_source, BlockedSourceComponents)
    assert blocked_source.values.shape == (*leading, N_BLOCKS, C_PER_BLOCK)
    assert sources["blocked_a"].delta.shape == leading
    dense_source = sources["dense"].components
    assert isinstance(dense_source, jax.Array) and dense_source.shape == (*leading, DENSE_C)

    # The draws happen directly in the honest shapes off the split site key —
    # components first, delta second — with no packed intermediate.
    keys = jax.random.split(jax.random.PRNGKey(0), len(sites))
    components_key, delta_key = jax.random.split(keys[0])
    np.testing.assert_array_equal(
        blocked_source.values,
        jax.random.uniform(components_key, (*leading, N_BLOCKS, C_PER_BLOCK), jnp.float32),
    )
    np.testing.assert_array_equal(
        sources["blocked_a"].delta, jax.random.uniform(delta_key, leading, jnp.float32)
    )


def test_full_emission_masks_read_the_flat_block_major_view():
    sites = _sites()
    leading = (2, 3)
    sources = init_persistent_sources(sites, leading, jnp.float32, jax.random.PRNGKey(1)).per_site()
    ci_lower = {
        site.name: jax.random.uniform(
            jax.random.fold_in(jax.random.PRNGKey(2), idx), (*leading, site.C)
        )
        for idx, site in enumerate(sites)
    }
    masking = materialize_masking(source_masking(ci_lower, sources))
    masks = masking.component_masks
    assert masking.weight_delta_masks is not None
    delta_masks = masking.weight_delta_masks
    blocked_components = sources["blocked_a"].components
    assert isinstance(blocked_components, BlockedSourceComponents)
    np.testing.assert_array_equal(
        require_full_emission(masks["blocked_a"]),
        ci_lower["blocked_a"] + (1.0 - ci_lower["blocked_a"]) * blocked_components.flat,
    )
    np.testing.assert_array_equal(delta_masks["blocked_a"], sources["blocked_a"].delta)
    assert require_full_emission(masks["dense"]).shape == (*leading, DENSE_C)


def test_adam_state_mirrors_the_block_structure():
    sources = init_persistent_sources(_sites(), (1, 2), jnp.float32, jax.random.PRNGKey(4))
    state = init_sources_adam_state(sources)
    m_components = state.m.stacks["blocked"].components
    assert isinstance(m_components, BlockedSourceComponents)
    assert m_components.values.shape == (2, 1, 2, N_BLOCKS, C_PER_BLOCK)
    assert state.m.site_stack_indices == sources.site_stack_indices


@pytest.mark.parametrize("init", ["random", "ones", "zeroes"])
def test_fresh_pgd_sources_take_the_block_arm(init: PGDInitStrategy):
    sources = init_fresh_pgd_sources(_sites(), init, "c", (2, 3), jax.random.PRNGKey(5))
    components = sources["blocked_a"].components
    assert isinstance(components, BlockedSourceComponents)
    assert components.values.shape == (1, 1, N_BLOCKS, C_PER_BLOCK)
    assert sources["blocked_a"].delta.shape == (1, 1)
    dense = sources["dense"].components
    assert isinstance(dense, jax.Array) and dense.shape == (1, 1, DENSE_C)


def test_sgd_src_step_is_stateless_and_projects():
    """SGD projects sources to [0, 1] without optimizer buffers.

    Bf16 sources retain their dtype, and a mismatched config/state pair is rejected."""
    sgd = SgdPGDConfig(lr_schedule=ScheduleConfig.constant(0.5))
    sources = init_persistent_sources(_sites(), (2, 3), jnp.float32, jax.random.PRNGKey(6))
    opt_state = init_sources_opt_state(sgd, sources)
    assert isinstance(opt_state, SourcesSgdState)
    assert len(jax.tree.leaves(opt_state)) == 0

    grad = jax.tree.map(jnp.ones_like, sources)
    key = jax.random.PRNGKey(9)
    ascended, state_after = sources_ascend_project(
        sources, grad, opt_state, jnp.float32(0.25), sgd, key
    )
    assert state_after is opt_state
    after = ascended.per_site()["blocked_a"].components
    before = sources.per_site()["blocked_a"].components
    assert isinstance(after, BlockedSourceComponents) and isinstance(
        before, BlockedSourceComponents
    )
    np.testing.assert_array_equal(
        np.asarray(after.values), np.clip(np.asarray(before.values) + 0.25, 0.0, 1.0)
    )

    descended, _ = sources_ascend_project(
        sources, jax.tree.map(lambda g: -4.0 * g, grad), opt_state, jnp.float32(0.5), sgd, key
    )
    assert float(jnp.max(np.asarray(descended.per_site()["dense"].components))) == 0.0

    bf16_sources = init_persistent_sources(_sites(), (2, 3), jnp.bfloat16, jax.random.PRNGKey(7))
    bf16_grad = jax.tree.map(jnp.ones_like, bf16_sources)
    bf16_ascended, _ = sources_ascend_project(
        bf16_sources, bf16_grad, SourcesSgdState(), jnp.float32(0.25), sgd, key
    )
    assert all(leaf.dtype == jnp.bfloat16 for leaf in jax.tree.leaves(bf16_ascended))

    with pytest.raises(AssertionError, match="source optimizer config/state mismatch"):
        sources_ascend_project(
            sources, grad, init_sources_adam_state(sources), jnp.float32(1), sgd, key
        )


def test_sharded_init_matches_eager_with_mixed_geometries():
    """`init_sources_sharded` over a mixed dense/block-factored site set reproduces the
    eager stacked init bit-for-bit, and the block-factored group lands block-dim with its
    slot axis leading."""
    from jax.sharding import AxisType, Mesh

    from param_decomp.core.init_placed import init_sources_sharded
    from param_decomp.core.model import Positioned

    sites = _sites()
    mesh = Mesh(
        np.asarray(jax.devices()[:1]).reshape(1, 1),
        ("data", "tp"),
        axis_types=(AxisType.Explicit,) * 2,
    )
    sharded = init_sources_sharded(
        sites, Positioned(5), "bsc", 2, jnp.float32, jax.random.PRNGKey(3), mesh
    )
    eager = init_persistent_sources(sites, (2, 5), jnp.float32, jax.random.PRNGKey(3))
    blocked = sharded.stacks["blocked"].components
    assert isinstance(blocked, BlockedSourceComponents)
    assert blocked.values.shape == (2, 2, 5, N_BLOCKS, C_PER_BLOCK)
    assert sharded.site_stack_indices == eager.site_stack_indices
    for got, want in zip(jax.tree.leaves(sharded), jax.tree.leaves(eager), strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_momentum_sgd_src_step_carries_one_float_velocity():
    """Momentum SGD carries one signed velocity tree matching the source shapes.

    Velocity stays bf16 for either 16-bit source format and fp32 for fp32 sources."""
    from param_decomp.core.adversary import SourcesMomentumState
    from param_decomp.core.configs import MomentumSgdPGDConfig

    cfg = MomentumSgdPGDConfig(momentum=0.5, lr_schedule=ScheduleConfig.constant(0.1))
    sources = init_persistent_sources(_sites(), (2, 3), jnp.float32, jax.random.PRNGKey(8))
    state = init_sources_opt_state(cfg, sources)
    assert isinstance(state, SourcesMomentumState)
    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(state.velocity))
    assert len(jax.tree.leaves(state.velocity)) == len(jax.tree.leaves(sources))

    grad = jax.tree.map(jnp.ones_like, sources)
    key = jax.random.PRNGKey(10)
    lr = jnp.float32(0.25)
    once, state1 = sources_ascend_project(sources, grad, state, lr, cfg, key)
    assert isinstance(state1, SourcesMomentumState)
    # first ascent: v = 0.5*0 + 1 = 1 -> +lr
    np.testing.assert_allclose(
        np.asarray(once.per_site()["dense"].components),
        np.clip(np.asarray(sources.per_site()["dense"].components) + 0.25, 0.0, 1.0),
        rtol=1e-6,
    )
    twice, state2 = sources_ascend_project(once, grad, state1, lr, cfg, key)
    # second ascent: v = 0.5*1 + 1 = 1.5 -> +1.5*lr on the unclipped entries
    dense_before = np.asarray(once.per_site()["dense"].components)
    unclipped = dense_before + 0.375 <= 1.0
    np.testing.assert_allclose(
        np.asarray(twice.per_site()["dense"].components)[unclipped],
        (dense_before + 0.375)[unclipped],
        rtol=1e-6,
    )
    assert isinstance(state2, SourcesMomentumState)
    velocity = state2.velocity.stacks["shared"].components
    assert isinstance(velocity, jax.Array)
    np.testing.assert_allclose(np.asarray(velocity), 1.5, rtol=1e-6)

    # bf16 storage rides a bf16 velocity; uint16 storage does too (signed float buffer)
    bf16_sources = init_persistent_sources(_sites(), (2, 3), jnp.bfloat16, jax.random.PRNGKey(8))
    bf16_state = init_sources_opt_state(cfg, bf16_sources)
    assert isinstance(bf16_state, SourcesMomentumState)
    assert all(leaf.dtype == jnp.bfloat16 for leaf in jax.tree.leaves(bf16_state.velocity))
    u16_sources = init_persistent_sources(_sites(), (2, 3), jnp.uint16, jax.random.PRNGKey(8))
    u16_state = init_sources_opt_state(cfg, u16_sources)
    assert isinstance(u16_state, SourcesMomentumState)
    assert all(leaf.dtype == jnp.bfloat16 for leaf in jax.tree.leaves(u16_state.velocity))


def test_uint16_fixed_point_representation_round_trips_and_rounds_unbiased():
    """The uint16 unit-interval representation: init quantizes round-to-nearest, the
    float view dequantizes `u/65535` (error <= 1/(2*65535) at init), ascents store back
    saturating, and the stochastic rounding is UNBIASED — a sub-half-step update lands
    with probability proportional to its size instead of stalling to zero."""
    from param_decomp.core.adversary import UINT16_UNIT_SCALE, source_values_to_float

    sgd = SgdPGDConfig(lr_schedule=ScheduleConfig.constant(0.5))
    sources = init_persistent_sources(_sites(), (2, 3), jnp.uint16, jax.random.PRNGKey(11))
    assert all(leaf.dtype == jnp.uint16 for leaf in jax.tree.leaves(sources))
    fp32 = init_persistent_sources(_sites(), (2, 3), jnp.float32, jax.random.PRNGKey(11))
    view = source_values_to_float(sources)
    for got, want in zip(jax.tree.leaves(view), jax.tree.leaves(fp32), strict=True):
        assert got.dtype == jnp.float32
        np.testing.assert_allclose(
            np.asarray(got), np.asarray(want), atol=0.5 / UINT16_UNIT_SCALE, rtol=0
        )

    # saturating projection: a huge ascent pins every coordinate at exactly 1.0
    grad = jax.tree.map(jnp.ones_like, view)
    conditioning, _ = sources_ascend_project(
        sources,
        jax.tree.map(lambda g: 10.0 * g, grad),
        SourcesSgdState(),
        jnp.float32(1),
        sgd,
        jax.random.PRNGKey(12),
    )
    assert all(int(leaf.max()) == 65535 for leaf in jax.tree.leaves(conditioning))
    assert float(jnp.max(jax.tree.leaves(source_values_to_float(conditioning))[0])) == 1.0

    # unbiased sub-half-step: delta = 0.2 quantization steps never lands under
    # round-to-nearest; stochastically it lands ~20% of the time, mean ~delta
    delta = 0.2 / UINT16_UNIT_SCALE
    tiny_grad = jax.tree.map(lambda v: jnp.full_like(v, delta / 0.5), view)
    landed = []
    for trial in range(200):
        stepped, _ = sources_ascend_project(
            sources,
            tiny_grad,
            SourcesSgdState(),
            jnp.float32(0.5),
            sgd,
            jax.random.fold_in(jax.random.PRNGKey(13), trial),
        )
        moved = np.asarray(stepped.stacks["shared"].components, np.int64) - np.asarray(
            sources.stacks["shared"].components, np.int64
        )
        assert set(np.unique(moved)) <= {0, 1}
        landed.append(moved.mean())
    mean_step = float(np.mean(landed))
    assert 0.12 < mean_step < 0.28, mean_step  # ~Binomial(200*36, .2) mean, generous rails


def test_uint16_stochastic_ascent_tracks_fp32_better_than_bf16():
    """The ruling's fixture: PGD-style ascent toward a target vector, fp32 vs bf16 vs
    uint16+stochastic-rounding VALUES (same fp32 update math). bf16 stalls whenever
    |lr*grad| falls under half an ulp of the value (~2^-9 near 1.0); uint16 keeps moving
    in expectation at uniform resolution, so its endpoint error must land closer to the
    fp32 trajectory's than bf16's does."""
    from param_decomp.core.adversary import SourcesSgdState as Stateless
    from param_decomp.core.adversary import source_values_to_float, store_unit_float

    sgd = SgdPGDConfig(lr_schedule=ScheduleConfig.constant(1.0))
    lr = jnp.float32(3e-4)
    shape = (512,)
    target = jax.random.uniform(jax.random.PRNGKey(20), shape, jnp.float32)
    start = jax.random.uniform(jax.random.PRNGKey(21), shape, jnp.float32)
    site = SiteSpec(
        name="s", factorization=DenseFactorization(d_in=4, d_out=4, C=shape[0]), group="g"
    )

    def run(storage_dtype: jnp.dtype) -> np.ndarray:
        base = init_persistent_sources((site,), (), storage_dtype, jax.random.PRNGKey(0))
        stored = SourceStacks(
            stacks={
                "g": SourceStack(
                    components=store_unit_float(start, storage_dtype)[None],
                    delta=base.stacks["g"].delta,
                )
            },
            site_stack_indices=base.site_stack_indices,
        )
        for step in range(400):
            view = source_values_to_float(stored)
            values = view.stacks["g"].components
            assert isinstance(values, jax.Array)
            # ascent on -(s-t)^2/2: d/ds = (t - s)
            grad = SourceStacks(
                stacks={
                    "g": SourceStack(
                        components=(target[None] - values).astype(values.dtype),
                        delta=jnp.zeros_like(view.stacks["g"].delta),
                    )
                },
                site_stack_indices=stored.site_stack_indices,
            )
            stored, _ = sources_ascend_project(
                stored, grad, Stateless(), lr, sgd, jax.random.fold_in(jax.random.PRNGKey(1), step)
            )
        end = source_values_to_float(stored).per_site()[site.name].components
        assert isinstance(end, jax.Array)
        return np.asarray(end, np.float32)

    fp32_end = run(jnp.float32)
    bf16_end = run(jnp.bfloat16)
    u16_end = run(jnp.uint16)
    err = lambda end: float(np.sqrt(np.mean((end - fp32_end) ** 2)))  # noqa: E731
    bf16_err, u16_err = err(bf16_end), err(u16_end)
    assert u16_err < bf16_err, (u16_err, bf16_err)
    # and both track the analytic pull toward the target
    assert float(np.mean(np.abs(u16_end - np.asarray(target)))) < float(
        np.mean(np.abs(start - target))
    )


# ── the selected read: a take along the block axis ────────────────────────────

_SELECTED_K = 2
_SELECTED_SITES = ("blocked_a", "blocked_b")


def _selected_ci_lower(
    sites: tuple[SiteSpec, ...], leading: tuple[int, ...], dtype: jnp.dtype
) -> dict[str, SiteCI]:
    """Block-factored sites emit `SelectedCI` at `dtype` with DISTINCT block indices per
    token (the top-k of random logits); the dense site emits full width."""
    lower: dict[str, SiteCI] = {}
    for idx, site in enumerate(sites):
        key = jax.random.fold_in(jax.random.PRNGKey(30), idx)
        match site.factorization:
            case BlockedFactorization(n_blocks=n_blocks, c_per_block=c_per_block):
                logits = jax.random.normal(key, (*leading, n_blocks))
                top_k = jnp.argsort(-logits, axis=-1)[..., :_SELECTED_K]
                lower[site.name] = SelectedCI(
                    values=jnp.zeros((*leading, _SELECTED_K * c_per_block), dtype),
                    block_indices=top_k.astype(jnp.int32),
                    n_blocks=n_blocks,
                )
            case DenseFactorization(C=c):
                lower[site.name] = jax.random.uniform(key, (*leading, c), dtype)
    return lower


def _one_hot_selected_read(table: jax.Array, ci: SelectedCI, out_sharding: NamedSharding | None):
    """The test's own expectation: the selected rows as a one-hot contraction over the
    block axis at the table's fp32 view — exact for 0/1 weights — then cast to the
    CI's dtype, matching the source-mask admission cast."""
    one_hot = jax.nn.one_hot(ci.block_indices, ci.n_blocks, dtype=table.dtype)
    if out_sharding is None:
        rows = jnp.einsum("...ke,...ec->...kc", one_hot, table)
    else:
        rows = jnp.einsum("...ke,...ec->...kc", one_hot, table, out_sharding=out_sharding)
    return rows.reshape(ci.values.shape).astype(ci.values.dtype)


def _assert_selected_read_is_the_one_hot_contraction(
    ci_lower: dict[str, SiteCI],
    sources: Sources,
    out_sharding: NamedSharding | None,
    *,
    exact_grad: bool,
) -> None:
    """Forward AND source gradient of the selected read, compiled, against the one-hot
    spelling at fp32. The forward is bit-identical (an exact select; a pointwise cast
    commutes with it). The gradient is bit-identical wherever no two selected cotangents
    meet on one stored entry — every lead, at the fp32 CI dtype, and the full `bsc` lead
    at any dtype; a broadcast lead at a 16-bit CI dtype sums its cotangents in that
    dtype, so it tracks the fp32 sum within one rounding."""

    def read(sources: Sources) -> dict[str, jax.Array]:
        ingredients = source_mask_ingredients(ci_lower, sources)
        out: dict[str, jax.Array] = {}
        for site in _SELECTED_SITES:
            pair = ingredients[site]
            assert isinstance(pair.ci, SelectedCI)
            out[site] = pair.source_values
        return out

    def expected(sources: Sources) -> dict[str, jax.Array]:
        out: dict[str, jax.Array] = {}
        for site in _SELECTED_SITES:
            components = sources[site].components
            ci = ci_lower[site]
            assert isinstance(components, BlockedSourceComponents) and isinstance(ci, SelectedCI)
            out[site] = _one_hot_selected_read(components.values, ci, out_sharding)
        return out

    def loss(fn: Callable[[Sources], dict[str, jax.Array]], sources: Sources) -> jax.Array:
        return sum(
            (jnp.sum(jnp.sin(v.astype(jnp.float32))) for v in fn(sources).values()),
            jnp.float32(0.0),
        )

    got, want = jax.jit(read)(sources), jax.jit(expected)(sources)
    assert set(got) == set(_SELECTED_SITES) == set(want)
    for site in _SELECTED_SITES:
        assert got[site].dtype == want[site].dtype
        np.testing.assert_array_equal(np.asarray(got[site]), np.asarray(want[site]))
    got_grad = jax.jit(jax.grad(lambda s: loss(read, s)))(sources)
    want_grad = jax.jit(jax.grad(lambda s: loss(expected, s)))(sources)
    got_leaves, want_leaves = jax.tree.leaves(got_grad), jax.tree.leaves(want_grad)
    assert len(got_leaves) == len(want_leaves) > 0
    for got_leaf, want_leaf in zip(got_leaves, want_leaves, strict=True):
        assert got_leaf.dtype == want_leaf.dtype
        if exact_grad:
            np.testing.assert_array_equal(np.asarray(got_leaf), np.asarray(want_leaf))
        else:
            np.testing.assert_allclose(
                np.asarray(got_leaf), np.asarray(want_leaf), rtol=2**-8, atol=1e-7
            )


_SELECTED_CI_DTYPES = pytest.mark.parametrize(
    "ci_dtype", [jnp.float32, jnp.bfloat16], ids=["f32", "bf16"]
)


@_SELECTED_CI_DTYPES
@pytest.mark.parametrize("leading", [(2, 5), (1, 5), (2, 1)], ids=["bsc", "sc", "bc"])
def test_selected_source_read_is_the_exact_select_of_the_one_hot_contraction(
    leading: tuple[int, ...], ci_dtype: jnp.dtype
):
    """uint16 sources on every `SourceShape` lead (full, batch-shared `sc`,
    position-shared `bc`), read by a selected CI over a (2, 5) waist at both CI dtypes."""
    sites = _sites()
    waist = (2, 5)
    ci_lower = _selected_ci_lower(sites, waist, ci_dtype)
    stored = init_persistent_sources(sites, leading, jnp.uint16, jax.random.PRNGKey(31))
    _assert_selected_read_is_the_one_hot_contraction(
        ci_lower,
        source_values_to_float(stored).per_site(),
        None,
        exact_grad=ci_dtype == jnp.float32 or leading == waist,
    )


@pytest.mark.skipif(len(jax.devices()) < 8, reason="requires eight local devices")
@pytest.mark.multidevice
@_SELECTED_CI_DTYPES
@pytest.mark.parametrize("leading", [(4, 5), (1, 5), (4, 1)], ids=["bsc", "sc", "bc"])
def test_placed_selected_source_read_is_the_exact_select_of_the_one_hot_contraction(
    leading: tuple[int, ...], ci_dtype: jnp.dtype
):
    """The placed read on the `(data, tp)` mesh at the persist layout ({batch: data,
    expert: tp}): each shard takes its own blocks' rows and the typed shard sum lands
    the selected rows — against the one-hot contraction, forward and gradient."""
    devices = np.asarray(jax.devices()[:8]).reshape(4, 2)
    mesh = Mesh(devices, ("data", "tp"), axis_types=(AxisType.Explicit,) * 2)
    sites = _sites()
    waist = (4, 5)
    lead_spec = tuple("data" if extent == waist[0] else None for extent in leading)
    with jax.set_mesh(mesh):
        ci_lower: dict[str, SiteCI] = {}
        for site, ci in _selected_ci_lower(sites, waist, ci_dtype).items():
            match ci:
                case SelectedCI():
                    ci_lower[site] = SelectedCI(
                        values=jax.device_put(
                            ci.values, NamedSharding(mesh, P("data", None, None))
                        ),
                        block_indices=jax.device_put(
                            ci.block_indices, NamedSharding(mesh, P("data", None, None))
                        ),
                        n_blocks=ci.n_blocks,
                    )
                case _:
                    ci_lower[site] = jax.device_put(ci, NamedSharding(mesh, P("data", None, "tp")))
        stored = init_persistent_sources(sites, leading, jnp.uint16, jax.random.PRNGKey(31))
        sources: Sources = {}
        for site, source in source_values_to_float(stored).per_site().items():
            match source.components:
                case BlockedSourceComponents(values=values):
                    components: jax.Array | BlockedSourceComponents = BlockedSourceComponents(
                        values=jax.device_put(
                            values, NamedSharding(mesh, P(*lead_spec, "tp", None))
                        )
                    )
                case jax.Array() as dense:
                    components = jax.device_put(dense, NamedSharding(mesh, P(*lead_spec, "tp")))
            sources[site] = SiteSource(
                components=components,
                delta=jax.device_put(source.delta, NamedSharding(mesh, P(*lead_spec))),
            )
        _assert_selected_read_is_the_one_hot_contraction(
            ci_lower,
            sources,
            NamedSharding(mesh, P("data", None, None, None)),
            exact_grad=ci_dtype == jnp.float32 or leading == waist,
        )


def test_momentum_velocity_ema_forms_in_fp32_and_rounds_once():
    """Under bf16 velocity the EMA `momentum·v + grad` is formed in fp32 and rounded ONCE
    into the buffer dtype. Forming it in bf16 would round the coefficient itself before
    the multiply, so the two spellings must disagree somewhere on a random buffer — and
    the stored velocity must match the fp32 spelling everywhere."""
    from param_decomp.core.adversary import SourcesMomentumState
    from param_decomp.core.configs import MomentumSgdPGDConfig

    momentum = 0.9
    cfg = MomentumSgdPGDConfig(momentum=momentum, lr_schedule=ScheduleConfig.constant(0.1))
    sources = init_persistent_sources(_sites(), (4, 64), jnp.bfloat16, jax.random.PRNGKey(14))
    state = init_sources_opt_state(cfg, sources)
    assert isinstance(state, SourcesMomentumState)
    velocity = jax.tree.map(
        lambda v: jax.random.normal(jax.random.PRNGKey(15), v.shape, jnp.float32).astype(v.dtype),
        state.velocity,
    )
    grad = jax.tree.map(
        lambda v: jax.random.normal(jax.random.PRNGKey(16), v.shape, jnp.float32).astype(v.dtype),
        sources,
    )
    _, after = sources_ascend_project(
        sources,
        grad,
        SourcesMomentumState(velocity=velocity),
        jnp.float32(0.1),
        cfg,
        jax.random.PRNGKey(17),
    )
    assert isinstance(after, SourcesMomentumState)
    fp32_once = jax.tree.map(
        lambda v, g: (momentum * v.astype(jnp.float32) + g.astype(jnp.float32)).astype(v.dtype),
        velocity,
        grad,
    )
    bf16_math = jax.tree.map(lambda v, g: momentum * v + g, velocity, grad)
    for got, want in zip(jax.tree.leaves(after.velocity), jax.tree.leaves(fp32_once), strict=True):
        assert got.dtype == jnp.bfloat16
        np.testing.assert_array_equal(np.asarray(got, np.float32), np.asarray(want, np.float32))
    assert any(
        bool(jnp.any(want != rounded))
        for want, rounded in zip(
            jax.tree.leaves(fp32_once), jax.tree.leaves(bf16_math), strict=True
        )
    ), "the fixture must distinguish the two spellings"
