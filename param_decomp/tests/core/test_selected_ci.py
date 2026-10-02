"""The SelectedCI bundle's contract: constructibility, dispatch at every constraint site,
and bit-level loss equivalence against the full-width SCATTER ORACLE.

`_scatter_to_full` below is the ONE place in the tree that materializes a bundle's full
`[.., C]` view — deliberately inside this test module, never in production code: every
consumer either dispatches on the bundle or refuses (`require_full_emission`)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from param_decomp.core.ci_fn.interface import CI
from param_decomp.core.components import (
    SelectedCI,
    block_selection_counts,
    map_site_ci,
    require_full_emission,
    selected_component_sums,
    site_ci_leading,
    site_ci_values,
)
from param_decomp.core.losses import (
    ema_frequency_penalty,
    importance_minimality_terms,
    per_component_frequencies,
)

E, K, C_PER_BLOCK = 6, 2, 3
C = E * C_PER_BLOCK
B, S = 2, 5


def _scatter_to_full(bundle: SelectedCI) -> jax.Array:
    """THE test oracle: the bundle's full `[.., C]` view, zeros at unselected slots."""
    lead = bundle.values.shape[:-1]
    k = bundle.block_indices.shape[-1]
    values = bundle.values.reshape(*lead, k, bundle.c_per_block)
    one_hot = jax.nn.one_hot(bundle.block_indices, bundle.n_blocks, dtype=values.dtype)
    full = jnp.einsum("...kc,...ke->...ec", values, one_hot)
    return full.reshape(*lead, bundle.C)


def _bundle(key: jax.Array) -> SelectedCI:
    ids_key, values_key = jax.random.split(key)
    # distinct blocks per token, as a real top-k guarantees
    order = jnp.argsort(jax.random.uniform(ids_key, (B, S, E)), axis=-1)
    values = jax.random.normal(values_key, (B, S, K * C_PER_BLOCK))
    return SelectedCI(values=values, block_indices=order[..., :K], n_blocks=E)


def test_bundle_pytree_roundtrip_and_helpers():
    bundle = _bundle(jax.random.key(0))
    leaves, treedef = jax.tree.flatten(bundle)
    assert len(leaves) == 2
    rebuilt = jax.tree.unflatten(treedef, leaves)
    assert rebuilt.n_blocks == E and rebuilt.c_per_block == C_PER_BLOCK and rebuilt.C == C
    assert site_ci_leading(bundle) == (B, S)
    assert site_ci_values(bundle) is bundle.values

    doubled = map_site_ci(lambda v: 2.0 * v, bundle)
    assert isinstance(doubled, SelectedCI)
    np.testing.assert_array_equal(doubled.block_indices, bundle.block_indices)
    np.testing.assert_allclose(doubled.values, 2.0 * bundle.values)

    full = jnp.zeros((B, S, C))
    assert require_full_emission(full) is full
    with pytest.raises(NotImplementedError, match="selected-emission"):
        require_full_emission(bundle)


def test_squashings_commute_with_the_scatter():
    """`CI.from_preactivations` maps both squashings over the bundle's values; scattering
    the squashed bundle equals squashing the scattered oracle (both squashings fix 0)."""
    bundle = _bundle(jax.random.key(1))
    ci = CI.from_preactivations({"selected": bundle, "full": _scatter_to_full(bundle)})
    for squashed in (ci.lower, ci.upper):
        selected_value = squashed["selected"]
        assert isinstance(selected_value, SelectedCI)
        np.testing.assert_allclose(
            _scatter_to_full(selected_value),
            require_full_emission(squashed["full"]),
            rtol=1e-6,
            atol=1e-7,
        )


def test_selected_component_sums_match_the_oracle():
    bundle = _bundle(jax.random.key(2))
    sums = selected_component_sums(bundle, bundle.values**2)
    oracle = (_scatter_to_full(bundle) ** 2).reshape(-1, C).sum(0)
    np.testing.assert_allclose(sums, oracle, rtol=1e-5, atol=1e-6)
    counts = block_selection_counts(bundle)
    oracle_counts = (
        jax.nn.one_hot(bundle.block_indices, E).reshape(-1, E).sum(0).astype(jnp.float32)
    )
    np.testing.assert_array_equal(counts, oracle_counts)


@pytest.mark.parametrize("reference_datapoint_count", [None, 4096])
def test_smooth_l0_terms_match_full_width_oracle(reference_datapoint_count: int | None):
    """Smooth-L0's `psi(0) = 0`: activity and freq on the bundle equal the full-width
    oracle, on BOTH the no-frequency (direct-sum) and frequency (segment) paths."""
    bundle = map_site_ci(jnp.abs, _bundle(jax.random.key(3)))
    assert isinstance(bundle, SelectedCI)
    gamma = jnp.asarray(0.3, jnp.float32)
    selected = importance_minimality_terms(
        {"site": bundle}, gamma, reference_datapoint_count, normalize_at_one=False
    )
    full = importance_minimality_terms(
        {"site": _scatter_to_full(bundle)}, gamma, reference_datapoint_count, normalize_at_one=False
    )
    np.testing.assert_allclose(selected[0], full[0], rtol=1e-6)
    np.testing.assert_allclose(selected[1], full[1], rtol=1e-6)


def test_smooth_l0_activity_gradient_matches_full_width_oracle():
    bundle = map_site_ci(jnp.abs, _bundle(jax.random.key(5)))
    assert isinstance(bundle, SelectedCI)
    gamma = jnp.asarray(0.3, jnp.float32)

    def selected_activity(values: jax.Array) -> jax.Array:
        b = SelectedCI(values, bundle.block_indices, E)
        return importance_minimality_terms({"site": b}, gamma, None, normalize_at_one=False)[0]

    def full_activity(values: jax.Array) -> jax.Array:
        b = SelectedCI(values, bundle.block_indices, E)
        return importance_minimality_terms(
            {"site": _scatter_to_full(b)}, gamma, None, normalize_at_one=False
        )[0]

    np.testing.assert_allclose(
        jax.grad(selected_activity)(bundle.values),
        jax.grad(full_activity)(bundle.values),
        rtol=1e-5,
        atol=1e-7,
    )


def test_per_component_frequencies_and_ema_match_oracle():
    bundle = map_site_ci(jnp.abs, _bundle(jax.random.key(6)))
    assert isinstance(bundle, SelectedCI)
    gamma = jnp.asarray(0.3, jnp.float32)
    selected_f = per_component_frequencies({"site": bundle}, gamma, normalize_at_one=False)
    full_f = per_component_frequencies(
        {"site": _scatter_to_full(bundle)}, gamma, normalize_at_one=False
    )
    assert selected_f["site"].shape == (C,)
    np.testing.assert_allclose(selected_f["site"], full_f["site"], rtol=1e-5, atol=1e-7)

    ema = {"site": jnp.full((C,), 0.01, jnp.float32)}
    selected_ema = ema_frequency_penalty(selected_f, ema, jnp.asarray(3.0), 100.0, 4096)
    full_ema = ema_frequency_penalty(full_f, ema, jnp.asarray(3.0), 100.0, 4096)
    np.testing.assert_allclose(selected_ema[0], full_ema[0], rtol=1e-5)
    np.testing.assert_allclose(selected_ema[1]["site"], full_ema[1]["site"], rtol=1e-5)


def test_alive_counts_identical_on_selected_across_the_ci_domain():
    """Both L0 readouts — the per-position `CI_L0` count and the slow tier's per-component
    density count — equal their full-width oracle at every threshold in `[0, 1]`, the
    domain's ends included (each prices the unselected pairs analytically)."""
    from param_decomp.core.ci_l0_eval import ci_l0_scalars
    from param_decomp.core.slow_eval import make_ci_reduction_step

    bundle = map_site_ci(lambda v: jnp.clip(v, 0.0, 1.0), _bundle(jax.random.key(7)))
    assert isinstance(bundle, SelectedCI)
    full = _scatter_to_full(bundle)
    for threshold in (0.0, 0.4, 1.0):
        selected_l0 = ci_l0_scalars({"site": bundle}, ("site",), threshold, {}, jnp.mean)
        full_l0 = ci_l0_scalars({"site": full}, ("site",), threshold, {}, jnp.mean)
        np.testing.assert_allclose(
            selected_l0[f"l0/{threshold}_site"], full_l0[f"l0/{threshold}_site"], rtol=1e-6
        )
        # the reduction step squashes preactivations itself; the clipped values are their
        # own lower squash, so feeding them as preactivations reads them back unchanged
        reduction = jax.jit(make_ci_reduction_step(threshold, None, None))
        selected_density = reduction({"site": bundle})[0]["site"]
        full_density = reduction({"site": full})[0]["site"]
        assert selected_density.shape == (C,)
        np.testing.assert_array_equal(selected_density, full_density)


def test_mask_builders_carry_the_bundle():
    """Every mask builder is pointwise on the values: selected masks leave as bundles with
    the SAME block indices (the seam contract travels in the type)."""
    from param_decomp.core.adversary import (
        BlockedSourceComponents,
        SiteSource,
    )
    from param_decomp.core.masking import (
        constant_delta_pinned_masking,
        materialize_masking,
        source_masking,
        stochastic_delta_pinned_masking,
        unmasked_no_delta_masking,
    )

    bundle = map_site_ci(lambda v: jnp.clip(jnp.abs(v), 0.0, 1.0), _bundle(jax.random.key(8)))
    assert isinstance(bundle, SelectedCI)
    ci_lower = {"site": bundle}

    masking = stochastic_delta_pinned_masking(ci_lower, jax.random.key(9))
    masks = masking.component_masks
    assert masking.weight_delta_masks is not None
    deltas = masking.weight_delta_masks
    mask = masks["site"]
    assert isinstance(mask, SelectedCI)
    np.testing.assert_array_equal(mask.block_indices, bundle.block_indices)
    assert bool(jnp.all(mask.values >= bundle.values))
    assert deltas["site"].shape == (B, S)

    masking = constant_delta_pinned_masking(0.5, ci_lower)
    masks = masking.component_masks
    mask = masks["site"]
    assert isinstance(mask, SelectedCI)
    np.testing.assert_allclose(mask.values, bundle.values + (1.0 - bundle.values) * 0.5)

    masking = unmasked_no_delta_masking(ci_lower)
    masks = masking.component_masks
    assert masking.weight_delta_masks is None
    mask = masks["site"]
    assert isinstance(mask, SelectedCI)
    np.testing.assert_array_equal(mask.values, jnp.ones_like(bundle.values))

    # block-dim sources: the selected mask reads its selected entries by the bundle's
    # indices — gathered source == the oracle's pointwise view at the selected slots.
    source_values = jax.random.uniform(jax.random.key(10), (1, S, E, C_PER_BLOCK), jnp.float32)
    source: dict[str, SiteSource] = {
        "site": SiteSource(
            components=BlockedSourceComponents(values=source_values),
            delta=jax.random.uniform(jax.random.key(11), (1, S), jnp.float32),
        )
    }
    masking = materialize_masking(source_masking(ci_lower, source))
    masks = masking.component_masks
    assert masking.weight_delta_masks is not None
    deltas = masking.weight_delta_masks
    mask = masks["site"]
    assert isinstance(mask, SelectedCI)
    full_ci = _scatter_to_full(bundle)
    full_mask = full_ci + (1.0 - full_ci) * source_values.reshape(1, S, C)
    # compare at the selected slots via the oracle scatter of the selected mask minus the
    # structural (1-0)*source term at unselected slots
    one_hot = jax.nn.one_hot(bundle.block_indices, E, dtype=jnp.float32)
    selected = jnp.einsum("bske,bsec->bskc", one_hot, full_mask.reshape(B, S, E, C_PER_BLOCK))
    np.testing.assert_allclose(
        mask.values, selected.reshape(B, S, K * C_PER_BLOCK), rtol=1e-6, atol=1e-6
    )


def test_selected_component_maxes_match_the_scatter_oracle():
    """The segment-max equals the full-width view's max over every leading axis — the
    unselected-zero contribution included. Signed data exercises both arms: a block with
    unselected tokens clamps at 0, one selected by EVERY token keeps its (possibly negative)
    selected max."""
    from param_decomp.core.components import selected_component_maxes

    # Slot 0 selects block 0 on EVERY token (that block has no unselected entries); the
    # remaining slots draw distinct blocks from 1..E-1, keeping the top-k invariant.
    order = jnp.argsort(jax.random.uniform(jax.random.key(14), (B, S, E - 1)), axis=-1) + 1
    ids = jnp.concatenate([jnp.zeros((B, S, 1), order.dtype), order[..., : K - 1]], axis=-1)
    values = jax.random.normal(jax.random.key(15), (B, S, K * C_PER_BLOCK))
    bundle = SelectedCI(values=values, block_indices=ids, n_blocks=E)
    data = jax.random.normal(jax.random.key(18), bundle.values.shape) - 0.5
    # Slot 0 (block 0) all-negative: its exact selected max must come through un-clamped.
    data = data.at[..., :C_PER_BLOCK].set(-jnp.abs(data[..., :C_PER_BLOCK]) - 0.1)

    full = _scatter_to_full(SelectedCI(values=data, block_indices=ids, n_blocks=E))
    oracle = jnp.max(full.reshape(-1, C), axis=0)
    np.testing.assert_allclose(selected_component_maxes(bundle, data), oracle, rtol=1e-6)
    assert bool(jnp.any(oracle < 0.0)), "the all-selected arm was not exercised"


def test_per_component_batch_max_dispatches_on_emission():
    """Per-component max CI matches the dense oracle for mixed selected and full emissions."""
    from param_decomp.core.train import _per_component_batch_max

    bundle = _bundle(jax.random.key(16)).map_values(jax.nn.sigmoid)
    full = jax.random.uniform(jax.random.key(17), (B, S, C))
    maxes = _per_component_batch_max({"selected": bundle, "full": full})
    oracle = jnp.max(_scatter_to_full(bundle).reshape(-1, C), axis=0)
    np.testing.assert_allclose(maxes["selected"], oracle, rtol=1e-6)
    np.testing.assert_allclose(maxes["full"], jnp.max(full.reshape(-1, C), axis=0), rtol=1e-6)


multidevice = pytest.mark.skipif(len(jax.devices()) < 8, reason="requires eight local devices")


@multidevice
@pytest.mark.multidevice
def test_selected_reductions_accept_a_batch_sharded_lead():
    """Regression: the slow-eval tier feeds `selected_component_sums`
    dp-sharded selected values (`float32[B@data, S, k·c]`), whose leading-collapse reshape
    explicit sharding refused. The no-collapse spellings must accept the sharded lead
    and agree exactly with the same reduction on the unsharded bundle."""
    from jax.sharding import AxisType, Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from param_decomp.core.components import selected_component_maxes

    mesh = Mesh(np.asarray(jax.devices()[:8]), ("data",), axis_types=(AxisType.Explicit,))
    bundle = _bundle(jax.random.key(21))
    data = bundle.values**2
    expected_sums = selected_component_sums(bundle, data)
    expected_maxes = selected_component_maxes(bundle, bundle.values)
    expected_counts = block_selection_counts(bundle)

    def shard(x: jax.Array) -> jax.Array:
        return jax.device_put(x, NamedSharding(mesh, P("data", *(None,) * (x.ndim - 1))))

    # B=2 does not tile 8 devices; widen the batch by tiling, then scale the oracles.
    wide = SelectedCI(
        jnp.tile(bundle.values, (8 // B, 1, 1)),
        jnp.tile(bundle.block_indices, (8 // B, 1, 1)),
        E,
    )
    wide_data = jnp.tile(data, (8 // B, 1, 1))
    with jax.set_mesh(mesh):
        placed = SelectedCI(shard(wide.values), shard(wide.block_indices), E)
        got_sums = jax.jit(lambda b, d: selected_component_sums(b, d))(placed, shard(wide_data))
        got_maxes = jax.jit(lambda b: selected_component_maxes(b, b.values))(placed)
        got_counts = jax.jit(block_selection_counts)(placed)
    scale = 8 // B
    np.testing.assert_allclose(np.asarray(got_sums), scale * expected_sums, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(got_maxes), expected_maxes, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(got_counts), scale * expected_counts, rtol=0)
