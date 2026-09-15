"""The NarrowCI bundle's contract: constructibility, dispatch at every constraint site,
and bit-level loss equivalence against the full-width SCATTER ORACLE.

`_scatter_to_full` below is the ONE place in the tree that materializes a bundle's full
`[.., C]` view — deliberately inside this test module, never in production code: every
consumer either dispatches on the bundle or refuses (`require_full_emission`)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from param_decomp.core.ci_fn import CI
from param_decomp.core.components import (
    NarrowCI,
    map_site_ci,
    narrow_component_sums,
    narrow_routed_counts,
    require_full_emission,
    site_ci_leading,
    site_ci_values,
)
from param_decomp.core.losses import (
    ema_frequency_penalty,
    importance_minimality_terms,
    per_component_frequencies,
)

E, K, C_PER_EXPERT = 6, 2, 3
C = E * C_PER_EXPERT
B, S = 2, 5


def _scatter_to_full(bundle: NarrowCI) -> jax.Array:
    """THE test oracle: the bundle's full `[.., C]` view, zeros at unrouted slots."""
    lead = bundle.values.shape[:-1]
    k = bundle.router_indices.shape[-1]
    values = bundle.values.reshape(*lead, k, bundle.c_per_expert)
    one_hot = jax.nn.one_hot(bundle.router_indices, bundle.n_experts, dtype=values.dtype)
    full = jnp.einsum("...kc,...ke->...ec", values, one_hot)
    return full.reshape(*lead, bundle.C)


def _bundle(key: jax.Array) -> NarrowCI:
    ids_key, values_key = jax.random.split(key)
    # distinct experts per token, as a real top-k guarantees
    order = jnp.argsort(jax.random.uniform(ids_key, (B, S, E)), axis=-1)
    values = jax.random.normal(values_key, (B, S, K * C_PER_EXPERT))
    return NarrowCI(values=values, router_indices=order[..., :K], n_experts=E)


def test_bundle_pytree_roundtrip_and_helpers():
    bundle = _bundle(jax.random.key(0))
    leaves, treedef = jax.tree.flatten(bundle)
    assert len(leaves) == 2
    rebuilt = jax.tree.unflatten(treedef, leaves)
    assert rebuilt.n_experts == E and rebuilt.c_per_expert == C_PER_EXPERT and rebuilt.C == C
    assert site_ci_leading(bundle) == (B, S)
    assert site_ci_values(bundle) is bundle.values

    doubled = map_site_ci(lambda v: 2.0 * v, bundle)
    assert isinstance(doubled, NarrowCI)
    np.testing.assert_array_equal(doubled.router_indices, bundle.router_indices)
    np.testing.assert_allclose(doubled.values, 2.0 * bundle.values)

    full = jnp.zeros((B, S, C))
    assert require_full_emission(full) is full
    with pytest.raises(NotImplementedError, match="narrow"):
        require_full_emission(bundle)


def test_squashings_commute_with_the_scatter():
    """`CI.from_preactivations` maps both squashings over the bundle's values; scattering
    the squashed bundle equals squashing the scattered oracle (both squashings fix 0)."""
    bundle = _bundle(jax.random.key(1))
    ci = CI.from_preactivations({"narrow": bundle, "full": _scatter_to_full(bundle)})
    for squashed in (ci.lower, ci.upper):
        narrow_value = squashed["narrow"]
        assert isinstance(narrow_value, NarrowCI)
        np.testing.assert_allclose(
            _scatter_to_full(narrow_value),
            require_full_emission(squashed["full"]),
            rtol=1e-6,
            atol=1e-7,
        )


def test_narrow_component_sums_match_the_oracle():
    bundle = _bundle(jax.random.key(2))
    sums = narrow_component_sums(bundle, bundle.values**2)
    oracle = (_scatter_to_full(bundle) ** 2).reshape(-1, C).sum(0)
    np.testing.assert_allclose(sums, oracle, rtol=1e-5, atol=1e-6)
    counts = narrow_routed_counts(bundle)
    oracle_counts = (
        jax.nn.one_hot(bundle.router_indices, E).reshape(-1, E).sum(0).astype(jnp.float32)
    )
    np.testing.assert_array_equal(counts, oracle_counts)


@pytest.mark.parametrize("reference_datapoint_count", [None, 4096])
def test_smooth_l0_terms_match_full_width_oracle(reference_datapoint_count: int | None):
    """Smooth-L0's `psi(0) = 0`: activity and freq on the bundle equal the full-width
    oracle, on BOTH the no-frequency (direct-sum) and frequency (segment) paths."""
    bundle = map_site_ci(jnp.abs, _bundle(jax.random.key(3)))
    assert isinstance(bundle, NarrowCI)
    gamma = jnp.asarray(0.3, jnp.float32)
    narrow = importance_minimality_terms(
        {"site": bundle}, gamma, reference_datapoint_count, normalize_at_one=False
    )
    full = importance_minimality_terms(
        {"site": _scatter_to_full(bundle)}, gamma, reference_datapoint_count, normalize_at_one=False
    )
    np.testing.assert_allclose(narrow[0], full[0], rtol=1e-6)
    np.testing.assert_allclose(narrow[1], full[1], rtol=1e-6)


def test_smooth_l0_activity_gradient_matches_full_width_oracle():
    bundle = map_site_ci(jnp.abs, _bundle(jax.random.key(5)))
    assert isinstance(bundle, NarrowCI)
    gamma = jnp.asarray(0.3, jnp.float32)

    def narrow_activity(values: jax.Array) -> jax.Array:
        b = NarrowCI(values, bundle.router_indices, E)
        return importance_minimality_terms({"site": b}, gamma, None, normalize_at_one=False)[0]

    def full_activity(values: jax.Array) -> jax.Array:
        b = NarrowCI(values, bundle.router_indices, E)
        return importance_minimality_terms(
            {"site": _scatter_to_full(b)}, gamma, None, normalize_at_one=False
        )[0]

    np.testing.assert_allclose(
        jax.grad(narrow_activity)(bundle.values),
        jax.grad(full_activity)(bundle.values),
        rtol=1e-5,
        atol=1e-7,
    )


def test_per_component_frequencies_and_ema_match_oracle():
    bundle = map_site_ci(jnp.abs, _bundle(jax.random.key(6)))
    assert isinstance(bundle, NarrowCI)
    gamma = jnp.asarray(0.3, jnp.float32)
    narrow_f = per_component_frequencies({"site": bundle}, gamma, normalize_at_one=False)
    full_f = per_component_frequencies(
        {"site": _scatter_to_full(bundle)}, gamma, normalize_at_one=False
    )
    assert narrow_f["site"].shape == (C,)
    np.testing.assert_allclose(narrow_f["site"], full_f["site"], rtol=1e-5, atol=1e-7)

    ema = {"site": jnp.full((C,), 0.01, jnp.float32)}
    narrow_ema = ema_frequency_penalty(narrow_f, ema, jnp.asarray(3.0), 100.0, 4096)
    full_ema = ema_frequency_penalty(full_f, ema, jnp.asarray(3.0), 100.0, 4096)
    np.testing.assert_allclose(narrow_ema[0], full_ema[0], rtol=1e-5)
    np.testing.assert_allclose(narrow_ema[1]["site"], full_ema[1]["site"], rtol=1e-5)


def test_alive_counts_identical_on_narrow_across_the_ci_domain():
    """Both L0 readouts — the per-position `CI_L0` count and the slow tier's per-component
    density count — equal their full-width oracle at every threshold in `[0, 1]`, the
    domain's ends included (each prices the unrouted pairs analytically)."""
    from param_decomp.core.ci_l0_eval import ci_l0_scalars
    from param_decomp.core.slow_eval import make_ci_reduction_step

    bundle = map_site_ci(lambda v: jnp.clip(v, 0.0, 1.0), _bundle(jax.random.key(7)))
    assert isinstance(bundle, NarrowCI)
    full = _scatter_to_full(bundle)
    for threshold in (0.0, 0.4, 1.0):
        narrow_l0 = ci_l0_scalars({"site": bundle}, ("site",), threshold, {}, jnp.mean)
        full_l0 = ci_l0_scalars({"site": full}, ("site",), threshold, {}, jnp.mean)
        np.testing.assert_allclose(
            narrow_l0[f"l0/{threshold}_site"], full_l0[f"l0/{threshold}_site"], rtol=1e-6
        )
        # the reduction step squashes preactivations itself; the clipped values are their
        # own lower squash, so feeding them as preactivations reads them back unchanged
        reduction = make_ci_reduction_step(threshold, None, None)
        narrow_density = reduction({"site": bundle})[0]["site"]
        full_density = reduction({"site": full})[0]["site"]
        assert narrow_density.shape == (C,)
        np.testing.assert_array_equal(narrow_density, full_density)


def test_mask_builders_carry_the_bundle():
    """Every mask builder is pointwise on the values: narrow masks leave as bundles with
    the SAME router indices (the seam contract travels in the type)."""
    from param_decomp.core.adversary import (
        ExpertBlockedSource,
        SiteSource,
    )
    from param_decomp.core.masking import (
        constant_delta_pinned_masks,
        masks_from_sources,
        stochastic_delta_pinned_masks,
        unmasked_no_delta_masks,
    )

    bundle = map_site_ci(lambda v: jnp.clip(jnp.abs(v), 0.0, 1.0), _bundle(jax.random.key(8)))
    assert isinstance(bundle, NarrowCI)
    ci_lower = {"site": bundle}

    masks, deltas = stochastic_delta_pinned_masks(ci_lower, jax.random.key(9))
    mask = masks["site"]
    assert isinstance(mask, NarrowCI)
    np.testing.assert_array_equal(mask.router_indices, bundle.router_indices)
    assert bool(jnp.all(mask.values >= bundle.values))
    assert deltas["site"].shape == (B, S)

    masks, _ = constant_delta_pinned_masks(0.5, ci_lower)
    mask = masks["site"]
    assert isinstance(mask, NarrowCI)
    np.testing.assert_allclose(mask.values, bundle.values + (1.0 - bundle.values) * 0.5)

    masks, deltas = unmasked_no_delta_masks(ci_lower)
    mask = masks["site"]
    assert isinstance(mask, NarrowCI)
    np.testing.assert_array_equal(mask.values, jnp.ones_like(bundle.values))
    np.testing.assert_array_equal(deltas["site"], jnp.zeros((B, S)))

    # block-dim sources: the narrow mask reads its routed entries by the bundle's
    # indices — gathered source == the oracle's pointwise view at the routed slots.
    source_values = jax.random.uniform(jax.random.key(10), (1, S, E, C_PER_EXPERT), jnp.float32)
    source: dict[str, SiteSource] = {
        "site": SiteSource(
            components=ExpertBlockedSource(values=source_values),
            delta=jax.random.uniform(jax.random.key(11), (1, S), jnp.float32),
        )
    }
    masks, deltas = masks_from_sources(ci_lower, source)
    mask = masks["site"]
    assert isinstance(mask, NarrowCI)
    full_ci = _scatter_to_full(bundle)
    full_mask = full_ci + (1.0 - full_ci) * source_values.reshape(1, S, C)
    # compare at the routed slots via the oracle scatter of the narrow mask minus the
    # structural (1-0)*source term at unrouted slots
    one_hot = jax.nn.one_hot(bundle.router_indices, E, dtype=jnp.float32)
    routed = jnp.einsum("bske,bsec->bskc", one_hot, full_mask.reshape(B, S, E, C_PER_EXPERT))
    np.testing.assert_allclose(
        mask.values, routed.reshape(B, S, K * C_PER_EXPERT), rtol=1e-6, atol=1e-6
    )


def test_refusal_arms():
    from param_decomp.core.masking import mixed_persistent_stochastic_masks

    bundle = _bundle(jax.random.key(12))
    with pytest.raises(NotImplementedError, match="narrow"):
        mixed_persistent_stochastic_masks(
            key=jax.random.key(13),
            ci_lower={"site": bundle},
            persistent_sources={},  # unreachable: the refusal fires first
            leading=(B, S),
            adv_fraction=jnp.asarray(0.5),
            stochastic_routes=None,
        )


def test_narrow_component_maxes_match_the_scatter_oracle():
    """The segment-max equals the full-width view's max over every leading axis — the
    unrouted-zero contribution included. Signed data exercises both arms: an expert with
    unrouted tokens clamps at 0, one routed by EVERY token keeps its (possibly negative)
    routed max."""
    from param_decomp.core.components import narrow_component_maxes

    # Slot 0 routes expert 0 on EVERY token (its block has no unrouted entries); the
    # remaining slots draw distinct experts from 1..E-1, keeping the top-k invariant.
    order = jnp.argsort(jax.random.uniform(jax.random.key(14), (B, S, E - 1)), axis=-1) + 1
    ids = jnp.concatenate([jnp.zeros((B, S, 1), order.dtype), order[..., : K - 1]], axis=-1)
    values = jax.random.normal(jax.random.key(15), (B, S, K * C_PER_EXPERT))
    bundle = NarrowCI(values=values, router_indices=ids, n_experts=E)
    data = jax.random.normal(jax.random.key(18), bundle.values.shape) - 0.5
    # Slot 0 (expert 0) all-negative: its exact routed max must come through un-clamped.
    data = data.at[..., :C_PER_EXPERT].set(-jnp.abs(data[..., :C_PER_EXPERT]) - 0.1)

    full = _scatter_to_full(NarrowCI(values=data, router_indices=ids, n_experts=E))
    oracle = jnp.max(full.reshape(-1, C), axis=0)
    np.testing.assert_allclose(narrow_component_maxes(bundle, data), oracle, rtol=1e-6)
    assert bool(jnp.any(oracle < 0.0)), "the all-routed arm was not exercised"


def test_per_component_batch_max_dispatches_on_emission():
    """The CI-scaled weight-decay statistic agrees with the full-width oracle on a mixed narrow/full dict."""
    from param_decomp.core.train import _per_component_batch_max

    bundle = _bundle(jax.random.key(16)).map_values(jax.nn.sigmoid)
    full = jax.random.uniform(jax.random.key(17), (B, S, C))
    maxes = _per_component_batch_max({"narrow": bundle, "full": full})
    oracle = jnp.max(_scatter_to_full(bundle).reshape(-1, C), axis=0)
    np.testing.assert_allclose(maxes["narrow"], oracle, rtol=1e-6)
    np.testing.assert_allclose(maxes["full"], jnp.max(full.reshape(-1, C), axis=0), rtol=1e-6)


multidevice = pytest.mark.skipif(len(jax.devices()) < 8, reason="requires eight local devices")


@multidevice
@pytest.mark.multidevice
def test_narrow_reductions_accept_a_batch_sharded_lead():
    """The bsc smoke's launch blocker: the slow-eval tier feeds `narrow_component_sums`
    dp-sharded narrow values (`float32[B@data, S, k·c]`), whose leading-collapse reshape
    explicit sharding refused. The no-collapse spellings must accept the sharded lead
    and agree exactly with the same reduction on the unsharded bundle."""
    from jax.sharding import AxisType, Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from param_decomp.core.components import narrow_component_maxes

    mesh = Mesh(np.asarray(jax.devices()[:8]), ("data",), axis_types=(AxisType.Explicit,))
    bundle = _bundle(jax.random.key(21))
    data = bundle.values**2
    expected_sums = narrow_component_sums(bundle, data)
    expected_maxes = narrow_component_maxes(bundle, bundle.values)
    expected_counts = narrow_routed_counts(bundle)

    def shard(x: jax.Array) -> jax.Array:
        return jax.device_put(x, NamedSharding(mesh, P("data", *(None,) * (x.ndim - 1))))

    # B=2 does not tile 8 devices; widen the batch by tiling, then scale the oracles.
    wide = NarrowCI(
        jnp.tile(bundle.values, (8 // B, 1, 1)),
        jnp.tile(bundle.router_indices, (8 // B, 1, 1)),
        E,
    )
    wide_data = jnp.tile(data, (8 // B, 1, 1))
    with jax.set_mesh(mesh):
        placed = NarrowCI(shard(wide.values), shard(wide.router_indices), E)
        got_sums = jax.jit(lambda b, d: narrow_component_sums(b, d))(placed, shard(wide_data))
        got_maxes = jax.jit(lambda b: narrow_component_maxes(b, b.values))(placed)
        got_counts = jax.jit(narrow_routed_counts)(placed)
    scale = 8 // B
    np.testing.assert_allclose(np.asarray(got_sums), scale * expected_sums, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(got_maxes), expected_maxes, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(got_counts), scale * expected_counts, rtol=0)
