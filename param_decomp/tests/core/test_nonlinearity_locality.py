"""Numerics, validation, eval, and training plumbing for the nonlinearity loss."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec
from pydantic import ValidationError

from param_decomp.core.components import (
    BlockedFactorization,
    ComponentStacks,
    DenseFactorization,
    SiteSpec,
    component_stacks_from_sites,
    nonlinearity_partitions,
    site_stack_indices_for,
)
from param_decomp.core.configs import (
    AnyLossMetricConfig,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    NonlinearityLocalityLossConfig,
    StochasticReconLossConfig,
)
from param_decomp.core.losses import nonlinearity_loss
from param_decomp.core.nonlinearity import (
    ComponentSide,
    DeltaNetHeads,
    KVHeads,
    Neurons,
    NonlinearityAlignment,
    NonlinearityPartition,
    NonlinearityUnitKind,
    QueryHeads,
)
from param_decomp.core.nonlinearity_eval import (
    NONLINEARITY_EVAL_EFFECTIVE_COUNT_KEY,
    NONLINEARITY_EVAL_RELATIVE_THRESHOLD,
    NONLINEARITY_EVAL_SOFT_COUNT_KEY,
    SiteNonlinearityStats,
    component_nonlinearity_stats,
    make_nonlinearity_eval_step,
    nonlinearity_log_entries,
    site_nonlinearity_stats,
)
from param_decomp.core.objective import ResolvedNonlinearity, build_objective
from param_decomp.core.schedule import Knot, ScheduleConfig


def _stacked(us: dict[str, jax.Array]) -> ComponentStacks:
    return component_stacks_from_sites(
        {name: (jnp.zeros((1, u.shape[0])), u) for name, u in us.items()}
    )


def _loss(
    u: jax.Array,
    partition: NonlinearityPartition,
    threshold: float,
    *,
    normalize_at_one: bool = False,
) -> float:
    value, _ = nonlinearity_loss(
        _stacked({"s": u}),
        {"s": NonlinearityAlignment("output", partition)},
        jnp.asarray(threshold, jnp.float32),
        {partition.unit_kind: 1.0},
        normalize_at_one=normalize_at_one,
    )
    return float(value)


def test_one_hot_and_uniform_soft_counts_follow_formula():
    u_count, t = 8, 4.0
    one_hot = jnp.zeros((1, u_count)).at[0, 3].set(2.5)
    # a single loaded unit: share 1, count = 1/(1 + t/U)
    assert _loss(one_hot, Neurons(), t) == pytest.approx(1 / (1 + t / u_count), abs=1e-6)
    uniform = jnp.full((1, u_count), 0.7)
    assert _loss(uniform, Neurons(), t) == pytest.approx(u_count / (1 + t), rel=1e-6)


def test_normalize_at_one_makes_one_hot_count_exact_across_thresholds():
    unit_count = 8
    one_hot = jnp.zeros((1, unit_count)).at[0, 3].set(2.5)
    for threshold in (8.0, 4.0, 2.0):
        assert _loss(one_hot, Neurons(), threshold, normalize_at_one=True) == pytest.approx(
            1.0, abs=1e-6
        )


def test_normalize_at_one_preserves_gqa_use_multiplicity():
    n_kv_head, use_multiplicity, head_dim = 2, 4, 3
    one_hot = jnp.zeros((1, n_kv_head * head_dim)).at[0, :head_dim].set(1.0)
    for threshold in (8.0, 4.0, 2.0):
        assert _loss(
            one_hot,
            KVHeads(n_kv_head, use_multiplicity),
            threshold,
            normalize_at_one=True,
        ) == pytest.approx(use_multiplicity, abs=1e-6)


def test_relative_threshold_tracks_the_uniform_share_at_any_unit_count():
    """For m loaded units out of U, the count is m/(1 + t·m/U)."""
    t, m = 4.0, 3
    for u_count in (16, 256, 4096):
        u = jnp.zeros((1, u_count)).at[0, :m].set(1.0)
        assert _loss(u, Neurons(), t) == pytest.approx(m / (1 + t * m / u_count), rel=1e-5)


def test_scale_invariance_per_component_exact_across_wide_range():
    u = jax.random.normal(jax.random.PRNGKey(0), (5, 12))
    base = _loss(u, QueryHeads(4), 2.0)
    for scale in (1e-6, 1e-3, 137.0, 1e6):
        assert _loss(scale * u, QueryHeads(4), 2.0) == pytest.approx(base, rel=1e-5)


def test_radial_gradient_is_exactly_zero():
    """Degree-0 homogeneity (Euler): <dL/dU, U> = 0 — shrinking a component is never
    rewarded, at any norm. This is the invariant an absolute eps floor broke."""
    u = jax.random.normal(jax.random.PRNGKey(2), (3, 12))
    for scale in (1.0, 1e-5):
        scaled = scale * u
        grad = jax.grad(
            lambda x: nonlinearity_loss(
                _stacked({"s": x}),
                {"s": NonlinearityAlignment("output", QueryHeads(4))},
                jnp.asarray(3.0),
                {"attention_head": 1.0},
                normalize_at_one=False,
            )[0]
        )(scaled)
        radial = float(jnp.abs((grad * scaled).sum(-1)).max())
        tangential = float(jnp.abs(grad * scaled).sum(-1).max())
        assert tangential > 0.0
        assert radial <= 1e-5 * max(tangential, 1e-30)


def test_unit_blocks_are_contiguous():
    # mass on columns 0..2 = the whole of unit 0 (d_out 12, U 4, block 3) -> count ~1
    u = jnp.zeros((1, 12)).at[0, :3].set(1.0)
    assert _loss(u, QueryHeads(4), 4.0) == pytest.approx(1 / (1 + 4.0 / 4), rel=1e-6)
    # the same mass straddling units 0 and 1 -> two half-loaded units, higher count
    straddle = jnp.zeros((1, 12)).at[0, 2:5].set(1.0)
    assert _loss(straddle, QueryHeads(4), 4.0) > _loss(u, QueryHeads(4), 4.0)


def test_zero_component_contributes_zero_with_exactly_zero_gradient():
    u = jnp.stack([jnp.zeros(8), jnp.ones(8)])
    with_dead = _loss(u, Neurons(), 4.0)
    alive_only = _loss(jnp.ones((1, 8)), Neurons(), 4.0)
    assert with_dead == pytest.approx(alive_only / 2, rel=1e-6)  # dead adds 0 to the sum

    grad_fn = jax.grad(
        lambda x: nonlinearity_loss(
            _stacked({"s": x}),
            {"s": NonlinearityAlignment("output", Neurons())},
            jnp.asarray(4.0),
            {"neuron": 1.0},
            normalize_at_one=False,
        )[0]
    )
    grad = grad_fn(u)
    assert bool(jnp.all(jnp.isfinite(grad)))
    assert float(jnp.abs(grad[0]).max()) == 0.0  # the exact-zero branch: no pull either way

    # a tiny-but-nonzero component is just a component: finite grads that obey the
    # degree-(−1) homogeneity ∇f(s·u) = ∇f(u)/s (the max-preconditioning keeps fp32 sane)
    base_u = jnp.arange(1.0, 9.0)[None, :]
    tiny = grad_fn(1e-15 * base_u)
    assert bool(jnp.all(jnp.isfinite(tiny)))
    np.testing.assert_allclose(1e-15 * tiny, grad_fn(base_u), rtol=1e-4)


@pytest.mark.parametrize("normalize_at_one", [False, True])
def test_mean_over_components_across_input_and_output_sites(normalize_at_one: bool):
    output_vectors = jnp.zeros((2, 8)).at[:, 0].set(1.0)
    input_vectors = jnp.ones((6, 4))
    components = component_stacks_from_sites(
        {
            "output": (jnp.zeros((1, 2)), output_vectors),
            "input": (input_vectors.T, jnp.zeros((6, 1))),
        }
    )
    combined, _ = nonlinearity_loss(
        components,
        {
            "output": NonlinearityAlignment("output", Neurons()),
            "input": NonlinearityAlignment("input", Neurons()),
        },
        jnp.asarray(4.0),
        {"neuron": 1.0},
        normalize_at_one=normalize_at_one,
    )
    output_count = (1 + 4.0 / 8 if normalize_at_one else 1.0) / (1 + 4.0 / 8)
    input_count = (1 + 4.0 / 4 if normalize_at_one else 1.0) * 4 / (1 + 4.0)
    assert float(combined) == pytest.approx((2 * output_count + 6 * input_count) / 8, rel=1e-6)


def test_same_shape_sites_have_sitewise_equivalent_value_and_gradient():
    partition = QueryHeads(4)
    partitions = {"a": partition, "b": partition}
    a = jax.random.normal(jax.random.PRNGKey(3), (3, 12))
    b = jax.random.normal(jax.random.PRNGKey(4), (3, 12))

    def together(a: jax.Array, b: jax.Array) -> jax.Array:
        return nonlinearity_loss(
            _stacked({"a": a, "b": b}),
            {
                name: NonlinearityAlignment("output", partition)
                for name, partition in (partitions).items()
            },
            jnp.asarray(2.0),
            {"attention_head": 1.0},
            normalize_at_one=False,
        )[0]

    def separately(a: jax.Array, b: jax.Array) -> jax.Array:
        a_loss = nonlinearity_loss(
            _stacked({"a": a}),
            {"a": NonlinearityAlignment("output", partition)},
            jnp.asarray(2.0),
            {"attention_head": 1.0},
            normalize_at_one=False,
        )[0]
        b_loss = nonlinearity_loss(
            _stacked({"b": b}),
            {"b": NonlinearityAlignment("output", partition)},
            jnp.asarray(2.0),
            {"attention_head": 1.0},
            normalize_at_one=False,
        )[0]
        return (a_loss + b_loss) / 2

    actual = jax.value_and_grad(together, argnums=(0, 1))(a, b)
    expected = jax.value_and_grad(separately, argnums=(0, 1))(a, b)
    jax.tree.map(lambda x, y: np.testing.assert_allclose(x, y, rtol=1e-6), actual, expected)


def test_per_kind_reduction_and_attention_head_multiplier():
    partitions = {
        "mlp": Neurons(),
        "q": QueryHeads(2),
    }
    us = {
        "mlp": jnp.ones((2, 8)),
        "q": jnp.zeros((2, 8)).at[:, :4].set(1.0),
    }
    total, by_kind = nonlinearity_loss(
        _stacked(us),
        {
            name: NonlinearityAlignment("output", partition)
            for name, partition in (partitions).items()
        },
        jnp.asarray(4.0),
        {"neuron": 1.0, "attention_head": 0.25},
        normalize_at_one=False,
    )
    assert float(by_kind["neuron"]) == pytest.approx(8 / 5, rel=1e-6)
    assert float(by_kind["attention_head"]) == pytest.approx(1 / 3, rel=1e-6)
    assert float(total) == pytest.approx(8 / 5 + 0.25 / 3, rel=1e-6)


def test_concentrated_kv_component_counts_its_uses():
    """A kv component fully in one kv head feeds G = n_head/n_kv_head attention
    nonlinearities, so at small t its soft count is ≈ G, not ≈ 1."""
    n_head, n_kv_head, head_dim = 8, 2, 3
    g = n_head // n_kv_head
    kv = KVHeads(n_kv_head, g)
    concentrated = jnp.zeros((1, n_kv_head * head_dim)).at[0, :head_dim].set(1.0)
    t = 1e-3
    assert _loss(concentrated, kv, t) == pytest.approx(g / (1 + t / n_kv_head), rel=1e-6)
    assert _loss(concentrated, kv, t) == pytest.approx(g, rel=1e-3)


def test_uniform_kv_matches_uniform_q_at_equal_n_head():
    """Uniform over kv heads counts n_head/(1+t) uses — the same footing as uniform
    over n_head query heads."""
    n_head, n_kv_head, head_dim, t = 8, 2, 3, 4.0
    g = n_head // n_kv_head
    kv_loss = _loss(
        jnp.ones((1, n_kv_head * head_dim)),
        KVHeads(n_kv_head, g),
        t,
    )
    q_loss = _loss(
        jnp.ones((1, n_head * head_dim)),
        QueryHeads(n_head),
        t,
    )
    assert kv_loss == pytest.approx(n_head / (1 + t), rel=1e-6)
    assert kv_loss == pytest.approx(q_loss, rel=1e-6)


def test_normalized_uniform_q_and_kv_keep_their_site_local_endpoints():
    n_head, n_kv_head, head_dim, threshold = 8, 2, 3, 4.0
    use_multiplicity = n_head // n_kv_head
    kv_loss = _loss(
        jnp.ones((1, n_kv_head * head_dim)),
        KVHeads(n_kv_head, use_multiplicity),
        threshold,
        normalize_at_one=True,
    )
    q_loss = _loss(
        jnp.ones((1, n_head * head_dim)),
        QueryHeads(n_head),
        threshold,
        normalize_at_one=True,
    )

    raw_uniform_count = n_head / (1 + threshold)
    assert kv_loss == pytest.approx(raw_uniform_count * (1 + threshold / n_kv_head), rel=1e-6)
    assert q_loss == pytest.approx(raw_uniform_count * (1 + threshold / n_head), rel=1e-6)
    assert kv_loss > q_loss


def test_unpartitioned_stack_member_has_zero_gradient_and_no_effect():
    partition = Neurons()
    penalized = jax.random.normal(jax.random.PRNGKey(5), (3, 8))
    bystander = jax.random.normal(jax.random.PRNGKey(6), (3, 8))

    def loss_of(p: jax.Array, b: jax.Array) -> jax.Array:
        return nonlinearity_loss(
            _stacked({"p": p, "b": b}),
            {"p": NonlinearityAlignment("output", partition)},
            jnp.asarray(4.0),
            {"neuron": 1.0},
            normalize_at_one=False,
        )[0]

    value, (p_grad, b_grad) = jax.value_and_grad(loss_of, argnums=(0, 1))(penalized, bystander)
    alone = nonlinearity_loss(
        _stacked({"p": penalized}),
        {"p": NonlinearityAlignment("output", partition)},
        jnp.asarray(4.0),
        {"neuron": 1.0},
        normalize_at_one=False,
    )[0]
    assert float(value) == pytest.approx(float(alone), rel=1e-6)
    assert float(jnp.abs(p_grad).max()) > 0.0
    assert float(jnp.abs(b_grad).max()) == 0.0


def test_site_spec_rejects_bad_partitions():
    with pytest.raises(AssertionError):  # 10 % 4 != 0
        SiteSpec(
            "s",
            DenseFactorization(d_in=4, d_out=10, C=3),
            "s",
            alignment=NonlinearityAlignment("output", QueryHeads(4)),
        )
    with pytest.raises(AssertionError):
        QueryHeads(0)
    with pytest.raises(AssertionError):
        KVHeads(4, use_multiplicity=0)


def _loss_metrics(
    extra: tuple[NonlinearityLocalityLossConfig, ...] = (),
) -> list[AnyLossMetricConfig]:
    schedule = ScheduleConfig.constant(1.0)
    return [
        FaithfulnessLossConfig(coeff=1.0),
        ImportanceMinimalityLossConfig(coeff=1.0, gamma=schedule),
        StochasticReconLossConfig(coeff=1.0),
        *extra,
    ]


def test_config_validation_and_the_optional_singleton():
    threshold = ScheduleConfig(
        max_val=8.0,
        points=(
            Knot(at=0.0, frac=1.0),
            Knot(at=1.0, frac=0.25),
        ),
    )
    coefficients: dict[NonlinearityUnitKind, float | None] = {"neuron": 1.0}
    nl_cfg = NonlinearityLocalityLossConfig(
        coeff=0.1, relative_threshold=threshold, unit_kind_coefficients=coefficients
    )
    surface = build_objective(
        _loss_metrics((nl_cfg,)),
        tuple(
            SiteSpec(name=name, factorization=DenseFactorization(d_in=4, d_out=4, C=4), group="g")
            for name in ("site_a",)
        ),
    )
    assert nl_cfg.normalize_at_one is False
    assert surface.nonlinearity is not None
    assert (
        float(surface.nonlinearity.coeff.at(jnp.float32(0))) == pytest.approx(0.1)
        and surface.nonlinearity.relative_threshold == threshold
    )

    assert (
        build_objective(
            _loss_metrics(),
            tuple(
                SiteSpec(
                    name=name, factorization=DenseFactorization(d_in=4, d_out=4, C=4), group="g"
                )
                for name in ("site_a",)
            ),
        ).nonlinearity
        is None
    )

    with pytest.raises(AssertionError):  # the singleton refuses a second instance
        build_objective(
            _loss_metrics((nl_cfg, nl_cfg)),
            tuple(
                SiteSpec(
                    name=name, factorization=DenseFactorization(d_in=4, d_out=4, C=4), group="g"
                )
                for name in ("site_a",)
            ),
        )

    with pytest.raises(ValidationError):
        NonlinearityLocalityLossConfig(
            coeff=-0.1, relative_threshold=threshold, unit_kind_coefficients=coefficients
        )
    with pytest.raises(ValidationError):
        NonlinearityLocalityLossConfig(
            coeff=0.1,
            relative_threshold=ScheduleConfig(
                max_val=8.0,
                points=(
                    Knot(at=0.0, frac=0.0),
                    Knot(at=1.0, frac=1.0),
                ),
            ),
            unit_kind_coefficients=coefficients,
        )
    with pytest.raises(ValidationError):
        NonlinearityLocalityLossConfig(
            coeff=0.1,
            relative_threshold=ScheduleConfig(
                max_val=8.0,
                points=(
                    Knot(at=0.0, frac=1.0),
                    Knot(at=1.0, frac=0.0),
                ),
            ),
            unit_kind_coefficients=coefficients,
        )


def test_normalize_at_one_dispatches_from_resolved_config():
    schedule = ScheduleConfig.constant(4.0)
    config = NonlinearityLocalityLossConfig(
        name="nonlinearity",
        coeff=1.0,
        relative_threshold=schedule,
        unit_kind_coefficients={"neuron": 1.0},
        normalize_at_one=True,
    )
    surface = build_objective(
        _loss_metrics((config,)),
        tuple(
            SiteSpec(name=name, factorization=DenseFactorization(d_in=4, d_out=4, C=4), group="g")
            for name in ("s",)
        ),
    )
    assert surface.nonlinearity is not None
    site = SiteSpec(
        "s",
        DenseFactorization(d_in=4, d_out=8, C=1),
        "s",
        alignment=NonlinearityAlignment("output", Neurons()),
    )
    resolved = ResolvedNonlinearity.resolve(surface.nonlinearity, (site,))
    one_hot = jnp.zeros((1, 8)).at[0, 3].set(2.5)

    result = resolved.evaluate(jnp.asarray(0.0, jnp.float32), _stacked({"s": one_hot}))

    assert float(result.weighted_loss) == pytest.approx(1.0, abs=1e-6)
    assert float(result.loss) == pytest.approx(1.0, abs=1e-6)


def test_unit_kind_coefficients_validation():
    threshold = ScheduleConfig(
        max_val=8.0,
        points=(Knot(at=0.0, frac=1.0), Knot(at=1.0, frac=0.25)),
    )

    def cfg(
        coefficients: dict[NonlinearityUnitKind, float | None],
    ) -> NonlinearityLocalityLossConfig:
        return NonlinearityLocalityLossConfig(
            coeff=0.1, relative_threshold=threshold, unit_kind_coefficients=coefficients
        )

    excluded = cfg({"neuron": 1.0, "attention_head": None})
    assert excluded.unit_kind_coefficients == {"neuron": 1.0, "attention_head": None}

    with pytest.raises(ValidationError):  # weights are strictly positive; off is None, never 0.0
        cfg({"neuron": 0.0})
    with pytest.raises(ValidationError):  # must train at least one kind
        cfg({"neuron": None, "attention_head": None})
    with pytest.raises(ValidationError):  # keys are the closed unit-kind vocabulary
        NonlinearityLocalityLossConfig.model_validate(
            {
                "coeff": 0.1,
                "relative_threshold": threshold,
                "unit_kind_coefficients": {"banana": 1.0},
            }
        )


def test_component_nonlinearity_stats_endpoints_and_long_tail():
    u_count = 8
    one_hot = component_nonlinearity_stats(jnp.zeros((1, u_count)).at[0, 5].set(3.0), Neurons())
    assert float(one_hot.effective_use_count_per_subcomponent[0]) == pytest.approx(1.0)

    uniform = component_nonlinearity_stats(jnp.ones((1, u_count)), Neurons())
    assert float(uniform.effective_use_count_per_subcomponent[0]) == pytest.approx(float(u_count))

    zero = component_nonlinearity_stats(jnp.zeros((1, u_count)), Neurons())
    assert float(zero.soft_use_count[0]) == 0.0
    assert float(zero.effective_use_count_per_subcomponent[0]) == 0.0

    long_tail = jnp.concatenate((jnp.ones(1), jnp.full(99, 0.1)))[None]
    tail_stats = component_nonlinearity_stats(long_tail, Neurons())
    expected = (1.0 + 99 * 0.1) ** 2 / (1.0 + 99 * 0.1**2)
    assert float(tail_stats.effective_use_count_per_subcomponent[0]) == pytest.approx(
        expected, rel=1e-6
    )


def test_nonlinearity_eval_compiles_replicated_reductions_for_host_consumers():
    site = SiteSpec(
        "in",
        DenseFactorization(d_in=4, d_out=8, C=2),
        "in",
        alignment=NonlinearityAlignment("output", Neurons()),
    )
    components = component_stacks_from_sites(
        {"in": (jnp.ones((4, 2)), jnp.stack((jax.nn.one_hot(0, 8), jnp.ones(8))))}
    )
    mesh = Mesh(np.asarray(jax.devices()), ("devices",), axis_types=(AxisType.Explicit,))
    replicated = NamedSharding(mesh, PartitionSpec())
    with jax.set_mesh(mesh):
        placed_components = jax.device_put(components, replicated)
        lowered = jax.jit(make_nonlinearity_eval_step((site,), replicated)).lower(placed_components)
        compiled = lowered.compile()
        output_shapes = compiled.out_info
        assert output_shapes["in"].soft_use_count.sharding.is_equivalent_to(replicated, 2)
        stack_stats = compiled(placed_components)
    assert stack_stats["in"].soft_use_count.is_fully_replicated
    stats = site_nonlinearity_stats(stack_stats, (site,))
    np.testing.assert_allclose(
        stats["in"].effective_use_count_per_subcomponent, [1.0, 8.0], rtol=1e-6
    )
    np.testing.assert_allclose(stats["in"].soft_use_count, [8 / 12, 8 / 5], rtol=1e-6)


def test_component_nonlinearity_stats_scale_by_use_multiplicity():
    n_kv_head, head_dim, g = 2, 4, 4
    kv = KVHeads(n_kv_head, g)
    stats = component_nonlinearity_stats(jnp.ones((1, n_kv_head * head_dim)), kv)
    assert float(stats.soft_use_count[0]) == pytest.approx(
        g * n_kv_head / (1 + NONLINEARITY_EVAL_RELATIVE_THRESHOLD), rel=1e-6
    )
    assert float(stats.effective_use_count_per_subcomponent[0]) == pytest.approx(
        g * n_kv_head, rel=1e-6
    )


def test_nonlinearity_eval_step_and_log_entries():
    sites = (
        SiteSpec(
            "in",
            DenseFactorization(d_in=4, d_out=8, C=3),
            "in",
            alignment=NonlinearityAlignment("output", Neurons()),
        ),
        SiteSpec("out", DenseFactorization(d_in=8, d_out=4, C=3), "out"),
    )
    partitions = nonlinearity_partitions(sites)
    assert set(partitions) == {"in"}
    vu = {
        name: (jnp.ones((spec.d_in, spec.C)), jnp.ones((spec.C, spec.d_out)))
        for name, spec in (("in", sites[0]), ("out", sites[1]))
    }
    components = component_stacks_from_sites(vu)
    stack_stats = jax.jit(make_nonlinearity_eval_step(sites, None))(components)
    assert set(stack_stats) == {"in"}
    assert stack_stats["in"].soft_use_count.shape == (1, 3)
    stats = site_nonlinearity_stats(stack_stats, sites)
    assert set(stats) == {"in"}

    prefix = "eval/nonlinearity/sites/in"
    entries = nonlinearity_log_entries(
        stats, {"in": np.array([0.5, 0.0, 0.9]), "out": np.ones(3)}, partitions
    )
    stratum = f"{prefix}/mean_ci_gt_0"
    assert entries[f"{stratum}/n_components"] == 2.0
    assert entries[f"{prefix}/all/{NONLINEARITY_EVAL_SOFT_COUNT_KEY}"] == pytest.approx(
        8 / (1 + NONLINEARITY_EVAL_RELATIVE_THRESHOLD), rel=1e-6
    )
    assert entries[f"{stratum}/{NONLINEARITY_EVAL_EFFECTIVE_COUNT_KEY}"] == pytest.approx(
        8.0, rel=1e-6
    )
    aggregate = "eval/nonlinearity/aggregates/neuron"
    assert entries[f"{aggregate}/all/{NONLINEARITY_EVAL_EFFECTIVE_COUNT_KEY}"] == pytest.approx(
        8.0, rel=1e-6
    )

    dead = nonlinearity_log_entries(stats, {"in": np.zeros(3)}, partitions)
    assert dead[f"{stratum}/n_components"] == 0.0
    assert f"{stratum}/{NONLINEARITY_EVAL_SOFT_COUNT_KEY}" not in dead


def test_nonlinearity_log_aggregates_pool_components_within_unit_kind():
    stats = {
        "n0": SiteNonlinearityStats(
            soft_use_count=np.array([1.0, 3.0]),
            effective_use_count_per_subcomponent=np.array([2.0, 4.0]),
        ),
        "n1": SiteNonlinearityStats(
            soft_use_count=np.array([5.0, 7.0]),
            effective_use_count_per_subcomponent=np.array([6.0, 8.0]),
        ),
        "h0": SiteNonlinearityStats(
            soft_use_count=np.array([9.0, 11.0]),
            effective_use_count_per_subcomponent=np.array([10.0, 12.0]),
        ),
    }
    ci_means = {
        "n0": np.array([1.0, 0.0]),
        "n1": np.array([2.0, 3.0]),
        "h0": np.array([0.0, 4.0]),
    }
    partitions = {
        "n0": Neurons(),
        "n1": Neurons(),
        "h0": KVHeads(16, 2),
    }
    entries = nonlinearity_log_entries(stats, ci_means, partitions)
    assert next(iter(entries)).startswith("eval/nonlinearity/aggregates/")

    neuron = "eval/nonlinearity/aggregates/neuron"
    assert entries[f"{neuron}/all/{NONLINEARITY_EVAL_SOFT_COUNT_KEY}"] == pytest.approx(4.0)
    assert entries[
        f"{neuron}/mean_ci_gt_0/{NONLINEARITY_EVAL_EFFECTIVE_COUNT_KEY}"
    ] == pytest.approx((2.0 + 6.0 + 8.0) / 3)
    attention = "eval/nonlinearity/aggregates/attention_head"
    assert entries[f"{attention}/mean_ci_gt_0/{NONLINEARITY_EVAL_SOFT_COUNT_KEY}"] == pytest.approx(
        11.0
    )


def test_block_factored_site_stats_land_in_flat_block_major_order():
    """An expert-blocked site's U is block-dim `[E, c, d]`; its statistics must land as
    one `[C]` vector in the flat expert-major order every per-component consumer emits
    (`selected_component_sums`, the CI means): component `(e, j)` at `e·c + j`."""
    n_experts, c_per_expert, d_out = 3, 2, 8
    site = SiteSpec(
        "moe",
        BlockedFactorization(n_blocks=n_experts, d_in=4, d_out=d_out, c_per_block=c_per_expert),
        "experts",
        alignment=NonlinearityAlignment("output", Neurons()),
    )
    # component (e, j) writes uniformly to its first `e·c + j + 1` neurons, so its
    # effective use count IS its flat expert-major index + 1
    flat_index = jnp.arange(n_experts * c_per_expert).reshape(n_experts, c_per_expert)
    u = (jnp.arange(d_out)[None, None, :] <= flat_index[..., None]).astype(jnp.float32)
    components = ComponentStacks(
        stacks={"experts": (jnp.zeros((1, n_experts, 4, c_per_expert)), u[None])},
        site_stack_indices=site_stack_indices_for((site,)),
    )
    stack_stats = jax.jit(make_nonlinearity_eval_step((site,), None))(components)
    assert stack_stats["experts"].soft_use_count.shape == (1, n_experts, c_per_expert)
    stats = site_nonlinearity_stats(stack_stats, (site,))["moe"]
    np.testing.assert_allclose(
        stats.effective_use_count_per_subcomponent,
        np.arange(1, n_experts * c_per_expert + 1),
        rtol=1e-6,
    )


def test_expert_blocked_components_match_flat_component_reduction():
    n_experts, c_per_expert, d_out = 3, 2, 8
    expert_us = jax.random.normal(jax.random.PRNGKey(7), (n_experts, c_per_expert, d_out))
    threshold = jnp.asarray(4.0)

    def expert_loss(us: jax.Array) -> jax.Array:
        stacks = ComponentStacks(
            stacks={"experts": (jnp.zeros((1, n_experts, 4, c_per_expert)), us[None])},
            site_stack_indices=(("site", "experts", 0),),
        )
        return nonlinearity_loss(
            stacks,
            {"site": NonlinearityAlignment("output", Neurons())},
            threshold,
            {"neuron": 1.0},
            normalize_at_one=False,
        )[0]

    def flat_loss(us: jax.Array) -> jax.Array:
        return nonlinearity_loss(
            _stacked({"site": us.reshape(n_experts * c_per_expert, d_out)}),
            {"site": NonlinearityAlignment("output", Neurons())},
            threshold,
            {"neuron": 1.0},
            normalize_at_one=False,
        )[0]

    actual = jax.value_and_grad(expert_loss)(expert_us)
    expected = jax.value_and_grad(flat_loss)(expert_us)
    jax.tree.map(lambda x, y: np.testing.assert_allclose(x, y, rtol=1e-6), actual, expected)


@pytest.mark.parametrize("side", ["input", "output"])
@pytest.mark.parametrize("blocked", [False, True])
@pytest.mark.parametrize("normalize_at_one", [False, True])
@pytest.mark.parametrize(
    ("partition", "unit_count", "use_multiplicity"),
    [
        (DeltaNetHeads(3, 1), 3, 1),
        (DeltaNetHeads(3, 2), 3, 2),
        (QueryHeads(3), 3, 1),
        (Neurons(), 12, 1),
    ],
)
def test_locality_uses_declared_factor_for_loss_gradients_and_eval(
    side: ComponentSide,
    blocked: bool,
    normalize_at_one: bool,
    partition: NonlinearityPartition,
    unit_count: int,
    use_multiplicity: int,
):
    from param_decomp.core.components import (
        component_stacks_from_site_arrays,
        nonlinearity_alignments,
    )

    d_in, d_out = (12, 5) if side == "input" else (5, 12)
    factorization = (
        BlockedFactorization(n_blocks=2, d_in=d_in, d_out=d_out, c_per_block=4)
        if blocked
        else DenseFactorization(d_in=d_in, d_out=d_out, C=4)
    )
    site = SiteSpec(
        "site",
        factorization,
        "group",
        alignment=NonlinearityAlignment(side, partition),
    )
    shape = (2, 4, 12) if blocked else (4, 12)
    vectors = jax.random.uniform(jax.random.key(51), shape, minval=0.1, maxval=2.0)
    opposite = jnp.zeros((*shape[:-2], 4, 5))
    factors = (
        (jnp.swapaxes(vectors, -1, -2), opposite)
        if side == "input"
        else (jnp.swapaxes(opposite, -1, -2), vectors)
    )

    def loss(factors: tuple[jax.Array, jax.Array]) -> jax.Array:
        return nonlinearity_loss(
            component_stacks_from_site_arrays((site,), {site.name: factors}),
            nonlinearity_alignments((site,)),
            jnp.asarray(4.0),
            {partition.unit_kind: 1.0},
            normalize_at_one=normalize_at_one,
        )[0]

    def expected_counts(vectors: jax.Array) -> jax.Array:
        energy = (
            jnp.square(vectors).reshape(*vectors.shape[:-1], unit_count, 12 // unit_count).sum(-1)
        )
        fractions = energy / energy.sum(-1, keepdims=True)
        return use_multiplicity * (fractions / (fractions + 4.0 / unit_count)).sum(-1)

    value, gradients = jax.value_and_grad(loss)(factors)
    normalizer = 1 + 4.0 / unit_count if normalize_at_one else 1.0
    expected_value, expected_gradient = jax.value_and_grad(
        lambda v: normalizer * expected_counts(v).mean()
    )(vectors)
    np.testing.assert_allclose(value, expected_value, rtol=1e-6)
    active_gradient = jnp.swapaxes(gradients[0], -1, -2) if side == "input" else gradients[1]
    np.testing.assert_allclose(active_gradient, expected_gradient, rtol=2e-5, atol=1e-7)
    assert float(jnp.abs(active_gradient).max()) > 0
    np.testing.assert_array_equal(gradients[1 if side == "input" else 0], 0)

    components = component_stacks_from_site_arrays((site,), {site.name: factors})
    stats = jax.jit(make_nonlinearity_eval_step((site,), None))(components)[site.group]
    np.testing.assert_allclose(stats.soft_use_count[0], expected_counts(vectors), rtol=1e-6)
    zeros = jax.tree.map(jnp.zeros_like, factors)
    zero_value, zero_grad = jax.value_and_grad(loss)(zeros)
    assert float(zero_value) == 0
    for gradient in zero_grad:
        np.testing.assert_array_equal(gradient, 0)


@pytest.mark.parametrize("blocked", [False, True])
def test_head_partition_checks_the_declared_side_within_each_block(blocked: bool):
    factorization = (
        BlockedFactorization(n_blocks=2, d_in=12, d_out=5, c_per_block=4)
        if blocked
        else DenseFactorization(d_in=12, d_out=5, C=4)
    )
    SiteSpec(
        "site", factorization, "group", alignment=NonlinearityAlignment("input", QueryHeads(3))
    )
    with pytest.raises(AssertionError):
        SiteSpec(
            "site", factorization, "group", alignment=NonlinearityAlignment("output", QueryHeads(3))
        )
