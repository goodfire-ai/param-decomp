import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from param_decomp.core.adversary import SiteSource
from param_decomp.core.ci_fn import (
    Chunk,
    ChunkwiseTransformerCIArch,
    ChunkwiseTransformerCIFn,
    MHACIAttention,
    PlacedCIFn,
    evaluate_ci,
    evaluate_compute_ci,
    materialize_ci_compute_weights,
)
from param_decomp.core.components import SiteSpec
from param_decomp.core.decomposed_linear import (
    PlannedComponentLinear,
    constrain_component_activation,
    site_forward,
)
from param_decomp.core.init_placed import init_ci_fn_placed
from param_decomp.core.linear_plan import LinearPlan, placed_linear
from param_decomp.core.masking import masks_from_sources
from param_decomp.core.placement import TargetLinearPlacement, from_config
from param_decomp.core.tools.hlo_census import collective_census

pytestmark = [
    pytest.mark.multidevice,
    pytest.mark.skipif(
        jax.default_backend() != "cpu" or jax.device_count() < 4,
        reason="requires a four-device CPU topology from make test-multidevice",
    ),
]


def test_ci_ffn_operand_gather_preserves_the_semantic_bias_basis():
    mesh = Mesh(
        np.asarray(jax.devices()[:4]).reshape(1, 2, 2),
        ("replicate", "fsdp", "tp"),
        axis_types=(AxisType.Explicit,) * 3,
    )
    rules = from_config("zero1", mesh, (SiteSpec("site", 4, 4, 2, "group"),))
    plan = rules.ci_fn.linear_plan(
        "ffn",
        ("d_model", "ffn_hidden"),
        3,
        transposed=False,
    )
    x = jnp.arange(32, dtype=jnp.float32).reshape(4, 2, 4)
    weight = jnp.arange(32, dtype=jnp.float32).reshape(4, 8)
    bias = jnp.arange(8, dtype=jnp.float32)
    sharded_x = jax.device_put(
        x,
        NamedSharding(mesh, P(("replicate", "fsdp"), None, None)),
    )
    sharded_weight = jax.device_put(weight, NamedSharding(mesh, plan.resident_weight))

    with jax.set_mesh(mesh):
        actual = jax.jit(lambda a, w: placed_linear(a, w, plan) + bias)(sharded_x, sharded_weight)
    expected = x @ weight + bias
    np.testing.assert_array_equal(actual, expected)


def test_ci_tap_rms_precedes_the_local_tp_input_slice():
    mesh = Mesh(
        np.asarray(jax.devices()[:4]).reshape(1, 2, 2),
        ("replicate", "fsdp", "tp"),
        axis_types=(AxisType.Explicit,) * 3,
    )
    site = SiteSpec("layers.0.site", 8, 8, 8, "site")
    rules = from_config("zero1", mesh, (site,))
    arch = ChunkwiseTransformerCIArch(
        chunks=(Chunk(input_taps=("tap",), output_sites=(site.name,)),),
        input_dim=8,
        d_model=8,
        n_blocks=0,
        attention=MHACIAttention(n_heads=2),
        ffn_hidden=8,
        ffn_kind="gelu",
        learned_norm_scale=False,
    )
    ci_fn = init_ci_fn_placed(arch, (site,), jax.random.PRNGKey(0), mesh, rules)
    tap = jax.device_put(
        jnp.ones((2, 4, 8), jnp.bfloat16),
        NamedSharding(mesh, P(("replicate", "fsdp"), None, None)),
    )

    @jax.jit
    def lower(cf: ChunkwiseTransformerCIFn, value: jax.Array) -> jax.Array:
        ci = evaluate_ci(PlacedCIFn(fn=cf, placement=rules.ci_fn), {"tap": value}, remat=False)
        return constrain_component_activation(ci.lower[site.name], rules)

    compiled = lower.lower(ci_fn, tap).compile()
    output = compiled(ci_fn, tap)
    assert output.sharding.spec == P(("replicate", "fsdp"), None, "tp")

    hlo = compiled.as_text()
    assert hlo is not None
    collectives = [
        line
        for line in hlo.splitlines()
        if re.search(
            r"= .*\b(all-reduce|all-gather|collective-permute|all-to-all|reduce-scatter)\(",
            line,
        )
    ]
    assert not any("collective-permute" in line or "all-to-all" in line for line in collectives)
    assert not any('op_name="jit(lower)/reduce_sum"' in line for line in collectives), "\n".join(
        collectives
    )
    tp_reductions = [
        line
        for line in collectives
        if "all-reduce" in line and "replica_groups={{0,1},{2,3}}" in line
    ]
    assert len(tp_reductions) == 1, "\n".join(collectives)


def test_ci_operand_gathers_slice_the_scanned_stack_first():
    mesh = Mesh(
        np.asarray(jax.devices()[:4]).reshape(1, 2, 2),
        ("replicate", "fsdp", "tp"),
        axis_types=(AxisType.Explicit,) * 3,
    )
    sites = tuple(SiteSpec(f"layers.{i}.site", 8, 8, 8, "site") for i in range(2))
    rules = from_config("zero1", mesh, sites)
    arch = ChunkwiseTransformerCIArch(
        chunks=tuple(Chunk(input_taps=("tap",), output_sites=(site.name,)) for site in sites),
        input_dim=8,
        d_model=8,
        n_blocks=1,
        attention=MHACIAttention(n_heads=2),
        ffn_hidden=8,
        ffn_kind="gelu",
        learned_norm_scale=False,
    )
    ci_fn = init_ci_fn_placed(arch, sites, jax.random.PRNGKey(0), mesh, rules)
    tap = jax.device_put(
        jnp.ones((2, 4, 8), jnp.bfloat16),
        NamedSharding(mesh, P(("replicate", "fsdp"), None, None)),
    )

    def loss(value: ChunkwiseTransformerCIFn) -> jax.Array:
        ci = evaluate_ci(PlacedCIFn(fn=value, placement=rules.ci_fn), {"tap": tap}, remat=False)
        return jnp.stack([jnp.sum(ci.lower[site.name]) for site in sites]).sum()

    hlo = jax.jit(jax.grad(loss)).lower(ci_fn).compile().as_text()
    assert hlo is not None
    operand_gathers = [
        line for line in hlo.splitlines() if "all-gather(" in line and "while/body" in line
    ]
    assert operand_gathers
    assert not any(re.search(r"= (f32|bf16)\[2,", line) for line in operand_gathers), "\n".join(
        operand_gathers
    )


def test_ci_replica_residency_bounds_cross_replica_collectives_outside_the_scan():
    mesh = Mesh(
        np.asarray(jax.devices()[:4]).reshape(2, 2, 1),
        ("replicate", "fsdp", "tp"),
        axis_types=(AxisType.Explicit,) * 3,
    )
    sites = tuple(SiteSpec(f"layers.{i}.site", 8, 8, 8, "site") for i in range(2))
    rules = from_config("zero1", mesh, sites)
    arch = ChunkwiseTransformerCIArch(
        chunks=tuple(Chunk(input_taps=("tap",), output_sites=(site.name,)) for site in sites),
        input_dim=8,
        d_model=8,
        n_blocks=1,
        attention=MHACIAttention(n_heads=2),
        ffn_hidden=8,
        ffn_kind="gelu",
        learned_norm_scale=False,
    )
    ci_fn = init_ci_fn_placed(arch, sites, jax.random.PRNGKey(0), mesh, rules)
    tap = jax.device_put(
        jnp.ones((4, 2, 8), jnp.bfloat16),
        NamedSharding(mesh, P(("replicate", "fsdp"), None, None)),
    )

    def loss(value: ChunkwiseTransformerCIFn) -> jax.Array:
        # The step's CI lifecycle: the masters->residents gather (cross-replica) runs
        # once in entry via materialize; only per-chunk fsdp gathers ride the scan.
        compute = materialize_ci_compute_weights(PlacedCIFn(fn=value, placement=rules.ci_fn))
        ci = evaluate_compute_ci(compute, {"tap": tap}, remat=False)
        return sum(
            (jnp.sum(ci.lower[site.name]) for site in sites), jnp.zeros((), jnp.bfloat16)
        ).astype(jnp.float32)

    with jax.set_mesh(mesh):
        compiled = jax.jit(jax.value_and_grad(loss)).lower(ci_fn).compile()
        value, _grad = compiled(ci_fn)
    assert jnp.isfinite(value)

    hlo = compiled.as_text() or ""
    census = collective_census(hlo, replica_stride=2, n_devices=4)
    # The one sanctioned in-loop cross-replica collective: the replicated-persisted
    # bias/norm-scale grads' whole-batch sum ([d]-sized; no ÷N master to defer to).
    # The size bound keeps a reintroduced V/U-shaped weight-grad reduction from
    # hiding behind the sanctioned arm; weight grads defer to the entry reductions.
    assert census.in_loop_cross_replicate <= 1, census.counts
    assert all(size <= 2**20 for size in census.in_loop_cross_replicate_bytes), (
        census.in_loop_cross_replicate_bytes
    )
    assert census.counts.get("loop:all-reduce[xrep]", 0) == census.in_loop_cross_replicate, (
        census.counts
    )
    assert census.exit_reductions > 0, census.counts
    assert not any("collective-permute" in key for key in census.counts), census.counts


def test_ci_vector_state_starts_at_its_declared_tp_layout():
    mesh = Mesh(
        np.asarray(jax.devices()[:4]).reshape(1, 2, 2),
        ("replicate", "fsdp", "tp"),
        axis_types=(AxisType.Explicit,) * 3,
    )
    site = SiteSpec("layers.0.site", 8, 8, 8, "site")
    rules = from_config("zero1", mesh, (site,))
    arch = ChunkwiseTransformerCIArch(
        chunks=(Chunk(input_taps=("tap",), output_sites=(site.name,)),),
        input_dim=8,
        d_model=8,
        n_blocks=1,
        attention=MHACIAttention(n_heads=2),
        ffn_hidden=8,
        ffn_kind="gelu",
        learned_norm_scale=False,
    )
    ci_fn = init_ci_fn_placed(arch, (site,), jax.random.PRNGKey(0), mesh, rules)
    assert isinstance(ci_fn, ChunkwiseTransformerCIFn)

    def spec(array: jax.Array) -> P:
        assert isinstance(array.sharding, NamedSharding)
        return array.sharding.spec

    assert spec(ci_fn.chunks.in_proj_b) == P(None, None)
    assert spec(ci_fn.chunks.blocks[0].b1) == P(None, "tp")
    assert spec(ci_fn.chunks.blocks[0].b2) == P(None, None)
    assert spec(ci_fn.chunks.out_bs[0]) == P(None, "tp")


def test_structured_source_materializes_masks_without_tp_redistribution():
    mesh = Mesh(
        np.asarray(jax.devices()[:4]).reshape(1, 2, 2),
        ("replicate", "fsdp", "tp"),
        axis_types=(AxisType.Explicit,) * 3,
    )
    ci = jax.device_put(
        jnp.full((2, 4, 8), 0.25, jnp.bfloat16),
        NamedSharding(mesh, P("fsdp", None, "tp")),
    )
    source = SiteSource(
        components=jax.device_put(
            jnp.full((1, 1, 8), 0.5, jnp.float32),
            NamedSharding(mesh, P(None, None, "tp")),
        ),
        delta=jax.device_put(jnp.full((1, 1), 0.75, jnp.float32), NamedSharding(mesh, P())),
    )

    @jax.jit
    def lower(ci_value: jax.Array, source_value: SiteSource[jax.Array]):
        masks, deltas = masks_from_sources({"site": ci_value}, {"site": source_value})
        return masks["site"], deltas["site"]

    compiled = lower.lower(ci, source).compile()
    mask, delta = compiled(ci, source)
    assert mask.sharding.spec == P("fsdp", None, "tp")
    assert delta.sharding.is_equivalent_to(NamedSharding(mesh, P()), delta.ndim)

    hlo = compiled.as_text()
    assert hlo is not None
    collectives = [
        line
        for line in hlo.splitlines()
        if re.search(
            r"= .*\b(all-reduce|all-gather|collective-permute|all-to-all|reduce-scatter)\(",
            line,
        )
    ]
    assert collectives == []


def test_component_delta_path_uses_one_u_and_preserves_value_and_gradients():
    mesh = Mesh(
        np.asarray(jax.devices()[:4]).reshape(1, 2, 2),
        ("replicate", "fsdp", "tp"),
        axis_types=(AxisType.Explicit,) * 3,
    )
    site = SiteSpec("layers.0.mlp.gate_proj", 8, 16, 8, "gate")
    rules = from_config("zero1", mesh, (site,))
    batch = P(("replicate", "fsdp"), None, None)
    x_host = jnp.arange(2 * 4 * 8, dtype=jnp.float32).reshape(2, 4, 8) / 64
    v_host = jnp.arange(8 * 8, dtype=jnp.float32).reshape(8, 8) / 64
    u_host = jnp.arange(8 * 16, dtype=jnp.float32).reshape(8, 16) / 128
    w_host = jnp.arange(16 * 8, dtype=jnp.float32).reshape(16, 8) / 128
    mask_host = jnp.full((2, 4, 8), 0.7, jnp.float32)
    delta_host = jnp.full((2, 4), 0.3, jnp.float32)
    args = (
        jax.device_put(x_host, NamedSharding(mesh, batch)),
        jax.device_put(v_host, NamedSharding(mesh, P("fsdp", "tp"))),
        jax.device_put(u_host, NamedSharding(mesh, P("tp", "fsdp"))),
        jax.device_put(w_host, NamedSharding(mesh, P("tp", "fsdp"))),
        jax.device_put(mask_host, NamedSharding(mesh, P(("replicate", "fsdp"), None, "tp"))),
        jax.device_put(delta_host, NamedSharding(mesh, P(("replicate", "fsdp"), None))),
    )
    frozen_plan = LinearPlan(
        mesh=mesh,
        input=batch,
        operand_input=batch,
        resident_weight=P("fsdp", "tp"),
        operand=P(None, "tp"),
        output=P(("replicate", "fsdp"), None, "tp"),
        weight_reduced=frozenset(),
    )

    def forward(
        x: jax.Array,
        v: jax.Array,
        u: jax.Array,
        w: jax.Array,
        mask: jax.Array,
        delta: jax.Array,
    ) -> jax.Array:
        return site_forward(x, v, u, w, mask, delta, None, rules, frozen_plan).output

    with jax.set_mesh(mesh):
        compiled = jax.jit(forward).lower(*args).compile()
        actual = compiled(*args)
        actual_grads = jax.jit(jax.grad(lambda *values: jnp.sum(forward(*values) ** 2), range(6)))(
            *args
        )

    assert actual.sharding.spec == batch
    expected = ((x_host @ v_host) * mask_host) @ u_host
    expected += delta_host[..., None] * (x_host @ w_host.T - (x_host @ v_host) @ u_host)
    expected_grads = jax.grad(
        lambda x, v, u, w, mask, delta: jnp.sum(
            (((x @ v) * mask) @ u + delta[..., None] * (x @ w.T - (x @ v) @ u)) ** 2
        ),
        range(6),
    )(x_host, v_host, u_host, w_host, mask_host, delta_host)
    assert jnp.allclose(actual, expected, rtol=1e-5, atol=1e-5)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        assert jnp.allclose(actual_grad, expected_grad, rtol=1e-5, atol=1e-4)

    hlo = compiled.as_text() or ""
    assert "collective-permute" not in hlo
    assert "all-to-all" not in hlo
    dots = [line for line in hlo.splitlines() if " dot(" in line]
    assert len(dots) == 3, "one x@V, one @U, and one frozen x@W.T\n" + "\n".join(dots)


def _tp_reductions(hlo: str) -> list[str]:
    """Reduction collectives over the tp axis at the (1, 2, 2) mesh, across XLA's
    replica_groups spellings ({{0,1},{2,3}}, iota, or named-mesh axis groups)."""
    tp_group_spellings = ("{{0,1},{2,3}}", "<=[4]", "'axis_2'}")
    return [
        line
        for line in hlo.splitlines()
        if ("reduce-scatter(" in line or "all-reduce(" in line)
        and any(spelling in line for spelling in tp_group_spellings)
    ]


def test_target_native_component_linears_match_megatron_column_and_row_waists():
    mesh = Mesh(
        np.asarray(jax.devices()[:4]).reshape(1, 2, 2),
        ("replicate", "fsdp", "tp"),
        axis_types=(AxisType.Explicit,) * 3,
    )
    sites = (
        SiteSpec("layers.0.mlp.gate_proj", 8, 16, 8, "gate"),
        SiteSpec("layers.0.mlp.down_proj", 16, 8, 8, "down"),
    )
    rules = from_config("zero1", mesh, sites)
    batch = P(("replicate", "fsdp"), None, None)
    intermediate = P(("replicate", "fsdp"), None, "tp")

    def plan(target: TargetLinearPlacement, v_ndim: int, u_ndim: int) -> PlannedComponentLinear:
        external_axes = ("batch", "position", "feature")
        component_axes = ("batch", "position", "C")
        del v_ndim, u_ndim
        return PlannedComponentLinear(
            v=rules.target_native_component_linear_plan(
                target,
                ("d_in", "C"),
                external_axes,
                component_axes,
            ),
            u=rules.target_native_component_linear_plan(
                target,
                ("C", "d_out"),
                component_axes,
                external_axes,
            ),
            component=rules.activations.component,
            output=target.output,
        )

    def frozen_plan(target: TargetLinearPlacement, transposed_weight: P) -> LinearPlan:
        axes = ("batch", "position", "feature")
        return LinearPlan(
            mesh=mesh,
            input=target.input.spec_for(axes),
            operand_input=target.input.spec_for(axes),
            resident_weight=transposed_weight,
            operand=P(*reversed(target.operand.spec_for(("d_out", "d_in")))),
            output=target.output.spec_for(axes),
            weight_reduced=frozenset(),
        )

    def column(
        x: jax.Array,
        v: jax.Array,
        u: jax.Array,
        w: jax.Array,
        mask: jax.Array,
        delta: jax.Array,
    ) -> jax.Array:
        target = rules.target.column
        return site_forward(
            x,
            v,
            u,
            w,
            mask,
            delta,
            None,
            plan(target, v.ndim, u.ndim),
            frozen_plan(target, P("fsdp", "tp")),
        ).output

    column_host = (
        jnp.arange(2 * 4 * 8, dtype=jnp.float32).reshape(2, 4, 8) / 64,
        jnp.arange(8 * 8, dtype=jnp.float32).reshape(8, 8) / 64,
        jnp.arange(8 * 16, dtype=jnp.float32).reshape(8, 16) / 128,
        jnp.arange(16 * 8, dtype=jnp.float32).reshape(16, 8) / 128,
        jnp.full((2, 4, 8), 0.7, jnp.float32),
        jnp.full((2, 4), 0.3, jnp.float32),
    )
    column_args = tuple(
        jax.device_put(value, NamedSharding(mesh, spec))
        for value, spec in zip(
            column_host,
            (
                batch,
                P("fsdp", "tp"),
                P("tp", "fsdp"),
                P("tp", "fsdp"),
                P(("replicate", "fsdp"), None, "tp"),
                P(("replicate", "fsdp"), None),
            ),
            strict=True,
        )
    )
    with jax.set_mesh(mesh):
        column_compiled = jax.jit(column).lower(*column_args).compile()
        column_actual = column_compiled(*column_args)
    x, v, u, w, mask, delta = column_host
    column_expected = ((x @ v) * mask) @ u + delta[..., None] * (x @ w.T - (x @ v) @ u)
    assert column_actual.sharding.is_equivalent_to(
        rules.target.column.output.sharding_for(("batch", "position", "feature")),
        ndim=3,
    )
    assert jnp.allclose(column_actual, column_expected, rtol=1e-5, atol=1e-5)
    column_hlo = column_compiled.as_text() or ""
    # The C contraction reduces over tp exactly once (reduce-scatter or all-reduce —
    # the typed-output lowering leaves the spelling to the compiler); never a
    # collective-permute (the grid-transpose failure mode).
    assert "collective-permute" not in column_hlo
    tp_reductions = _tp_reductions(column_hlo)
    assert len(tp_reductions) == 1, "\n".join(tp_reductions)

    def row(
        x: jax.Array,
        v: jax.Array,
        u: jax.Array,
        w: jax.Array,
        mask: jax.Array,
        delta: jax.Array,
    ) -> jax.Array:
        target = rules.target.row
        return site_forward(
            x,
            v,
            u,
            w,
            mask,
            delta,
            None,
            plan(target, v.ndim, u.ndim),
            frozen_plan(target, P("tp", "fsdp")),
        ).output

    row_host = (
        jnp.arange(2 * 4 * 16, dtype=jnp.float32).reshape(2, 4, 16) / 128,
        jnp.arange(16 * 8, dtype=jnp.float32).reshape(16, 8) / 128,
        jnp.arange(8 * 8, dtype=jnp.float32).reshape(8, 8) / 64,
        jnp.arange(8 * 16, dtype=jnp.float32).reshape(8, 16) / 128,
        jnp.full((2, 4, 8), 0.7, jnp.float32),
        jnp.full((2, 4), 0.3, jnp.float32),
    )
    row_args = tuple(
        jax.device_put(value, NamedSharding(mesh, spec))
        for value, spec in zip(
            row_host,
            (
                intermediate,
                P("fsdp", "tp"),
                P("tp", "fsdp"),
                P("fsdp", "tp"),
                P(("replicate", "fsdp"), None, "tp"),
                P(("replicate", "fsdp"), None),
            ),
            strict=True,
        )
    )
    with jax.set_mesh(mesh):
        row_compiled = jax.jit(row).lower(*row_args).compile()
        row_actual = row_compiled(*row_args)
    x, v, u, w, mask, delta = row_host
    row_expected = ((x @ v) * mask) @ u + delta[..., None] * (x @ w.T - (x @ v) @ u)
    assert row_actual.sharding.is_equivalent_to(
        rules.target.row.output.sharding_for(("batch", "position", "feature")),
        ndim=3,
    )
    assert jnp.allclose(row_actual, row_expected, rtol=1e-5, atol=1e-5)
    row_hlo = row_compiled.as_text() or ""
    assert "collective-permute" not in row_hlo
    row_reductions = _tp_reductions(row_hlo)
    assert len(row_reductions) == 1, "\n".join(row_reductions)
