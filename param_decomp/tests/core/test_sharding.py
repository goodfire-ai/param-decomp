"""Sharding tests. Run under simulated multi-device CPU via the env in conftest.

These guard the harness pitfall (NOTES): `shard_batch` must reconstruct the FULL
global array across the mesh, not replicate a per-process slice. The GPU-count
invariance of the whole step is validated end-to-end by
`experiments/distributed_stacked_sites.py` at 1 vs N devices (bit-identical);
that needs distinct process-level device counts so it lives in the runnable
experiment, not here.
"""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType
from jaxtyping import Array, TypeCheckError

from param_decomp.core.components import (
    DENSE_U_AXES,
    DENSE_V_AXES,
    ComponentStacks,
    DenseFactorization,
    SiteSpec,
    site_stack_indices_for,
)
from param_decomp.core.configs import PlacementPresetName
from param_decomp.core.decomposed_linear import SiteWeights, site_out
from param_decomp.core.dict_utils import FrozenMapping
from param_decomp.core.linear_plan import placed_linear
from param_decomp.core.model import Positioned, Positionless
from param_decomp.core.placement import (
    component_stacks_shardings,
    component_stacks_to_compute_weights,
    from_config,
)
from param_decomp.core.sharding import (
    data_parallel_size,
    hsdp_mesh,
    place_cpu_arrays,
    place_target,
    shard_batch,
)
from param_decomp.lm.batch import LMBatchWithDocuments
from param_decomp.targets.transformer import TransformerPreparedMasking, TransformerPreparedWeights

# Needs >1 jax device; hangs at the default 1 device, so gated behind --runmultidevice.
# Run via `make test-multidevice` (sets XLA_FLAGS for simulated CPU devices). See conftest.
pytestmark = pytest.mark.multidevice


def test_mesh_axes_are_authored_not_derived_from_device_locality():
    n = jax.device_count()
    assert n % 2 == 0
    assert hsdp_mesh(1, n, 1).shape == {"replicate": 1, "fsdp": n, "tp": 1}
    assert hsdp_mesh(2, n // 2, 1).shape == {"replicate": 2, "fsdp": n // 2, "tp": 1}


def test_frozen_target_fsdp_placement_preserves_the_clean_forward():
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from param_decomp.core.components import SiteC
    from param_decomp.targets.testing import tiny_glu_cfg, tiny_glu_decomposed_lm
    from param_decomp.targets.transformer import (
        GatedMLP,
        TransformerDecomposedModel,
        glu_site_specs,
    )

    mesh = hsdp_mesh(1, jax.device_count(), 1)
    cfg = tiny_glu_cfg()
    sites = glu_site_specs(cfg, (SiteC("layers.0.mlp.gate_proj", 8),))
    model = tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(91))
    rules = from_config("owner", mesh, sites)
    placed = place_target(model, rules)

    target = placed.model
    assert isinstance(target, TransformerDecomposedModel)
    placed_mlp = target.stacked.mlp
    assert isinstance(placed_mlp, GatedMLP)
    assert isinstance(target.embed.sharding, NamedSharding)
    assert isinstance(placed_mlp.Wg.sharding, NamedSharding)
    assert isinstance(placed_mlp.Wd.sharding, NamedSharding)
    assert isinstance(target.norm.sharding, NamedSharding)
    assert target.embed.sharding.spec == P(None, "fsdp")
    assert placed_mlp.Wg.sharding.spec == P(None, "tp", "fsdp")
    assert placed_mlp.Wd.sharding.spec == P(None, "fsdp", "tp")
    assert target.norm.sharding.spec == P(None)

    batch = jax.device_count()  # the (replicate, fsdp) data plane
    tokens = jnp.arange(batch * 8, dtype=jnp.int32).reshape(batch, 8) % cfg.vocab_size
    expected = jax.jit(
        lambda target_, x: target_.clean_forward(
            LMBatchWithDocuments.from_unsegmented_sequences(x), placement=None
        ).output
    )(model, tokens)
    with jax.set_mesh(mesh):
        actual = jax.jit(
            lambda placed_, x: placed_.clean_forward(
                LMBatchWithDocuments.from_unsegmented_sequences(x)
            ).output
        )(placed, tokens)
    assert jnp.allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_owner_fsdp_tp_masked_forward_keeps_frozen_megatron_plan():
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from param_decomp.core.components import SiteC, init_component_stacks
    from param_decomp.core.init_placed import init_component_stacks_placed
    from param_decomp.core.model import MaterializedMasking, PlacedModel
    from param_decomp.targets.lm_output import LMOutput
    from param_decomp.targets.testing import tiny_glu_cfg, tiny_glu_decomposed_lm
    from param_decomp.targets.transformer import (
        canonical_site_cs,
        glu_site_specs,
        site_name,
    )

    replicate = 2 if jax.device_count() >= 8 else 1
    mesh = Mesh(
        np.asarray(jax.devices()[: 4 * replicate]).reshape(replicate, 2, 2),
        ("replicate", "fsdp", "tp"),
        axis_types=(AxisType.Explicit,) * 3,
    )
    cfg = tiny_glu_cfg()
    sites = glu_site_specs(
        cfg,
        canonical_site_cs(
            tuple(
                SiteC(site_name(layer, kind), 8)
                for layer in range(4)
                for kind in ("q", "k", "v", "o", "gate", "up", "down")
            )
        ),
    )
    model = tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(20))
    components = init_component_stacks(sites, jax.random.PRNGKey(21))
    tokens = jnp.arange(32, dtype=jnp.int32).reshape(4, 8) % cfg.vocab_size
    component_masks = {site.name: jnp.full((4, 8, site.C), 0.7) for site in sites}
    delta_masks = {site.name: jnp.full((4, 8), 0.3) for site in sites}
    routes = {site.name: jnp.arange(32).reshape(4, 8) % 2 == 0 for site in sites}
    capture_keys = frozenset(f"resid.{layer + 1}" for layer in range(4))
    unplaced = PlacedModel(model=model, placement=None)
    expected = jax.jit(
        lambda target, prepared, batch, component, delta, route: target.masked_forward(
            prepared,
            LMBatchWithDocuments.from_unsegmented_sequences(batch),
            masking=target.model.prepare_masking(
                MaterializedMasking(component_masks=component, weight_delta_masks=delta)
            ),
            routes=route,
            capture_keys=capture_keys,
            remat=True,
        ).output
    )(
        unplaced,
        unplaced.prepare_compute_weights(components),
        tokens,
        component_masks,
        delta_masks,
        routes,
    )

    rules = from_config("owner", mesh, sites)
    placed_model = place_target(model, rules)
    placed_components = init_component_stacks_placed(sites, jax.random.PRNGKey(21), rules)
    placed_tokens = jax.device_put(tokens, NamedSharding(mesh, P(("replicate", "fsdp"), None)))
    placed_component, placed_delta, placed_routes = jax.tree.map(
        lambda value: jax.device_put(
            value,
            NamedSharding(mesh, P(("replicate", "fsdp"), *(None for _ in value.shape[1:]))),
        ),
        (component_masks, delta_masks, routes),
    )

    def masked(
        target: PlacedModel[
            LMBatchWithDocuments,
            LMOutput,
            TransformerPreparedWeights,
            LMBatchWithDocuments,
            TransformerPreparedMasking,
        ],
        prepared: TransformerPreparedWeights,
        batch: Array,
        component: dict[str, Array],
        delta: dict[str, Array],
        route: dict[str, Array],
    ) -> Array:
        output = target.masked_forward(
            prepared,
            LMBatchWithDocuments.from_unsegmented_sequences(batch),
            masking=target.model.prepare_masking(
                MaterializedMasking(component_masks=component, weight_delta_masks=delta)
            ),
            routes=route,
            capture_keys=capture_keys,
            remat=True,
        ).output
        assert isinstance(output, jax.Array)
        return output

    with jax.set_mesh(mesh):
        prepared = placed_model.prepare_compute_weights(placed_components)
        args = (
            placed_model,
            prepared,
            placed_tokens,
            placed_component,
            placed_delta,
            placed_routes,
        )
        lowered = jax.jit(masked).lower(*args)
        actual = jax.jit(masked)(*args)

        def component_loss(value: ComponentStacks) -> Array:
            prepared_value = placed_model.prepare_compute_weights(value)
            return jnp.sum(
                masked(
                    placed_model,
                    prepared_value,
                    placed_tokens,
                    placed_component,
                    placed_delta,
                    placed_routes,
                )
            )

        gradient_hlo = (
            jax.jit(jax.grad(component_loss)).lower(placed_components).compile().as_text()
        )

    assert jnp.allclose(actual, expected, rtol=3e-2, atol=3e-2)
    hlo = lowered.compile().as_text() or ""
    assert "all-gather" in hlo
    assert not any("all-gather" in line and "'tp'" in line for line in hlo.splitlines()), (
        "frozen target operands must retain their Megatron TP shard"
    )
    assert gradient_hlo is not None
    assert "collective-permute" not in gradient_hlo
    # The chained-reduced structural invariants: every cross-replicate collective stays
    # OUT of the scan loop, and the components' deferred backward reduction lands as
    # entry-region cross-replicate tensor reductions (the masters' exit reduce-scatter).
    from param_decomp.core.tools.hlo_census import collective_census

    census = collective_census(gradient_hlo, replica_stride=4, n_devices=mesh.devices.size)
    if replicate > 1:
        assert census.in_loop_cross_replicate == 0, census.counts
        assert census.exit_reductions > 0, census.counts
    forbidden_operand_stacks = (
        "f32[3,32,32]",
        "f32[3,16,32]",
        "f32[3,8,32]",
        "f32[3,32,16]",
        "bf16[3,1,64,4]",
        "bf16[3,1,4,64]",
    )
    scan_carries = "\n".join(line for line in gradient_hlo.splitlines() if " while(" in line)
    assert not any(shape in scan_carries for shape in forbidden_operand_stacks), scan_carries


def test_placed_masked_forward_decomposes_only_the_mlp_kinds():
    """Placement must not widen the decomposed kind set: a kind with no sites runs
    frozen on the placed linear plan, and only decomposed kinds carry per-kind inputs."""
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from param_decomp.core.components import init_component_stacks
    from param_decomp.core.init_placed import init_component_stacks_placed
    from param_decomp.core.model import MaterializedMasking, PlacedModel
    from param_decomp.targets.lm_output import LMOutput
    from param_decomp.targets.testing import tiny_glu_cfg, tiny_glu_decomposed_lm
    from param_decomp.targets.transformer import (
        glu_site_specs,
        mlp_family_site_cs,
    )

    cfg = tiny_glu_cfg()
    sites = glu_site_specs(cfg, mlp_family_site_cs(2, 3, 8))
    model = tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(30))
    components = init_component_stacks(sites, jax.random.PRNGKey(31))
    batch, seq = jax.device_count(), 8
    tokens = jnp.arange(batch * seq, dtype=jnp.int32).reshape(batch, seq) % cfg.vocab_size
    component_masks = {site.name: jnp.full((batch, seq, site.C), 0.7) for site in sites}
    delta_masks = {site.name: jnp.full((batch, seq), 0.3) for site in sites}
    routes = {site.name: tokens % 2 == 0 for site in sites}
    mesh = hsdp_mesh(1, jax.device_count(), 1)
    rules = from_config("ddp", mesh, sites)

    def masked(
        target: PlacedModel[
            LMBatchWithDocuments,
            LMOutput,
            TransformerPreparedWeights,
            LMBatchWithDocuments,
            TransformerPreparedMasking,
        ],
        prepared: TransformerPreparedWeights,
        batch_: Array,
        component: dict[str, Array],
        delta: dict[str, Array],
        route: dict[str, Array],
    ) -> Array:
        output = target.masked_forward(
            prepared,
            LMBatchWithDocuments.from_unsegmented_sequences(batch_),
            masking=target.model.prepare_masking(
                MaterializedMasking(component_masks=component, weight_delta_masks=delta)
            ),
            routes=route,
            capture_keys=frozenset(),
            remat=True,
        ).output
        assert isinstance(output, jax.Array)
        return output

    unplaced = PlacedModel(model=model, placement=None)
    expected = jax.jit(masked)(
        unplaced,
        unplaced.prepare_compute_weights(components),
        tokens,
        component_masks,
        delta_masks,
        routes,
    )

    placed_model = place_target(model, rules)
    placed_components = init_component_stacks_placed(sites, jax.random.PRNGKey(31), rules)
    placed_tokens = jax.device_put(tokens, NamedSharding(mesh, P(("replicate", "fsdp"), None)))
    placed_component, placed_delta, placed_routes = jax.tree.map(
        lambda value: jax.device_put(
            value, NamedSharding(mesh, P(("replicate", "fsdp"), *(None,) * (value.ndim - 1)))
        ),
        (component_masks, delta_masks, routes),
    )
    with jax.set_mesh(mesh):
        prepared = placed_model.prepare_compute_weights(placed_components)
        actual = jax.jit(masked)(
            placed_model,
            prepared,
            placed_tokens,
            placed_component,
            placed_delta,
            placed_routes,
        )
    assert jnp.allclose(actual, expected, rtol=3e-2, atol=3e-2)


def test_glu_prepared_weights_follow_the_declared_compute_row():
    from dataclasses import replace

    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    from param_decomp.core.components import SiteC, init_component_stacks
    from param_decomp.core.placement import PlacedRule
    from param_decomp.targets.testing import tiny_glu_cfg, tiny_glu_decomposed_lm
    from param_decomp.targets.transformer import canonical_site_cs, glu_site_specs, site_name

    mesh = Mesh(
        np.asarray(jax.devices()[:4]).reshape(1, 2, 2),
        ("replicate", "fsdp", "tp"),
        axis_types=(AxisType.Explicit,) * 3,
    )
    cfg = tiny_glu_cfg()
    sites = glu_site_specs(cfg, canonical_site_cs((SiteC(site_name(0, "gate"), 8),)))
    model = tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(30))
    components = init_component_stacks(sites, jax.random.PRNGKey(31))
    rules = from_config("zero1", mesh, sites)
    compute = PlacedRule(
        mesh=mesh,
        label_for_log="test/swapped_compute",
        rule=FrozenMapping({"d_in": ("tp",), "d_out": ("tp",), "C": ("fsdp",)}),
    )
    rules = replace(rules, components=replace(rules.components, compute_weights=compute))

    with jax.set_mesh(mesh):
        prepared = jax.jit(lambda value: model.prepare_compute_weights(value, rules))(components)

    # The swapped compute row still gathers the zero1 masters over `replicate`, so the
    # residents carry the chained-reduced typing (size-1 replicate included — jax's dot
    # transpose matches reduced sets against contraction specs as written).
    assert prepared.per_kind["gate"]["V"].sharding.spec == P(
        None, "tp", "fsdp", reduced=frozenset({"replicate"})
    )
    assert prepared.per_kind["gate"]["U"].sharding.spec == P(
        None, "fsdp", "tp", reduced=frozenset({"replicate"})
    )


def test_chained_reduced_transpose_defers_master_reduction_past_the_scan():
    """The chained-reduced spelling: the compute residents are typed `reduced` over the
    gathered axes, so the backward's cross-replicate reduction is ONE entry-region
    collective at the master boundary — never inside the layer scan — and the gradient
    is exact."""
    from jax.sharding import Mesh, NamedSharding

    from param_decomp.core.tools.hlo_census import collective_census

    mesh = Mesh(
        np.asarray(jax.devices()[:4]).reshape(2, 2, 1),
        ("replicate", "fsdp", "tp"),
        axis_types=(AxisType.Explicit,) * 3,
    )
    sites = tuple(
        SiteSpec(
            name=f"linear.{i}",
            factorization=DenseFactorization(d_in=8, d_out=8, C=4),
            group="linear",
        )
        for i in range(2)
    )
    rules = from_config("owner", mesh, sites)
    host = jnp.arange(2 * 8 * 4, dtype=jnp.float32).reshape(2, 8, 4) / 64
    owner = rules.components.optimizer_state
    master = jax.device_put(host, owner.sharding_for(DENSE_V_AXES))
    x_host = jnp.arange(8 * 8, dtype=jnp.float32).reshape(8, 8) / 32
    x = jax.device_put(
        x_host, NamedSharding(mesh, rules.target.component.input.spec_for(("batch", "feature")))
    )

    linear_plan = rules.component_linear_plan(
        ("d_in", "C"),
        ("batch", "feature"),
        ("batch", "C"),
    )

    def loss(value: Array) -> Array:
        components = ComponentStacks(
            stacks={"linear": (value, jnp.zeros((2, 4, 8), value.dtype))},
            site_stack_indices=site_stack_indices_for(sites),
        )
        compute = component_stacks_to_compute_weights(components, rules.components).stacks[
            "linear"
        ][0]

        def layer(carry: Array, weight: Array) -> tuple[Array, None]:
            return carry + jnp.sum(placed_linear(x, weight, linear_plan)), None

        total, _ = jax.lax.scan(layer, jnp.zeros((), value.dtype), compute)
        return total

    with jax.set_mesh(mesh):
        gradient = jax.jit(
            jax.grad(loss),
            in_shardings=master.sharding,
            out_shardings=master.sharding,
        )
        actual = gradient(master)
        hlo = gradient.lower(master).compile().as_text()
    expected = jax.grad(lambda value: sum(jnp.sum(x_host @ weight) for weight in value))(host)
    assert jnp.array_equal(actual, expected)

    assert hlo is not None
    census = collective_census(hlo, replica_stride=2, n_devices=4)
    assert census.in_loop_cross_replicate == 0, census.counts
    assert census.exit_reductions == 1, census.counts
    assert not re.search(r"f32\[2,8,8\].*all-gather", hlo)


@pytest.mark.parametrize("preset", ["zero1-replicated-resident", "owner-replicated-resident"])
def test_replicated_resident_masked_forward_gathers_no_weights_in_loop(
    preset: PlacementPresetName,
):
    """The residency census (T20, explicit-mesh spelling): with the resident rows equal
    to the operand rows, the once-per-step entry gather is the ONLY weight collective —
    every in-loop all-gather is activation-shaped (batch-leading), zero cross-replicate
    collectives inside any while body, and the deferred master reduction still lands as
    entry reduce-scatters. Values and master gradients match the hsdp-resident `zero1`
    lowering of the equivalent three-axis (replicate, 1, tp) mesh — for BOTH persistence
    schemes of the 2x2 (zero1 masters and owner stack-cut masters). The resident arm
    runs on the genuine two-axis (data, tp) mesh."""
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from param_decomp.core.components import SiteC, init_component_stacks
    from param_decomp.core.init_placed import init_component_stacks_placed
    from param_decomp.core.model import MaterializedMasking, PlacedModel
    from param_decomp.core.placement import batch_axes
    from param_decomp.core.tools.hlo_census import (
        _computations,
        _loop_computations,
        collective_census,
    )
    from param_decomp.targets.lm_output import LMOutput
    from param_decomp.targets.testing import tiny_glu_cfg, tiny_glu_decomposed_lm
    from param_decomp.targets.transformer import (
        canonical_site_cs,
        glu_site_specs,
        site_name,
    )

    replicate = 4 if jax.device_count() >= 8 else 2
    devices = np.asarray(jax.devices()[: 2 * replicate])
    meshes = {
        "resident": Mesh(
            devices.reshape(replicate, 2),
            ("data", "tp"),
            axis_types=(AxisType.Explicit,) * 2,
        ),
        "zero1": Mesh(
            devices.reshape(replicate, 1, 2),
            ("replicate", "fsdp", "tp"),
            axis_types=(AxisType.Explicit,) * 3,
        ),
    }
    cfg = tiny_glu_cfg()
    sites = glu_site_specs(
        cfg,
        canonical_site_cs(
            tuple(
                SiteC(site_name(layer, kind), 8)
                for layer in range(4)
                for kind in ("q", "k", "v", "o", "gate", "up", "down")
            )
        ),
    )
    model = tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(20))
    components = init_component_stacks(sites, jax.random.PRNGKey(21))
    batch = 4
    tokens = jnp.arange(batch * 8, dtype=jnp.int32).reshape(batch, 8) % cfg.vocab_size
    component_masks = {site.name: jnp.full((batch, 8, site.C), 0.7) for site in sites}
    delta_masks = {site.name: jnp.full((batch, 8), 0.3) for site in sites}
    routes = {site.name: (jnp.arange(batch * 8).reshape(batch, 8) % 2 == 0) for site in sites}
    capture_keys = frozenset(f"resid.{layer + 1}" for layer in range(4))

    def masked(
        target: PlacedModel[
            LMBatchWithDocuments,
            LMOutput,
            TransformerPreparedWeights,
            LMBatchWithDocuments,
            TransformerPreparedMasking,
        ],
        prepared: TransformerPreparedWeights,
        batch_: Array,
        component: dict[str, Array],
        delta: dict[str, Array],
        route: dict[str, Array],
    ) -> Array:
        output = target.masked_forward(
            prepared,
            LMBatchWithDocuments.from_unsegmented_sequences(batch_),
            masking=target.model.prepare_masking(
                MaterializedMasking(component_masks=component, weight_delta_masks=delta)
            ),
            routes=route,
            capture_keys=capture_keys,
            remat=True,
        ).output
        assert isinstance(output, jax.Array)
        return output

    unplaced = PlacedModel(model=model, placement=None)
    expected = jax.jit(masked)(
        unplaced,
        unplaced.prepare_compute_weights(components),
        tokens,
        component_masks,
        delta_masks,
        routes,
    )

    outputs: dict[str, Array] = {}
    gradients: dict[str, ComponentStacks] = {}
    hlos: dict[str, str] = {}
    for arm, mesh in meshes.items():
        rules = from_config(preset if arm == "resident" else "zero1", mesh, sites)
        data_axes = batch_axes(mesh)
        placed_tokens = jax.device_put(tokens, NamedSharding(mesh, P(data_axes, None)))
        placed_component, placed_delta, placed_routes = jax.tree.map(
            lambda value, mesh=mesh, data_axes=data_axes: jax.device_put(
                value,
                NamedSharding(mesh, P(data_axes, *(None for _ in value.shape[1:]))),
            ),
            (component_masks, delta_masks, routes),
        )
        placed_model = place_target(model, rules)
        placed_components = init_component_stacks_placed(sites, jax.random.PRNGKey(21), rules)

        def component_loss(
            value: ComponentStacks,
            *,
            placed_model: PlacedModel[
                LMBatchWithDocuments,
                LMOutput,
                TransformerPreparedWeights,
                LMBatchWithDocuments,
                TransformerPreparedMasking,
            ] = placed_model,
            placed_tokens: Array = placed_tokens,
            placed_component: dict[str, Array] = placed_component,
            placed_delta: dict[str, Array] = placed_delta,
            placed_routes: dict[str, Array] = placed_routes,
        ) -> Array:
            prepared_value = placed_model.prepare_compute_weights(value)
            return jnp.sum(
                masked(
                    placed_model,
                    prepared_value,
                    placed_tokens,
                    placed_component,
                    placed_delta,
                    placed_routes,
                )
            )

        with jax.set_mesh(mesh):
            prepared = placed_model.prepare_compute_weights(placed_components)
            outputs[arm] = jax.jit(masked)(
                placed_model,
                prepared,
                placed_tokens,
                placed_component,
                placed_delta,
                placed_routes,
            )
            grad_fn = jax.jit(jax.grad(component_loss))
            gradients[arm] = grad_fn(placed_components)
            hlo = grad_fn.lower(placed_components).compile().as_text()
            assert hlo is not None
            hlos[arm] = hlo

    # host-side compares: the two arms live on DIFFERENT meshes (two-axis resident vs
    # three-axis zero1), and Explicit-mesh broadcasting refuses mixed shardings by type
    assert np.allclose(np.asarray(outputs["resident"]), np.asarray(expected), rtol=3e-2, atol=3e-2)
    assert np.allclose(
        np.asarray(outputs["resident"]), np.asarray(outputs["zero1"]), rtol=1e-6, atol=1e-6
    )
    for resident_leaf, zero1_leaf in zip(
        jax.tree.leaves(gradients["resident"]), jax.tree.leaves(gradients["zero1"]), strict=True
    ):
        assert np.allclose(np.asarray(resident_leaf), np.asarray(zero1_leaf), rtol=1e-5, atol=1e-7)

    census = collective_census(hlos["resident"], replica_stride=2, n_devices=2 * replicate)
    assert census.in_loop_cross_replicate == 0, census.counts
    assert census.exit_reductions > 0, census.counts
    # the once-per-step masters->resident gather: cross-replicate, in entry
    assert census.counts.get("entry:all-gather[xrep]", 0) > 0, census.counts

    # zero WEIGHT gathers in any while body: every surviving in-loop all-gather result
    # is activation-shaped (leading dim = the per-shard batch); weight shapes lead with
    # the stack (4) or a d dim (16/32/64), never the batch shard.
    comps = _computations(hlos["resident"])
    batch_shard = batch // replicate
    for name in _loop_computations(comps):
        for line in comps[name]:
            m = re.search(r"all-gather(?:-start)?\(", line)
            if m is None:
                continue
            for dims_text in re.findall(
                r"\b(?:pred|bf16|f16|f32|s32|u32)\[([\d,]+)\]", line[: m.start()]
            ):
                dims = tuple(int(d) for d in dims_text.split(","))
                assert len(dims) >= 3 and dims[0] == batch_shard, (
                    f"in-loop all-gather is not activation-shaped — a weight gather "
                    f"survived residency: {dims} in {line.strip()}"
                )


def test_owner_replicated_resident_faithfulness_transition_is_identity():
    """Owner persistence's measured win, pinned in the fourth corner: the faithfulness
    rows ARE the optimizer rows (stack-cut), so materializing the faithfulness weights
    from the masters lowers with ZERO collectives — an identity read, not a transition."""
    from jax.sharding import Mesh

    from param_decomp.core.placement import component_stacks_to_faithfulness_weights
    from param_decomp.core.tools.hlo_census import _COLLECTIVE_OP

    mesh = Mesh(
        np.asarray(jax.devices()[:8]).reshape(4, 2),
        ("data", "tp"),
        axis_types=(AxisType.Explicit,) * 2,
    )
    sites = tuple(
        SiteSpec(
            name=f"linear.{i}",
            factorization=DenseFactorization(d_in=8, d_out=8, C=8),
            group="linear",
        )
        for i in range(4)
    )
    rules = from_config("owner-replicated-resident", mesh, sites)
    components_rows = rules.components
    assert components_rows.faithfulness_weights.rule == components_rows.optimizer_state.rule

    host_v = jnp.arange(4 * 8 * 8, dtype=jnp.float32).reshape(4, 8, 8) / 256
    host_u = jnp.arange(4 * 8 * 8, dtype=jnp.float32).reshape(4, 8, 8) / 512
    components = ComponentStacks(
        stacks={
            "linear": (
                jax.device_put(host_v, components_rows.optimizer_state.sharding_for(DENSE_V_AXES)),
                jax.device_put(host_u, components_rows.optimizer_state.sharding_for(DENSE_U_AXES)),
            )
        },
        site_stack_indices=site_stack_indices_for(sites),
    )
    with jax.set_mesh(mesh):
        materialize = jax.jit(
            lambda value: component_stacks_to_faithfulness_weights(value, components_rows)
        )
        result = materialize(components)
        hlo = materialize.lower(components).compile().as_text()
    assert hlo is not None
    assert _COLLECTIVE_OP.search(hlo) is None, "faithfulness transition emitted a collective"
    faith_v, faith_u = result.stacks["linear"]
    assert jnp.array_equal(faith_v, host_v) and jnp.array_equal(faith_u, host_u)
    assert faith_v.sharding == components.stacks["linear"][0].sharding


def test_place_cpu_arrays_preserves_values_and_assigns_device_slices() -> None:
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    mesh = Mesh(
        np.asarray(jax.devices()[:4]).reshape(2, 2),
        ("replicate", "fsdp"),
        axis_types=(AxisType.Explicit,) * 2,
    )
    full = jnp.arange(8 * 12, dtype=jnp.bfloat16).reshape(8, 12)
    tree = {"weights": (full, full), "step": jnp.array(3, dtype=jnp.int32), "absent": None}
    shardings = {
        "weights": (NamedSharding(mesh, P()), NamedSharding(mesh, P(None, "fsdp"))),
        "step": NamedSharding(mesh, P()),
        "absent": None,
    }
    placed = place_cpu_arrays(tree, shardings)

    assert jax.tree.structure(placed) == jax.tree.structure(tree)
    for original, actual, expected_sharding in zip(
        jax.tree.leaves(tree), jax.tree.leaves(placed), jax.tree.leaves(shardings), strict=True
    ):
        assert actual.sharding == expected_sharding
        assert actual.shape == original.shape
        assert actual.dtype == original.dtype
        np.testing.assert_array_equal(actual, original)
        for shard in actual.addressable_shards:
            np.testing.assert_array_equal(shard.data, np.asarray(original)[shard.index])


def test_place_cpu_arrays_rejects_mismatched_structure_and_nonarray_leaves() -> None:
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    sharding = NamedSharding(hsdp_mesh(1, 1, jax.device_count()), P())
    array = jnp.ones(4)
    with pytest.raises(TypeCheckError):
        place_cpu_arrays({"weight": array}, {"different": sharding})
    with pytest.raises(TypeCheckError):
        place_cpu_arrays({"weight": array}, sharding)
    with pytest.raises(TypeCheckError):
        place_cpu_arrays({"weight": array, "count": 3}, {"weight": sharding, "count": sharding})
    with pytest.raises(TypeCheckError):
        place_cpu_arrays({"weight": np.ones(4)}, {"weight": sharding})
    with pytest.raises(TypeCheckError):
        place_cpu_arrays({"weight": array}, {"weight": array})


def test_shard_batch_preserves_global_data():
    mesh = hsdp_mesh(1, jax.device_count(), 1)
    n = mesh.devices.size
    B = 8 * n
    full = jax.random.normal(jax.random.PRNGKey(0), (3, B, 5))
    sharded = shard_batch(full, mesh, batch_axis=1)
    assert sharded.shape == full.shape
    # the sharded array must equal the original global array (the harness pitfall
    # replicated a single slice instead, which this catches when n > 1).
    assert jnp.allclose(jnp.asarray(sharded), full)


def test_shard_batch_requires_divisible_batch():
    mesh = hsdp_mesh(1, jax.device_count(), 1)
    n = mesh.devices.size
    if n == 1:
        return  # any batch divides 1
    full = jax.random.normal(jax.random.PRNGKey(1), (2, n + 1, 4))
    try:
        shard_batch(full, mesh, batch_axis=1)
    except AssertionError:
        return
    raise AssertionError("expected non-divisible batch to fail")


def test_tp_replicas_do_not_count_as_batch_shards():
    mesh = hsdp_mesh(1, jax.device_count() // 2, 2)
    n_data = data_parallel_size(mesh)
    assert n_data * 2 == mesh.devices.size
    full = jax.random.normal(jax.random.PRNGKey(2), (n_data, 5))
    sharded = shard_batch(full, mesh, batch_axis=0)
    assert jnp.array_equal(sharded, full)
    assert isinstance(sharded.sharding, jax.sharding.NamedSharding)
    assert sharded.sharding.spec == jax.sharding.PartitionSpec(("replicate", "fsdp"), None)


def test_decomposed_linear_forward_and_backward_are_tp_invariant():
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    x = jnp.arange(4 * 3 * 8, dtype=jnp.float32).reshape(4, 3, 8) / 100
    v = jnp.arange(8 * 6, dtype=jnp.float32).reshape(8, 6) / 50
    u = jnp.arange(6 * 12, dtype=jnp.float32).reshape(6, 12) / 70
    mask = jax.nn.sigmoid(jnp.arange(4 * 3 * 6).reshape(4, 3, 6) / 20)

    def run(tp: int) -> tuple[Array, tuple[Array, Array, Array, Array]]:
        # a fixed four-device mesh: the varied axis is tp, and the toy dims tile ÷4
        mesh = Mesh(
            np.asarray(jax.devices()[:4]).reshape(1, 4 // tp, tp),
            ("replicate", "fsdp", "tp"),
            axis_types=(AxisType.Explicit,) * 3,
        )
        batch = NamedSharding(mesh, P(("replicate", "fsdp"), None, None))
        args = (
            jax.device_put(x, batch),
            jax.device_put(v, NamedSharding(mesh, P(None, "tp"))),
            jax.device_put(u, NamedSharding(mesh, P("tp", None))),
            jax.device_put(mask, batch),
        )
        rules = from_config(
            "owner",
            mesh,
            (
                SiteSpec(
                    name="linear",
                    factorization=DenseFactorization(d_in=8, d_out=12, C=6),
                    group="linear",
                ),
            ),
        )

        unread_target = jnp.zeros((12, 8))

        def forward(x: Array, v: Array, u: Array, mask: Array) -> Array:
            return site_out(x, SiteWeights(unread_target, v, u, None, rules), mask, None, None)

        def loss(x: Array, v: Array, u: Array, mask: Array) -> Array:
            return jnp.sum(
                site_out(x, SiteWeights(unread_target, v, u, None, rules), mask, None, None) ** 2
            )

        with jax.set_mesh(mesh):
            output = jax.jit(forward)(*args)
            gradient = jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3)))(*args)
        return output, gradient

    # Host-side comparison: the two runs' outputs are committed to different meshes,
    # and explicit-mode ops refuse cross-mesh operands.
    tp1, tp2 = jax.tree.map(np.asarray, (run(1), run(2)))
    assert np.allclose(tp1[0], tp2[0], rtol=2e-6, atol=2e-6)
    assert jax.tree.all(
        jax.tree.map(lambda a, b: np.allclose(a, b, rtol=2e-6, atol=1e-4), tp1[1], tp2[1])
    )


def test_jitted_sharded_inits_match_eager_values():
    """`init_*_placed` (model-owned `.shardings`) must be a placement-only change: same
    values as the host (unsharded) init fns (threefry is partitionable, so generating under
    jit with `out_shardings` cannot perturb the stream — only op fusion can reassociate the
    scaling, rel ~1e-7), with the expected pure-HSDP placements (V FSDP d_in on
    `fsdp`, U FSDP d_out on `fsdp`, C never sharded) — for a heterogeneous-C site set
    spanning attention and MLP matrices."""
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from param_decomp.core.adversary import full_source_components, init_persistent_sources
    from param_decomp.core.ci_fn.implementations.chunkwise.arch import (
        Chunk,
        ChunkwiseTransformerCIFnArch,
    )
    from param_decomp.core.ci_fn.implementations.transformer.layers import MHACIFnAttention
    from param_decomp.core.components import SiteC, init_component_stacks
    from param_decomp.core.init_placed import (
        init_component_stacks_placed,
        init_sources_sharded,
    )
    from param_decomp.targets.testing import tiny_glu_cfg
    from param_decomp.targets.transformer import canonical_site_cs, glu_site_specs
    from param_decomp.tests.placed_ci_fn import placed_ci_fn

    # The HSDP mesh `(replicate, fsdp)`: on the n-device CPU sim with n not a multiple of 8,
    # `fsdp` takes the full count and `replicate` is 1. V FSDP-shards d_in on `fsdp`, U FSDP
    # d_out on `fsdp`; C is never sharded. The tiny target's matrix dims (n_embd=32,
    # n_intermediate=64, qkv head dims) all tile the sim device counts (1/2/4).
    n = jax.device_count()
    mesh = hsdp_mesh(1, jax.device_count(), 1)
    cfg = tiny_glu_cfg()
    # layers.{2,3}.mlp.gate_proj share (d_in, d_out, C), so the stacked init has a REAL
    # multi-site V/U semantic group (stack + unstack across sites, not just groups of one);
    # they also make a 3-site C group for the stacked sources init below.
    sites = glu_site_specs(
        cfg,
        canonical_site_cs(
            (
                SiteC("layers.2.self_attn.q_proj", 8 * n),
                SiteC("layers.2.self_attn.o_proj", 16 * n),
                SiteC("layers.2.mlp.gate_proj", 8 * n),
                SiteC("layers.3.mlp.gate_proj", 8 * n),
                SiteC("layers.3.mlp.down_proj", 16 * n),
            )
        ),
    )
    from param_decomp.core.components import vu_groups

    assert max(len(g.specs) for g in vu_groups(sites).values()) >= 2
    # Placement is MODEL-OWNED, owner-partitioned ÷N (hybrid HSDP rule): the STACK axis
    # shards over `replicate` (whole matrices owned per node-group), matrix d dims over
    # `fsdp`, C over `tp` — total ÷(replicate·fsdp·tp) = ÷N. On the sim mesh replicate is 1
    # (every stack length tiles it), so every group takes the stack rule.
    vu_placed = init_component_stacks_placed(
        sites, jax.random.PRNGKey(1), from_config("owner", mesh, sites)
    )
    vu_eager = init_component_stacks(sites, jax.random.PRNGKey(1))
    for Vs, Us in vu_placed.stacks.values():
        assert isinstance(Vs.sharding, NamedSharding) and isinstance(Us.sharding, NamedSharding)
        assert Vs.sharding.spec == P("replicate", "fsdp", "tp"), Vs.sharding.spec
        assert Us.sharding.spec == P("replicate", "tp", "fsdp"), Us.sharding.spec
    # The placed init must be BIT-identical to the same init jitted WITHOUT placement
    # (threefry is partitionable: `out_shardings` cannot perturb the stream) — the
    # trajectory anchor. The unjitted eager reference differs from either jitted path by
    # ~1 ULP (XLA fusion), so it only gets allclose.
    from functools import partial

    vu_jitted_unplaced = jax.jit(partial(init_component_stacks, sites))(jax.random.PRNGKey(1))
    for got, want in zip(
        jax.tree.leaves(vu_placed), jax.tree.leaves(vu_jitted_unplaced), strict=True
    ):
        assert got.shape == want.shape and got.dtype == want.dtype
        assert jnp.array_equal(jnp.asarray(got), jnp.asarray(want))
    for got, want in zip(jax.tree.leaves(vu_placed), jax.tree.leaves(vu_eager), strict=True):
        assert jnp.allclose(jnp.asarray(got), want, rtol=1e-6, atol=0)

    # A declared ÷N shard dim that does NOT tile the device count is a loud crash at
    # `PlacementRules` construction (fail-fast), not a silent replicate. (Only observable
    # at n > 1.)
    if n > 1:
        from param_decomp.core.components import SiteSpec

        # d_in = n+1 (does not tile N=n) -> placement construction must crash.
        indivisible = (
            SiteSpec(
                "layers.2.mlp.gate_proj",
                DenseFactorization(d_in=n + 1, d_out=8 * n, C=16),
                group="gate_proj",
            ),
        )
        try:
            init_component_stacks_placed(
                indivisible, jax.random.PRNGKey(1), from_config("owner", mesh, indivisible)
            )
        except AssertionError:
            pass
        else:
            raise AssertionError("expected a non-dividing d_in to fail at placement construction")

    first_block = min(int(s.name.split(".")[1]) for s in sites)
    arch = ChunkwiseTransformerCIFnArch(
        chunks=(
            Chunk(input_taps=(f"resid.{first_block}",), output_sites=tuple(s.name for s in sites)),
        ),
        input_dim=cfg.n_embd,
        d_model=16,
        n_blocks=1,
        attention=MHACIFnAttention(mask="bidirectional", implementation="xla", n_heads=2),
        ffn_hidden=8 * n,
        ffn_kind="gelu",
        learned_norm_scale=False,
    )
    ci_placed = placed_ci_fn(
        arch, sites, jax.random.PRNGKey(2), mesh, from_config("zero1", mesh, sites)
    )
    ci_eager = arch.initialize(sites, None, jax.random.PRNGKey(2))
    for got, want in zip(jax.tree.leaves(ci_placed), jax.tree.leaves(ci_eager), strict=True):
        assert got.shape == want.shape and got.dtype == want.dtype
        assert jnp.allclose(jnp.asarray(got), want, rtol=1e-6, atol=0)

    site_names = tuple(s.name for s in sites)
    src_sharded = init_sources_sharded(
        sites,
        Positioned(16),
        "bsc",
        n,
        jnp.float32,
        jax.random.PRNGKey(3),
        mesh,
    )
    src_eager = init_persistent_sources(sites, (n, 16), jnp.float32, jax.random.PRNGKey(3))
    # The placed init must be BIT-identical to the eager stacked init (same per-site
    # keys): uniform draws are pure threefry bit-ops, so even eager-vs-jit is exact. The
    # storage is one slot-major stack per semantic group: slot axis replicated, C on tp.
    assert src_sharded.site_stack_indices == src_eager.site_stack_indices
    for group, stack in src_sharded.stacks.items():
        components = full_source_components(stack.components)
        assert isinstance(components.sharding, NamedSharding)
        assert isinstance(stack.delta.sharding, NamedSharding)
        assert components.sharding.spec == P(None, ("replicate", "fsdp"), None, "tp"), group
        assert stack.delta.sharding.spec == P(None, ("replicate", "fsdp"), None), group
    for got, want in zip(jax.tree.leaves(src_sharded), jax.tree.leaves(src_eager), strict=True):
        assert jnp.array_equal(got, want)
    for name in site_names:
        for got, want in zip(
            jax.tree.leaves(src_sharded.per_site()[name]),
            jax.tree.leaves(src_eager.per_site()[name]),
            strict=True,
        ):
            assert jnp.array_equal(got, want), name

    # bsc: one source per batch element, batch-sharded over the full mesh (the leading
    # axis after the slot axis), no cross-rank sync.
    bsc_global_batch = 4 * n
    src_bsc = init_sources_sharded(
        sites,
        Positioned(16),
        "bsc",
        bsc_global_batch,
        jnp.float32,
        jax.random.PRNGKey(3),
        mesh,
    )
    for name, C in zip(site_names, (s.C for s in sites), strict=True):
        site = src_bsc.per_site()[name]
        assert full_source_components(site.components).shape == (bsc_global_batch, 16, C), name
        assert site.delta.shape == (bsc_global_batch, 16), name
    for group, stack in src_bsc.stacks.items():
        components = full_source_components(stack.components)
        assert isinstance(components.sharding, NamedSharding)
        assert isinstance(stack.delta.sharding, NamedSharding)
        assert components.sharding.spec == P(None, ("replicate", "fsdp"), None, "tp"), group
        assert stack.delta.sharding.spec == P(None, ("replicate", "fsdp"), None), group


def test_fresh_pgd_c_bc_sources_are_replica_identical():
    """Fresh-PGD `c`/`bc` sources must be REPLICA-IDENTICAL across every shard: the `c` -> `(1,1,C)` / `bc` -> `(B,1,C)` component leaf carries no
    sharded leading axis, so the adversarial source the masks see must hold the same
    values on every device. Replica-identity follows from the init key being replicated
    (the trainer derives it from `fold_in(run_key, step)`, identical on all processes).

    Asserts: (a) the per-shard buffers of the replicated init all equal the eager
    single-device init, and (b) the placement is fully replicated (`P()`), at the
    test's device count (run at 1 AND `--xla_force_host_platform_device_count=4`).
    """
    from functools import partial

    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from param_decomp.core.adversary import init_fresh_pgd_sources
    from param_decomp.core.components import SiteSpec

    mesh = hsdp_mesh(1, jax.device_count(), 1)
    n = mesh.devices.size
    batch = 4 * n
    seq = 7
    sites = (
        SiteSpec(
            "layers.2.self_attn.q_proj", DenseFactorization(d_in=16, d_out=16, C=8), group="q_proj"
        ),
        SiteSpec(
            "layers.3.mlp.down_proj", DenseFactorization(d_in=8, d_out=16, C=13), group="down_proj"
        ),
    )

    for scope in ("c", "bc"):
        key = jax.random.PRNGKey(660)
        eager = init_fresh_pgd_sources(sites, "random", scope, (batch, seq), key)
        init = partial(init_fresh_pgd_sources, sites, "random", scope, (batch, seq))
        repl = NamedSharding(mesh, P())
        sharded = jax.jit(init, out_shardings=repl)(key)
        for site in sites:
            for leaf, expected in zip(
                jax.tree.leaves(sharded[site.name]),
                jax.tree.leaves(eager[site.name]),
                strict=True,
            ):
                assert isinstance(leaf.sharding, NamedSharding)
                assert leaf.sharding.spec == P(), (scope, site.name)
                for shard in leaf.addressable_shards:
                    assert jnp.array_equal(jnp.asarray(shard.data), expected), (
                        scope,
                        site.name,
                    )


def test_init_sources_sharded_shape_and_placement_per_arm():
    """Every (positions x source_shape) arm: the stored shape and `PartitionSpec` per
    `configs.BatchSourceShape`, and the positionless raise for `bsc`."""
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from param_decomp.core.adversary import full_source_components
    from param_decomp.core.components import DenseFactorization, SiteSpec
    from param_decomp.core.configs import BatchSourceShape
    from param_decomp.core.init_placed import init_sources_sharded
    from param_decomp.core.model import PositionAxis

    mesh = hsdp_mesh(1, jax.device_count(), 1)
    C, B = 8 * jax.device_count(), 4 * jax.device_count()
    site = SiteSpec(
        name="layers.2.mlp.gate_proj",
        factorization=DenseFactorization(d_in=C, d_out=C, C=C),
        group="g",
    )

    def init(positions: PositionAxis, source_shape: BatchSourceShape):
        return init_sources_sharded(
            (site,), positions, source_shape, B, jnp.float32, jax.random.PRNGKey(3), mesh
        )

    batch_sharded = ("replicate", "fsdp")
    cases: list[tuple[PositionAxis, BatchSourceShape, tuple[int, ...], P]] = [
        (Positioned(16), "bc", (B, 1), P(batch_sharded, None)),
        (Positioned(16), "bsc", (B, 16), P(batch_sharded, None)),
        (Positionless(), "bc", (B,), P(batch_sharded)),
    ]
    for positions, source_shape, want_leading, want_delta_spec in cases:
        stacks = init(positions, source_shape)
        src = stacks.per_site()[site.name]
        assert full_source_components(src.components).shape == (*want_leading, C)
        assert src.delta.shape == want_leading
        stack = stacks.stacks["g"]
        components = full_source_components(stack.components)
        assert isinstance(components.sharding, NamedSharding)
        assert components.sharding.spec == P(None, *want_delta_spec, "tp")
        assert isinstance(stack.delta.sharding, NamedSharding)
        assert stack.delta.sharding.spec == P(None, *want_delta_spec)
    with pytest.raises(ValueError, match="positionless"):
        init(Positionless(), "bsc")


def test_batch_source_pool_placed_gather_and_scatter_add_grad():
    """Each batch element updates its selected particle under pure tensor parallelism."""
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from param_decomp.core.adversary import SourceStacks, full_source_components
    from param_decomp.core.components import DenseFactorization, SiteSpec
    from param_decomp.core.configs import SourcePoolConfig
    from param_decomp.core.init_placed import init_source_pool_sharded
    from param_decomp.core.masking import sample_source_pool

    mesh = hsdp_mesh(1, 1, jax.device_count())
    C, B, T, N = 8 * jax.device_count(), 4 * jax.device_count(), 6, 8
    site = SiteSpec(
        name="layers.2.mlp.gate_proj",
        factorization=DenseFactorization(d_in=C, d_out=C, C=C),
        group="g",
    )
    pool_cfg = SourcePoolConfig(size_per_batch_element=N)
    pool = init_source_pool_sharded((site,), pool_cfg, B, jnp.float32, jax.random.PRNGKey(3), mesh)
    stack = pool.stacks["g"]
    components = full_source_components(stack.components)
    assert components.shape == (1, B, N, C)
    assert isinstance(components.sharding, NamedSharding)
    assert components.sharding.spec == P(None, ("replicate", "fsdp"), None, "tp")
    assert stack.delta.shape == (1, B, N)
    assert isinstance(stack.delta.sharding, NamedSharding)
    assert stack.delta.sharding.spec == P(None, ("replicate", "fsdp"), None)

    batch_sharded = ("replicate", "fsdp")
    with jax.set_mesh(mesh):
        sampled = jax.jit(lambda p: sample_source_pool(jax.random.PRNGKey(0), p, (B, T)))(pool)[
            site.name
        ]
        assert isinstance(sampled.components, jax.Array)
        assert sampled.components.shape == (B, 1, C)
        assert sampled.delta.shape == (B, 1)
        assert isinstance(sampled.components.sharding, NamedSharding)
        assert isinstance(sampled.delta.sharding, NamedSharding)
        assert sampled.components.sharding.is_equivalent_to(
            NamedSharding(mesh, P(batch_sharded, None, "tp")), ndim=3
        )
        assert sampled.delta.sharding.is_equivalent_to(
            NamedSharding(mesh, P(batch_sharded, None)), ndim=2
        )

        def summed(p: SourceStacks) -> Array:
            source = sample_source_pool(jax.random.PRNGKey(0), p, (B, T))[site.name]
            assert isinstance(source.components, jax.Array)
            return jnp.sum(source.components) + jnp.sum(source.delta)

        grads = jax.jit(jax.grad(summed))(pool)
        assert float(jnp.sum(full_source_components(grads.stacks["g"].components))) == B * C
        assert float(jnp.sum(grads.stacks["g"].delta)) == B


def test_faithfulness_deltas_lower_piecewise_without_full_master_gathers():
    """The faithfulness lifecycle (masters -> faithfulness weights -> `W - V@U` deltas)
    must never materialize a full-C f32 master stack: the transition permutes shards
    and the delta einsum reduce-scatters its C contraction onto d_in (the delta rows'
    spelling). A replicated-d_in einsum output instead makes XLA all-gather the full-C
    f32 operands — measured ~18.5 GB/rank of the faith segment's peak at the 32L
    production shape."""
    from jax.sharding import AxisType, Mesh

    from param_decomp.core.components import SiteC, init_component_stacks
    from param_decomp.core.model import faithfulness_weight_deltas
    from param_decomp.targets.testing import tiny_glu_cfg, tiny_glu_decomposed_lm
    from param_decomp.targets.transformer import canonical_site_cs, glu_site_specs, site_name

    mesh = Mesh(
        np.asarray(jax.devices()[:8]).reshape(2, 2, 2),
        ("replicate", "fsdp", "tp"),
        axis_types=(AxisType.Explicit,) * 3,
    )
    cfg = tiny_glu_cfg()
    sites = glu_site_specs(
        cfg,
        canonical_site_cs(
            tuple(
                SiteC(site_name(layer, kind), 8)
                for layer in range(cfg.n_layer)
                for kind in ("gate", "up", "down")
            )
        ),
    )
    model = tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(40))
    components = init_component_stacks(sites, jax.random.PRNGKey(41))
    expected = jax.jit(lambda vu: model.weight_deltas(vu))(components)

    rules = from_config("zero1", mesh, sites)
    placed_model = place_target(model, rules)
    placed = jax.device_put(components, component_stacks_shardings(components, rules))

    with jax.set_mesh(mesh):
        deltas_fn = jax.jit(lambda vu: faithfulness_weight_deltas(placed_model, vu))
        deltas = deltas_fn(placed)
        hlo = deltas_fn.lower(placed).compile().as_text() or ""

    for group, delta in deltas.items():
        assert np.allclose(np.asarray(delta), np.asarray(expected[group]), atol=1e-5), group

    # The sanctioned moves are the shard-permuting all-to-all transition and the
    # reduce-scattered contraction; a cross-replicate all-gather is the full-C
    # materialization this lowering exists to avoid (tp-local gathers are fine).
    from param_decomp.core.tools.hlo_census import _COLLECTIVE_OP, _spans_replicate

    for line in hlo.splitlines():
        m = _COLLECTIVE_OP.search(line)
        if m is None or m.group(1) != "all-gather":
            continue
        assert not _spans_replicate(line, "all-gather", replica_stride=4, n_devices=8), (
            f"faithfulness path all-gathers across replicate: {line.strip()[:160]}"
        )


def test_per_site_metric_readers_trace_under_stack_owned_placement():
    """`_grad_norm_metrics` / `uv_norm_ratio_metrics` read one slot per site off tiny
    per-group [g] vectors whose stack axis is REPLICATE-SHARDED under `sharding: owner`;
    a static slice of a sharded dim is unimplemented, so each reader must replicate its
    vector first. Trace-time regression for the production owner relaunch (the reno
    trainer died at its first slot read)."""
    from jax.sharding import NamedSharding

    from param_decomp.core.components import SiteC
    from param_decomp.core.init_placed import init_component_stacks_placed
    from param_decomp.core.train import _grad_norm_metrics, uv_norm_ratio_metrics
    from param_decomp.targets.testing import tiny_glu_cfg
    from param_decomp.targets.transformer import canonical_site_cs, glu_site_specs

    n = jax.device_count()
    assert n % 2 == 0, n
    mesh = hsdp_mesh(2, n // 2, 1)
    cfg = tiny_glu_cfg()
    # Two layers x {gate, down}: every semantic group has stack length 2, tiling
    # replicate=2, so `owner` shards the STACK axis — the failing precondition.
    sites = glu_site_specs(
        cfg,
        canonical_site_cs(
            (
                SiteC("layers.2.mlp.gate_proj", 8 * n),
                SiteC("layers.2.mlp.down_proj", 16 * n),
                SiteC("layers.3.mlp.gate_proj", 8 * n),
                SiteC("layers.3.mlp.down_proj", 16 * n),
            )
        ),
    )
    rules = from_config("owner", mesh, sites)
    vu = init_component_stacks_placed(sites, jax.random.PRNGKey(1), rules)
    for Vs, _ in vu.stacks.values():
        assert isinstance(Vs.sharding, NamedSharding) and Vs.sharding.spec[0] is not None

    with jax.set_mesh(mesh):
        grad_norms = jax.jit(lambda c: _grad_norm_metrics(c, {}, mesh))(vu)
        ratios = uv_norm_ratio_metrics(vu)

    for name, _, _ in vu.site_stack_indices:
        assert np.isfinite(float(grad_norms[f"grad_norms/components.vu['{name}'][0]"])), name
        assert np.isfinite(float(ratios[f"uv_norm_ratio['{name}']"])), name
