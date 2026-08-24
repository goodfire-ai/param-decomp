"""Reverse-mode memory pins for the GLU masked forward at the production topology.

`remat_recon_forwards` is the checkpoint policy of the per-block scan: True =
`nothing_saveable` (re-forward each block in the backward), False = `dots_saveable`
(store batch-scaled activation dots, re-forward nothing). Three properties are
load-bearing at scale and pinned here on a tiny fully-decomposed GLU over the
`(replicate, fsdp, tp) = (1, 2, 2)` mesh:

  * gradients agree with the unplaced reference under BOTH policies;
  * under `remat=False` every stored residual scales with the batch — a weight-shaped
    residual (a gathered operand kept alive into the backward) must never be saved;
  * no compiled op materializes a `[n_blocks, ...]` weight stack in operand layout —
    the GSPMD failure mode where per-block gathers hoist out of the scan and the
    executable allocates the whole gathered model at once.
"""

import dataclasses
import math
import re

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Public `jax.ad_checkpoint` exposes only `print_saved_residuals`; the returning variant
# this audit needs lives in the private module.
from jax._src.ad_checkpoint import saved_residuals
from jax.core import ShapedArray
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array

from param_decomp.core.components import ComponentStacks, SiteC, init_component_stacks
from param_decomp.core.init_placed import init_component_stacks_placed
from param_decomp.core.model import (
    MaterializedMasking,
    PlacedModel,
    prepare_compute_weights,
)
from param_decomp.core.placement import from_config
from param_decomp.core.sharding import place_target
from param_decomp.targets.glu_transformer import (
    KIND_ORDER,
    GLUDecomposedModel,
    canonical_site_cs,
    glu_site_specs,
    site_name,
)
from param_decomp.targets.testing import tiny_glu_cfg, tiny_glu_decomposed_lm
from param_decomp.vendored_jax.llama import AttentionImplementation

pytestmark = [
    pytest.mark.multidevice,
    pytest.mark.skipif(
        jax.default_backend() != "cpu" or jax.device_count() < 4,
        reason="requires a four-device CPU topology from make test-multidevice",
    ),
]

# The batch divides the (replicate, fsdp) data plane, and batch/sequence sizes match no
# weight dimension of `tiny_glu_cfg` (d=32, d_ff=64, qd=32, kvd=16, C=8, n_layer=8), so
# "carries a batch axis" is decidable from a residual's shape alone.
_BATCH, _SEQ = 6, 5


def _fixture(implementation: AttentionImplementation = "auto"):
    cfg = tiny_glu_cfg()
    sites = glu_site_specs(
        cfg,
        canonical_site_cs(
            tuple(
                SiteC(site_name(layer, kind), 8)
                for layer in range(cfg.n_layer)
                for kind in KIND_ORDER
            )
        ),
    )
    model = tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(7))
    if implementation != "auto":
        attn = dataclasses.replace(model.stacked.attn, implementation=implementation)
        model = eqx.tree_at(lambda m: m.stacked.attn, model, attn)
    tokens = jnp.arange(_BATCH * _SEQ, dtype=jnp.int32).reshape(_BATCH, _SEQ) % cfg.vocab_size
    masks = {site.name: jnp.full((_BATCH, _SEQ, site.C), 0.7) for site in sites}
    deltas = {site.name: jnp.full((_BATCH, _SEQ), 0.3) for site in sites}
    routes = {site.name: (tokens % 2 == 0) for site in sites}
    capture_keys = frozenset(f"resid.{layer + 1}" for layer in range(cfg.n_layer))
    return cfg, sites, model, tokens, masks, deltas, routes, capture_keys


def _mesh() -> Mesh:
    return Mesh(
        np.asarray(jax.devices()[:4]).reshape(1, 2, 2),
        ("replicate", "fsdp", "tp"),
        axis_types=(AxisType.Explicit,) * 3,
    )


def _place_batched[T](mesh: Mesh, tree: T) -> T:
    return jax.tree.map(
        lambda value: jax.device_put(
            value,
            NamedSharding(mesh, P(("replicate", "fsdp"), *(None for _ in value.shape[1:]))),
        ),
        tree,
    )


def _loss_fn(
    model: PlacedModel,
    tokens: Array,
    masks: dict[str, Array],
    deltas: dict[str, Array],
    routes: dict[str, Array],
    capture_keys: frozenset[str],
    remat: bool,
):
    def loss(components: ComponentStacks) -> Array:
        prepared = prepare_compute_weights(model, components)
        result = model.masked_forward(
            prepared,
            tokens,
            masking=MaterializedMasking(
                component_masks=masks, weight_delta_masks=deltas, routes=routes
            ),
            capture_keys=capture_keys,
            remat=remat,
        )
        capture_sum = sum(
            jnp.sum(value.astype(jnp.float32) ** 2) for value in result.captures.values()
        )
        return jnp.sum(result.output.astype(jnp.float32) ** 2) / 1e3 + capture_sum

    return loss


@pytest.mark.parametrize("remat", (True, False))
def test_masked_forward_gradients_match_the_unplaced_reference(remat: bool):
    _, sites, model, tokens, masks, deltas, routes, capture_keys = _fixture()
    components = init_component_stacks(sites, jax.random.PRNGKey(11))

    unplaced = PlacedModel(model=model, placement=None)
    reference_loss = _loss_fn(unplaced, tokens, masks, deltas, routes, capture_keys, remat=remat)
    expected_value, expected_grads = jax.jit(jax.value_and_grad(reference_loss))(components)

    mesh = _mesh()
    rules = from_config("owner", mesh, sites)
    placed_model = place_target(model, rules)
    placed_components = init_component_stacks_placed(sites, jax.random.PRNGKey(11), rules)
    placed_tokens = jax.device_put(tokens, NamedSharding(mesh, P(("replicate", "fsdp"), None)))
    placed_masks, placed_deltas, placed_routes = _place_batched(mesh, (masks, deltas, routes))

    with jax.set_mesh(mesh):
        placed_loss = _loss_fn(
            placed_model,
            placed_tokens,
            placed_masks,
            placed_deltas,
            placed_routes,
            capture_keys,
            remat=remat,
        )
        actual_value, actual_grads = jax.jit(jax.value_and_grad(placed_loss))(placed_components)

    assert jnp.allclose(actual_value, expected_value, rtol=3e-2)
    for group, (expected_v, expected_u) in expected_grads.stacks.items():
        actual_v, actual_u = actual_grads.stacks[group]
        for actual, expected in ((actual_v, expected_v), (actual_u, expected_u)):
            expected_host = np.asarray(expected)
            # BF16 compute: placed and unplaced graphs reassociate differently (SPEC D4),
            # so small-magnitude entries carry absolute noise scaled by the group's grads.
            np.testing.assert_allclose(
                np.asarray(actual),
                expected_host,
                rtol=3e-2,
                atol=3e-2 * float(np.abs(expected_host).max()),
                err_msg=group,
            )


@pytest.mark.parametrize("implementation", ("auto", "xla"))
def test_remat_off_saves_only_batch_scaled_residuals(implementation: AttentionImplementation):
    cfg, sites, model, tokens, masks, deltas, routes, capture_keys = _fixture(implementation)
    mesh = _mesh()
    rules = from_config("owner", mesh, sites)
    placed_model = place_target(model, rules)
    placed_components = init_component_stacks_placed(sites, jax.random.PRNGKey(11), rules)
    placed_tokens = jax.device_put(tokens, NamedSharding(mesh, P(("replicate", "fsdp"), None)))
    placed_masks, placed_deltas, placed_routes = _place_batched(mesh, (masks, deltas, routes))

    placed_target = placed_model.model
    assert isinstance(placed_target, GLUDecomposedModel)
    weight_shapes = _weight_matrix_shapes(placed_target, placed_components)
    with jax.set_mesh(mesh):
        prepared = prepare_compute_weights(placed_model, placed_components)

        # Differentiate from the PREPARED compute weights: the once-per-step resident
        # relayout stays outside the audited graph, so every residual below is something
        # the masked forward itself asks the backward to store.
        def loss(prepared_weights: dict[str, dict[str, Array]]) -> Array:
            result = placed_model.masked_forward(
                prepared_weights,
                placed_tokens,
                masking=MaterializedMasking(
                    component_masks=placed_masks,
                    weight_delta_masks=placed_deltas,
                    routes=placed_routes,
                ),
                capture_keys=capture_keys,
                remat=False,
            )
            capture_sum = sum(
                jnp.sum(value.astype(jnp.float32) ** 2) for value in result.captures.values()
            )
            return jnp.sum(result.output.astype(jnp.float32) ** 2) / 1e3 + capture_sum

        residuals = saved_residuals(loss, prepared)

    # Arguments, constants, and literals are the trace's inputs (the frozen model and the
    # probe batch live regardless of checkpoint policy). The audit is over everything the
    # backward STORES from the forward computation — a hoisted gathered operand is always
    # a computed residual, never an input.
    shaped = [(aval, source) for aval, source in residuals if isinstance(aval, ShapedArray)]
    assert len(shaped) == len(residuals), residuals
    computed = [
        (aval, source)
        for aval, source in shaped
        if not any(
            tag in source for tag in ("from the argument", "from a constant", "from a literal")
        )
    ]
    assert computed
    n_blocks = cfg.n_layer

    # Residuals the block scan stacks (leading n_blocks axis) are the per-block store —
    # exactly where a retained gathered operand would land. Every one must scale with the
    # batch and none may be weight-shaped. Batch shows up at three extents: global,
    # shard_map-local (the data-plane divisor), and doubled (XLA packs two `[B, T, d]`
    # saves into one `[2B, T, d]` buffer).
    data_plane = mesh.shape["replicate"] * mesh.shape["fsdp"]
    batch_extents = {_BATCH, _BATCH // data_plane, 2 * _BATCH}
    assert batch_extents.isdisjoint({dim for shape in weight_shapes for dim in shape})
    scan_saved = [(aval, source) for aval, source in computed if aval.shape[:1] == (n_blocks,)]
    assert scan_saved
    offenders = [
        (aval.shape, source)
        for aval, source in scan_saved
        if not batch_extents.intersection(aval.shape[1:]) or tuple(aval.shape[-2:]) in weight_shapes
    ]
    assert not offenders, offenders

    # Residuals outside the scan are once-per-forward buffers (RoPE table, norm scale,
    # the lm_head transpose): at most one matrix each, never a stack.
    single = max(a * b for a, b in weight_shapes)
    outside = [
        (aval.shape, source)
        for aval, source in computed
        if aval.shape[:1] != (n_blocks,) and math.prod(aval.shape) > max(single, _BATCH * _SEQ * 64)
    ]
    assert not outside, outside


def _weight_matrix_shapes(
    model: GLUDecomposedModel, components: ComponentStacks
) -> frozenset[tuple[int, ...]]:
    """Trailing-2D shapes of every frozen weight and component matrix (and transposes)."""
    shapes: set[tuple[int, ...]] = set()
    for leaf in jax.tree.leaves((model.stacked, components)):
        if leaf.ndim >= 2:
            trailing = tuple(leaf.shape[-2:])
            shapes.add(trailing)
            shapes.add(trailing[::-1])
    return frozenset(shapes)


@pytest.mark.parametrize("remat", (True, False))
def test_no_op_materializes_a_full_weight_stack_in_operand_layout(remat: bool):
    cfg, sites, model, tokens, masks, deltas, routes, capture_keys = _fixture()
    mesh = _mesh()
    rules = from_config("owner", mesh, sites)
    placed_model = place_target(model, rules)
    placed_components = init_component_stacks_placed(sites, jax.random.PRNGKey(11), rules)
    placed_tokens = jax.device_put(tokens, NamedSharding(mesh, P(("replicate", "fsdp"), None)))
    placed_masks, placed_deltas, placed_routes = _place_batched(mesh, (masks, deltas, routes))

    with jax.set_mesh(mesh):
        loss = _loss_fn(
            placed_model,
            placed_tokens,
            placed_masks,
            placed_deltas,
            placed_routes,
            capture_keys,
            remat=remat,
        )
        compiled = jax.jit(jax.value_and_grad(loss)).lower(placed_components).compile()

    hlo = compiled.as_text()
    assert hlo is not None
    # The pathology: GSPMD hoists the per-block operand gathers out of the scan, so a
    # REAL collective (group size > 1) produces a full `[n_blocks, d, d]` weight stack.
    # Group-size-1 gathers are shard_map lowering artifacts of the once-per-step
    # compute-weight relayout over the trivial replicate=1 axis, not materializations.
    n_blocks = cfg.n_layer
    stacked = re.compile(rf"= \w+\[{n_blocks},\d+,\d+\]")
    real_group = re.compile(r"replica_groups=\{\{\d+(,\d+)+\}")
    offenders = [
        line
        for line in hlo.splitlines()
        if stacked.search(line)
        and re.search(r"\b(all-gather|all-reduce|collective-permute|all-to-all)\(", line)
        and real_group.search(line)
    ]
    assert not offenders, "\n".join(offenders)

    # Backstop to the collective scan above (the actual hoist detector): the legitimate
    # graph's temp arena is activation-scale (measured 0.11/0.30 MiB across policies).
    memory = compiled.memory_analysis()
    assert memory is not None
    assert memory.temp_size_in_bytes < 2**19, memory.temp_size_in_bytes
