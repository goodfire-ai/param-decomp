"""Blocked sources retain expert ownership through reads and their transpose."""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array

from param_decomp.routed.experts import ep_gather_job_blocks, expert_sharded_jobs


@pytest.mark.multidevice
@pytest.mark.skipif(len(jax.devices()) < 4, reason="requires four devices")
@pytest.mark.parametrize("source_batch", (1, 4))
@pytest.mark.parametrize("source_position", (1, 3))
@pytest.mark.parametrize("skewed", (False, True))
def test_source_jobs_preserve_values_gradients_and_expert_ownership(
    source_batch: int, source_position: int, skewed: bool
) -> None:
    b, t, e, k, c, shards = 4, 3, 4, 2, 3, 2
    mesh = Mesh(
        np.asarray(jax.devices()[:4]).reshape(2, 2),
        ("data", "tp"),
        axis_types=(AxisType.Explicit,) * 2,
    )
    ids = np.empty((b, t, k), dtype=np.int32)
    for row in range(b):
        for token in range(t):
            ids[row, token] = (
                (0, 1 if skewed and row == 0 else 2)
                if skewed
                else (
                    (row + token) % e,
                    (row + token + 1) % e,
                )
            )
    source = np.linspace(-0.7, 0.9, source_batch * source_position * e * c, dtype=np.float32)
    source = source.reshape(source_batch, source_position, e, c)
    cotangent = np.sin(np.arange(b * shards * t * k * c, dtype=np.float32)).reshape(
        b, shards, t * k, c
    )
    expected = np.zeros_like(cotangent)
    selected_cotangent = np.zeros((b, t, k, c), dtype=np.float32)
    dense_source = np.broadcast_to(source, (b, t, e, c))
    for row in range(b):
        flat_ids = ids[row].reshape(-1)
        for owner in range(shards):
            local = np.where(flat_ids // (e // shards) == owner, flat_ids % (e // shards), e)
            order = np.argsort(local, stable=True)
            for job, slot in enumerate(order):
                expert = flat_ids[slot]
                if expert // (e // shards) == owner:
                    token, pick = divmod(int(slot), k)
                    expected[row, owner, job] = dense_source[row, token, expert]
                    selected_cotangent[row, token, pick] = cotangent[row, owner, job]

    def dense_selected(table: Array) -> Array:
        full = jnp.broadcast_to(table, (b, t, e, c))
        return jnp.take_along_axis(full, jnp.asarray(ids)[..., None], axis=2)

    _, dense_pullback = jax.vjp(dense_selected, jnp.asarray(source))
    (expected_gradient,) = dense_pullback(jnp.asarray(selected_cotangent))
    with jax.set_mesh(mesh):
        placed_ids = jax.device_put(ids, NamedSharding(mesh, P("data", None, None)))
        source_sharding = NamedSharding(
            mesh, P(None if source_batch == 1 else "data", None, "tp", None)
        )
        placed_source = jax.device_put(source, source_sharding)
        placed_cotangent = jax.device_put(
            cotangent, NamedSharding(mesh, P("data", "tp", None, None))
        )

        def gather(table: Array, routing: Array) -> Array:
            jobs = expert_sharded_jobs(routing, e, shards)
            return ep_gather_job_blocks(table, jobs, "tp")

        def backward(table: Array, routing: Array, ct: Array) -> Array:
            _, pullback = jax.vjp(lambda values: gather(values, routing), table)
            return pullback(ct)[0]

        forward_executable = jax.jit(gather).lower(placed_source, placed_ids).compile()
        backward_executable = (
            jax.jit(backward).lower(placed_source, placed_ids, placed_cotangent).compile()
        )
        actual = forward_executable(placed_source, placed_ids)
        gradient = backward_executable(placed_source, placed_ids, placed_cotangent)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(gradient, expected_gradient, atol=3e-7, rtol=3e-7)
    assert actual.sharding.is_equivalent_to(
        NamedSharding(mesh, P("data", "tp", None, None)), actual.ndim
    )
    assert gradient.sharding.is_equivalent_to(source_sharding, gradient.ndim)
    collective_pattern = re.compile(
        r"\b(all-reduce|all-gather|all-to-all|reduce-scatter|collective-permute)\("
    )
    forward_hlo = forward_executable.as_text()
    backward_hlo = backward_executable.as_text()
    assert forward_hlo is not None and backward_hlo is not None
    assert not collective_pattern.search(forward_hlo)
    for line in backward_hlo.splitlines():
        if collective_pattern.search(line):
            assert source_batch == 1, line
            assert "all-reduce(" in line, line
            data_groups = (
                "replica_groups={{0,2},{1,3}}",
                "replica_groups=mesh['axis_0'=2,'axis_1'=2,'axis_2'=1,'axis_3'=1] {'axis_0'}",
            )
            assert any(group in line for group in data_groups), line
