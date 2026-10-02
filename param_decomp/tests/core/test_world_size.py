import jax
import pytest
from pydantic import ValidationError

from param_decomp.core import sharding
from param_decomp.core.world_size import (
    MultiNode,
    SingleNode,
    WorldSize,
    world_size_from_device_count,
)


@pytest.mark.parametrize("devices", range(1, 9))
def test_single_node_world(devices: int):
    world = world_size_from_device_count(devices)
    assert world == SingleNode(n_gpus=devices)
    assert world.device_count == devices
    assert world.gpus_per_node == devices


@pytest.mark.parametrize("nodes", [2, 3, 16])
def test_multi_node_world(nodes: int):
    world = world_size_from_device_count(nodes * 8)
    assert world == MultiNode(n_nodes=nodes)
    assert world.device_count == nodes * 8
    assert world.gpus_per_node == 8


@pytest.mark.parametrize("devices", [-1, 0, 9, 12, 15, 17])
def test_unsupported_world_sizes_refuse(devices: int):
    with pytest.raises(ValueError, match="World size must"):
        world_size_from_device_count(devices)


@pytest.mark.parametrize("devices", [-1, 0, 9, 16, True, 2.5])
def test_single_node_validates_at_construction(devices: object):
    with pytest.raises(ValidationError):
        SingleNode.model_validate({"n_gpus": devices})


@pytest.mark.parametrize("nodes", [-1, 0, 1, True, 2.5])
def test_multi_node_validates_at_construction(nodes: object):
    with pytest.raises(ValidationError):
        MultiNode.model_validate({"n_nodes": nodes})


@pytest.mark.parametrize(
    ("world", "processes"),
    [(SingleNode(n_gpus=2), 1), (SingleNode(n_gpus=8), 1), (MultiNode(n_nodes=2), 2)],
)
def test_startup_uses_the_declared_physical_world(
    monkeypatch: pytest.MonkeyPatch, world: WorldSize, processes: int
):
    initialized: list[list[int]] = []
    monkeypatch.setattr(
        jax.distributed,
        "initialize",
        lambda *, local_device_ids: initialized.append(local_device_ids),
    )
    monkeypatch.setattr(jax, "process_count", lambda: processes)
    monkeypatch.setattr(jax, "device_count", lambda: world.device_count)
    monkeypatch.setattr(jax, "local_device_count", lambda: world.gpus_per_node)
    monkeypatch.setattr(jax, "devices", lambda backend=None: [])
    monkeypatch.setattr(sharding, "checked_device_kind", lambda devices: None)
    sharding.initialize_topology(world, world.gpus_per_node)
    assert initialized == ([list(range(8))] if processes > 1 else [])


@pytest.mark.parametrize("local_devices", [0, 4, 16])
def test_startup_refuses_incompatible_local_device_counts(local_devices: int):
    with pytest.raises(AssertionError):
        sharding.initialize_topology(MultiNode(n_nodes=2), local_devices)
