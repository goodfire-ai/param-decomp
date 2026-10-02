"""A signal on any device stops every participant at the same collective boundary."""

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from param_decomp.core import run as engine


def test_local_consensus_observes_signals_received_after_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert jax.process_count() == 1
    mesh = Mesh(np.asarray(jax.devices()), ("device",))
    monkeypatch.setattr(engine, "_sigterm_received", False)
    consensus = engine._prepare_sigterm_consensus(mesh)
    assert not consensus()
    monkeypatch.setattr(engine, "_sigterm_received", True)
    assert consensus()


@pytest.mark.multidevice
def test_compiled_consensus_reduces_a_signal_from_each_device() -> None:
    mesh = Mesh(np.asarray(jax.devices()), ("device",))
    sharding = NamedSharding(mesh, P("device"))
    abstract_flags = jax.ShapeDtypeStruct((mesh.size,), np.bool_, sharding=sharding)
    with jax.set_mesh(mesh):
        consensus = jax.jit(engine._any_sigterm_received).lower(abstract_flags).compile()

    flags = np.zeros(mesh.size, dtype=np.bool_)
    assert not bool(consensus(jax.device_put(flags, sharding)))
    for device_index in range(mesh.size):
        flags[device_index] = True
        assert bool(consensus(jax.device_put(flags, sharding)))
        flags[device_index] = False
    assert not bool(consensus(jax.device_put(flags, sharding)))
