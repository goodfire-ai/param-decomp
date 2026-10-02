"""CPU staging for checkpoint readers and model assembly."""

from collections.abc import Iterator
from contextlib import contextmanager

import jax
import numpy as np
from jax.sharding import Mesh

from param_decomp.core.sharding import require_configured_backends


@contextmanager
def cpu_staging() -> Iterator[None]:
    require_configured_backends()
    cpu = jax.local_devices(backend="cpu")[0]
    with jax.set_mesh(Mesh(np.empty((), dtype=object), ())), jax.default_device(cpu):
        yield
