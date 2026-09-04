"""The finished-run loader restores a small generic offline CI consumer."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from param_decomp.core import placement
from param_decomp.core.components import SiteC, init_component_stacks
from param_decomp.core.model import PlacedModel
from param_decomp.core.train import Decomposition
from param_decomp.experiments.lm import load_run
from param_decomp.targets.glu_transformer import KIND_ORDER, glu_site_specs, site_name
from param_decomp.targets.testing import (
    tiny_glu_cfg,
    tiny_glu_chunkwise_ci_fn,
    tiny_glu_decomposed_lm,
)


def test_restore_decomposition_uses_consumer_sharded_abstract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    abstract = object()
    manager = SimpleNamespace(latest_step=lambda: 17)
    restored = object()
    monkeypatch.setattr(load_run, "_consumer_decomposition_abstract", lambda *_args: abstract)
    monkeypatch.setattr(load_run, "make_read_only_checkpoint_manager", lambda _path: manager)

    def restore(actual_manager: object, step: int, reference: object) -> object:
        assert actual_manager is manager
        assert step == 17
        assert reference is abstract
        return restored

    monkeypatch.setattr(load_run, "restore_decomposition", restore)
    actual, step = load_run._restore_decomposition(
        cast(Any, object()), cast(Any, object()), cast(Any, object()), tmp_path, None
    )

    assert actual is restored
    assert step == 17


def test_consumer_decomposition_abstract_shards_eight_device_cpu_mesh() -> None:
    probe = r"""
import jax
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P, SingleDeviceSharding

from param_decomp.core import placement
from param_decomp.core.ci_fn import Chunk, ChunkwiseTransformerCIArch, MHACIAttention
from param_decomp.core.components import SiteC
from param_decomp.core.model import PlacedModel
from param_decomp.core.run_state import init_decomposition
from param_decomp.core.sharding import hsdp_mesh
from param_decomp.experiments.lm.load_run import (
    _consumer_decomposition_abstract,
    _prepare_read_only_consumer,
)
from param_decomp.targets.glu_transformer import KIND_ORDER, glu_site_specs, site_name
from param_decomp.targets.testing import tiny_glu_cfg, tiny_glu_decomposed_lm

cfg = tiny_glu_cfg()
sites = glu_site_specs(
    cfg, tuple(SiteC(site_name(2, kind), 8) for kind in KIND_ORDER)
)
model = tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
ci_fn = ChunkwiseTransformerCIArch(
    chunks=(Chunk(input_taps=("resid.2",), output_sites=model.site_names),),
    input_dim=cfg.n_embd,
    d_model=16,
    n_blocks=1,
    attention=MHACIAttention(n_heads=2),
    ffn_hidden=32,
    ffn_kind="gelu",
    learned_norm_scale=False,
)
mesh = hsdp_mesh(1, 8, 1)
assert mesh.size == 8, mesh
placed = PlacedModel(model=model, placement=placement.from_config("zero1", mesh, model.sites))
abstract = _consumer_decomposition_abstract(ci_fn, placed, mesh)
leaves = jax.tree.leaves(abstract)
assert leaves
assert all(isinstance(leaf.sharding, NamedSharding) for leaf in leaves)
assert not any(isinstance(leaf.sharding, SingleDeviceSharding) for leaf in leaves)

def nbytes(leaf):
    return int(np.prod(leaf.shape)) * np.dtype(leaf.dtype).itemsize

large = [leaf for leaf in leaves if nbytes(leaf) >= 1024]
assert large
assert all(leaf.sharding.spec != P() for leaf in large)
host_bytes = sum(nbytes(leaf) for leaf in leaves)
max_addressable_device_bytes = max(
    sum(
        int(np.prod(leaf.sharding.shard_shape(leaf.shape)))
        * np.dtype(leaf.dtype).itemsize
        for leaf in leaves
        if device in leaf.sharding.addressable_devices
    )
    for device in mesh.devices.flat
)
assert max_addressable_device_bytes < host_bytes, (
    max_addressable_device_bytes,
    host_bytes,
)

abstract_shardings = jax.tree.map(lambda leaf: leaf.sharding, abstract)
decomposition = jax.jit(
    lambda: init_decomposition(placed, ci_fn, jax.random.PRNGKey(1)),
    out_shardings=abstract_shardings,
)()
with jax.set_mesh(mesh):
    prepared_weights, compute_ci_fn = _prepare_read_only_consumer(
        placed, decomposition.components, decomposition.ci_fn
    )
    jax.block_until_ready((prepared_weights, compute_ci_fn))
prepared_leaves = jax.tree.leaves((prepared_weights, compute_ci_fn))
assert prepared_leaves
assert all(leaf.dtype == jax.numpy.bfloat16 for leaf in prepared_leaves)
assert all(isinstance(leaf.sharding, NamedSharding) for leaf in prepared_leaves)
assert not any(isinstance(leaf.sharding, SingleDeviceSharding) for leaf in prepared_leaves)
"""
    env = os.environ | {
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=8",
    }
    subprocess.run([sys.executable, "-c", probe], env=env, check=True)


def test_open_jax_run_restores_generic_ci_consumer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tiny_glu_cfg()
    sites = glu_site_specs(cfg, tuple(SiteC(site_name(2, kind), 2) for kind in KIND_ORDER))
    model = tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
    decomposition = Decomposition(
        components=init_component_stacks(sites, jax.random.PRNGKey(1)),
        ci_fn=tiny_glu_chunkwise_ci_fn(model, jax.random.PRNGKey(2), n_blocks=1),
    )
    monkeypatch.setattr(
        load_run,
        "load_deliverable",
        lambda *_args: SimpleNamespace(target=SimpleNamespace(), ci_fn=object()),
    )
    mesh = jax.sharding.Mesh(
        np.asarray([jax.devices()[0]]).reshape(1, 1, 1),
        ("replicate", "fsdp", "tp"),
    )
    monkeypatch.setattr(load_run, "hsdp_mesh", lambda *_args: mesh)
    placed = PlacedModel(model=model, placement=placement.from_config("ddp", mesh, model.sites))
    monkeypatch.setattr(load_run, "build_target", lambda *_args: placed)
    monkeypatch.setattr(
        load_run,
        "_restore_decomposition",
        lambda *_args: (decomposition, 7),
    )

    run = load_run.open_jax_run(tmp_path / "p-ci", data_root=tmp_path)
    assert run.step == 7
    assert run.placed is placed
    assert run.model is model
    assert all(
        leaf.dtype == jnp.bfloat16
        for leaf in jax.tree.leaves((run.prepared_weights, run.ci_fn))
        if eqx.is_inexact_array(leaf)
    )
