"""The finished-run loader restores a run onto the layout its caller names."""

import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pytest

from param_decomp.core.ci_fn.implementations.chunkwise.arch import (
    Chunk,
    ChunkwiseTransformerCIFnArch,
)
from param_decomp.core.ci_fn.implementations.transformer.layers import MHACIFnAttention
from param_decomp.core.components import SiteC, init_component_stacks
from param_decomp.core.configs import ResidentMeshShape
from param_decomp.core.precision import COMPUTE_DT, cast_floating
from param_decomp.core.train import Decomposition
from param_decomp.experiments.lm import load_run
from param_decomp.experiments.lm.load_run import (
    SINGLE_DEVICE_RESIDENT_LAYOUT,
    ConsumerLayout,
)
from param_decomp.experiments.lm.resolved import LMCIFnArch
from param_decomp.lm.batch import LMBatchWithDocuments
from param_decomp.targets.qwen36_moe import (
    Qwen36MoeDecomposedModel,
    full_site_cs,
    is_expert_kind,
    qwen36_moe_site_specs,
)
from param_decomp.targets.testing import (
    TINY_QWEN36_CS,
    tiny_glu_cfg,
    tiny_glu_decomposed_lm,
    tiny_qwen36_cfg,
    tiny_qwen36_decomposed_model,
    tiny_qwen36_moe_ci_fn_arch,
)
from param_decomp.targets.transformer import (
    KIND_ORDER,
    TransformerDecomposedModel,
    glu_site_specs,
    site_name,
)
from param_decomp.targets.transformer_taps import resid_tap_key


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
from param_decomp.core.ci_fn.implementations.chunkwise.arch import Chunk, ChunkwiseTransformerCIFnArch
from param_decomp.core.ci_fn.implementations.transformer.layers import MHACIFnAttention
from param_decomp.core.components import SiteC
from param_decomp.core.init_placed import seeded_ci_fn_initializer
from param_decomp.core.model import PlacedModel
from param_decomp.core.run_state import init_decomposition
from param_decomp.core.sharding import hsdp_mesh
from param_decomp.experiments.lm.load_run import (
    _consumer_decomposition_abstract,
    _prepare_read_only_consumer,
)
from param_decomp.targets.transformer import KIND_ORDER, glu_site_specs, site_name
from param_decomp.targets.testing import tiny_glu_cfg, tiny_glu_decomposed_lm

cfg = tiny_glu_cfg()
sites = glu_site_specs(
    cfg, tuple(SiteC(site_name(2, kind), 8) for kind in KIND_ORDER)
)
model = tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
ci_fn = ChunkwiseTransformerCIFnArch(
    chunks=(Chunk(input_taps=("resid.2",), output_sites=model.site_names),),
    input_dim=cfg.n_embd,
    d_model=16,
    n_blocks=1,
    attention=MHACIFnAttention(mask="bidirectional", implementation="xla", n_heads=2),
    ffn_hidden=32,
    ffn_kind="gelu",
    learned_norm_scale=False,
)
mesh = hsdp_mesh(1, 8, 1)
assert mesh.size == 8, mesh
rules = placement.from_config("zero1", mesh, model.sites)
placed = PlacedModel(model=model, placement=rules)
with jax.set_mesh(mesh):
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
with jax.set_mesh(mesh):
    decomposition = jax.jit(
        lambda: init_decomposition(
            placed, seeded_ci_fn_initializer(ci_fn, model.sites, rules), jax.random.PRNGKey(1)
        ),
        out_shardings=abstract_shardings,
    )()
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


def _write_decomposition_checkpoint[Conditioning](
    run_dir: Path, step: int, decomposition: Decomposition[Conditioning]
) -> None:
    manager = ocp.CheckpointManager(
        (run_dir / "ckpts").resolve(),
        options=ocp.CheckpointManagerOptions(enable_async_checkpointing=False),
    )
    manager.save(step, args=ocp.args.Composite(decomposition=ocp.args.StandardSave(decomposition)))
    manager.wait_until_finished()
    manager.close()


def _open_checkpointed(
    run_dir: Path,
    model: TransformerDecomposedModel | Qwen36MoeDecomposedModel,
    ci_fn_arch: LMCIFnArch,
    layout: ConsumerLayout,
) -> load_run.LoadedLMRun:
    """A fresh decomposition of `model` checkpointed under `run_dir`, then opened through
    the REAL `open_run` path on `layout`; only the deliverable read and the HF target
    load are stood in for by the fixture."""
    decomposition = Decomposition(
        components=init_component_stacks(model.sites, jax.random.PRNGKey(1)),
        ci_fn=ci_fn_arch.initialize(model.sites, None, jax.random.PRNGKey(2)),
    )
    _write_decomposition_checkpoint(run_dir, 3, decomposition)
    deliverable = SimpleNamespace(target=object(), ci_fn=ci_fn_arch)
    with (
        mock.patch.object(load_run, "load_deliverable", lambda *_args: deliverable),
        mock.patch.object(load_run, "load_target", lambda *_args: model),
    ):
        return load_run.open_run(run_dir, None, data_root=run_dir, layout=layout)


def _plain_chunkwise_arch(
    model: TransformerDecomposedModel | Qwen36MoeDecomposedModel, first_block: int, input_dim: int
) -> ChunkwiseTransformerCIFnArch:
    return ChunkwiseTransformerCIFnArch(
        chunks=(Chunk(input_taps=(resid_tap_key(first_block),), output_sites=model.site_names),),
        input_dim=input_dim,
        d_model=16,
        n_blocks=1,
        attention=MHACIFnAttention(mask="bidirectional", implementation="xla", n_heads=2),
        ffn_hidden=32,
        ffn_kind="gelu",
        learned_norm_scale=False,
    )


def open_tiny_glu_run(run_dir: Path, layout: ConsumerLayout) -> load_run.LoadedLMRun:
    cfg = tiny_glu_cfg()
    sites = glu_site_specs(cfg, tuple(SiteC(site_name(2, kind), 8) for kind in KIND_ORDER))
    model = tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
    return _open_checkpointed(run_dir, model, _plain_chunkwise_arch(model, 2, cfg.n_embd), layout)


def open_tiny_qwen36_run(
    run_dir: Path, c_of: Mapping[str, int], layout: ConsumerLayout
) -> load_run.LoadedLMRun:
    """A routed site set carries the MoE chunkwise CI (whose expert banks the `-moe`
    presets place); a shared-only set carries the plain chunkwise CI, as a run
    decomposing no experts would."""
    cfg = tiny_qwen36_cfg()
    sites = qwen36_moe_site_specs(cfg, full_site_cs(cfg, c_of))
    model = tiny_qwen36_decomposed_model(cfg, sites, jax.random.PRNGKey(0))
    ci_fn_arch: LMCIFnArch = (
        tiny_qwen36_moe_ci_fn_arch(model)
        if any(is_expert_kind(kind) for kind in c_of)
        else _plain_chunkwise_arch(model, 0, cfg.n_embd)
    )
    return _open_checkpointed(run_dir, model, ci_fn_arch, layout)


SHARED_ONLY_QWEN36_CS = {kind: c for kind, c in TINY_QWEN36_CS.items() if kind.startswith("shared")}

# Cs that tile the tp=2 rank: whole expert blocks per rank, dense C ÷tp.
ROUTED_QWEN36_CS = {
    "experts_gate": 16,
    "experts_up": 16,
    "experts_down": 16,
    "shared_gate": 8,
    "shared_up": 8,
    "shared_down": 8,
}


def assert_opened_on(run: load_run.LoadedLMRun, layout: ConsumerLayout) -> None:
    assert run.step == 3
    assert run.mesh.shape == layout.mesh.model_dump()
    assert run.placed.placement is not None and run.placed.placement.mesh == run.mesh
    leaves = jax.tree.leaves((run.prepared_weights, run.prepared_ci_fn))
    assert leaves
    assert all(leaf.dtype == jnp.bfloat16 for leaf in leaves if eqx.is_inexact_array(leaf))
    assert all(leaf.sharding.mesh == run.mesh for leaf in leaves)


def test_open_run_restores_a_glu_run_on_the_single_device_layout(tmp_path: Path) -> None:
    run = open_tiny_glu_run(tmp_path / "p-glu", SINGLE_DEVICE_RESIDENT_LAYOUT)
    assert_opened_on(run, SINGLE_DEVICE_RESIDENT_LAYOUT)
    assert run.run_id == "p-glu"
    assert isinstance(run.model, TransformerDecomposedModel)


def test_shared_only_qwen36_run_opens_on_the_single_device_layout(tmp_path: Path) -> None:
    run = open_tiny_qwen36_run(
        tmp_path / "p-shared", SHARED_ONLY_QWEN36_CS, SINGLE_DEVICE_RESIDENT_LAYOUT
    )
    assert_opened_on(run, SINGLE_DEVICE_RESIDENT_LAYOUT)
    assert isinstance(run.model, Qwen36MoeDecomposedModel)


def test_moe_preset_refuses_a_run_without_routed_expert_sites(tmp_path: Path) -> None:
    layout = ConsumerLayout(
        mesh=ResidentMeshShape(data=1, tp=1), sharding="zero1-replicated-resident-moe"
    )
    with pytest.raises(AssertionError, match="name no semantic axis"):
        open_tiny_qwen36_run(tmp_path / "p-shared", SHARED_ONLY_QWEN36_CS, layout)


def test_layout_must_span_the_process_devices(tmp_path: Path) -> None:
    layout = ConsumerLayout(
        mesh=ResidentMeshShape(data=1, tp=jax.device_count() + 1),
        sharding="zero1-replicated-resident",
    )
    with pytest.raises(AssertionError, match="this process has"):
        load_run.open_run(tmp_path, None, data_root=tmp_path, layout=layout)


def test_routed_qwen36_run_opens_on_the_moe_preset_over_tp(tmp_path: Path) -> None:
    """`{data: 1, tp: N}` with N the whole process (whole expert blocks per tp rank), in
    a two-device subprocess — the tiny CI arch's two attention heads fix the fixture's
    tp: the expert-blocked V/U land sharded over tp."""
    probe = f"""
from pathlib import Path
import jax
from param_decomp.core.configs import ResidentMeshShape
from param_decomp.experiments.lm.load_run import ConsumerLayout
from param_decomp.tests.experiments.lm.test_load_run import (
    ROUTED_QWEN36_CS,
    assert_opened_on,
    open_tiny_qwen36_run,
)

assert jax.device_count() == 2, jax.device_count()
layout = ConsumerLayout(
    mesh=ResidentMeshShape(data=1, tp=2), sharding="zero1-replicated-resident-moe"
)
run = open_tiny_qwen36_run(Path({str(tmp_path / "p-routed")!r}), ROUTED_QWEN36_CS, layout)
assert_opened_on(run, layout)
assert any(
    not leaf.sharding.is_fully_replicated for leaf in jax.tree.leaves(run.prepared_weights)
)
"""
    env = os.environ | {
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
    }
    subprocess.run([sys.executable, "-c", probe], env=env, check=True)


def test_hf_target_load_passes_output_and_attention_to_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    from jax.typing import DTypeLike

    from param_decomp.attention import AttentionImplementation
    from param_decomp.core.components import SiteSpec
    from param_decomp.experiments.lm.config import hf_model_variant
    from param_decomp.experiments.lm.resolved import TargetConfig
    from param_decomp.targets.lm_output import OutputEdge, StreamedLinearOutput, StreamedOutputEdge
    from param_decomp.targets.transformer import (
        FrozenAttn,
        TransformerArch,
        TransformerLayer,
        build_decomposed_lm,
    )

    cfg = tiny_glu_cfg()
    model = tiny_glu_decomposed_lm(cfg, (), jax.random.PRNGKey(0))
    requested_settings: list[tuple[OutputEdge, AttentionImplementation]] = []

    def load(
        model_name: str,
        cfg: TransformerArch,
        sites: tuple[SiteSpec, ...],
        weights_dtype: DTypeLike,
        output_edge: OutputEdge,
        attention_implementation: AttentionImplementation,
    ) -> TransformerDecomposedModel:
        assert model_name == "meta-llama/Llama-3.1-8B"
        assert weights_dtype == jnp.float32
        requested_settings.append((output_edge, attention_implementation))
        layers = [
            TransformerLayer(
                ln1=layer.ln1,
                ln2=layer.ln2,
                attn=FrozenAttn(
                    wq=layer.attn.wq,
                    wk=layer.attn.wk,
                    wv=layer.attn.wv,
                    wo=layer.attn.wo,
                    n_head=cfg.n_head,
                    n_kv_head=cfg.n_kv_head,
                    head_dim=cfg.head_dim,
                    n_rep=cfg.n_rep,
                    implementation=attention_implementation,
                ),
                mlp=layer.mlp,
            )
            for layer in model.layers
        ]
        runtime = build_decomposed_lm(
            embed=model.embed,
            layers=layers,
            norm=model.norm,
            lm_head=model.lm_head,
            inv_freq=model.inv_freq,
            cfg=cfg,
            sites=sites,
            output_edge=output_edge,
        )

        return runtime

    variant = replace(
        hf_model_variant("meta-llama/Llama-3.1-8B"),
        arch=cfg,
        load_weights=load,
    )
    monkeypatch.setattr(load_run, "hf_model_variant", lambda _name: variant)
    edge = StreamedOutputEdge(n_vocab_chunks=4)
    target = TargetConfig(
        model_name="meta-llama/Llama-3.1-8B",
        sites=(),
        weights_dtype="float32",
        attention_implementation="xla",
        output_edge=edge,
        component_initialization="random",
    )
    loaded = load_run.load_target(target, tmp_path)
    assert isinstance(loaded, TransformerDecomposedModel)
    assert requested_settings == [(edge, "xla")]
    assert loaded.output_edge == edge
    assert loaded.stacked.attn.implementation == "xla"
    output = loaded.clean_forward(
        LMBatchWithDocuments.from_unsegmented_sequences(jnp.zeros((1, 4), dtype=jnp.int32)),
        placement=None,
    ).output
    assert isinstance(output, StreamedLinearOutput)
    np.testing.assert_array_equal(output.head, model.head_weight)
    assert output.n_chunks == 4


def test_open_target_and_components_restores_components_without_the_ci_fn(tmp_path: Path) -> None:
    cfg = tiny_glu_cfg()
    sites = glu_site_specs(cfg, tuple(SiteC(site_name(2, kind), 8) for kind in KIND_ORDER))
    model = tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
    components = init_component_stacks(model.sites, jax.random.PRNGKey(1))
    ci_fn = _plain_chunkwise_arch(model, 2, cfg.n_embd).initialize(
        model.sites, None, jax.random.PRNGKey(2)
    )
    _write_decomposition_checkpoint(tmp_path, 3, Decomposition(components=components, ci_fn=ci_fn))
    with (
        mock.patch.object(load_run, "load_dense_target", lambda *_args: object()),
        mock.patch.object(load_run, "load_target", lambda *_args: model),
    ):
        loaded = load_run.open_target_and_components(tmp_path, 3, data_root=tmp_path)
    assert loaded.step == 3 and loaded.run_id == tmp_path.name
    expected = model.prepare_compute_weights(cast_floating(components, COMPUTE_DT), None)
    for actual, want in zip(
        jax.tree.leaves(loaded.prepared_weights), jax.tree.leaves(expected), strict=True
    ):
        assert jnp.array_equal(actual, want)
