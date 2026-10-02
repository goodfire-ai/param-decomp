"""Shared geometry, CI architecture and assertions for placed Qwen tests."""

import re

import jax
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array

from param_decomp.core.components import (
    ComponentStacks,
    init_component_stacks,
)
from param_decomp.core.placement import batch_axes
from param_decomp.core.tools.hlo_census import (
    _computations,
    _loop_computations,
)
from param_decomp.targets.qwen36_moe import (
    Qwen36MoeConfig,
    Qwen36MoeDecomposedModel,
    full_site_cs,
    qwen36_moe_site_specs,
    site_name,
)
from param_decomp.targets.testing import (
    tiny_qwen36_cfg,
    tiny_qwen36_decomposed_model,
)
from param_decomp.targets.transformer_taps import resid_tap_key

# data=4, tp=2 on the 8 simulated devices; every extent of the tiny config tiles it
# (4 experts ÷2, 2 q-heads ÷2, 2/4 DeltaNet k/v-heads ÷2, fused 32 ÷2, shared 12 ÷2),
# expert C_block=4 ÷data, dense C=8 ÷(tp·data).
DATA, TP = 4, 2
CENSUS_CS: dict[str, int] = {
    "experts_gate": 16,
    "experts_up": 16,
    "experts_down": 16,
    "shared_gate": 8,
    "shared_up": 8,
    "shared_down": 8,
}
BATCH, SEQ = 4, 8

multidevice = pytest.mark.skipif(len(jax.devices()) < 8, reason="requires eight local devices")


def placement_mesh() -> Mesh:
    devices = np.asarray(jax.devices()[: DATA * TP]).reshape(DATA, TP)
    return Mesh(devices, ("data", "tp"), axis_types=(AxisType.Explicit,) * 2)


def model_and_components(
    c_of: dict[str, int],
) -> tuple[Qwen36MoeDecomposedModel, ComponentStacks]:
    cfg = tiny_qwen36_cfg()
    sites = qwen36_moe_site_specs(cfg, full_site_cs(cfg, c_of))
    model = tiny_qwen36_decomposed_model(cfg, sites, jax.random.PRNGKey(0))
    return model, init_component_stacks(sites, jax.random.PRNGKey(1))


def batch_placed(value: Array, mesh: Mesh) -> Array:
    spec = P(batch_axes(mesh), *(None for _ in value.shape[1:]))
    return jax.device_put(value, NamedSharding(mesh, spec))


def assert_no_weight_gather_in_any_loop(hlo: str, batch_shard: int) -> None:
    """Every all-gather inside a while body must be activation-shaped (leading dim = the
    per-shard batch) — a weight shape leading with a stack/expert/matrix dim means
    GSPMD sank a weight gather into the loop."""
    comps = _computations(hlo)
    for name in _loop_computations(comps):
        for line in comps[name]:
            m = re.search(r"all-gather(?:-start)?\(", line)
            if m is None:
                continue
            for dims_text in re.findall(
                r"\b(?:pred|bf16|f16|f32|s32|u32)\[([\d,]+)\]", line[: m.start()]
            ):
                dims = tuple(int(d) for d in dims_text.split(","))
                assert len(dims) >= 2 and dims[0] == batch_shard, (
                    f"in-loop all-gather is not activation-shaped — a weight gather "
                    f"survived residency: {dims} in {line.strip()[:160]}"
                )


def moe_ci_fn_arch(cfg: Qwen36MoeConfig):
    """One chunk per stage, expert kinds narrow, shared kinds full — dims tiling the
    (data=4, tp=2) mesh: expert_ffn_hidden ÷data, C_block=4 ÷data, q-heads ÷tp."""
    from param_decomp.core.ci_fn.implementations.block_selected.arch import (
        BlockSelectedChunk,
        BlockSelectedChunkwiseTransformerCIFnArch,
        FullSlot,
        SelectedSlot,
    )
    from param_decomp.core.ci_fn.implementations.transformer.layers import MHACIFnAttention
    from param_decomp.targets.qwen36_moe import is_expert_kind

    interval = cfg.full_attention_interval
    chunks = []
    for start in range(0, cfg.n_layer, interval):
        layers = tuple(range(start, start + interval))
        slots = []
        for router, layer in enumerate(layers):
            for kind in CENSUS_CS:
                site = site_name(layer, kind)
                slots.append(
                    SelectedSlot(site=site, selection=router)
                    if is_expert_kind(kind)
                    else FullSlot(site=site)
                )
        chunks.append(
            BlockSelectedChunk(
                input_taps=(resid_tap_key(start),), layers=layers, slots=tuple(slots)
            )
        )
    return BlockSelectedChunkwiseTransformerCIFnArch(
        chunks=tuple(chunks),
        input_dim=cfg.n_embd,
        d_model=8,
        n_blocks=2,
        attention=MHACIFnAttention(mask="bidirectional", implementation="xla", n_heads=2),
        table_size=cfg.n_experts,
        selected_ffn_hidden=8,
        shared_ffn_hidden=8,
        learned_norm_scale=False,
        expert_implementation="ragged_dot",
    )
