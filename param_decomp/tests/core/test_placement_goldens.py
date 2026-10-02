"""Golden pin of placement CONSTRUCTION: every preset plus two explicit tables, over a
mesh x site-set grid, each cell serialized (or its refusal message recorded) against the
committed `placement_goldens.json` — refusals are pinned behavior, message included, and
the census' resolved `stack_pad` counts are pinned per cell. Fallback-bearing tables are
unrepresentable, so their cells pin the SCHEMA refusal (pydantic parse errors). Each
transformer CI architecture's rows per preset, bound on every mesh, are pinned alongside
(`ci|<arch>|<preset>|<mesh>`, unsupported presets and mesh refusals by their message;
a preset the global transformer refuses before any mesh, once as `ci|global|<preset>`),
with one resolved chunk-stack census per architecture (`ci_census|...`).
Regenerate only when placement semantics deliberately change:
`python -m param_decomp.tests.core.gen_placement_goldens`."""

import dataclasses
import json
from collections.abc import Callable
from pathlib import Path
from typing import cast, get_args

import pytest
from jax.sharding import AbstractMesh
from pydantic import ValidationError

from param_decomp.core.ci_fn.implementations.block_selected.arch import (
    BlockSelectedChunk,
    BlockSelectedChunkwiseTransformerCIFnArch,
    FullSlot,
    SelectedSlot,
)
from param_decomp.core.ci_fn.implementations.block_selected.placement import (
    bind_block_selected_rows,
)
from param_decomp.core.ci_fn.implementations.block_selected.placement import (
    preset_rows as block_selected_preset_rows,
)
from param_decomp.core.ci_fn.implementations.chunkwise.arch import (
    Chunk,
    ChunkwiseTransformerCIFnArch,
)
from param_decomp.core.ci_fn.implementations.chunkwise.placement import bind_chunkwise_rows
from param_decomp.core.ci_fn.implementations.chunkwise.placement import (
    preset_rows as chunkwise_preset_rows,
)
from param_decomp.core.ci_fn.implementations.global_transformer.placement import bind_global_rows
from param_decomp.core.ci_fn.implementations.global_transformer.placement import (
    preset_rows as global_preset_rows,
)
from param_decomp.core.ci_fn.implementations.transformer.component_conditioning import (
    bind_readout_row,
    preset_readout_row,
)
from param_decomp.core.ci_fn.implementations.transformer.layers import MHACIFnAttention
from param_decomp.core.components import BlockedFactorization, DenseFactorization, SiteSpec
from param_decomp.core.configs import PlacementPresetName, PlacementSpec, PlacementTableConfig
from param_decomp.core.placement import (
    PlacedRule,
    PlacementRules,
    StackCensus,
    from_config,
    reachable_placed_rules,
)

GOLDENS_PATH = Path(__file__).parent / "placement_goldens.json"

_MESH_AXES = ("replicate", "fsdp", "tp")
# The `(data, tp)` mesh: home of the `*-replicated-resident` presets; every three-axis
# spec on it (and every resident preset on a three-axis mesh) pins the binder's
# mesh-vocabulary refusal.
MESHES = {
    "replicate4_fsdp8_tp1": AbstractMesh((4, 8, 1), _MESH_AXES),
    "replicate2_fsdp2_tp2": AbstractMesh((2, 2, 2), _MESH_AXES),
    "data4_tp2": AbstractMesh((4, 2), ("data", "tp")),
}


def _sites(group_sizes: dict[tuple[int, int, int], int]) -> tuple[SiteSpec, ...]:
    """One semantic group per `(d_in, d_out, C): g` entry."""
    return tuple(
        SiteSpec(
            f"s{d_in}x{d_out}x{c}.{i}",
            DenseFactorization(d_in=d_in, d_out=d_out, C=c),
            f"{d_in}x{d_out}x{c}",
        )
        for (d_in, d_out, c), g in group_sizes.items()
        for i in range(g)
    )


# `tiling` tiles every preset's stack sharding on both meshes; `mixed` adds a 1-stack
# group that tiles neither mesh's replicate extent, so stack-sharded specs resolve a
# persist pad for it — the pinned census carries the pad count.
SITE_SETS = {
    "tiling": _sites({(64, 32, 8): 4}),
    "mixed": _sites({(64, 32, 8): 4, (128, 64, 8): 1}),
}

_TARGET_ROWS = {
    "embedding": {"persist": {"d_model": "fsdp"}, "operand": {}},
    "normalization": {},
    "position_encoding": {},
    "column": {
        "persist": {"d_in": "fsdp", "d_out": "tp"},
        "operand": {"d_out": "tp"},
        "input": "external",
        "output": "intermediate",
    },
    "row": {
        "persist": {"d_out": "fsdp", "d_in": "tp"},
        "operand": {"d_in": "tp"},
        "input": "intermediate",
        "output": "external",
    },
    "output": {"persist": {"d_model": "fsdp"}, "operand": {}},
    "intermediate": {
        "batch": ["replicate", "fsdp"],
        "feature": "tp",
        "q_head": "tp",
        "kv_head": "tp",
    },
    "component": {"input": "external", "output": "external"},
}
_OWNER_COMPONENT_ROWS = {
    "optimizer_state": {"stack": "replicate", "d_in": "fsdp", "d_out": "fsdp", "C": "tp"},
    "compute_weights": {"d_in": "fsdp", "d_out": "fsdp", "C": "tp"},
    "faithfulness_weights": {
        "stack": "replicate",
        "d_in": "fsdp",
        "d_out": "fsdp",
        "C": "tp",
    },
    "faithfulness_deltas": {"stack": "replicate", "d_out": "fsdp"},
    "operands": {"C": "tp"},
    "ns_compute": {"stack": "replicate"},
}
_EXPLICIT_OWNER = PlacementTableConfig.model_validate(
    {
        "components": _OWNER_COMPONENT_ROWS,
        "ci_fn": "zero1",
        "activations": {
            "external": {"batch": ["replicate", "fsdp"]},
            "component": {"batch": ["replicate", "fsdp"], "C": "tp"},
        },
        "target": _TARGET_ROWS,
    }
)
_EXPLICIT_FSDP_ONLY = PlacementTableConfig.model_validate(
    {
        "components": {
            "optimizer_state": {"d_in": "fsdp", "d_out": "fsdp"},
            "compute_weights": {"d_in": "fsdp", "d_out": "fsdp"},
            "faithfulness_weights": {"d_in": "fsdp", "d_out": "fsdp"},
            "faithfulness_deltas": {"d_out": "fsdp"},
            "operands": {},
            "ns_compute": {},
        },
        "ci_fn": "zero1",
        "activations": {
            "external": {"batch": ["replicate", "fsdp"]},
            "component": {"batch": ["replicate", "fsdp"]},
        },
        "target": {
            "embedding": {"persist": {"d_model": "fsdp"}, "operand": {}},
            "normalization": {},
            "position_encoding": {},
            "column": {
                "persist": {"d_in": "fsdp"},
                "operand": {},
                "input": "external",
                "output": "intermediate",
            },
            "row": {
                "persist": {"d_out": "fsdp"},
                "operand": {},
                "input": "intermediate",
                "output": "external",
            },
            "output": {"persist": {"d_model": "fsdp"}, "operand": {}},
            "intermediate": {"batch": ["replicate", "fsdp"]},
            "component": {"input": "external", "output": "external"},
        },
    }
)

SPECS: dict[str, PlacementSpec] = {
    "preset_owner": "owner",
    # deleted preset: pins the unknown-name refusal (a str, so the widening cast holds)
    "preset_owner_zero1": cast(PlacementSpec, cast(str, "owner+zero1")),
    "preset_zero1": "zero1",
    "preset_zero1_replicated_resident": "zero1-replicated-resident",
    "preset_owner_replicated_resident": "owner-replicated-resident",
    # the MoE presets over these DENSE site sets pin their fail-closed refusals (their
    # expert/C_block keys name axes no dense tensor consumes)
    "preset_zero1_replicated_resident_moe": "zero1-replicated-resident-moe",
    "preset_owner_replicated_resident_moe": "owner-replicated-resident-moe",
    "preset_zero1_replicated_resident_moe_replicated_ns": (
        "zero1-replicated-resident-moe-replicated-ns"
    ),
    "preset_ddp": "ddp",
    "explicit_owner": _EXPLICIT_OWNER,
    "explicit_fsdp_only": _EXPLICIT_FSDP_ONLY,
}

# Fallback rows are unrepresentable: these raw tables die at PARSE, and the pydantic
# error records (type, loc, msg) are pinned — a schema loosening would change a golden.
SCHEMA_REFUSALS: dict[str, dict[str, object]] = {
    "schema_optimizer_state_fallback": {
        "components": _OWNER_COMPONENT_ROWS
        | {"optimizer_state_fallback": {"d_in": "fsdp", "C": ["tp", "replicate"]}},
        "ci_fn": "zero1",
        "activations": {
            "external": {"batch": ["replicate", "fsdp"]},
            "component": {"batch": ["replicate", "fsdp"], "C": "tp"},
        },
        "target": _TARGET_ROWS,
    },
    "schema_faithfulness_fallback_pair": {
        "components": _OWNER_COMPONENT_ROWS
        | {
            "faithfulness_weights_fallback": {"d_in": "fsdp", "C": ["tp", "replicate"]},
            "faithfulness_deltas_fallback": {"d_out": "fsdp", "d_in": ["tp", "replicate"]},
        },
        "ci_fn": "zero1",
        "activations": {
            "external": {"batch": ["replicate", "fsdp"]},
            "component": {"batch": ["replicate", "fsdp"], "C": "tp"},
        },
        "target": _TARGET_ROWS,
    },
}

CI_FN_BOUND_ROWS: dict[
    str, Callable[[PlacementPresetName, AbstractMesh], tuple[PlacedRule, ...]]
] = {
    "chunkwise": lambda preset, mesh: reachable_placed_rules(
        bind_chunkwise_rows(chunkwise_preset_rows(preset), mesh)
    ),
    "block_selected": lambda preset, mesh: reachable_placed_rules(
        bind_block_selected_rows(block_selected_preset_rows(preset), mesh)
    ),
    "global": lambda preset, mesh: reachable_placed_rules(
        bind_global_rows(global_preset_rows(preset), mesh)
    ),
    "conditioning": lambda preset, mesh: (bind_readout_row(preset_readout_row(preset), mesh),),
}
# The global transformer names rows for `owner` alone; every other preset refuses before
# any mesh is involved, so each refusal is pinned once.
_GLOBAL_PRESETS: tuple[PlacementPresetName, ...] = ("owner",)
CI_FN_KEYS = tuple(
    f"ci|{arch}|{preset}|{mesh}"
    for arch in CI_FN_BOUND_ROWS
    for preset in (_GLOBAL_PRESETS if arch == "global" else get_args(PlacementPresetName))
    for mesh in MESHES
)
CI_FN_REFUSAL_KEYS = tuple(
    f"ci|global|{preset}"
    for preset in get_args(PlacementPresetName)
    if preset not in _GLOBAL_PRESETS
)

_ATTENTION = MHACIFnAttention(mask="bidirectional", implementation="xla", n_heads=2)
_CENSUS_MESH = MESHES["data4_tp2"]
_DENSE_CHUNK_SITES = _sites({(8, 8, 8): 3})
_SELECTED_CHUNK_SITES = tuple(
    site
    for i in range(3)
    for site in (
        SiteSpec(f"dense.{i}", DenseFactorization(d_in=8, d_out=8, C=8), "dense"),
        SiteSpec(
            f"selected.{i}",
            BlockedFactorization(n_blocks=4, d_in=8, d_out=8, c_per_block=4),
            "selected",
        ),
    )
)


def _chunkwise_census(preset: PlacementPresetName) -> StackCensus:
    arch = ChunkwiseTransformerCIFnArch(
        chunks=tuple(
            Chunk(input_taps=("tap",), output_sites=(site.name,)) for site in _DENSE_CHUNK_SITES
        ),
        input_dim=8,
        d_model=8,
        n_blocks=1,
        attention=_ATTENTION,
        ffn_hidden=16,
        ffn_kind="gelu",
        learned_norm_scale=False,
    )
    rules = from_config(preset, _CENSUS_MESH, _DENSE_CHUNK_SITES)
    return arch.resolve_placement(rules).chunks


def _block_selected_census(preset: PlacementPresetName) -> StackCensus:
    arch = BlockSelectedChunkwiseTransformerCIFnArch(
        chunks=tuple(
            BlockSelectedChunk(
                input_taps=("tap",),
                layers=(0,),
                slots=(FullSlot(f"dense.{i}"), SelectedSlot(f"selected.{i}", 0)),
            )
            for i in range(3)
        ),
        input_dim=8,
        d_model=8,
        n_blocks=1,
        attention=_ATTENTION,
        table_size=4,
        selected_ffn_hidden=8,
        shared_ffn_hidden=8,
        learned_norm_scale=False,
        expert_implementation="ragged_dot",
    )
    rules = from_config(preset, _CENSUS_MESH, _SELECTED_CHUNK_SITES)
    return arch.resolve_placement(rules).chunks


# Three chunks on data=4: owner-cut masters pad the stack to 4, zero1 masters pad nothing.
CI_FN_CENSUSES: dict[str, Callable[[], StackCensus]] = {
    "ci_census|chunkwise|owner-replicated-resident": lambda: _chunkwise_census(
        "owner-replicated-resident"
    ),
    "ci_census|chunkwise|zero1-replicated-resident": lambda: _chunkwise_census(
        "zero1-replicated-resident"
    ),
    "ci_census|block_selected|zero1-replicated-resident-moe": lambda: _block_selected_census(
        "zero1-replicated-resident-moe"
    ),
}

GRID_KEYS = (
    tuple(f"{spec}|{mesh}|{sites}" for spec in SPECS for mesh in MESHES for sites in SITE_SETS)
    + tuple(SCHEMA_REFUSALS)
    + CI_FN_KEYS
    + CI_FN_REFUSAL_KEYS
    + tuple(CI_FN_CENSUSES)
)


def _row_json(row: PlacedRule) -> dict[str, object]:
    return {
        "label": row.label_for_log,
        "rule": {axis: list(assignment) for axis, assignment in sorted(row.rule.items())},
    }


def serialize_rules(rules: PlacementRules) -> dict[str, object]:
    """The full construction result as JSON-native data: every row's label + rule, the
    resolved group census, and the target linears' activation aliases. Deliberately a
    hand transcription of the structure, independent of `reachable_placed_rules`."""
    components = rules.components
    target = rules.target
    return {
        "mesh": {axis: int(size) for axis, size in rules.mesh.shape.items()},
        "components": {
            "optimizer_state": _row_json(components.optimizer_state),
            "compute_weights": _row_json(components.compute_weights),
            "faithfulness_weights": _row_json(components.faithfulness_weights),
            "faithfulness_deltas": _row_json(components.faithfulness_deltas),
            "operands": _row_json(components.operands),
            "ns_compute": _row_json(components.ns_compute),
            "group_census": {
                name: {
                    "factorization": {
                        "kind": type(entry.factorization).__name__,
                        **dataclasses.asdict(entry.factorization),
                    },
                    "stack_len": entry.stack_len,
                    "stack_pad": entry.stack_pad,
                }
                for name, entry in sorted(components.group_census.items())
            },
        },
        "ci_fn": rules.ci_fn,
        "activations": {
            "external": _row_json(rules.activations.external),
            "component": _row_json(rules.activations.component),
        },
        "target": {
            "embedding": {
                "persist": _row_json(target.embedding.persist),
                "operand": _row_json(target.embedding.operand),
            },
            "normalization": _row_json(target.normalization),
            "position_encoding": _row_json(target.position_encoding),
            "column": {
                "persist": _row_json(target.column.persist),
                "operand": _row_json(target.column.operand),
                "input": target.column.input.label_for_log,
                "output": target.column.output.label_for_log,
            },
            "row": {
                "persist": _row_json(target.row.persist),
                "operand": _row_json(target.row.operand),
                "input": target.row.input.label_for_log,
                "output": target.row.output.label_for_log,
            },
            "output": {
                "persist": _row_json(target.output.persist),
                "operand": _row_json(target.output.operand),
            },
            "intermediate": _row_json(target.intermediate),
            "component": {
                "input": target.component.input.label_for_log,
                "output": target.component.output.label_for_log,
            },
        },
    }


def build_cell(key: str) -> dict[str, object]:
    """One grid cell: the serialized rules, the construction refusal message (unknown
    preset, non-tiling groups), the schema refusal's pydantic error records, or one CI
    architecture's rows under a preset."""
    if key in CI_FN_KEYS:
        _, arch, preset, mesh = key.split("|")
        try:
            rows = CI_FN_BOUND_ROWS[arch](cast(PlacementPresetName, preset), MESHES[mesh])
        except (NotImplementedError, AssertionError) as refusal:
            return {"refused": str(refusal)}
        return {"ci_rows": [_row_json(row) for row in rows]}
    if key in CI_FN_REFUSAL_KEYS:
        _, _, preset = key.split("|")
        with pytest.raises(NotImplementedError) as refusal:
            global_preset_rows(cast(PlacementPresetName, preset))
        return {"refused": str(refusal.value)}
    if key in CI_FN_CENSUSES:
        census = CI_FN_CENSUSES[key]()
        return {"stack_len": census.stack_len, "stack_pad": census.stack_pad}
    if key in SCHEMA_REFUSALS:
        with pytest.raises(ValidationError) as excinfo:
            PlacementTableConfig.model_validate(SCHEMA_REFUSALS[key])
        return {
            "schema_refused": [
                {"type": e["type"], "loc": list(e["loc"]), "msg": e["msg"]}
                for e in excinfo.value.errors()
            ]
        }
    spec, mesh, sites = key.split("|")
    try:
        rules = from_config(SPECS[spec], MESHES[mesh], SITE_SETS[sites])
    except AssertionError as refusal:
        return {"refused": str(refusal)}
    return {"rules": serialize_rules(rules)}


def _goldens() -> dict[str, object]:
    return json.loads(GOLDENS_PATH.read_text())


def test_goldens_cover_the_grid_exactly():
    assert set(_goldens()) == set(GRID_KEYS)


@pytest.mark.parametrize("key", GRID_KEYS)
def test_construction_matches_golden(key: str):
    assert build_cell(key) == _goldens()[key]
