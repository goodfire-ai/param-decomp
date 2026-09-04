"""Golden pin of placement CONSTRUCTION: every preset plus two explicit tables, over a
mesh x site-set grid, each cell serialized (or its refusal message recorded) against the
committed `placement_goldens.json` — refusals are pinned behavior, the strict refusal
message included. Fallback-bearing tables are unrepresentable, so their cells pin the
SCHEMA refusal (pydantic parse errors). Regenerate only when placement semantics
deliberately change: `python -m param_decomp.tests.core.gen_placement_goldens`."""

import json
from pathlib import Path

import pytest
from jax.sharding import AbstractMesh
from pydantic import ValidationError

from param_decomp.core.axes import MeshAssignment
from param_decomp.core.components import SiteSpec
from param_decomp.core.configs import PlacementTableConfig
from param_decomp.core.placement import (
    CIWeightPlacement,
    PlacedRule,
    PlacementRules,
    from_config,
)

GOLDENS_PATH = Path(__file__).parent / "placement_goldens.json"

_MESH_AXES = ("replicate", "fsdp", "tp")
MESHES = {
    "replicate4_fsdp8_tp1": AbstractMesh((4, 8, 1), _MESH_AXES),
    "replicate2_fsdp2_tp2": AbstractMesh((2, 2, 2), _MESH_AXES),
}


def _sites(group_sizes: dict[tuple[int, int, int], int]) -> tuple[SiteSpec, ...]:
    """One semantic group per `(d_in, d_out, C): g` entry."""
    return tuple(
        SiteSpec(f"s{d_in}x{d_out}x{c}.{i}", d_in, d_out, c, f"{d_in}x{d_out}x{c}")
        for (d_in, d_out, c), g in group_sizes.items()
        for i in range(g)
    )


# `tiling` tiles every preset's stack sharding on both meshes; `mixed` adds a 1-stack
# group that tiles neither mesh's replicate extent, so stack-sharded specs refuse it —
# the refusal message is pinned behavior.
SITE_SETS = {
    "tiling": _sites({(64, 32, 8): 4}),
    "mixed": _sites({(64, 32, 8): 4, (128, 64, 8): 1}),
}

_CI_ROWS = {
    "attention": {
        "optimizer_state": {"d_model": ["fsdp", "replicate"], "q_head": "tp", "kv_head": "tp"},
        "compute_weights": {"d_model": "fsdp", "q_head": "tp", "kv_head": "tp"},
        "operands": {"q_head": "tp", "kv_head": "tp"},
        "ns_compute": {},
    },
    "ffn": {
        "optimizer_state": {"ffn_hidden": ["fsdp", "tp", "replicate"]},
        "compute_weights": {"ffn_hidden": ["fsdp", "tp"]},
        "operands": {"ffn_hidden": "tp"},
        "ns_compute": {},
    },
    "input": {
        "optimizer_state": {"input": "tp", "d_model": ["fsdp", "replicate"]},
        "compute_weights": {"input": "tp", "d_model": "fsdp"},
        "operands": {"input": "tp"},
        "ns_compute": {},
    },
    "output": {
        "optimizer_state": {"d_model": ["fsdp", "replicate"], "C": "tp"},
        "compute_weights": {"d_model": "fsdp", "C": "tp"},
        "operands": {"C": "tp"},
        "ns_compute": {},
    },
    "vectors": {"ffn_hidden": "tp", "C": "tp"},
    "activations": {
        "batch": ["replicate", "fsdp"],
        "input": "tp",
        "q_head": "tp",
        "kv_head": "tp",
        "ffn_hidden": "tp",
        "C": "tp",
    },
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
        "ci_fn": _CI_ROWS,
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
        "ci_fn": {
            "attention": {
                "optimizer_state": {"d_model": "fsdp"},
                "compute_weights": {"d_model": "fsdp"},
                "operands": {},
                "ns_compute": {},
            },
            "ffn": {
                "optimizer_state": {"ffn_hidden": "fsdp"},
                "compute_weights": {"ffn_hidden": "fsdp"},
                "operands": {},
                "ns_compute": {},
            },
            "input": {
                "optimizer_state": {"d_model": "fsdp"},
                "compute_weights": {"d_model": "fsdp"},
                "operands": {},
                "ns_compute": {},
            },
            "output": {
                "optimizer_state": {"d_model": "fsdp"},
                "compute_weights": {"d_model": "fsdp"},
                "operands": {},
                "ns_compute": {},
            },
            "vectors": {},
            "activations": {"batch": ["replicate", "fsdp"]},
        },
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

SPECS: dict[str, str | PlacementTableConfig] = {
    "preset_owner": "owner",
    "preset_owner_zero1": "owner+zero1",  # deleted preset: pins the unknown-name refusal
    "preset_zero1": "zero1",
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
        "ci_fn": _CI_ROWS,
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
        "ci_fn": _CI_ROWS,
        "activations": {
            "external": {"batch": ["replicate", "fsdp"]},
            "component": {"batch": ["replicate", "fsdp"], "C": "tp"},
        },
        "target": _TARGET_ROWS,
    },
}

GRID_KEYS = tuple(
    f"{spec}|{mesh}|{sites}" for spec in SPECS for mesh in MESHES for sites in SITE_SETS
) + tuple(SCHEMA_REFUSALS)


def _assignment_json(assignment: MeshAssignment) -> str | list[str] | None:
    match assignment:
        case None | str():
            return assignment
        case tuple():
            return list(assignment)


def _row_json(row: PlacedRule) -> dict[str, object]:
    return {
        "label": row.label,
        "rule": {axis: _assignment_json(a) for axis, a in sorted(row.rule.items())},
    }


def _ci_weight_json(placement: CIWeightPlacement) -> dict[str, object]:
    return {
        "optimizer_state": _row_json(placement.optimizer_state),
        "compute_weights": _row_json(placement.compute_weights),
        "operands": _row_json(placement.operands),
        "ns_compute": _row_json(placement.ns_compute),
    }


def serialize_rules(rules: PlacementRules) -> dict[str, object]:
    """The full construction result as JSON-native data: every row's label + rule, the
    resolved group census, and the target linears' activation aliases. Deliberately a
    hand transcription of the structure, independent of `PlacementRules._rows`."""
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
            "group_stack_lens": dict(sorted(components.group_stack_lens.items())),
        },
        "ci_fn": {
            "attention": _ci_weight_json(rules.ci_fn.attention),
            "ffn": _ci_weight_json(rules.ci_fn.ffn),
            "input": _ci_weight_json(rules.ci_fn.input),
            "output": _ci_weight_json(rules.ci_fn.output),
            "vectors": _row_json(rules.ci_fn.vectors),
            "activations": _row_json(rules.ci_fn.activations),
        },
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
                "input": target.column.input.label,
                "output": target.column.output.label,
            },
            "row": {
                "persist": _row_json(target.row.persist),
                "operand": _row_json(target.row.operand),
                "input": target.row.input.label,
                "output": target.row.output.label,
            },
            "output": {
                "persist": _row_json(target.output.persist),
                "operand": _row_json(target.output.operand),
            },
            "intermediate": _row_json(target.intermediate),
            "component": {
                "input": target.component.input.label,
                "output": target.component.output.label,
            },
        },
    }


def build_cell(key: str) -> dict[str, object]:
    """One grid cell: the serialized rules, the construction refusal message (unknown
    preset, non-tiling groups), or the schema refusal's pydantic error records."""
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
