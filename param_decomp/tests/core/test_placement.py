"""Declarative placement, build-time group-census validation, and strict refusals."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from pydantic import ValidationError

from param_decomp.core.ci_fn import (
    Chunk,
    ChunkwiseTransformerCIArch,
    GQACIAttention,
    MHACIAttention,
    resolve_ci_placement,
)
from param_decomp.core.components import (
    ComponentStacks,
    SiteSpec,
    component_stacks_from_site_arrays,
)
from param_decomp.core.configs import PlacementTableConfig
from param_decomp.core.decomposed_linear import constrain_component_activation
from param_decomp.core.placement import (
    PlacedRule,
    component_stacks_audit,
    component_stacks_shardings,
    component_stacks_to_compute_weights,
    component_stacks_to_faithfulness_weights,
    constrain_faithfulness_deltas,
    dropped_mesh_axes,
    from_config,
)

MESH = jax.sharding.AbstractMesh((4, 8, 1), ("replicate", "fsdp", "tp"))
SINGLE_DEVICE_MESH = jax.sharding.AbstractMesh((1, 1, 1), ("replicate", "fsdp", "tp"))
V_AXES = ("stack", "d_in", "C")
U_AXES = ("stack", "C", "d_out")
TARGET_ROWS = {
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
CI_ROWS = {
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


def _sites(group_sizes: dict[tuple[int, int, int], int]) -> tuple[SiteSpec, ...]:
    """One semantic group per `(d_in, d_out, C): g` test entry."""
    return tuple(
        SiteSpec(f"s{d_in}x{d_out}x{c}.{i}", d_in, d_out, c, f"{d_in}x{d_out}x{c}")
        for (d_in, d_out, c), g in group_sizes.items()
        for i in range(g)
    )


# All dims tile MESH's zero1 assignment (d ÷fsdp=8, C ÷tp·replicate=4) so shape
# validation never masks the assignment/claim behaviour under test.
TILING = _sites({(64, 32, 8): 4})  # 4 tiles ÷replicate=4
MIXED = _sites({(64, 32, 8): 4, (128, 64, 8): 1})  # + a non-tiling group of 1


def test_owner_preset_derives_the_d4_layout():
    rules = from_config("owner", MESH, TILING)
    assert rules.components.optimizer_state.spec_for(V_AXES) == P("replicate", "fsdp", "tp")
    assert rules.components.optimizer_state.spec_for(U_AXES) == P("replicate", "tp", "fsdp")
    assert rules.components.compute_weights.spec_for(V_AXES) == P(None, "fsdp", "tp")
    assert rules.components.faithfulness_weights.spec_for(V_AXES) == P("replicate", "fsdp", "tp")
    assert rules.components.operands.spec_for(("d_in", "C")) == P(None, "tp")
    assert rules.components.operands.spec_for(("C", "d_out")) == P("tp", None)
    assert rules.target.embedding.persist.spec_for(("vocab", "d_model")) == P(None, "fsdp")
    assert rules.target.embedding.operand.spec_for(("vocab", "d_model")) == P(None, None)
    assert rules.target.normalization.spec_for(("d_model",)) == P(None)
    assert rules.target.position_encoding.spec_for(("rope_frequency",)) == P(None)
    assert rules.target.column.persist.spec_for(("d_out", "d_in")) == P("tp", "fsdp")
    assert rules.target.column.operand.spec_for(("d_out", "d_in")) == P("tp", None)
    assert rules.target.row.persist.spec_for(("d_out", "d_in")) == P("fsdp", "tp")
    assert rules.target.row.operand.spec_for(("d_out", "d_in")) == P(None, "tp")
    assert rules.target.output.persist.spec_for(("vocab", "d_model")) == P(None, "fsdp")
    assert rules.target.output.operand.spec_for(("vocab", "d_model")) == P(None, None)
    assert rules.target.intermediate.spec_for(("batch", "feature")) == P(
        ("replicate", "fsdp"), "tp"
    )
    assert rules.target.column.input is rules.activations.external
    assert rules.target.column.output is rules.target.intermediate
    assert rules.target.row.input is rules.target.intermediate
    assert rules.target.row.output is rules.activations.external
    assert rules.target.component.input is rules.target.component.output
    assert rules.target.component.input is rules.activations.external


def test_zero1_preset_is_intra_matrix_everywhere():
    # No zero1 row shards the component stack axis, so any stack length is placeable —
    # MIXED's 1-stack group included. The masters park `replicate` MINOR on C: entry to
    # the ÷fsdp compute weights is a pure minor-axis gather over `replicate`
    # (PLACEMENT_DESIGN.md invariant 5), and the faithfulness rows ARE the master
    # layout, so that transition is the identity.
    rules = from_config("zero1", MESH, MIXED)
    owner = from_config("owner", MESH, TILING)
    assert rules.components.compute_weights == owner.components.compute_weights
    assert rules.components.operands == owner.components.operands
    assert rules.activations == owner.activations
    assert rules.components.optimizer_state.spec_for(V_AXES) == P(None, "fsdp", ("tp", "replicate"))
    assert rules.components.optimizer_state.spec_for(U_AXES) == P(None, ("tp", "replicate"), "fsdp")
    assert rules.components.faithfulness_weights.rule == rules.components.optimizer_state.rule


def test_ddp_preset_replicates_params_and_shards_batch():
    rules = from_config("ddp", MESH, TILING)
    assert rules.components.optimizer_state.spec_for(V_AXES) == P(None, None, None)
    assert rules.activations.external.spec_for(("batch", "position", "feature")) == P(
        ("replicate", "fsdp"), None, None
    )
    assert rules.activations.component.spec_for(("batch", "position", "C")) == P(
        ("replicate", "fsdp"), None, "tp"
    )


def test_unlisted_axis_is_replicated():
    rules = from_config("owner", MESH, TILING)
    # an axis name the row does not list -> None (replicated), silently
    assert rules.components.compute_weights.spec_for(("stack", "d_in", "vocab")) == P(
        None, "fsdp", None
    )


def test_rule_validation_unknown_axes_loud_and_per_tensor_uniqueness():
    with pytest.raises(AssertionError, match="unknown mesh axes"):
        PlacedRule(mesh=MESH, label="x", rule={"d_in": "data"})
    # a rule MAY reuse one mesh axis under several semantic names (d_in/d_out -> fsdp:
    # no single tensor carries both) — uniqueness is per-TENSOR, at spec derivation
    row = PlacedRule(mesh=MESH, label="x", rule={"d_in": "fsdp", "d_out": ("fsdp", "tp")})
    assert row.spec_for(("d_in", "vocab")) == P("fsdp", None)
    with pytest.raises(AssertionError, match="mesh axis twice"):
        row.spec_for(("d_in", "d_out"))


def test_target_vector_rows_refuse_implicit_sharded_execution():
    for row in ("normalization", "position_encoding"):
        target = dict(TARGET_ROWS)
        target[row] = {"d_model": "fsdp"}
        table = PlacementTableConfig.model_validate(
            {
                "components": {
                    "optimizer_state": {"stack": "replicate", "d_in": "fsdp"},
                    "compute_weights": {"d_in": "fsdp"},
                    "faithfulness_weights": {"stack": "replicate", "d_in": "fsdp"},
                    "faithfulness_deltas": {"stack": "replicate", "d_out": "fsdp"},
                    "operands": {},
                    "ns_compute": {"stack": "replicate"},
                },
                "ci_fn": CI_ROWS,
                "activations": {"external": {}, "component": {}},
                "target": target,
            }
        )
        with pytest.raises(AssertionError, match=f"target.{row}"):
            from_config(table, MESH, TILING)


def test_shape_validation_divisibility():
    persist = from_config("owner", MESH, TILING).components.optimizer_state
    persist.validate_shape(V_AXES, (32, 4096, 24576))  # 32%4, 4096%8 ok
    with pytest.raises(AssertionError, match="does not tile"):
        persist.validate_shape(V_AXES, (32, 4097, 24576))
    with pytest.raises(AssertionError, match="does not tile"):
        persist.validate_shape(V_AXES, (30, 4096, 24576))  # 30 % 4 != 0


def test_construction_validates_group_shapes():
    # d_in = 65 does not tile the owner persist d assignment (fsdp=8): construction — not
    # some later `.shardings` call — is where the divisibility dies.
    with pytest.raises(AssertionError, match="does not tile"):
        from_config("owner", MESH, _sites({(65, 32, 8): 4}))
    # the C-minor matrix MASTERS carry the C % (tp·replicate) gate (here ÷4), for every
    # group under zero1 — not just the transient faithfulness rows
    with pytest.raises(AssertionError, match="does not tile"):
        from_config("zero1", MESH, _sites({(64, 32, 6): 4}))


def test_describe_prints_rules_and_derived_audit():
    rules = from_config("owner", MESH, TILING)
    out = rules.describe(
        tensors={
            "V[q_proj family]": (rules.components.optimizer_state, V_AXES, (32, 4096, 512)),
            "U[q_proj family]": (rules.components.optimizer_state, U_AXES, (32, 512, 4096)),
        }
    )
    assert "mesh: replicate=4, fsdp=8, tp=1" in out
    assert "components/optimizer_state" in out and "derived placements:" in out
    assert "per-device" in out
    # the audit shares the fail-fast path: a bad shape refuses to print
    with pytest.raises(AssertionError, match="does not tile"):
        rules.describe(tensors={"bad": (rules.components.optimizer_state, V_AXES, (31, 4096, 512))})


def test_duplicate_mesh_axis_within_one_assignment_rejected_statically():
    with pytest.raises(AssertionError, match="repeats a mesh axis"):
        PlacedRule(mesh=MESH, label="x", rule={"d_in": ("fsdp", "fsdp")})


def test_from_config_presets_and_explicit_table():
    with pytest.raises(AssertionError, match="unknown placement preset"):
        from_config("fsdp2", MESH, TILING)
    table = PlacementTableConfig.model_validate(
        {
            "components": {
                "optimizer_state": {"stack": ["replicate", "fsdp"]},
                "compute_weights": {"d_in": "fsdp"},
                "faithfulness_weights": {"stack": ["replicate", "fsdp"]},
                "faithfulness_deltas": {"stack": ["replicate", "fsdp"]},
                "operands": {},
                "ns_compute": {},
            },
            "ci_fn": CI_ROWS,
            "activations": {
                "external": {"batch": ["replicate", "fsdp"]},
                "component": {"batch": ["replicate", "fsdp"]},
            },
            "target": TARGET_ROWS,
        }
    )
    rules = from_config(table, MESH, _sites({(64, 32, 8): 32}))  # 32 tiles ÷(replicate·fsdp)
    # YAML lists arrive as ordered tuples — nested-axis ORDER is semantics
    assert rules.components.optimizer_state.spec_for(V_AXES) == P(("replicate", "fsdp"), None, None)


def test_unconsumed_rule_key_fails_closed():
    # A misspelled axis ('d_inn') dies at the config schema's closed `SemanticAxis`
    # Literal, before any placement construction.
    with pytest.raises(ValidationError, match="d_inn"):
        PlacementTableConfig.model_validate(
            {
                "components": {
                    "optimizer_state": {"stack": ["replicate", "fsdp"], "d_inn": "tp"},
                    "compute_weights": {"d_in": "fsdp"},
                    "faithfulness_weights": {"stack": ["replicate", "fsdp"]},
                    "faithfulness_deltas": {"stack": ["replicate", "fsdp"]},
                    "operands": {},
                    "ns_compute": {},
                },
                "ci_fn": CI_ROWS,
                "activations": {
                    "external": {"batch": ["replicate", "fsdp"]},
                    "component": {"batch": ["replicate", "fsdp"]},
                },
                "target": TARGET_ROWS,
            }
        )
    # 'feature' is a legal SemanticAxis but consumed by no tensor at the optimizer-state
    # row: without the construction-time key check the rule would silently place nothing
    # (exact-name lookup, unlisted-axis-replicates default).
    table = PlacementTableConfig.model_validate(
        {
            "components": {
                "optimizer_state": {"stack": ["replicate", "fsdp"], "feature": "tp"},
                "compute_weights": {"d_in": "fsdp"},
                "faithfulness_weights": {"stack": ["replicate", "fsdp"]},
                "faithfulness_deltas": {"stack": ["replicate", "fsdp"]},
                "operands": {},
                "ns_compute": {},
            },
            "ci_fn": CI_ROWS,
            "activations": {
                "external": {"batch": ["replicate", "fsdp"]},
                "component": {"batch": ["replicate", "fsdp"]},
            },
            "target": TARGET_ROWS,
        }
    )
    with pytest.raises(AssertionError, match="name no semantic axis"):
        from_config(table, MESH, _sites({(64, 32, 8): 32}))


def _chunkwise_arch(attention: GQACIAttention | MHACIAttention) -> ChunkwiseTransformerCIArch:
    return ChunkwiseTransformerCIArch(
        chunks=(Chunk(input_taps=("tap",), output_sites=("site",)),),
        input_dim=64,
        d_model=1024,
        n_blocks=1,
        attention=attention,
        ffn_hidden=128,
        ffn_kind="gelu",
        learned_norm_scale=False,
    )


def test_ci_head_split_refuses_a_non_tiling_kv_head_assignment():
    # GQA K/V heads that don't tile the `kv_head` assignment refuse at resolution — the
    # standard non-tiling message, never a silent feature-replicated fallback.
    mesh = jax.sharding.AbstractMesh((1, 1, 16), ("replicate", "fsdp", "tp"))
    rules = from_config("owner", mesh, _sites({(64, 32, 16): 2}))
    with pytest.raises(AssertionError, match=r"'kv_head' \(dim 8\) does not tile"):
        resolve_ci_placement(_chunkwise_arch(GQACIAttention(n_heads=16, n_kv_heads=8)), rules)
    assert resolve_ci_placement(_chunkwise_arch(MHACIAttention(n_heads=16)), rules) is rules.ci_fn


@pytest.mark.parametrize("tp", (1, 2, 4, 8))
def test_ci_head_split_resolves_at_the_llama_family_head_geometry(tp: int):
    # The 8B HF GLU variants (Llama-3.1 and Qwen3) run 32 query / 8 K/V heads: every tp ≤ 8
    # tiles both counts, so a family-shaped GQA CI arch resolves on those meshes.
    mesh = jax.sharding.AbstractMesh((1, 8 // tp, tp), ("replicate", "fsdp", "tp"))
    rules = from_config("owner", mesh, _sites({(64, 32, 8): 2}))
    arch = _chunkwise_arch(GQACIAttention(n_heads=32, n_kv_heads=8))
    assert resolve_ci_placement(arch, rules) is rules.ci_fn


def test_explicit_table_with_kv_head_unmapped_is_the_authored_kv_replication():
    # Wanting replicated K/V heads is a conscious authored state: an explicit table whose
    # attention/activations rows simply omit `kv_head`. It constructs, K/V tensors derive
    # unsharded on that axis, any K/V head count resolves, and describe() shows the rows.
    mesh = jax.sharding.AbstractMesh((1, 2, 4), ("replicate", "fsdp", "tp"))
    kv_replicated_ci = {
        **CI_ROWS,
        "attention": {
            "optimizer_state": {"d_model": ["fsdp", "replicate"], "q_head": "tp"},
            "compute_weights": {"d_model": "fsdp", "q_head": "tp"},
            "operands": {"q_head": "tp"},
            "ns_compute": {},
        },
        "activations": {
            "batch": ["replicate", "fsdp"],
            "input": "tp",
            "q_head": "tp",
            "ffn_hidden": "tp",
            "C": "tp",
        },
    }
    table = PlacementTableConfig.model_validate(
        {
            "components": _OWNER_TABLE_ROWS,
            "ci_fn": kv_replicated_ci,
            "activations": _ACTIVATION_ROWS,
            "target": TARGET_ROWS,
        }
    )
    rules = from_config(table, mesh, _sites({(64, 32, 8): 2}))
    arch = _chunkwise_arch(GQACIAttention(n_heads=8, n_kv_heads=2))  # 2 does not tile tp=4
    assert resolve_ci_placement(arch, rules) is rules.ci_fn
    kv_axes = ("stack", "kv_head", "d_model")
    assert rules.ci_fn.attention.optimizer_state.spec_for(kv_axes) == P(
        None, None, ("fsdp", "replicate")
    )
    assert rules.ci_fn.attention.operands.spec_for(kv_axes) == P(None, None, None)
    assert rules.ci_fn.activations.spec_for(("batch", "position", "kv_head", "head_dim")) == P(
        ("replicate", "fsdp"), None, None, None
    )
    ci_lines = [
        line
        for line in rules.describe().splitlines()
        if "ci_fn/attention" in line or "ci_fn/activations" in line
    ]
    assert any("q_head->tp" in line for line in ci_lines)
    assert all("kv_head" not in line for line in ci_lines), ci_lines


def test_fsdp_only_table_replicates_parameter_state_over_replicate():
    fsdp_only = {"d_in": "fsdp", "d_out": "fsdp"}
    ci_fsdp_only: dict[str, object] = {
        role: {
            "optimizer_state": {axis: "fsdp"},
            "compute_weights": {axis: "fsdp"},
            "operands": {},
            "ns_compute": {},
        }
        for role, axis in {
            "attention": "d_model",
            "ffn": "ffn_hidden",
            "input": "d_model",
            "output": "d_model",
        }.items()
    }
    ci_fsdp_only["vectors"] = {}
    ci_fsdp_only["activations"] = {"batch": ["replicate", "fsdp"]}
    table = PlacementTableConfig.model_validate(
        {
            "components": {
                "optimizer_state": fsdp_only,
                "compute_weights": fsdp_only,
                "faithfulness_weights": fsdp_only,
                "faithfulness_deltas": {"d_out": "fsdp"},
                "operands": {},
                "ns_compute": {},
            },
            "ci_fn": ci_fsdp_only,
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
    rules = from_config(table, MESH, TILING)

    assert rules.components.optimizer_state.spec_for(V_AXES) == P(None, "fsdp", None)
    assert rules.components.compute_weights.spec_for(U_AXES) == P(None, None, "fsdp")
    assert rules.ci_fn.attention.optimizer_state.spec_for(("stack", "q_head", "d_model")) == P(
        None, None, "fsdp"
    )
    assert rules.ci_fn.ffn.compute_weights.spec_for(("stack", "d_model", "ffn_hidden")) == P(
        None, None, "fsdp"
    )
    assert rules.target.column.persist.spec_for(("d_out", "d_in")) == P(None, "fsdp")
    assert rules.activations.external.spec_for(("batch", "position", "d_model")) == P(
        ("replicate", "fsdp"), None, None
    )


def test_component_linear_and_ci_constraints_are_derived_from_authored_rows():
    mesh = Mesh(
        np.asarray(jax.devices()).reshape(1, 1, 1),
        ("replicate", "fsdp", "tp"),
        axis_types=(AxisType.Explicit,) * 3,
    )
    table = PlacementTableConfig.model_validate(
        {
            "components": {
                "optimizer_state": {"C": "tp"},
                "compute_weights": {"C": "tp"},
                "faithfulness_weights": {"C": "tp"},
                "faithfulness_deltas": {},
                "operands": {"C": "tp"},
                "ns_compute": {},
            },
            "ci_fn": CI_ROWS,
            "activations": {
                "external": {"batch": ["replicate", "fsdp"]},
                "component": {"batch": ["replicate", "fsdp"], "C": "tp"},
            },
            "target": TARGET_ROWS,
        }
    )
    rules = from_config(table, mesh, (SiteSpec("linear", 4, 6, 2, "linear"),))
    v_plan = rules.component_linear_plan(
        ("d_in", "C"),
        ("batch", "feature"),
        ("batch", "C"),
    )
    u_plan = rules.component_linear_plan(
        ("C", "d_out"),
        ("batch", "C"),
        ("batch", "feature"),
    )
    assert (v_plan.input, v_plan.operand, v_plan.output) == (
        P(("replicate", "fsdp"), None),
        P(None, "tp"),
        P(("replicate", "fsdp"), "tp"),
    )
    assert (u_plan.input, u_plan.operand, u_plan.output) == (
        P(("replicate", "fsdp"), "tp"),
        P("tp", None),
        P(("replicate", "fsdp"), None),
    )

    ci_jaxpr = jax.make_jaxpr(lambda x: constrain_component_activation(x, rules))(
        jnp.ones((3, 5, 2))
    )
    ci_constraints = [
        equation.params["dst_sharding"].spec
        for equation in ci_jaxpr.jaxpr.eqns
        if equation.primitive.name == "reshard"
    ]
    assert ci_constraints == [P(("replicate", "fsdp"), None, "tp")]


def test_ci_transformer_derives_megatron_row_and_column_plans():
    mesh = Mesh(
        np.asarray(jax.devices()).reshape(1, 1, 1),
        ("replicate", "fsdp", "tp"),
        axis_types=(AxisType.Explicit,) * 3,
    )
    table = PlacementTableConfig.model_validate(
        {
            "components": {
                "optimizer_state": {"C": "tp"},
                "compute_weights": {"C": "tp"},
                "faithfulness_weights": {"C": "tp"},
                "faithfulness_deltas": {},
                "operands": {"C": "tp"},
                "ns_compute": {},
            },
            "ci_fn": CI_ROWS,
            "activations": {
                "external": {"batch": ["replicate", "fsdp"]},
                "component": {"batch": ["replicate", "fsdp"], "C": "tp"},
            },
            "target": TARGET_ROWS,
        }
    )
    ci = from_config(table, mesh, (SiteSpec("linear", 4, 6, 2, "linear"),)).ci_fn
    ndim = 3  # [batch, position, feature]

    q = ci.linear_plan("attention", ("q_head", "d_model"), ndim, transposed=True)
    kv = ci.linear_plan("attention", ("kv_head", "d_model"), ndim, transposed=True)
    attention_output = ci.linear_plan("attention", ("d_model", "q_head"), ndim, transposed=True)
    ffn_up = ci.linear_plan("ffn", ("d_model", "ffn_hidden"), ndim, transposed=False)
    ffn_down = ci.linear_plan("ffn", ("ffn_hidden", "d_model"), ndim, transposed=False)
    input_projection = ci.linear_plan("input", ("input", "d_model"), ndim, transposed=False)
    output_head = ci.linear_plan("output", ("d_model", "C"), ndim, transposed=False)

    for qkv in (q, kv):
        assert (qkv.input, qkv.operand, qkv.output) == (
            P(("replicate", "fsdp"), None, None),
            P(None, "tp"),
            P(("replicate", "fsdp"), None, "tp"),
        )
    assert (attention_output.input, attention_output.operand, attention_output.output) == (
        P(("replicate", "fsdp"), None, "tp"),
        P("tp", None),
        P(("replicate", "fsdp"), None, None),
    )
    assert (ffn_up.input, ffn_up.operand, ffn_up.output) == (
        P(("replicate", "fsdp"), None, None),
        P(None, "tp"),
        P(("replicate", "fsdp"), None, "tp"),
    )
    assert (ffn_down.input, ffn_down.operand, ffn_down.output) == (
        P(("replicate", "fsdp"), None, "tp"),
        P("tp", None),
        P(("replicate", "fsdp"), None, None),
    )
    assert (input_projection.input, input_projection.operand, input_projection.output) == (
        P(("replicate", "fsdp"), None, "tp"),
        P("tp", None),
        P(("replicate", "fsdp"), None, None),
    )
    assert (output_head.input, output_head.operand, output_head.output) == (
        P(("replicate", "fsdp"), None, None),
        P(None, "tp"),
        P(("replicate", "fsdp"), None, "tp"),
    )


def test_describe_marks_every_execution_row_enforced_and_flags_large_replication():
    rules = from_config("ddp", MESH, TILING)
    big = {"V big": (rules.components.optimizer_state, V_AXES, (32, 4096, 24576))}
    out = rules.describe(tensors=big, not_audited=("ci_fn", "frozen target"))
    # ddp persists replicated: a >10M-elem tensor must be flagged, loudly
    assert "FULLY REPLICATED" in out
    assert "NOT yet enforced" not in out
    assert "components/optimizer_state " in out or "components/optimizer_state\t" in out
    assert "NOT AUDITED" in out and "ci_fn" in out


# ── strict construction: the group census + non-tiling refusals ──────────────


def test_strict_owner_refuses_a_non_tiling_group_at_build():
    # the refusal is the UX: it names every non-tiling group with its stack length,
    # the sharded extent, and both remedies (mesh change or a stack-free placement)
    with pytest.raises(
        AssertionError,
        match=r"128x64x8 \(stacks 1\).*no fallback.*divides 4.*sharding: zero1",
    ):
        from_config("owner", MESH, MIXED)


def test_owner_zero1_is_not_a_preset():
    with pytest.raises(AssertionError, match="unknown placement preset 'owner\\+zero1'"):
        from_config("owner+zero1", MESH, MIXED)


def test_construction_resolves_the_group_census():
    rules = from_config("owner", MESH, TILING)
    assert dict(rules.components.group_stack_lens) == {"64x32x8": 4}
    mixed = from_config("zero1", MESH, MIXED)
    assert dict(mixed.components.group_stack_lens) == {"64x32x8": 4, "128x64x8": 1}


_OWNER_TABLE_ROWS = {
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
_ZERO1_ROW = {"d_in": "fsdp", "d_out": "fsdp", "C": ["tp", "replicate"]}
_ACTIVATION_ROWS = {
    "external": {"batch": ["replicate", "fsdp"]},
    "component": {"batch": ["replicate", "fsdp"], "C": "tp"},
}


def test_explicit_table_stack_sharding_refuses_non_tiling_groups():
    strict = PlacementTableConfig.model_validate(
        {
            "components": _OWNER_TABLE_ROWS,
            "ci_fn": CI_ROWS,
            "activations": _ACTIVATION_ROWS,
            "target": TARGET_ROWS,
        }
    )
    with pytest.raises(AssertionError, match=r"128x64x8 \(stacks 1\).*no fallback"):
        from_config(strict, MESH, MIXED)
    assert from_config(strict, MESH, TILING) is not None


def test_fallback_rows_are_unrepresentable_in_the_schema():
    # mixed per-group state cannot be spelled: the fallback row vocabulary is gone,
    # and the closed schema refuses it at parse — not at construction
    for fallback_rows in (
        {"optimizer_state_fallback": _ZERO1_ROW},
        {
            "faithfulness_weights_fallback": {
                "d_in": "fsdp",
                "d_out": "fsdp",
                "C": ["tp", "replicate"],
            },
            "faithfulness_deltas_fallback": {"d_out": "fsdp", "d_in": ["tp", "replicate"]},
        },
    ):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            PlacementTableConfig.model_validate(
                {
                    "components": _OWNER_TABLE_ROWS | fallback_rows,
                    "ci_fn": CI_ROWS,
                    "activations": _ACTIVATION_ROWS,
                    "target": TARGET_ROWS,
                }
            )


def test_zero1_faithfulness_rows_are_the_master_layout():
    components = from_config("zero1", MESH, MIXED).components
    assert components.faithfulness_weights.spec_for(V_AXES) == P(None, "fsdp", ("tp", "replicate"))
    assert components.faithfulness_weights.spec_for(U_AXES) == P(None, ("tp", "replicate"), "fsdp")
    assert components.faithfulness_deltas.spec_for(("stack", "d_out", "d_in")) == P(
        None, "fsdp", ("tp", "replicate")
    )
    # masters rest in the faithfulness layout: that transition is the identity
    for axes in (V_AXES, U_AXES):
        assert components.optimizer_state.spec_for(
            axes
        ) == components.faithfulness_weights.spec_for(axes)


def test_single_device_construction_tiles_trivially():
    # a consumer re-placing a finished run on one device: every stack length divides 1,
    # so even `owner` constructs over the mixed census
    rules = from_config("owner", SINGLE_DEVICE_MESH, MIXED)
    assert dict(rules.components.group_stack_lens) == {"64x32x8": 4, "128x64x8": 1}


# ── the consumer boundary: validation of the received assignment ─────────────


def _stacks_with_tiling_and_non_tiling_groups():
    """Stacks matching MIXED: one group of 4 (tiles ÷replicate=4) and one of 1."""
    vu = {
        spec.name: (jnp.zeros((spec.d_in, spec.C)), jnp.zeros((spec.C, spec.d_out)))
        for spec in MIXED
    }
    return component_stacks_from_site_arrays(MIXED, vu)


@pytest.mark.multidevice
@pytest.mark.skipif(len(jax.devices()) < 4, reason="requires four local devices")
@pytest.mark.parametrize("preset", ("owner", "zero1"))
def test_component_compute_weight_transition_executes_gather_and_transpose(preset: str):
    mesh = Mesh(
        np.asarray(jax.devices()[:4]).reshape(2, 2, 1),
        ("replicate", "fsdp", "tp"),
        axis_types=(AxisType.Explicit,) * 3,
    )
    sites = _sites({(4, 4, 2): 2})
    rules = from_config(preset, mesh, sites)
    expected = component_stacks_from_site_arrays(
        sites,
        {
            spec.name: (
                jnp.arange(8, dtype=jnp.float32).reshape(4, 2) + i * 16,
                jnp.arange(8, dtype=jnp.float32).reshape(2, 4) + i * 16 + 8,
            )
            for i, spec in enumerate(sites)
        },
    )
    persistent = jax.device_put(expected, component_stacks_shardings(expected, rules))

    with jax.set_mesh(mesh):
        compute_weights = jax.jit(
            lambda x: component_stacks_to_compute_weights(x, rules.components)
        )(persistent)
        gradient = jax.jit(
            jax.grad(
                lambda x: sum(
                    jnp.sum(y * y)
                    for y in jax.tree.leaves(
                        component_stacks_to_compute_weights(x, rules.components)
                    )
                )
            )
        )(persistent)
        faithfulness_weights = jax.jit(
            lambda x: component_stacks_to_faithfulness_weights(x, rules.components)
        )(persistent)
        constrained_deltas = jax.jit(
            lambda x: constrain_faithfulness_deltas(
                {
                    group: jnp.einsum(
                        "gic,gco->goi",
                        vs,
                        us,
                        out_sharding=rules.components.faithfulness_deltas.spec_for(
                            ("stack", "d_out", "d_in")
                        ),
                    )
                    for group, (vs, us) in component_stacks_to_faithfulness_weights(
                        x, rules.components
                    ).stacks.items()
                },
                rules.components,
            )
        )(persistent)
        faithfulness_gradient = jax.jit(
            jax.grad(
                lambda x: sum(
                    jnp.sum(delta * delta)
                    for delta in constrain_faithfulness_deltas(
                        {
                            group: jnp.einsum(
                                "gic,gco->goi",
                                vs,
                                us,
                                out_sharding=rules.components.faithfulness_deltas.spec_for(
                                    ("stack", "d_out", "d_in")
                                ),
                            )
                            for group, (vs, us) in component_stacks_to_faithfulness_weights(
                                x, rules.components
                            ).stacks.items()
                        },
                        rules.components,
                    ).values()
                )
            )
        )(persistent)
        reference_faithfulness_gradient = jax.jit(
            jax.grad(
                lambda x: sum(
                    jnp.sum(jnp.einsum("gic,gco->goi", vs, us) ** 2) for vs, us in x.stacks.values()
                )
            )
        )(expected)

    assert jax.tree.all(
        jax.tree.map(
            lambda x, y: np.array_equal(np.asarray(x), np.asarray(y)), compute_weights, expected
        )
    )
    assert jax.tree.all(
        jax.tree.map(
            lambda x, y: np.array_equal(np.asarray(x), 2 * np.asarray(y)), gradient, expected
        )
    )
    assert jax.tree.all(
        jax.tree.map(
            lambda x, y: jnp.allclose(x, y),
            faithfulness_gradient,
            reference_faithfulness_gradient,
        )
    )
    compute = rules.components.compute_weights
    owner_row = rules.components.optimizer_state
    for _group, (vs, us) in compute_weights.stacks.items():
        assert vs.sharding.spec == P(
            *compute.spec_for(V_AXES), reduced=dropped_mesh_axes(owner_row, compute, V_AXES)
        )
        assert us.sharding.spec == P(
            *compute.spec_for(U_AXES), reduced=dropped_mesh_axes(owner_row, compute, U_AXES)
        )
    row = rules.components.faithfulness_weights
    for group, (vs, us) in faithfulness_weights.stacks.items():
        assert vs.sharding.is_equivalent_to(NamedSharding(mesh, row.spec_for(V_AXES)), vs.ndim)
        assert us.sharding.is_equivalent_to(NamedSharding(mesh, row.spec_for(U_AXES)), us.ndim)
        delta = constrained_deltas[group]
        assert delta.sharding.is_equivalent_to(
            NamedSharding(
                mesh,
                rules.components.faithfulness_deltas.spec_for(("stack", "d_out", "d_in")),
            ),
            delta.ndim,
        )


@pytest.mark.multidevice
@pytest.mark.skipif(len(jax.devices()) < 4, reason="requires four local devices")
def test_zero1_compute_weight_entry_roundtrip_is_gather_shaped_and_exact():
    """The C-minor masters reach the resident compute weights by a minor-axis all-gather
    over `replicate` (on C); the transpose reduction is exact and never lowers as a
    collective-permute or all-to-all (the historical grid-transpose failure mode)."""
    mesh = Mesh(
        np.asarray(jax.devices()[:4]).reshape(2, 2, 1),
        ("replicate", "fsdp", "tp"),
        axis_types=(AxisType.Explicit,) * 3,
    )
    sites = _sites({(4, 4, 2): 2})
    rules = from_config("zero1", mesh, sites)
    stacks = component_stacks_from_site_arrays(
        sites,
        {
            spec.name: (
                jnp.arange(8, dtype=jnp.float32).reshape(4, 2),
                jnp.arange(8, dtype=jnp.float32).reshape(2, 4),
            )
            for spec in sites
        },
    )
    persistent = jax.device_put(stacks, component_stacks_shardings(stacks, rules))

    def loss(x: ComponentStacks) -> jax.Array:
        resident = component_stacks_to_compute_weights(x, rules.components)
        # NOT jnp.stack: stacking reduced-typed scalars mis-transposes (grad x replicate,
        # jax 0.10.1); a python sum keeps the cotangent path per-leaf and exact.
        return sum((jnp.sum(leaf * leaf) for leaf in jax.tree.leaves(resident)), jnp.float32(0))

    with jax.set_mesh(mesh):
        gradient = jax.jit(jax.grad(loss))(persistent)
        hlo = jax.jit(jax.grad(loss)).lower(persistent).compile().as_text()

    assert jax.tree.all(
        jax.tree.map(
            lambda g, m: np.array_equal(np.asarray(g), 2 * np.asarray(m)), gradient, persistent
        )
    )
    assert hlo is not None
    assert "all-to-all" not in hlo
    assert "collective-permute" not in hlo
    assert " all-gather(" in hlo


def _non_tiling_group_stacks_and_rules(mesh: Mesh):
    """One group whose 1-stack cannot tile replicate>1 — placeable under zero1 because
    no zero1 row shards the stack axis — with arange V/U so any row permutation changes
    the logical content."""
    sites = _sites({(8, 4, 8): 1})
    rules = from_config("zero1", mesh, sites)
    stacks = component_stacks_from_site_arrays(
        sites,
        {
            sites[0].name: (
                jnp.arange(64, dtype=jnp.float32).reshape(8, 8),
                jnp.arange(32, dtype=jnp.float32).reshape(8, 4),
            )
        },
    )
    return stacks, rules


@pytest.mark.multidevice
@pytest.mark.skipif(len(jax.devices()) < 4, reason="requires four local devices")
@pytest.mark.parametrize("mesh_shape", ((2, 2, 1), (2, 2, 2), (1, 2, 2)))
def test_faithfulness_weights_preserve_semantic_row_order(mesh_shape: tuple[int, int, int]):
    n_devices = int(np.prod(mesh_shape))
    if len(jax.devices()) < n_devices:
        pytest.skip(f"requires {n_devices} local devices")
    mesh = Mesh(
        np.asarray(jax.devices()[:n_devices]).reshape(*mesh_shape),
        ("replicate", "fsdp", "tp"),
        axis_types=(AxisType.Explicit,) * 3,
    )
    stacks, rules = _non_tiling_group_stacks_and_rules(mesh)
    (group,) = stacks.stacks
    persistent = jax.device_put(stacks, component_stacks_shardings(stacks, rules))

    with jax.set_mesh(mesh):
        weights = jax.jit(lambda x: component_stacks_to_faithfulness_weights(x, rules.components))(
            persistent
        )

    for expected, actual in zip(stacks.stacks[group], weights.stacks[group], strict=True):
        np.testing.assert_array_equal(jax.device_get(actual), jax.device_get(expected))


@pytest.mark.multidevice
@pytest.mark.skipif(len(jax.devices()) < 4, reason="requires four local devices")
def test_matrix_faithfulness_matches_unsharded_reference():
    mesh = Mesh(
        np.asarray(jax.devices()[:4]).reshape(2, 2, 1),
        ("replicate", "fsdp", "tp"),
        axis_types=(AxisType.Explicit,) * 3,
    )
    stacks, rules = _non_tiling_group_stacks_and_rules(mesh)
    (group,) = stacks.stacks
    persistent = jax.device_put(stacks, component_stacks_shardings(stacks, rules))

    def loss(x: ComponentStacks) -> jax.Array:
        weights = component_stacks_to_faithfulness_weights(x, rules.components)
        deltas = constrain_faithfulness_deltas(
            {
                g: jnp.einsum(
                    "gic,gco->goi",
                    vs,
                    us,
                    out_sharding=rules.components.faithfulness_deltas.spec_for(
                        ("stack", "d_out", "d_in")
                    ),
                )
                for g, (vs, us) in weights.stacks.items()
            },
            rules.components,
        )
        return jnp.stack(tuple(jnp.sum(delta**2) for delta in deltas.values())).sum()

    with jax.set_mesh(mesh):
        value, grads = jax.jit(jax.value_and_grad(loss))(persistent)

    def reference_loss(vs: jax.Array, us: jax.Array) -> jax.Array:
        return jnp.sum(jnp.einsum("gic,gco->goi", vs, us) ** 2)

    reference_value, reference_grads = jax.value_and_grad(reference_loss, argnums=(0, 1))(
        *stacks.stacks[group]
    )
    np.testing.assert_allclose(float(value), float(reference_value), rtol=1e-6)
    for actual, expected in zip(grads.stacks[group], reference_grads, strict=True):
        np.testing.assert_allclose(jax.device_get(actual), expected, rtol=1e-5)


@pytest.mark.multidevice
@pytest.mark.skipif(len(jax.devices()) < 4, reason="requires four local devices")
def test_matrix_faithfulness_transition_is_identity_in_hlo():
    """The matrix masters rest in the faithfulness layout, so the transition compiles to
    NO weight collective at all — the only cross-replicate traffic is the delta einsum's
    own C-contraction reduction."""
    mesh = Mesh(
        np.asarray(jax.devices()[:4]).reshape(2, 2, 1),
        ("replicate", "fsdp", "tp"),
        axis_types=(AxisType.Explicit,) * 3,
    )
    stacks, rules = _non_tiling_group_stacks_and_rules(mesh)
    shardings = component_stacks_shardings(stacks, rules)
    persistent = jax.device_put(stacks, shardings)

    def loss(x: ComponentStacks) -> jax.Array:
        weights = component_stacks_to_faithfulness_weights(x, rules.components)
        deltas = constrain_faithfulness_deltas(
            {
                group: jnp.einsum(
                    "gic,gco->goi",
                    vs,
                    us,
                    out_sharding=rules.components.faithfulness_deltas.spec_for(
                        ("stack", "d_out", "d_in")
                    ),
                )
                for group, (vs, us) in weights.stacks.items()
            },
            rules.components,
        )
        return jnp.stack(tuple(jnp.sum(delta**2) for delta in deltas.values())).sum()

    with jax.set_mesh(mesh):
        # grads pinned back to the masters' layout, as the optimizer update requires
        hlo = jax.jit(jax.grad(loss), out_shardings=shardings).lower(persistent).compile().as_text()

    assert hlo is not None
    assert "all-to-all" not in hlo
    assert "collective-permute" not in hlo
    assert "shard_map" not in hlo, "the identity transition must not enter a reshard"
    # the full-rank per-group delta ([g, d_out, d_in] unsharded) this lifecycle exists
    # to prevent must not materialize anywhere in the compiled module
    assert "f32[1,4,8]" not in hlo


@pytest.mark.multidevice
@pytest.mark.skipif(len(jax.devices()) < 4, reason="requires four local devices")
def test_zero1_faithfulness_transition_lowers_without_collective_permute():
    mesh = Mesh(
        np.asarray(jax.devices()[:4]).reshape(2, 2, 1),
        ("replicate", "fsdp", "tp"),
        axis_types=(AxisType.Explicit,) * 3,
    )
    sites = _sites({(4, 4, 2): 2})
    rules = from_config("zero1", mesh, sites)
    stacks = component_stacks_from_site_arrays(
        sites,
        {
            spec.name: (
                jnp.arange(8, dtype=jnp.float32).reshape(4, 2),
                jnp.arange(8, dtype=jnp.float32).reshape(2, 4),
            )
            for spec in sites
        },
    )
    persistent = jax.device_put(stacks, component_stacks_shardings(stacks, rules))

    def loss(x: ComponentStacks) -> jax.Array:
        weights = component_stacks_to_faithfulness_weights(x, rules.components)
        deltas = constrain_faithfulness_deltas(
            {
                group: jnp.einsum(
                    "gic,gco->goi",
                    vs,
                    us,
                    out_sharding=rules.components.faithfulness_deltas.spec_for(
                        ("stack", "d_out", "d_in")
                    ),
                )
                for group, (vs, us) in weights.stacks.items()
            },
            rules.components,
        )
        return jnp.stack(tuple(jnp.sum(delta**2) for delta in deltas.values())).sum()

    with jax.set_mesh(mesh):
        hlo = jax.jit(jax.grad(loss)).lower(persistent).compile().as_text()

    assert hlo is not None
    assert "collective-permute" not in hlo


def test_placement_audit_reports_the_persistence_row():
    stacks = _stacks_with_tiling_and_non_tiling_groups()
    audit = component_stacks_audit(stacks, from_config("zero1", MESH, MIXED))
    labels = {label: row.label for label, (row, _axes, _shape) in audit.items()}
    assert labels == {
        "V 64x32x8": "components/optimizer_state",
        "U 64x32x8": "components/optimizer_state",
        "V 128x64x8": "components/optimizer_state",
        "U 128x64x8": "components/optimizer_state",
    }


def test_boundary_refuses_an_assignment_built_for_other_shape_groups():
    stacks = _stacks_with_tiling_and_non_tiling_groups()
    other = from_config("owner", MESH, TILING)  # missing MIXED's (128, 64, 8) group
    with pytest.raises(AssertionError, match="built for component groups"):
        component_stacks_audit(stacks, other)
    with pytest.raises(AssertionError, match="built for component groups"):
        component_stacks_shardings(stacks, other)


def test_boundary_refuses_an_assignment_with_a_different_stack_length():
    # same semantic groups, different stack length: both tile ÷1 so construction passes and
    # only the boundary can catch the drift
    stacks = _stacks_with_tiling_and_non_tiling_groups()
    shrunk = _sites({(64, 32, 8): 2, (128, 64, 8): 1})
    rules = from_config("owner", SINGLE_DEVICE_MESH, shrunk)
    with pytest.raises(AssertionError, match="expects a 2-stack"):
        component_stacks_audit(stacks, rules)
