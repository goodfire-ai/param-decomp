"""The declarative placement layer: derived specs, fail-fast validation, transition costs."""

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import PartitionSpec as P

from param_decomp.placement import PlacedRule, from_config, preset

MESH = jax.sharding.AbstractMesh((4, 8, 1), ("replicate", "fsdp", "tp"))
V_AXES = ("stack", "d_in", "C")
U_AXES = ("stack", "C", "d_out")


def test_owner_preset_derives_the_d4_layout():
    rules = preset("owner", MESH)
    assert rules.params.persist.spec_for(V_AXES) == P("replicate", "fsdp", "tp")
    assert rules.params.persist.spec_for(U_AXES) == P("replicate", "tp", "fsdp")
    # forward: stack unlisted -> replicated across node-groups, d on fsdp
    assert rules.params.forward.spec_for(V_AXES) == P(None, "fsdp", "tp")
    # owner is STRICT: no ZeRO-1 opt-in row for non-tiling stacks
    assert rules.params.zero1 is None


def test_owner_zero1_preset_is_owner_plus_the_optin_row():
    rules = preset("owner+zero1", MESH)
    owner = preset("owner", MESH)
    assert rules.params.persist == owner.params.persist
    assert rules.params.forward == owner.params.forward
    assert rules.activations == owner.activations
    # the opt-in row = intra-matrix ZeRO-1 behind the stack axis; fsdp-major (PR #927:
    # replicate-major turns the ÷N→÷fsdp reconstruct into a grid-transpose permute)
    assert rules.params.zero1 is not None
    assert rules.params.zero1.spec_for(V_AXES) == P(None, ("fsdp", "replicate"), "tp")
    assert rules.params.zero1.spec_for(U_AXES) == P(None, "tp", ("fsdp", "replicate"))


def test_ddp_preset_replicates_params_and_shards_batch():
    rules = preset("ddp", MESH)
    assert rules.params.persist.spec_for(V_AXES) == P(None, None, None)
    assert rules.activations.spec_for(("batch", "seq", "d")) == P(("replicate", "fsdp"), None, None)


def test_unlisted_axis_is_replicated():
    rules = preset("owner", MESH)
    # unlisted axis name within a declared row -> None (replicated), silently
    assert rules.params.forward.spec_for(("stack", "d_in", "novel_axis")) == P(None, "fsdp", None)


def test_rule_validation_unknown_axes_loud_and_per_tensor_uniqueness():
    with pytest.raises(AssertionError, match="unknown mesh axes"):
        PlacedRule(mesh=MESH, label="x", rule={"a": "nonexistent"})
    # a rule MAY reuse one mesh axis under several semantic names (d_in/d_out -> fsdp:
    # no single tensor carries both) — uniqueness is per-TENSOR, at spec derivation
    row = PlacedRule(mesh=MESH, label="x", rule={"a": "fsdp", "b": ("fsdp", "tp")})
    assert row.spec_for(("a", "other")) == P("fsdp", None)
    with pytest.raises(AssertionError, match="mesh axis twice"):
        row.spec_for(("a", "b"))


def test_shape_validation_divisibility():
    persist = preset("owner", MESH).params.persist
    persist.validate_shape(V_AXES, (32, 4096, 24576))  # 32%4, 4096%8 ok
    with pytest.raises(AssertionError, match="does not tile"):
        persist.validate_shape(V_AXES, (32, 4097, 24576))
    with pytest.raises(AssertionError, match="does not tile"):
        persist.validate_shape(V_AXES, (30, 4096, 24576))  # 30 % 4 != 0


def test_transition_bytes_zero_iff_specs_match():
    persist = PlacedRule(
        mesh=MESH, label="params/persist", rule={"stack": "replicate", "d_in": "fsdp"}
    )
    ns_resident = PlacedRule(
        mesh=MESH, label="optim/ns.resident", rule={"stack": "replicate", "d_in": "fsdp"}
    )
    ns_spread = PlacedRule(
        mesh=MESH, label="optim/ns.spread", rule={"stack": ("replicate", "fsdp")}
    )
    shape = (32, 4096, 24576)
    assert persist.transition_bytes(ns_resident, V_AXES, shape, jnp.dtype(jnp.float32)) == 0
    # owner-resident NS: no reshard; transient re-spread: <= full tensor moves
    assert (
        persist.transition_bytes(ns_spread, V_AXES, shape, jnp.dtype(jnp.float32))
        == 32 * 4096 * 24576 * 4
    )


def test_describe_prints_rules_and_derived_audit():
    rules = preset("owner", MESH)
    out = rules.describe(
        tensors={
            "V[q_proj family]": (rules.params.persist, V_AXES, (32, 4096, 512)),
            "U[q_proj family]": (rules.params.persist, U_AXES, (32, 512, 4096)),
        }
    )
    assert "mesh: replicate=4, fsdp=8, tp=1" in out
    assert "params/persist" in out and "derived placements:" in out
    assert "per-device" in out
    # the audit shares the fail-fast path: a bad shape refuses to print
    with pytest.raises(AssertionError, match="does not tile"):
        rules.describe(tensors={"bad": (rules.params.persist, V_AXES, (31, 4096, 512))})


def test_duplicate_mesh_axis_within_one_assignment_rejected_statically():
    with pytest.raises(AssertionError, match="repeats a mesh axis"):
        PlacedRule(mesh=MESH, label="x", rule={"a": ("fsdp", "fsdp")})


def test_from_config_presets_and_explicit_table():
    from param_decomp.configs import PlacementTableConfig

    with pytest.raises(AssertionError, match="unknown placement preset"):
        from_config("fsdp2", MESH)
    table = PlacementTableConfig.model_validate(
        {
            "params": {
                "persist": {"stack": ["replicate", "fsdp"]},
                "forward": {"d_in": "fsdp"},
            },
            "activations": {"batch": ["replicate", "fsdp"]},
        }
    )
    rules = from_config(table, MESH)
    # YAML lists arrive as ordered tuples — nested-axis ORDER is semantics
    assert rules.params.persist.spec_for(V_AXES) == P(("replicate", "fsdp"), None, None)
    # zero1 undeclared -> the strict arm (absence IS strictness)
    assert rules.params.zero1 is None


def test_describe_marks_unenforced_rows_and_flags_large_replication():
    rules = preset("ddp", MESH)
    big = {"V big": (rules.params.persist, V_AXES, (32, 4096, 24576))}
    out = rules.describe(tensors=big, not_audited=("ci_fn", "frozen target"))
    # ddp persists replicated: a >10M-elem tensor must be flagged, loudly
    assert "FULLY REPLICATED" in out
    # rows nobody consumes yet must say so
    assert out.count("NOT yet enforced") >= 2  # activations + params/forward
    assert "params/persist " in out or "params/persist\t" in out
    # enforced rows must NOT carry the mark
    for line in out.splitlines():
        if line.strip().startswith("params/persist "):
            assert "NOT yet enforced" not in line
    assert "NOT AUDITED" in out and "ci_fn" in out


def _stacks_with_tiling_and_non_tiling_groups():
    """One shape group of 4 (tiles ÷replicate=4) and one of 1 (does not)."""
    from param_decomp.components import component_stacks_from_sites

    vu = {f"layers.{i}.mlp.gate_proj": (jnp.zeros((8, 3)), jnp.zeros((3, 8))) for i in range(4)}
    vu["layers.18.mlp.down_proj"] = (jnp.zeros((16, 3)), jnp.zeros((3, 16)))
    return component_stacks_from_sites(vu)


def test_strict_owner_fails_closed_on_a_non_tiling_group():
    stacks = _stacks_with_tiling_and_non_tiling_groups()
    with pytest.raises(AssertionError, match="declares no params.zero1"):
        stacks.placement_audit(preset("owner", MESH))


def test_owner_zero1_routes_only_non_tiling_groups_to_the_optin_row():
    stacks = _stacks_with_tiling_and_non_tiling_groups()
    audit = stacks.placement_audit(preset("owner+zero1", MESH))
    labels = {label: row.label for label, (row, _axes, _shape) in audit.items()}
    assert labels["V (d_in=8, d_out=8, C=3)"] == "params/persist"
    assert labels["V (d_in=16, d_out=16, C=3)"] == "params/persist.zero1"
    assert labels["U (d_in=16, d_out=16, C=3)"] == "params/persist.zero1"


def test_preset_names_match_runtime_config_literal():
    import typing

    from param_decomp.configs import RuntimeConfig
    from param_decomp.placement import PRESET_NAMES

    ann = RuntimeConfig.model_fields["sharding"].annotation
    literals = [a for a in typing.get_args(ann) if typing.get_origin(a) is typing.Literal]
    assert literals, ann
    assert set(typing.get_args(literals[0])) == set(PRESET_NAMES)
