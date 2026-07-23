"""The declarative placement layer: derived specs, fail-fast validation, transition costs,
and the build-time per-shape-group assignment + bidirectional claim checks."""

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import PartitionSpec as P

from param_decomp.core.components import SiteSpec, component_stacks_from_sites
from param_decomp.core.configs import PlacementTableConfig
from param_decomp.core.placement import (
    PlacedRule,
    component_stacks_audit,
    component_stacks_shardings,
    from_config,
    from_config_for_consumer,
)

MESH = jax.sharding.AbstractMesh((4, 8, 1), ("replicate", "fsdp", "tp"))
SINGLE_DEVICE_MESH = jax.sharding.AbstractMesh((1, 1, 1), ("replicate", "fsdp", "tp"))
V_AXES = ("stack", "d_in", "C")
U_AXES = ("stack", "C", "d_out")


def _sites(group_sizes: dict[tuple[int, int, int], int]) -> tuple[SiteSpec, ...]:
    """One site set with a `g`-stack shape group per `(d_in, d_out, C): g` entry."""
    return tuple(
        SiteSpec(f"s{d_in}x{d_out}x{c}.{i}", d_in, d_out, c)
        for (d_in, d_out, c), g in group_sizes.items()
        for i in range(g)
    )


# All d dims tile MESH's zero1 assignment (fsdp·replicate = 32) so shape validation never
# masks the assignment/claim behaviour under test.
TILING = _sites({(64, 32, 8): 4})  # 4 tiles ÷replicate=4
MIXED = _sites({(64, 32, 8): 4, (128, 64, 8): 1})  # + a non-tiling group of 1


def test_owner_preset_derives_the_d4_layout():
    rules = from_config("owner", MESH, TILING)
    assert rules.params.persist.spec_for(V_AXES) == P("replicate", "fsdp", "tp")
    assert rules.params.persist.spec_for(U_AXES) == P("replicate", "tp", "fsdp")
    # forward: stack unlisted -> replicated across node-groups, d on fsdp
    assert rules.params.forward.spec_for(V_AXES) == P(None, "fsdp", "tp")
    # owner is STRICT: no ZeRO-1 opt-in row for non-tiling stacks
    assert rules.params.zero1 is None


def test_owner_zero1_preset_is_owner_plus_the_optin_row():
    rules = from_config("owner+zero1", MESH, MIXED)
    owner = from_config("owner", MESH, TILING)
    assert rules.params.persist == owner.params.persist
    assert rules.params.forward == owner.params.forward
    assert rules.activations == owner.activations
    # the opt-in row = intra-matrix ZeRO-1 behind the stack axis; fsdp-major
    # (replicate-major turns the ÷N→÷fsdp reconstruct into a grid-transpose permute —
    # PLACEMENT_DESIGN.md lesson 4)
    assert rules.params.zero1 is not None
    assert rules.params.zero1.spec_for(V_AXES) == P(None, ("fsdp", "replicate"), "tp")
    assert rules.params.zero1.spec_for(U_AXES) == P(None, "tp", ("fsdp", "replicate"))


def test_ddp_preset_replicates_params_and_shards_batch():
    rules = from_config("ddp", MESH, TILING)
    assert rules.params.persist.spec_for(V_AXES) == P(None, None, None)
    assert rules.activations.spec_for(("batch", "seq", "d")) == P(("replicate", "fsdp"), None, None)


def test_unlisted_axis_is_replicated():
    rules = from_config("owner", MESH, TILING)
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
    persist = from_config("owner", MESH, TILING).params.persist
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
    rules = from_config("owner", MESH, TILING)
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
    with pytest.raises(AssertionError, match="unknown placement preset"):
        from_config("fsdp2", MESH, TILING)
    table = PlacementTableConfig.model_validate(
        {
            "params": {
                "persist": {"stack": ["replicate", "fsdp"]},
                "forward": {"d_in": "fsdp"},
            },
            "activations": {"batch": ["replicate", "fsdp"]},
        }
    )
    rules = from_config(table, MESH, _sites({(64, 32, 8): 32}))  # 32 tiles ÷(replicate·fsdp)
    # YAML lists arrive as ordered tuples — nested-axis ORDER is semantics
    assert rules.params.persist.spec_for(V_AXES) == P(("replicate", "fsdp"), None, None)
    # zero1 undeclared -> the strict arm (absence IS strictness)
    assert rules.params.zero1 is None


def test_describe_marks_unenforced_rows_and_flags_large_replication():
    rules = from_config("ddp", MESH, TILING)
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


# ── the build-time assignment + bidirectional claims ─────────────────────────


def test_strict_owner_refuses_a_non_tiling_group_at_build():
    with pytest.raises(AssertionError, match=r"\(d_in=128, d_out=64, C=8\) stacks 1"):
        from_config("owner", MESH, MIXED)


def test_owner_zero1_with_every_group_tiling_is_a_misconfiguration():
    with pytest.raises(AssertionError, match="declared-but-unreachable"):
        from_config("owner+zero1", MESH, TILING)
    # an owner+zero1 config built at a single-device topology: everything tiles ÷1, so
    # the zero1 claim is unexercisable — the message must say so
    with pytest.raises(AssertionError, match="single-device smoke cannot exercise"):
        from_config("owner+zero1", SINGLE_DEVICE_MESH, MIXED)


def test_owner_zero1_satisfied_assigns_only_non_tiling_groups():
    rules = from_config("owner+zero1", MESH, MIXED)
    assert rules.params.groups[(64, 32, 8)].row is rules.params.persist
    assert rules.params.groups[(128, 64, 8)].row is rules.params.zero1
    assert rules.params.groups[(128, 64, 8)].stack_len == 1
    assert rules.params.row_for((64, 32, 8)) is rules.params.persist
    with pytest.raises(AssertionError, match="not in this placement's assignment"):
        rules.params.row_for((1, 2, 3))


_OWNER_TABLE_ROWS = {
    "persist": {"stack": "replicate", "d_in": "fsdp", "d_out": "fsdp", "C": "tp"},
    "forward": {"d_in": "fsdp", "d_out": "fsdp", "C": "tp"},
}
_ZERO1_ROW = {"d_in": ["fsdp", "replicate"], "d_out": ["fsdp", "replicate"], "C": "tp"}


def test_explicit_table_claims_mirror_the_presets():
    strict = PlacementTableConfig.model_validate(
        {"params": _OWNER_TABLE_ROWS, "activations": {"batch": ["replicate", "fsdp"]}}
    )
    with_zero1 = PlacementTableConfig.model_validate(
        {
            "params": _OWNER_TABLE_ROWS | {"zero1": _ZERO1_ROW},
            "activations": {"batch": ["replicate", "fsdp"]},
        }
    )
    with pytest.raises(AssertionError, match="declares no params.zero1"):
        from_config(strict, MESH, MIXED)
    with pytest.raises(AssertionError, match="declared-but-unreachable"):
        from_config(with_zero1, MESH, TILING)
    rules = from_config(with_zero1, MESH, MIXED)
    assert rules.params.groups[(128, 64, 8)].row is rules.params.zero1


def test_zero1_and_ddp_presets_carry_no_claim():
    # no stack sharding -> every group takes persist trivially; nothing to claim
    for name in ("zero1", "ddp"):
        rules = from_config(name, MESH, MIXED)
        assert all(gp.row is rules.params.persist for gp in rules.params.groups.values())


def test_consumer_construction_skips_the_reachability_claim_only():
    # a finished owner+zero1 run re-placed on one device: everything tiles ÷1 -> all
    # persist, and the launch claim must NOT fire
    rules = from_config_for_consumer("owner+zero1", SINGLE_DEVICE_MESH, MIXED)
    assert all(gp.row is rules.params.persist for gp in rules.params.groups.values())
    # the fail-closed direction is unchanged: a non-tiling group still needs the row
    with pytest.raises(AssertionError, match="declares no params.zero1"):
        from_config_for_consumer("owner", MESH, MIXED)


# ── the consumer boundary: validation of the received assignment ─────────────


def _stacks_with_tiling_and_non_tiling_groups():
    """Stacks matching MIXED: one shape group of 4 (tiles ÷replicate=4) and one of 1."""
    vu = {
        spec.name: (jnp.zeros((spec.d_in, spec.C)), jnp.zeros((spec.C, spec.d_out)))
        for spec in MIXED
    }
    return component_stacks_from_sites(vu)


def test_placement_audit_looks_up_the_build_assignment():
    stacks = _stacks_with_tiling_and_non_tiling_groups()
    audit = component_stacks_audit(stacks, from_config("owner+zero1", MESH, MIXED))
    labels = {label: row.label for label, (row, _axes, _shape) in audit.items()}
    assert labels["V (d_in=64, d_out=32, C=8)"] == "params/persist"
    assert labels["V (d_in=128, d_out=64, C=8)"] == "params/persist.zero1"
    assert labels["U (d_in=128, d_out=64, C=8)"] == "params/persist.zero1"


def test_boundary_refuses_an_assignment_built_for_other_shape_groups():
    stacks = _stacks_with_tiling_and_non_tiling_groups()
    other = from_config("owner", MESH, TILING)  # missing MIXED's (128, 64, 8) group
    with pytest.raises(AssertionError, match="built for shape groups"):
        component_stacks_audit(stacks, other)
    with pytest.raises(AssertionError, match="built for shape groups"):
        component_stacks_shardings(stacks, other)


def test_boundary_refuses_an_assignment_with_a_different_stack_length():
    # same shape groups, different stack length: both tile ÷1 so construction passes and
    # only the boundary can catch the drift
    stacks = _stacks_with_tiling_and_non_tiling_groups()
    shrunk = _sites({(64, 32, 8): 2, (128, 64, 8): 1})
    rules = from_config("owner", SINGLE_DEVICE_MESH, shrunk)
    with pytest.raises(AssertionError, match="expects a 2-stack"):
        component_stacks_audit(stacks, rules)


def test_preset_names_match_runtime_config_literal():
    import typing

    from param_decomp.core.configs import RuntimeConfig
    from param_decomp.core.placement import PRESET_NAMES

    ann = RuntimeConfig.model_fields["sharding"].annotation
    literals = [a for a in typing.get_args(ann) if typing.get_origin(a) is typing.Literal]
    assert literals, ann
    assert set(typing.get_args(literals[0])) == set(PRESET_NAMES)
