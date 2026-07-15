"""The declarative placement layer: derived specs, fail-fast validation, transition costs."""

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import PartitionSpec as P

from param_decomp.placement import PlacementRules, preset

MESH = jax.sharding.AbstractMesh((4, 8, 1), ("replicate", "fsdp", "tp"))
V_AXES = ("stack", "d_in", "C")
U_AXES = ("stack", "C", "d_out")


def test_owner_preset_derives_the_d4_layout():
    rules = preset("owner", MESH)
    assert rules.spec_for("params/persist", V_AXES) == P("replicate", "fsdp", "tp")
    assert rules.spec_for("params/persist", U_AXES) == P("replicate", "tp", "fsdp")
    # subset fallback site = intra-matrix behind the stack axis
    assert rules.spec_for("params/persist.subset", V_AXES) == P(None, ("replicate", "fsdp"), "tp")
    # forward: stack unlisted -> replicated across node-groups, d on fsdp
    assert rules.spec_for("params/forward", V_AXES) == P(None, "fsdp", "tp")


def test_ddp_preset_replicates_params_and_shards_batch():
    rules = preset("ddp", MESH)
    assert rules.spec_for("params/persist", V_AXES) == P(None, None, None)
    assert rules.spec_for("activations", ("batch", "seq", "d")) == P(
        ("replicate", "fsdp"), None, None
    )


def test_unknown_site_is_loud_but_unlisted_axis_is_replicated():
    rules = preset("owner", MESH)
    with pytest.raises(AssertionError, match="no placement rule for site"):
        rules.spec_for("optim/muon.ns", V_AXES)
    # unlisted axis name within a declared site -> None (replicated), silently
    assert rules.spec_for("params/forward", ("stack", "d_in", "novel_axis")) == P(
        None, "fsdp", None
    )


def test_rule_validation_unknown_axes_loud_and_per_tensor_uniqueness():
    with pytest.raises(AssertionError, match="unknown mesh axes"):
        PlacementRules(mesh=MESH, sites={"x": {"a": "nonexistent"}})
    # a rule MAY reuse one mesh axis under several semantic names (d_in/d_out -> fsdp:
    # no single tensor carries both) — uniqueness is per-TENSOR, at spec derivation
    rules = PlacementRules(mesh=MESH, sites={"x": {"a": "fsdp", "b": ("fsdp", "tp")}})
    assert rules.spec_for("x", ("a", "other")) == P("fsdp", None)
    with pytest.raises(AssertionError, match="mesh axis twice"):
        rules.spec_for("x", ("a", "b"))


def test_shape_validation_divisibility():
    rules = preset("owner", MESH)
    rules.validate_shape("params/persist", V_AXES, (32, 4096, 24576))  # 32%4, 4096%8 ok
    with pytest.raises(AssertionError, match="does not tile"):
        rules.validate_shape("params/persist", V_AXES, (32, 4097, 24576))
    with pytest.raises(AssertionError, match="does not tile"):
        rules.validate_shape("params/persist", V_AXES, (30, 4096, 24576))  # 30 % 4 != 0


def test_transition_bytes_zero_iff_specs_match():
    rules = PlacementRules(
        mesh=MESH,
        sites={
            "params/persist": {"stack": "replicate", "d_in": "fsdp"},
            "optim/ns.resident": {"stack": "replicate", "d_in": "fsdp"},
            "optim/ns.spread": {"stack": ("replicate", "fsdp")},
        },
    )
    shape = (32, 4096, 24576)
    resident = rules.transition_bytes(
        "params/persist", "optim/ns.resident", V_AXES, shape, jnp.dtype(jnp.float32)
    )
    spread = rules.transition_bytes(
        "params/persist", "optim/ns.spread", V_AXES, shape, jnp.dtype(jnp.float32)
    )
    assert resident == 0  # owner-resident NS: no reshard
    assert spread == 32 * 4096 * 24576 * 4  # transient re-spread: <= full tensor moves


def test_describe_prints_rules_and_derived_audit():
    rules = preset("owner", MESH)
    out = rules.describe(
        tensors={
            "V[q_proj family]": ("params/persist", V_AXES, (32, 4096, 512)),
            "U[q_proj family]": ("params/persist", U_AXES, (32, 512, 4096)),
        }
    )
    assert "mesh: replicate=4, fsdp=8, tp=1" in out
    assert "params/persist" in out and "derived placements:" in out
    assert "per-device" in out
    # the audit shares the fail-fast path: a bad shape refuses to print
    with pytest.raises(AssertionError, match="does not tile"):
        rules.describe(tensors={"bad": ("params/persist", V_AXES, (31, 4096, 512))})
