"""EMA layout is independent of expert operands and CI execution."""

from dataclasses import asdict, replace
from typing import Any, Literal

import jax
import pytest
from jax.sharding import PartitionSpec as P

from param_decomp.core.ci_fn.implementations.block_selected.placement import preset_rows
from param_decomp.core.components import BlockedFactorization, DenseFactorization, SiteSpec
from param_decomp.core.configs import PlacementTableConfig
from param_decomp.core.dict_utils import FrozenMapping
from param_decomp.core.placement import PRESETS, from_config

_MOE_PRESET = "zero1-replicated-resident-moe"


def _authored_moe_table() -> dict[str, Any]:
    return {**asdict(PRESETS[_MOE_PRESET]), "ci_fn": _MOE_PRESET}


def _sites() -> tuple[SiteSpec, ...]:
    return (
        SiteSpec("dense", DenseFactorization(d_in=8, d_out=8, C=8), "dense"),
        SiteSpec(
            "expert", BlockedFactorization(n_blocks=4, d_in=8, d_out=8, c_per_block=2), "expert"
        ),
    )


@pytest.mark.parametrize("axes", ((), ("tp",)))
def test_ema_uses_the_authored_component_layout(axes: tuple[Literal["tp"], ...]):
    mesh = jax.sharding.AbstractMesh((2, 2), ("data", "tp"))
    rules = from_config("zero1-replicated-resident-moe", mesh, _sites())
    component = replace(
        rules.activations.component, rule=FrozenMapping({"batch": ("data",), "C": axes})
    )
    rules = replace(rules, activations=replace(rules.activations, component=component))
    assert rules.frequency_sharding.spec == P(axes)
    assert preset_rows(rules.ci_fn).expert_head.operands == {"expert": ("tp",)}
    assert rules.components.operands.assignment("expert") == ("tp",)


@pytest.mark.parametrize("internal_axis", ["expert", "C_block"])
def test_blocked_computation_can_differ_from_public_components(
    internal_axis: Literal["expert", "C_block"],
):
    mesh = jax.sharding.AbstractMesh((2, 2), ("data", "tp"))
    authored = _authored_moe_table()
    authored["activations"]["component"] = {"batch": "data", internal_axis: "tp"}
    table = PlacementTableConfig.model_validate(authored)
    rules = from_config(table, mesh, _sites())
    assert rules.activations.component.assignment(internal_axis) == ("tp",)
    assert rules.frequency_sharding.spec == P(None)
    assert preset_rows(rules.ci_fn).expert_head.operands == {"expert": ("tp",)}


def test_component_expert_operands_are_independent_of_the_ci_expert_rows():
    mesh = jax.sharding.AbstractMesh((2, 2), ("data", "tp"))
    table = PlacementTableConfig.model_validate(_authored_moe_table())
    table.components.operands["expert"] = "data"
    rules = from_config(table, mesh, _sites())
    ci_fn_rows = preset_rows(rules.ci_fn)
    assert ci_fn_rows.expert_head.operands == {"expert": ("tp",)}
    assert ci_fn_rows.expert_ffn.operands == {"expert": ("tp",)}
    assert rules.components.operands.assignment("expert") == ("data",)
    assert rules.frequency_sharding.spec == P("tp")
