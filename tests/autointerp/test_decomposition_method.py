from typing import get_args

from param_decomp.autointerp.schemas import DECOMPOSITION_DESCRIPTIONS, DecompositionMethod


def test_pd_is_only_param_decomp_method_key() -> None:
    method_values = set(get_args(DecompositionMethod))

    assert "pd" in method_values
    assert "param_decomp" not in method_values
    assert "spd" not in method_values
    assert set(DECOMPOSITION_DESCRIPTIONS) == method_values
    assert "PD" in DECOMPOSITION_DESCRIPTIONS["pd"]
