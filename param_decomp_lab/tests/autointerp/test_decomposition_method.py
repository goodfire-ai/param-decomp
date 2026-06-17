from typing import get_args

from param_decomp_lab.autointerp.schemas import (
    DECOMPOSITION_DESCRIPTIONS,
    DecompositionMethod,
)


def test_pd_is_only_param_decomp_method_key() -> None:
    method_values = set(get_args(DecompositionMethod))

    assert method_values == {"pd"}
    assert set(DECOMPOSITION_DESCRIPTIONS) == method_values
    assert "PD" in DECOMPOSITION_DESCRIPTIONS["pd"]
