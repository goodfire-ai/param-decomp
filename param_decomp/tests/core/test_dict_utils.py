import pytest

from param_decomp.core.dict_utils import dict_safe_update_


def test_collision_rejects_entire_update_even_when_values_agree():
    target = {"existing": 1}

    with pytest.raises(ValueError):
        dict_safe_update_(target, {"new": 3, "existing": 1})

    assert target == {"existing": 1}
