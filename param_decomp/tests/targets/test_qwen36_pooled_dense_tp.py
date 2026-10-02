"""Tensor-parallel dense-masked MoE pooled-source training."""

import pytest

from param_decomp.tests.targets.qwen36_placement_helpers import multidevice
from param_decomp.tests.targets.qwen36_pooled_source_helpers import (
    assert_placed_pooled_source_train_step,
)


@multidevice
@pytest.mark.multidevice
def test_placed_pooled_sources_dense_tp() -> None:
    assert_placed_pooled_source_train_step(
        implementation="dense_masked", tp=2, include_mixers=False
    )
