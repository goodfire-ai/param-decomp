"""Pooled-source training with dense-masked MoE, mixers and locality loss."""

import pytest

from param_decomp.tests.targets.qwen36_placement_helpers import multidevice
from param_decomp.tests.targets.qwen36_pooled_source_helpers import (
    assert_placed_pooled_source_train_step,
)


@multidevice
@pytest.mark.multidevice
def test_placed_pooled_sources_mixers() -> None:
    assert_placed_pooled_source_train_step(implementation="dense_masked", tp=1, include_mixers=True)
