"""EXECUTED replicate>1 gate: the full train step at an owner/ddp (2,2,2) mesh runs for
real steps and its metric trajectory matches the single-device layout by value (SPEC
D4), with the census asserting the cross-replicate collective placement (weight grads
defer to entry; in-loop cross-replicate collectives bounded to the sanctioned smalls)."""

import jax
import pytest

from param_decomp.targets.invariance_check import check_device_count_invariance

pytestmark = [
    pytest.mark.multidevice,
    pytest.mark.skipif(
        jax.default_backend() != "cpu" or jax.device_count() < 8,
        reason="requires an eight-device CPU topology from make test-multidevice",
    ),
]


@pytest.mark.parametrize("sharding", ("owner", "ddp"))
def test_owner_and_ddp_steps_match_single_device_trajectories(sharding: str):
    # rel widened one reassociation-growth step past the (1, n, 1) arm's envelope:
    # replicate>1 reduces in more orders, and drift grows ~5-10x per executed step.
    check_device_count_invariance(2, (2, 2, 2), sharding, census=True, rel=2e-3)
