"""The HFU pieces: the closed device-kind enumeration, its fleet check, and the
executed-FLOPs step cost."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest

from param_decomp.core.hardware_utilization import (
    ModelCost,
    StepCost,
    checked_device_kind,
    peak_bf16_dense_flops_per_second,
)


@dataclass(frozen=True)
class _Device:
    device_kind: str


def test_peak_lookup_knows_b200():
    assert peak_bf16_dense_flops_per_second("NVIDIA B200") == 2.25e15


def test_peak_lookup_refuses_cpu():
    with pytest.raises(ValueError, match="CPU has no declared BF16 peak"):
        peak_bf16_dense_flops_per_second("cpu")


def test_checked_device_kind_accepts_a_homogeneous_known_fleet():
    assert checked_device_kind([_Device("NVIDIA H200")] * 8) == "NVIDIA H200"
    assert checked_device_kind(jax.devices()) == "cpu"


def test_checked_device_kind_refuses_unknown_kind_naming_the_remedy():
    """The launch boundary's refusal: the kind, and where its row goes."""
    with pytest.raises(AssertionError, match=r"'TPU v9000'.*peak_bf16_dense_flops_per_second"):
        checked_device_kind([_Device("TPU v9000")])


def test_checked_device_kind_refuses_a_heterogeneous_fleet():
    with pytest.raises(AssertionError, match="heterogeneous"):
        checked_device_kind([_Device("NVIDIA B200"), _Device("NVIDIA H200")])


def test_step_cost_reads_executed_flops_off_the_compiled_step():
    def f(x: jax.Array, y: jax.Array) -> jax.Array:
        return x @ y

    compiled = jax.jit(f).lower(jnp.ones((8, 16)), jnp.ones((16, 4))).compile()
    cost = StepCost.of(compiled, jax.devices()[:1])
    assert cost.flops_per_step == 2 * 8 * 16 * 4
    assert cost.n_devices == 1
    assert cost.hfu(1.0) is None  # cpu: no declared peak


def test_hfu_is_executed_flops_over_peak():
    cost = StepCost(flops_per_step=8 * 2.25e15, n_devices=8, peak_flops_per_device=2.25e15)
    assert cost.hfu(2.0) == 0.5


@pytest.mark.parametrize("step_time", [0.0, -1.0, float("nan"), float("inf")])
def test_mfu_refuses_invalid_measurements(step_time: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        ModelCost(100, 8, "NVIDIA B200").mfu(step_time)


def test_model_cost_normalizes_global_work_once() -> None:
    cost = ModelCost(8 * 2_250_000_000_000_000, 8, "NVIDIA B200")
    assert cost.ideal_step_time_s == 1.0
    assert cost.mfu(2.0) == 0.5
    assert cost.mfu(0.5) == 2.0


def test_model_cost_refuses_cpu_at_construction() -> None:
    with pytest.raises(ValueError, match="CPU has no declared BF16 peak"):
        ModelCost(100, 1, "cpu")


@pytest.mark.parametrize(("flops", "devices"), [(-1, 1), (1, 0), (1, -1)])
def test_model_cost_refuses_invalid_work_or_fleet(flops: int, devices: int) -> None:
    with pytest.raises(ValueError):
        ModelCost(flops, devices, "NVIDIA B200")
