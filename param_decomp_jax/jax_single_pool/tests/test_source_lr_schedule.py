"""`scheduled_lr` (the persistent-source LR) mirrors torch's `get_scheduled_value`
across warmup + constant/linear/cosine decay, per step within fp tol (SPEC S13′)."""

import jax.numpy as jnp
import pytest

from jax_single_pool.losses import scheduled_lr
from param_decomp_config.schedule import ScheduleConfig, get_scheduled_value


@pytest.mark.parametrize("fn_type", ["constant", "linear", "cosine"])
@pytest.mark.parametrize("warmup_pct", [0.0, 0.1])
def test_scheduled_lr_matches_torch(fn_type: str, warmup_pct: float):
    final_val_frac = 1.0 if fn_type == "constant" else 0.1
    schedule = ScheduleConfig(
        start_val=2e-3,
        warmup_pct=warmup_pct,
        final_val_frac=final_val_frac,
        fn_type=fn_type,
    )
    total_steps = 200
    for step in range(total_steps + 1):
        jax_value = float(scheduled_lr(jnp.float32(step), total_steps, schedule))
        torch_value = get_scheduled_value(step, total_steps, schedule)
        assert jax_value == pytest.approx(torch_value, abs=1e-9), (
            f"step {step}: jax {jax_value} != torch {torch_value}"
        )


@pytest.mark.parametrize("fn_type", ["linear", "cosine"])
def test_decaying_schedule_actually_decays(fn_type: str):
    schedule = ScheduleConfig(start_val=2e-3, warmup_pct=0.0, final_val_frac=0.1, fn_type=fn_type)
    total_steps = 200
    first = float(scheduled_lr(jnp.float32(0), total_steps, schedule))
    last = float(scheduled_lr(jnp.float32(total_steps), total_steps, schedule))
    assert first == pytest.approx(2e-3)
    assert last == pytest.approx(2e-4)
    assert last < first
