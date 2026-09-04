"""Tests for the knot-based ScheduleConfig and its two evaluators (host + traced)."""

import jax.numpy as jnp
import numpy as np
import pytest
from pydantic import ValidationError

from param_decomp.core.losses import scheduled_value_traced
from param_decomp.core.schedule import Interp, Knot, ScheduleConfig, get_scheduled_value


def sched(
    max_val: float, *points: tuple[float, float] | tuple[float, float, Interp]
) -> ScheduleConfig:
    """`(at, frac)` or `(at, frac, interp)` tuples -> ScheduleConfig."""
    return ScheduleConfig(
        max_val=max_val,
        points=tuple(
            Knot(at=p[0], frac=p[1], interp=p[2] if len(p) > 2 else "linear") for p in points
        ),
    )


class TestValidation:
    def test_bare_float_coerces_to_constant(self):
        for raw in (0.01, 1, "1e-4"):
            config = ScheduleConfig.model_validate(raw)
            assert config.max_val == float(raw)
            assert config.is_constant

    def test_constant_classmethod(self):
        config = ScheduleConfig.constant(0.5)
        assert config.max_val == 0.5 and config.is_constant

    def test_rejects_single_knot(self):
        with pytest.raises(ValidationError, match=">= 2 knots"):
            ScheduleConfig(max_val=1.0, points=(Knot(at=0.0, frac=1.0),))

    def test_rejects_span_not_zero_to_one(self):
        with pytest.raises(ValidationError, match="span"):
            sched(1.0, (0.0, 1.0), (0.9, 0.5))
        with pytest.raises(ValidationError, match="span"):
            sched(1.0, (0.1, 1.0), (1.0, 0.5))

    def test_rejects_non_increasing_positions(self):
        with pytest.raises(ValidationError, match="strictly increase"):
            sched(1.0, (0.0, 1.0), (0.5, 0.5), (0.5, 0.2), (1.0, 0.1))

    def test_rejects_unattained_max(self):
        with pytest.raises(ValidationError, match="max_val"):
            sched(1.0, (0.0, 0.9), (1.0, 0.5))


class TestConstant:
    @pytest.mark.parametrize("step", [0, 50, 99, 100])
    def test_constant_everywhere(self, step: int):
        assert get_scheduled_value(step, 100, ScheduleConfig.constant(0.001)) == 0.001


def retired_value(
    step: int,
    total_steps: int,
    start_val: float,
    warmup_pct: float,
    final_val_frac: float,
    fn_type: str,
) -> float:
    """`get_scheduled_value` as it read before knot schedules — verbatim. This is the only
    remaining executable record of the trajectory the production seats used to run, and it
    is what the migration harness below measures every seat against."""
    warmup_steps = int(total_steps * warmup_pct)
    decay_steps = total_steps - warmup_steps
    if step < warmup_steps:
        return start_val * (step / warmup_steps)
    if decay_steps <= 1:
        return start_val
    progress = (step - warmup_steps) / (decay_steps - 1)
    match fn_type:
        case "constant":
            return start_val
        case "linear":
            return start_val * (final_val_frac + (1 - final_val_frac) * (1 - progress))
        case _:
            cosine = 0.5 * (1 + np.cos(np.pi * progress))
            return start_val * (final_val_frac + (1 - final_val_frac) * cosine)


# Every distinct no-warmup shape the 14 seat YAMLs author, as `(retired form, knot form)`.
# These carry SPEC S20's torch-parity endpoint through unchanged, so they must agree at
# EVERY step — a production anneal silently moving is the regression this pins.
POINTWISE_SEATS = [
    # main + CI-fn LRs (all 14 seats; max_val varies, the shape does not)
    ((7e-05, 0.1, "cosine"), sched(7e-05, (0.0, 1.0), (1.0, 0.1, "cosine"))),
    ((2e-03, 0.1, "cosine"), sched(2e-03, (0.0, 1.0), (1.0, 0.1, "cosine"))),
    ((1.0, 0.01, "linear"), sched(1.0, (0.0, 1.0), (1.0, 0.01))),  # the gamma seats
    ((0.5, 1.0, "constant"), ScheduleConfig.constant(0.5)),  # adv_fraction
]


@pytest.mark.parametrize("total_steps", [1, 2, 10, 100, 200_000])
@pytest.mark.parametrize("retired,migrated", POINTWISE_SEATS)
def test_migrated_seat_is_numerically_unchanged(
    retired: tuple[float, float, str], migrated: ScheduleConfig, total_steps: int
):
    """Each no-warmup seat evaluates to what its retired form did, step for step, over every
    step a run actually trains. (`step == total_steps` is excluded on purpose — see
    `test_final_value_is_held_past_the_end`.)"""
    start_val, final_val_frac, fn_type = retired
    steps = sorted({0, 1, total_steps // 3, total_steps // 2, total_steps - 1})
    for step in (s for s in steps if s >= 0):
        # Algebraically identical, so the only gap is double rounding.
        assert get_scheduled_value(step, total_steps, migrated) == pytest.approx(
            retired_value(step, total_steps, start_val, 0.0, final_val_frac, fn_type),
            rel=1e-14,
            abs=1e-14,
        ), (step, total_steps, retired, migrated)


@pytest.mark.parametrize("total_steps", [2, 10, 100, 200_000])
@pytest.mark.parametrize("retired,migrated", POINTWISE_SEATS)
def test_final_value_is_held_past_the_end(
    retired: tuple[float, float, str], migrated: ScheduleConfig, total_steps: int
):
    """At the one-past-the-end `step == total_steps` an optax count can reach, `t` clamps to
    1.0 and the schedule HOLDS its final value. The retired evaluator instead let `progress`
    run past 1 and extrapolated through the far endpoint (e.g. linear `2.0 -> 0.4` returned
    0.3919 at `step == 100` of 100). No update ever consumed that count, so no seat's
    trajectory moved — but holding is the honest behaviour, and this pins it."""
    _, final_val_frac, _ = retired
    assert get_scheduled_value(total_steps, total_steps, migrated) == pytest.approx(
        migrated.max_val * final_val_frac
    )


# The PPGD source LR is the ONE knowingly non-pointwise migration: the retired form ramped
# over `int(total_steps * 0.025)` whole steps, the knot form over normalized time, so the
# two differ INSIDE the warmup window and nowhere else. Per production seat: its step count
# and the measured max |Δ| relative to `max_val`. The deltas shrink as O(1/steps) — the same
# class SPEC S9/S20 already accept — except the 250-step smoke seat, which is coarse enough
# that one whole warmup step (of six) lands on the wrong side of the ramp.
PPGD_WARMUP_SEATS = [
    (250, 3.62e-02),  # llama8b_full32L_HSDP_b32_dp32_SAVESMOKE — smoke only, not a result seat
    (5_000, 1.99e-04),  # llama8b_l18_b128_cmp32
    (20_000, 4.99e-05),  # the four TMS seats
    (40_000, 2.50e-05),  # llama8b_l18-26_9layer_chunkwise
    (50_000, 2.00e-05),  # resid_mlp_1l
    (100_000, 1.00e-05),  # resid_mlp_2l, resid_mlp_3l
    (200_000, 5.00e-06),  # llama8b_full32L_HSDP_b64_dp64, llama8b_l18_C49k_200k
    (400_000, 2.50e-06),  # pile_llama_simple_mlp-4L, ss_llama_simple_mlp-2L
]


@pytest.mark.parametrize("total_steps,published_delta", PPGD_WARMUP_SEATS)
def test_ppgd_warmup_migration_moves_only_inside_the_warmup_window(
    total_steps: int, published_delta: float
):
    """The 2.5% source-LR warmup's reparameterization stays within its published bound, and
    is EXACTLY zero once the ramp completes — the run's steady state never moved."""
    migrated = sched(0.01, (0.0, 0.0), (0.025, 1.0), (1.0, 1.0))
    warmup_steps = int(total_steps * 0.025)
    worst = 0.0
    for step in range(total_steps):
        delta = (
            abs(
                get_scheduled_value(step, total_steps, migrated)
                - retired_value(step, total_steps, 0.01, 0.025, 1.0, "constant")
            )
            / 0.01
        )
        if step > warmup_steps:
            assert delta == 0.0, (step, warmup_steps, delta)
        worst = max(worst, delta)
    assert worst == pytest.approx(published_delta, rel=1e-2), (total_steps, worst)


class TestWarmup:
    config = sched(0.01, (0.0, 0.0), (0.025, 1.0), (1.0, 1.0))

    def test_starts_at_zero_and_holds_peak(self):
        total = 1000
        assert get_scheduled_value(0, total, self.config) == 0.0
        for step in [100, 500, 999]:
            assert get_scheduled_value(step, total, self.config) == pytest.approx(0.01)

    def test_ramp_is_linear_in_t(self):
        total = 1000
        for step in range(24):
            t = step / (total - 1)
            assert get_scheduled_value(step, total, self.config) == pytest.approx(0.01 * t / 0.025)

    def test_monotone_nondecreasing(self):
        total = 200
        values = [get_scheduled_value(s, total, self.config) for s in range(total)]
        assert all(a <= b + 1e-15 for a, b in zip(values, values[1:], strict=False))


class TestHold:
    config = sched(1.0, (0.0, 1.0), (0.5, 0.2, "hold"), (1.0, 0.2))

    def test_step_function_jumps_at_knot(self):
        total = 101  # t == 0.5 lands exactly on step 50
        assert get_scheduled_value(0, total, self.config) == 1.0
        assert get_scheduled_value(49, total, self.config) == 1.0
        assert get_scheduled_value(50, total, self.config) == pytest.approx(0.2)
        assert get_scheduled_value(100, total, self.config) == pytest.approx(0.2)

    def test_terminal_hold_jumps_at_final_step(self):
        config = sched(1.0, (0.0, 1.0), (1.0, 0.5, "hold"))
        total = 10
        for step in range(total - 1):
            assert get_scheduled_value(step, total, config) == 1.0
        assert get_scheduled_value(total - 1, total, config) == pytest.approx(0.5)


class TestMultiPhase:
    config = sched(1e-3, (0.0, 0.0), (0.05, 1.0), (0.6, 1.0), (1.0, 0.01, "cosine"))

    def test_phases(self):
        total = 1000
        assert get_scheduled_value(0, total, self.config) == 0.0
        plateau = get_scheduled_value(300, total, self.config)
        assert plateau == pytest.approx(1e-3)
        assert get_scheduled_value(999, total, self.config) == pytest.approx(1e-5)
        decay_values = [get_scheduled_value(s, total, self.config) for s in range(600, 1000)]
        assert all(a >= b - 1e-15 for a, b in zip(decay_values, decay_values[1:], strict=False))


class TestEdgeCases:
    def test_single_total_step(self):
        config = sched(1.0, (0.0, 1.0), (1.0, 0.1, "cosine"))
        assert get_scheduled_value(0, 1, config) == 1.0

    def test_step_may_equal_total_steps_and_holds_final(self):
        config = sched(1.0, (0.0, 1.0), (1.0, 0.1))
        assert get_scheduled_value(100, 100, config) == pytest.approx(0.1)

    def test_step_beyond_total_steps_asserts(self):
        with pytest.raises(AssertionError):
            get_scheduled_value(101, 100, ScheduleConfig.constant(1.0))


SCHEDULES = {
    "constant": ScheduleConfig.constant(0.7),
    "cosine_decay": sched(0.7, (0.0, 1.0), (1.0, 0.4, "cosine")),
    "linear_anneal": sched(2.0, (0.0, 1.0), (1.0, 0.2)),
    "warmup_flat": sched(0.01, (0.0, 0.0), (0.025, 1.0), (1.0, 1.0)),
    "warmup_cosine": sched(1e-3, (0.0, 0.0), (0.1, 1.0), (1.0, 0.1, "cosine")),
    "hold_staircase": sched(1.0, (0.0, 1.0), (0.3, 0.5, "hold"), (0.7, 0.25, "hold"), (1.0, 0.25)),
    "plateau_decay": sched(1e-3, (0.0, 0.0), (0.05, 1.0), (0.6, 1.0), (1.0, 0.01, "cosine")),
}


class TestTracedParity:
    """`scheduled_value_traced` matches the host `get_scheduled_value` pointwise."""

    @pytest.mark.parametrize("name", sorted(SCHEDULES))
    @pytest.mark.parametrize("total_steps", [1, 2, 10, 101])
    def test_matches_host_pointwise(self, name: str, total_steps: int):
        config = SCHEDULES[name]
        for step in range(total_steps + 1):
            host = get_scheduled_value(step, total_steps, config)
            traced = float(scheduled_value_traced(jnp.asarray(float(step)), total_steps, config))
            # rel 1e-5: the traced twin runs in fp32 (in-step), the host reference in
            # float64; a steep cosine interval amplifies the fp32 cos rounding.
            assert traced == pytest.approx(host, rel=1e-5, abs=1e-12), (step, config)
