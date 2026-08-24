"""The one schedule surface: `ScheduleConfig` — a knot-based piecewise curve — plus
`get_scheduled_value`, its host-numpy evaluator (the parity reference). Every scheduled
quantity (main LRs, PPGD source LR, imp-min `p`/`gamma`, nonlinearity threshold,
merged-loss `adv_fraction`, every loss coefficient via `configs.LossCoeff`) is configured by `ScheduleConfig` and
evaluated in-step by the jnp twin `losses.scheduled_value_traced` (jax lives there so
this module — imported by the config schema — stays jax-free); `test_schedule.py` pins
the pair pointwise."""

from typing import Literal, Self

import numpy as np
from pydantic import PositiveFloat, model_validator

from param_decomp.core.base_config import BaseConfig, Probability

Interp = Literal["linear", "cosine", "hold"]


class Knot(BaseConfig):
    """One `(at, frac)` point on the curve. `interp` is how the value travels FROM the
    previous knot to this one — `hold` keeps the previous knot's value, then jumps to
    `frac` exactly at `at`. Ignored on the first knot."""

    at: Probability
    frac: Probability
    interp: Interp = "linear"


class ScheduleConfig(BaseConfig):
    """A piecewise curve `step -> max_val * frac(t)` over normalized run time
    `t = step / (total_steps - 1)`, so the `at = 1.0` knot lands exactly ON the final
    step (the torch-parity convention, SPEC S20). `max_val` is the sweepable magnitude;
    the knots are the shape (`frac` in `[0, 1]`, attained at least once so `max_val` is
    honest). A bare float parses as the constant schedule at that value."""

    max_val: PositiveFloat
    points: tuple[Knot, ...]

    @model_validator(mode="before")
    @classmethod
    def _coerce_bare_float_to_constant(cls, data: object) -> object:
        if isinstance(data, str):
            data = float(data)  # YAML parses dotless scientific notation (`1e-4`) as a string
        if isinstance(data, int | float) and not isinstance(data, bool):
            return {
                "max_val": data,
                "points": ({"at": 0.0, "frac": 1.0}, {"at": 1.0, "frac": 1.0}),
            }
        return data

    @model_validator(mode="after")
    def _validate_points(self) -> Self:
        ats = [k.at for k in self.points]
        assert len(self.points) >= 2, f"a schedule needs >= 2 knots (first at 0, last at 1): {ats}"
        assert ats[0] == 0.0 and ats[-1] == 1.0, f"knots must span at=0 .. at=1, got {ats}"
        assert all(a < b for a, b in zip(ats, ats[1:], strict=False)), (
            f"knot positions must strictly increase, got {ats}"
        )
        assert any(k.frac == 1.0 for k in self.points), (
            "no knot attains frac=1.0 — max_val would overstate the curve's peak; "
            "rescale max_val so the peak knot is frac=1.0"
        )
        return self

    @classmethod
    def constant(cls, value: float) -> "ScheduleConfig":
        """The constant schedule at `value` — what a bare float parses to. For Python
        call sites; YAML writes the bare float directly."""
        return cls.model_validate(value)

    @property
    def is_constant(self) -> bool:
        return all(k.frac == 1.0 for k in self.points)


def _interval_frac_host(prev: Knot, knot: Knot, u: float) -> float:
    match knot.interp:
        case "linear":
            return prev.frac + (knot.frac - prev.frac) * u
        case "cosine":
            return prev.frac + (knot.frac - prev.frac) * 0.5 * (1 - float(np.cos(np.pi * u)))
        case "hold":
            return knot.frac if u >= 1.0 else prev.frac


def get_scheduled_value(step: int, total_steps: int, config: ScheduleConfig) -> float:
    """Compute the scheduled value at `step` (0-indexed, must be `<= total_steps`).

    `t` clamps to 1.0 at `step >= total_steps - 1`, so the one-past-the-end `step ==
    total_steps` an optax count can reach holds the final value. A knot's `at` owns the
    instant it names: at exactly `t == at_i` the value is knot i's `frac` (which is what
    makes `hold` a step function that jumps AT its knot)."""
    assert step >= 0, f"step must be non-negative, got {step}"
    assert total_steps > 0, f"total_steps must be positive, got {total_steps}"
    assert step <= total_steps, f"step ({step}) cannot exceed total_steps ({total_steps})"

    t = min(step / (total_steps - 1), 1.0) if total_steps > 1 else 0.0
    for prev, knot in zip(config.points, config.points[1:], strict=False):
        is_last = knot is config.points[-1]
        if t < knot.at or is_last:
            u = (t - prev.at) / (knot.at - prev.at)
            return config.max_val * _interval_frac_host(prev, knot, u)
    raise AssertionError("unreachable: the last interval owns every t <= 1.0")
