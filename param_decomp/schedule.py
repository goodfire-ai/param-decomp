"""Schedule config and value lookup used by `optimize()` and PGD metrics."""

from typing import Literal, Self

import numpy as np
from pydantic import Field, NonNegativeFloat, PositiveFloat, model_validator

from param_decomp.base_config import BaseConfig, Probability


class ScheduleConfig(BaseConfig):
    """Configuration for a schedule with linear warmup followed by an optional decay.

    Attributes:
        start_val: Starting/peak value reached at the end of warmup.
        warmup_pct: Fraction of total steps spent linearly ramping from 0 to ``start_val``.
        final_val_frac: End value as a fraction of ``start_val``. Must be 1.0 for the
            ``"constant"`` schedule.
        fn_type: Decay function applied after warmup. ``"constant"`` holds ``start_val``;
            ``"linear"`` linearly interpolates to ``final_val_frac * start_val``;
            ``"cosine"`` follows a half-cosine to the same endpoint.
    """

    start_val: PositiveFloat = Field(..., description="Starting/peak value (after warmup)")
    warmup_pct: Probability = Field(
        default=0.0, description="Fraction of total steps for linear warmup"
    )
    final_val_frac: NonNegativeFloat = Field(
        default=1.0,
        description="End value as fraction of start_val.",
    )
    fn_type: Literal["constant", "cosine", "linear"] = Field(
        default="constant", description="Decay function type after warmup"
    )

    @model_validator(mode="after")
    def validate_constant_schedule(self) -> Self:
        if self.fn_type == "constant" and self.final_val_frac != 1.0:
            raise ValueError("constant schedule requires final_val_frac == 1.0")
        return self


def get_scheduled_value(step: int, total_steps: int, config: ScheduleConfig) -> float:
    """Compute the scheduled value at ``step``.

    For ``step < warmup_steps`` the value ramps linearly from 0 to ``config.start_val``.
    After warmup, it follows ``config.fn_type``: ``"constant"`` holds ``start_val``,
    ``"linear"`` decays linearly, and ``"cosine"`` decays along a half-cosine. All decays
    end at ``config.final_val_frac * config.start_val``.

    Args:
        step: Current step, 0-indexed and at most ``total_steps``.
        total_steps: Total number of steps; sets the warmup and decay horizons.
        config: Schedule shape.
    """
    assert step >= 0, f"step must be non-negative, got {step}"
    assert total_steps > 0, f"total_steps must be positive, got {total_steps}"
    assert step <= total_steps, f"step ({step}) cannot exceed total_steps ({total_steps})"

    warmup_steps = int(total_steps * config.warmup_pct)
    decay_steps = total_steps - warmup_steps

    if step < warmup_steps:
        return config.start_val * (step / warmup_steps)

    if decay_steps <= 1:
        return config.start_val

    progress = (step - warmup_steps) / (decay_steps - 1)

    match config.fn_type:
        case "constant":
            return config.start_val
        case "linear":
            multiplier = config.final_val_frac + (1 - config.final_val_frac) * (1 - progress)
            return config.start_val * multiplier
        case "cosine":
            multiplier = config.final_val_frac + (1 - config.final_val_frac) * 0.5 * (
                1 + np.cos(np.pi * progress)
            )
            return config.start_val * multiplier
