"""Schedule config and value lookup used by `optimize()` and PGD metrics."""

from typing import Literal, Self

import numpy as np
from pydantic import Field, NonNegativeFloat, PositiveFloat, model_validator

from param_decomp.base_config import BaseConfig
from param_decomp.types import Probability


class ScheduleConfig(BaseConfig):
    """Configuration for a schedule with warmup and decay."""

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
    """Get the scheduled value at a given step.

    Handles warmup and decay according to the schedule config.

    Args:
        step: Current step (0-indexed)
        total_steps: Total number of steps
        config: Schedule configuration
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
