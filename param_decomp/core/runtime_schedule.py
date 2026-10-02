"""Schedule magnitudes carried as JIT inputs, separate from their static curves."""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float32, Int32, jaxtyped

from param_decomp.core.configs import LossCoeff
from param_decomp.core.schedule import Knot, ScheduleConfig


def _interval_frac_traced(prev: Knot, knot: Knot, t: Float32[Array, ""]) -> Float32[Array, ""]:
    u = (t - prev.at) / (knot.at - prev.at)
    match knot.interp:
        case "linear":
            return prev.frac + (knot.frac - prev.frac) * u
        case "cosine":
            return prev.frac + (knot.frac - prev.frac) * 0.5 * (1 - jnp.cos(jnp.pi * u))
        case "hold":
            return jnp.where(u >= 1.0, jnp.float32(knot.frac), jnp.float32(prev.frac))


@jaxtyped(typechecker=beartype)
def schedule_fraction_at(
    train_frac: Float32[Array, ""], points: tuple[Knot, ...]
) -> Float32[Array, ""]:
    """Interpolate a curve independently of its magnitude."""
    assert len(points) >= 2
    assert all(prev.at < knot.at for prev, knot in zip(points, points[1:], strict=False))
    frac = _interval_frac_traced(points[0], points[1], train_frac)
    for prev, knot in zip(points[1:], points[2:], strict=False):
        frac = jnp.where(train_frac >= prev.at, _interval_frac_traced(prev, knot, train_frac), frac)
    return frac


@jaxtyped(typechecker=beartype)
def scheduled_value_at(
    train_frac: Float32[Array, ""], config: ScheduleConfig
) -> Float32[Array, ""]:
    """Evaluate a schedule at one traced fraction of the run."""
    return config.max_val * schedule_fraction_at(train_frac, config.points)


def _train_frac_from_float_step(step: Float32[Array, ""], total_steps: int) -> Float32[Array, ""]:
    assert total_steps > 0, f"total_steps must be positive, got {total_steps}"
    if total_steps == 1:
        return jnp.zeros((), jnp.float32)
    return step / jnp.asarray(total_steps - 1, jnp.float32)


@jaxtyped(typechecker=beartype)
def train_frac_at(step: Int32[Array, ""], total_steps: int) -> Float32[Array, ""]:
    """Map the integer training counter to fraction-time."""
    return _train_frac_from_float_step(step.astype(jnp.float32), total_steps)


@jaxtyped(typechecker=beartype)
def scheduled_value_traced(
    step_f32: Float32[Array, ""], total_steps: int, config: ScheduleConfig
) -> Float32[Array, ""]:
    """Evaluate at a floating-point step, holding the final value past the last update."""
    train_frac = jnp.minimum(_train_frac_from_float_step(step_f32, total_steps), 1.0)
    return scheduled_value_at(train_frac, config)


class RuntimeSchedule(eqx.Module):
    magnitude: Float32[Array, ""]
    points: tuple[Knot, ...] | None = eqx.field(static=True)

    def __check_init__(self) -> None:
        assert self.magnitude.shape == (), self.magnitude.shape
        assert self.magnitude.dtype == jnp.float32, self.magnitude.dtype
        if self.points is not None:
            ScheduleConfig(max_val=1.0, points=self.points)

    @classmethod
    def from_coeff(cls, coeff: LossCoeff) -> "RuntimeSchedule":
        match coeff:
            case ScheduleConfig():
                return cls(jnp.asarray(coeff.max_val, jnp.float32), coeff.points)
            case float() | int():
                return cls(jnp.asarray(coeff, jnp.float32), None)

    @jaxtyped(typechecker=beartype)
    def at(self, train_frac: Float32[Array, ""]) -> Float32[Array, ""]:
        match self.points:
            case None:
                return self.magnitude
            case tuple():
                return self.magnitude * schedule_fraction_at(train_frac, self.points)
