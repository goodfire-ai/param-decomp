from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import jax
import optax
from jaxtyping import Array, Float32, Int32

from param_decomp.core.runtime_schedule import RuntimeSchedule


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ScheduledOptimizerState:
    schedule: RuntimeSchedule
    count: Int32[Array, ""]
    applied_learning_rate: Float32[Array, ""]
    inner_state: optax.OptState


class ScheduledOptimizer(NamedTuple):
    init: Callable[[optax.Params], ScheduledOptimizerState]
    update: Callable[
        [optax.Updates, ScheduledOptimizerState, optax.Params],
        tuple[optax.Updates, ScheduledOptimizerState],
    ]
