"""When an eval operation fires: the cadence vocabulary the engine dispatches on.

Pure value types, deliberately free of jax and of the engine that consumes them, so the
config layer can name a schedule (`experiments.eval_config.schedule_for`) without dragging
the trainer — and matplotlib, and optax — into every config parse.
"""

import dataclasses


@dataclasses.dataclass(frozen=True)
class Every:
    steps: int


@dataclasses.dataclass(frozen=True)
class FirstThenEvery:
    first: int
    steps: int


type EvalSchedule = Every | FirstThenEvery


def eval_due(schedule: EvalSchedule, step: int) -> bool:
    """Step 0 — the untrained baseline — is due only for a schedule naming it as `first`.
    Every cadence divides 0, so modular arithmetic alone would fire every operation there."""
    match schedule:
        case Every(steps):
            return step > 0 and step % steps == 0
        case FirstThenEvery(first, steps):
            return step == first or (step > 0 and step % steps == 0)
