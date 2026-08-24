from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast

import pytest

from param_decomp.core.eval_schedule import Every, FirstThenEvery
from param_decomp.core.run import (
    EvalInvocation,
    EvalOperation,
    Evaluation,
    _run_due_evaluation,
)
from param_decomp.core.train import TrainState


@dataclass(frozen=True)
class Context:
    step: int


def _dummy_state() -> TrainState:
    # `_run_due_evaluation` pairs `state.decomposition.ci_fn` with the resolved placement;
    # these scheduling tests never evaluate the fn, so a shape-only stand-in suffices.
    return cast(
        TrainState, cast(object, SimpleNamespace(decomposition=SimpleNamespace(ci_fn=None)))
    )


def test_core_schedules_operations_and_builds_one_context():
    contexts: list[int] = []

    def make_context(invocation: EvalInvocation) -> Context:
        contexts.append(invocation.now_step)
        return Context(invocation.now_step)

    evaluation = Evaluation(
        operations=(
            EvalOperation(schedule=Every(4), run=lambda ctx: {"four": ctx.step}),
            EvalOperation(schedule=Every(6), run=lambda ctx: {"six": ctx.step}),
        ),
        make_context=make_context,
    )
    state = _dummy_state()
    assert _run_due_evaluation(evaluation, state, 2, None) is None
    assert contexts == []
    assert _run_due_evaluation(evaluation, state, 4, None) == {"four": 4}
    assert contexts == [4]
    assert _run_due_evaluation(evaluation, state, 12, None) == {"four": 12, "six": 12}
    assert contexts == [4, 12]


def test_core_rejects_eval_output_collisions():
    evaluation = Evaluation(
        operations=(
            EvalOperation(schedule=Every(1), run=lambda _ctx: {"same": 1}),
            EvalOperation(schedule=Every(1), run=lambda _ctx: {"same": 2}),
        ),
        make_context=lambda invocation: Context(invocation.now_step),
    )
    with pytest.raises(AssertionError, match="colliding keys"):
        _run_due_evaluation(evaluation, _dummy_state(), 1, None)


def test_first_then_every_schedule_is_explicit():
    evaluation = Evaluation(
        operations=(
            EvalOperation(
                schedule=FirstThenEvery(first=2, steps=10),
                run=lambda ctx: {"slow": ctx.step},
            ),
        ),
        make_context=lambda invocation: Context(invocation.now_step),
    )
    state = _dummy_state()
    assert _run_due_evaluation(evaluation, state, 2, None) == {"slow": 2}
    assert _run_due_evaluation(evaluation, state, 4, None) is None
    assert _run_due_evaluation(evaluation, state, 10, None) == {"slow": 10}


def test_step_zero_runs_only_the_operations_that_name_it():
    evaluation = Evaluation(
        operations=(
            EvalOperation(schedule=Every(1000), run=lambda ctx: {"fast": ctx.step}),
            EvalOperation(
                schedule=FirstThenEvery(first=0, steps=5000), run=lambda ctx: {"slow": ctx.step}
            ),
        ),
        make_context=lambda invocation: Context(invocation.now_step),
    )
    assert _run_due_evaluation(evaluation, _dummy_state(), 0, None) == {"slow": 0}
