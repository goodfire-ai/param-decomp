from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast

import pytest

from param_decomp.core.eval_schedule import Every, FirstThenEvery
from param_decomp.core.run import (
    BatchedOperation,
    EvalInvocation,
    Evaluation,
    PassOperation,
    _run_due_evaluation,
    batched_operation,
    no_batch_contexts,
)
from param_decomp.core.train import TrainState


@dataclass(frozen=True)
class Pass:
    step: int


def _dummy_state() -> TrainState:
    """The engine only reads `state.decomposition.ci_fn` to build the invocation."""
    return cast(
        TrainState, cast(object, SimpleNamespace(decomposition=SimpleNamespace(ci_fn=None)))
    )


def test_core_schedules_operations_and_builds_one_pass():
    passes: list[int] = []

    def make_pass(invocation: EvalInvocation) -> Pass:
        passes.append(invocation.now_step)
        return Pass(invocation.now_step)

    evaluation = Evaluation(
        operations=(
            PassOperation(schedule=Every(4), run=lambda p: {"four": p.step}),
            PassOperation(schedule=Every(6), run=lambda p: {"six": p.step}),
        ),
        make_pass=make_pass,
        batch_contexts=no_batch_contexts,
    )
    state = _dummy_state()
    assert _run_due_evaluation(evaluation, state, 2, None) is None
    assert passes == []
    assert _run_due_evaluation(evaluation, state, 4, None) == {"four": 4}
    assert passes == [4]
    assert _run_due_evaluation(evaluation, state, 12, None) == {"four": 12, "six": 12}
    assert passes == [4, 12]


def test_core_rejects_eval_output_collisions():
    evaluation = Evaluation(
        operations=(
            PassOperation(schedule=Every(1), run=lambda _p: {"same": 1}),
            PassOperation(schedule=Every(1), run=lambda _p: {"same": 2}),
        ),
        make_pass=lambda invocation: Pass(invocation.now_step),
        batch_contexts=no_batch_contexts,
    )
    with pytest.raises(AssertionError, match="colliding keys"):
        _run_due_evaluation(evaluation, _dummy_state(), 1, None)


def test_first_then_every_schedule_is_explicit():
    evaluation = Evaluation(
        operations=(
            PassOperation(
                schedule=FirstThenEvery(first=2, steps=10),
                run=lambda p: {"slow": p.step},
            ),
        ),
        make_pass=lambda invocation: Pass(invocation.now_step),
        batch_contexts=no_batch_contexts,
    )
    state = _dummy_state()
    assert _run_due_evaluation(evaluation, state, 2, None) == {"slow": 2}
    assert _run_due_evaluation(evaluation, state, 4, None) is None
    assert _run_due_evaluation(evaluation, state, 10, None) == {"slow": 10}


def test_step_zero_runs_only_the_operations_that_name_it():
    evaluation = Evaluation(
        operations=(
            PassOperation(schedule=Every(1000), run=lambda p: {"fast": p.step}),
            PassOperation(
                schedule=FirstThenEvery(first=0, steps=5000), run=lambda p: {"slow": p.step}
            ),
        ),
        make_pass=lambda invocation: Pass(invocation.now_step),
        batch_contexts=no_batch_contexts,
    )
    assert _run_due_evaluation(evaluation, _dummy_state(), 0, None) == {"slow": 0}


def test_batched_operations_share_the_contexts_and_fold_in_order():
    produced: list[int] = []

    def batch_contexts(eval_pass: Pass) -> tuple[int, ...]:
        del eval_pass
        produced.append(1)
        return (10, 20, 30)

    def summing(name: str) -> BatchedOperation[Pass, int]:
        return batched_operation(
            schedule=Every(1),
            init=lambda: 0,
            update=lambda total, context: total + context,
            finish=lambda eval_pass, total: {name: float(total + eval_pass.step)},
        )

    evaluation = Evaluation(
        operations=(
            summing("a"),
            PassOperation(schedule=Every(1), run=lambda p: {"pass_level": p.step}),
            summing("b"),
        ),
        make_pass=lambda invocation: Pass(invocation.now_step),
        batch_contexts=batch_contexts,
    )
    record = _run_due_evaluation(evaluation, _dummy_state(), 1, None)
    # one shared context stream feeds every batched operation; pass-level ops see none
    assert produced == [1]
    assert record == {"a": 61.0, "b": 61.0, "pass_level": 1}


def test_pass_without_due_batched_operations_skips_the_batch_phase():
    def batch_contexts(_eval_pass: Pass) -> tuple[int, ...]:
        raise AssertionError("no batched operation is due; the batch phase must not run")

    evaluation = Evaluation(
        operations=(
            PassOperation(schedule=Every(1), run=lambda p: {"pass_level": p.step}),
            batched_operation(
                schedule=Every(1000),
                init=lambda: 0,
                update=lambda total, context: total + context,
                finish=lambda _eval_pass, total: {"batched": float(total)},
            ),
        ),
        make_pass=lambda invocation: Pass(invocation.now_step),
        batch_contexts=batch_contexts,
    )
    assert _run_due_evaluation(evaluation, _dummy_state(), 1, None) == {"pass_level": 1}
