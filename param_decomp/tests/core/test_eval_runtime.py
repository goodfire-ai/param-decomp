from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from param_decomp.core.adversary import PersistentAdversary
from param_decomp.core.ci_fn.implementations.layerwise_mlp import (
    LayerwiseMLPCIFnArch,
    init_layerwise_mlp_ci_fn,
)
from param_decomp.core.components import DenseFactorization, SiteSpec, init_component_stacks
from param_decomp.core.configs import (
    AdamWOptimizerConfig,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    StochasticReconLossConfig,
)
from param_decomp.core.eval_schedule import Every, FirstThenEvery
from param_decomp.core.losses import BatchFrequency
from param_decomp.core.objective import build_objective
from param_decomp.core.run import (
    EvalInvocation,
    Evaluation,
    SharedForwardOperation,
    StandaloneOperation,
    _run_due_evaluation,
    no_batch_contexts,
    shared_forward_operation,
)
from param_decomp.core.run_state import _adamw_optimizer
from param_decomp.core.schedule import ScheduleConfig
from param_decomp.core.train import Decomposition, PDState, PDTrainingState, TrainState


@dataclass(frozen=True)
class Pass:
    step: int


def _decomposition() -> Decomposition[object]:
    sites = (
        SiteSpec(name="site", group="site", factorization=DenseFactorization(d_in=2, d_out=2, C=2)),
    )
    return Decomposition(
        components=init_component_stacks(sites, jax.random.PRNGKey(0)),
        ci_fn=init_layerwise_mlp_ci_fn(
            LayerwiseMLPCIFnArch(
                hidden_dims=(2,), has_position_axis=False, input_names=("site.in",)
            ),
            sites,
            jax.random.PRNGKey(1),
        ),
    )


def _state[Conditioning](
    decomposition: Decomposition[Conditioning], adversaries: dict[str, PersistentAdversary]
) -> PDState[Conditioning]:
    optimizer = _adamw_optimizer(
        AdamWOptimizerConfig(lr_schedule=ScheduleConfig.constant(0.001)), 20
    )
    return TrainState(
        decomposition=decomposition,
        training=PDTrainingState(
            frequency=BatchFrequency(),
            objective=build_objective(
                (
                    FaithfulnessLossConfig(coeff=1.0),
                    ImportanceMinimalityLossConfig(coeff=1.0, gamma=ScheduleConfig.constant(1.0)),
                    StochasticReconLossConfig(coeff=1.0),
                ),
                (
                    SiteSpec(
                        name="site",
                        group="site",
                        factorization=DenseFactorization(d_in=2, d_out=2, C=2),
                    ),
                ),
            ),
            components_opt_state=optimizer.init(eqx.filter(decomposition.components, eqx.is_array)),
            ci_fn_opt_state=optimizer.init(eqx.filter(decomposition.ci_fn, eqx.is_array)),
            adversaries=adversaries,
            step=jnp.zeros((), jnp.int32),
        ),
    )


def test_core_schedules_operations_and_builds_one_pass():
    passes: list[int] = []

    def make_pass(invocation: EvalInvocation[object]) -> Pass:
        passes.append(invocation.now_step)
        return Pass(invocation.now_step)

    evaluation = Evaluation(
        operations=(
            StandaloneOperation(schedule=Every(4), run=lambda p: {"four": p.step}),
            StandaloneOperation(schedule=Every(6), run=lambda p: {"six": p.step}),
        ),
        make_pass=make_pass,
        batch_contexts=no_batch_contexts,
    )
    state = _state(_decomposition(), {})
    assert _run_due_evaluation(evaluation, state, 2) is None
    assert passes == []
    assert _run_due_evaluation(evaluation, state, 4) == {"four": 4}
    assert passes == [4]
    assert _run_due_evaluation(evaluation, state, 12) == {"four": 12, "six": 12}
    assert passes == [4, 12]


def test_core_rejects_eval_output_collisions():
    evaluation = Evaluation(
        operations=(
            StandaloneOperation(schedule=Every(1), run=lambda _p: {"same": 1}),
            StandaloneOperation(schedule=Every(1), run=lambda _p: {"same": 2}),
        ),
        make_pass=lambda invocation: Pass(invocation.now_step),
        batch_contexts=no_batch_contexts,
    )
    with pytest.raises(AssertionError, match="colliding keys"):
        _run_due_evaluation(evaluation, _state(_decomposition(), {}), 1)


def test_first_then_every_schedule_is_explicit():
    evaluation = Evaluation(
        operations=(
            StandaloneOperation(
                schedule=FirstThenEvery(first=2, steps=10),
                run=lambda p: {"slow": p.step},
            ),
        ),
        make_pass=lambda invocation: Pass(invocation.now_step),
        batch_contexts=no_batch_contexts,
    )
    state = _state(_decomposition(), {})
    assert _run_due_evaluation(evaluation, state, 2) == {"slow": 2}
    assert _run_due_evaluation(evaluation, state, 4) is None
    assert _run_due_evaluation(evaluation, state, 10) == {"slow": 10}


def test_step_zero_runs_only_the_operations_that_name_it():
    evaluation = Evaluation(
        operations=(
            StandaloneOperation(schedule=Every(1000), run=lambda p: {"fast": p.step}),
            StandaloneOperation(
                schedule=FirstThenEvery(first=0, steps=5000), run=lambda p: {"slow": p.step}
            ),
        ),
        make_pass=lambda invocation: Pass(invocation.now_step),
        batch_contexts=no_batch_contexts,
    )
    assert _run_due_evaluation(evaluation, _state(_decomposition(), {}), 0) == {"slow": 0}


def test_shared_forward_operations_share_the_contexts_and_fold_in_order():
    produced: list[int] = []

    def batch_contexts(eval_pass: Pass) -> tuple[int, ...]:
        del eval_pass
        produced.append(1)
        return (10, 20, 30)

    def summing(name: str) -> SharedForwardOperation[Pass, int]:
        return shared_forward_operation(
            schedule=Every(1),
            init=lambda: 0,
            update=lambda total, context: total + context,
            finish=lambda eval_pass, total: {name: float(total + eval_pass.step)},
        )

    evaluation = Evaluation(
        operations=(
            summing("a"),
            StandaloneOperation(schedule=Every(1), run=lambda p: {"standalone": p.step}),
            summing("b"),
        ),
        make_pass=lambda invocation: Pass(invocation.now_step),
        batch_contexts=batch_contexts,
    )
    record = _run_due_evaluation(evaluation, _state(_decomposition(), {}), 1)
    # one shared context stream feeds every shared-forward operation; standalone ops see none
    assert produced == [1]
    assert record == {"a": 61.0, "b": 61.0, "standalone": 1}


def test_pass_without_due_shared_forward_operations_skips_the_batch_phase():
    def batch_contexts(_eval_pass: Pass) -> tuple[int, ...]:
        raise AssertionError("no shared-forward operation is due; the batch phase must not run")

    evaluation = Evaluation(
        operations=(
            StandaloneOperation(schedule=Every(1), run=lambda p: {"standalone": p.step}),
            shared_forward_operation(
                schedule=Every(1000),
                init=lambda: 0,
                update=lambda total, context: total + context,
                finish=lambda _eval_pass, total: {"batched": float(total)},
            ),
        ),
        make_pass=lambda invocation: Pass(invocation.now_step),
        batch_contexts=batch_contexts,
    )
    assert _run_due_evaluation(evaluation, _state(_decomposition(), {}), 1) == {"standalone": 1}
