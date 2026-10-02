"""The training loop charges asynchronous work and overhead to the right clocks."""

import dataclasses
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from param_decomp.core import run as engine
from param_decomp.core.checkpoint import make_checkpoint_manager
from param_decomp.core.ci_fn.implementations.layerwise_mlp import LayerwiseMLPCIFn
from param_decomp.core.ci_fn.implementations.mlp import SiteMLP
from param_decomp.core.components import ComponentStacks, DenseFactorization, SiteSpec
from param_decomp.core.configs import (
    AdamWOptimizerConfig,
    Cadence,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    KeepLastNCheckpoints,
    PDConfigBase,
    PeriodicCheckpointing,
    StochasticReconLossConfig,
)
from param_decomp.core.eval_schedule import FirstThenEvery
from param_decomp.core.flops.types import UsefulFlops
from param_decomp.core.hardware_utilization import StepCost
from param_decomp.core.losses import BatchFrequency
from param_decomp.core.metrics import LogRecord
from param_decomp.core.objective import build_objective
from param_decomp.core.run_state import _adamw_optimizer
from param_decomp.core.schedule import ScheduleConfig
from param_decomp.core.train import Decomposition, PDState, PDTrainingState, TrainState
from param_decomp.core.training_performance import MfuAccounting


class ManualClock:
    def __init__(self) -> None:
        self.elapsed = 1000.0

    def advance(self, seconds: float) -> None:
        self.elapsed += seconds

    def now(self) -> float:
        return self.elapsed


def _state(step: int) -> PDState[object]:
    matrix = jnp.ones((1, 2, 2))
    components = ComponentStacks(
        stacks={"linear": (matrix, matrix)}, site_stack_indices=(("linear", "linear", 0),)
    )
    ci = LayerwiseMLPCIFn(
        site_mlps={"linear": SiteMLP(weights=[matrix[0]], biases=[jnp.zeros(2)])},
        input_names=("input",),
        output_names=("linear",),
        has_position_axis=False,
    )
    optimizer = _adamw_optimizer(AdamWOptimizerConfig(lr_schedule=ScheduleConfig.constant(0.1)), 4)
    sites = (
        SiteSpec(
            name="linear", group="linear", factorization=DenseFactorization(d_in=2, d_out=2, C=2)
        ),
    )
    return TrainState(
        decomposition=Decomposition(components, ci),
        training=PDTrainingState(
            objective=build_objective(
                (
                    FaithfulnessLossConfig(coeff=1.0),
                    ImportanceMinimalityLossConfig(coeff=1.0, gamma=ScheduleConfig.constant(1.0)),
                    StochasticReconLossConfig(coeff=1.0),
                ),
                sites,
            ),
            frequency=BatchFrequency(),
            components_opt_state=optimizer.init(eqx.filter(components, eqx.is_array)),
            ci_fn_opt_state=optimizer.init(eqx.filter(ci, eqx.is_array)),
            adversaries={},
            step=jnp.asarray(step),
        ),
    )


def _exercise_loop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, first_step: int, mfu_enabled: bool
) -> tuple[list[dict[str, float]], list[tuple[str, int]]]:
    clock = ManualClock()
    events: list[tuple[str, int]] = []
    counted_steps: list[int] = []
    pending_steps = 0
    one = jnp.asarray(1.0)
    state = _state(first_step)
    optimizer_config = AdamWOptimizerConfig(lr_schedule=ScheduleConfig.constant(0.1))
    pd = PDConfigBase(
        components_optimizer=optimizer_config,
        ci_fn_optimizer=optimizer_config,
        steps=4,
        batch_size=1,
    )
    retention = KeepLastNCheckpoints(n=1)
    cadence = Cadence(
        train_log_every=4,
        checkpointing=PeriodicCheckpointing(save_every=3, retention=retention),
    )
    manager = make_checkpoint_manager(tmp_path / "checkpoints", retention)
    optimizer = _adamw_optimizer(optimizer_config, pd.steps)
    prepared = engine._RunResources(
        opt_vu=optimizer,
        opt_ci=optimizer,
        init_key=jax.random.key(1),
        src_key=jax.random.key(2),
        run_key=jax.random.key(0),
        mesh=jax.sharding.Mesh(np.asarray(jax.devices()), ("device",)),
        saver=engine._PeriodicSaver(manager, 3),
        run_dir=tmp_path,
        start=engine.FreshTraining()
        if first_step == 0
        else engine.ResumeTraining(manager, first_step),
        sigterm_consensus=lambda: False,
    )

    def run_step(
        current: PDState[object], step: int
    ) -> tuple[PDState[object], dict[str, jax.Array]]:
        nonlocal pending_steps
        pending_steps += 1
        updated = dataclasses.replace(
            current, training=dataclasses.replace(current.training, step=jnp.asarray(step + 1))
        )
        return updated, {"total": one, "grad_norms/summary/total": one}

    def complete_pending[T](value: T) -> T:
        nonlocal pending_steps
        clock.advance(2.0 * pending_steps)
        pending_steps = 0
        return value

    def evaluate(invocation: engine.EvalInvocation[object]) -> LogRecord:
        assert pending_steps == 0, "Training must finish before evaluation timing begins"
        clock.advance(5.0)
        events.append(("evaluation", invocation.now_step))
        return {"eval/loss": 1.0}

    def save(_manager: object, step: int, _state: PDState[object]) -> None:
        assert pending_steps == 0, "Training must finish before checkpoint timing begins"
        clock.advance(11.0)
        events.append(("checkpoint", step))

    def finish_rendering() -> None:
        clock.advance(7.0)
        events.append(("render", 4))

    evaluation: engine.Evaluation[object, engine.EvalInvocation[object], None] = engine.Evaluation(
        operations=(engine.StandaloneOperation(FirstThenEvery(first=0, steps=2), evaluate),),
        make_pass=lambda invocation: invocation,
        batch_contexts=engine.no_batch_contexts,
    )

    def prepare_evaluation(
        invocation: engine.EvalInvocation[object],
    ) -> engine.Evaluation[object, engine.EvalInvocation[object], None]:
        assert invocation.decomposition is state.decomposition
        clock.advance(100.0)
        events.append(("prepare", invocation.now_step))
        return evaluation

    monkeypatch.setattr(engine, "monotonic", clock.now)
    monkeypatch.setattr(jax, "block_until_ready", complete_pending)
    monkeypatch.setattr(engine, "save_state", save)

    def step_flops(step: int) -> UsefulFlops:
        counted_steps.append(step)
        return UsefulFlops(model=100.0, optimizer=20.0)

    metrics_path = tmp_path / "metrics.jsonl"
    with metrics_path.open("w") as handle:
        sink = engine.MetricsSink(handle, None, "legacy")
        monkeypatch.setattr(sink, "wait_for_renderers", finish_rendering)
        engine._run_loop(
            pd,
            cadence,
            lambda _run_key: engine.EvaluationPlan(prepare_evaluation),
            sink,
            prepared,
            state,
            True,
            run_step,
            StepCost(
                flops_per_step=200.0,
                n_devices=1,
                peak_flops_per_device=100.0 if mfu_enabled else None,
            ),
            None,
            MfuAccounting(step_flops, peak_flops_per_second=100.0) if mfu_enabled else None,
        )
    manager.close()
    assert pending_steps == 0
    assert counted_steps == (list(range(first_step, pd.steps)) if mfu_enabled else [])
    records = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    return records, events


def test_off_cadence_overhead_and_final_checkpoint_use_matching_work_and_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    records, events = _exercise_loop(monkeypatch, tmp_path, first_step=0, mfu_enabled=True)
    assert [row["step"] for row in records] == [0, 2, 4]
    assert events == [
        ("prepare", 0),
        ("checkpoint", 0),
        ("evaluation", 0),
        ("evaluation", 2),
        ("checkpoint", 3),
        ("evaluation", 4),
        ("checkpoint", 4),
        ("render", 4),
    ]
    baseline, intermediate, final = records
    assert baseline["train/perf/runtime_s"] == 5.0
    assert baseline["train/perf/useful_flops"] == 0.0
    assert intermediate["train/perf/runtime_s"] == 14.0
    assert intermediate["train/perf/n_completed_steps"] == 2.0
    assert final["train/perf/runtime_s"] == 52.0
    assert final["train/perf/training_time_s"] == 8.0
    assert final["train/perf/n_window_steps"] == 4.0
    assert final["train/perf/step_time_s"] == 2.0
    assert final["train/perf/step_mfu_without_optimizer"] == 0.5
    assert final["train/perf/step_mfu_with_optimizer"] == 0.6
    assert final["train/perf/overall_mfu_without_optimizer"] == pytest.approx(400 / 5200)
    assert final["train/perf/overall_mfu_with_optimizer"] == pytest.approx(480 / 5200)


def test_resume_starts_an_empty_work_and_time_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    records, events = _exercise_loop(monkeypatch, tmp_path, first_step=2, mfu_enabled=True)
    assert [row["step"] for row in records] == [4]
    assert events[0] == ("prepare", 2)
    assert ("evaluation", 0) not in events
    (final,) = records
    assert final["train/perf/runtime_s"] == 38.0
    assert final["train/perf/training_time_s"] == 4.0
    assert final["train/perf/n_completed_steps"] == 2.0
    assert final["train/perf/useful_model_flops"] == 200.0
    assert final["train/perf/useful_optimizer_flops"] == 40.0
    assert final["train/perf/step_mfu_without_optimizer"] == 0.5
    assert final["train/perf/overall_mfu_without_optimizer"] == pytest.approx(200 / 3800)


@pytest.mark.parametrize(("first_step", "runtime"), [(0, 52.0), (2, 38.0)])
def test_cpu_loop_keeps_timing_without_mfu_accounting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, first_step: int, runtime: float
) -> None:
    records, _ = _exercise_loop(monkeypatch, tmp_path, first_step, mfu_enabled=False)
    final = records[-1]
    assert final["train/perf/runtime_s"] == runtime
    assert final["train/perf/step_time_s"] == 2.0
    assert final["train/perf/n_completed_steps"] == 4 - first_step
    assert all("mfu" not in name and "useful" not in name for record in records for name in record)
