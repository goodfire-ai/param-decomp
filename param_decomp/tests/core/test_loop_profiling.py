from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import nvtx
import orbax.checkpoint as ocp
import pytest

from param_decomp.core import run
from param_decomp.core.configs import (
    AdamWOptimizerConfig,
    Cadence,
    KeepAllCheckpoints,
    PDConfigBase,
    PeriodicCheckpointing,
)
from param_decomp.core.eval_schedule import FirstThenEvery
from param_decomp.core.hardware_utilization import StepCost
from param_decomp.core.run_state import _adamw_optimizer
from param_decomp.core.schedule import ScheduleConfig
from param_decomp.core.sharding import hsdp_mesh
from param_decomp.core.train import PDState
from param_decomp.tests.core.test_eval_runtime import _decomposition, _state


@pytest.mark.parametrize(
    ("profiling", "is_main", "start_step", "executed", "annotated", "blocked", "emitted", "saved"),
    [
        (
            None,
            True,
            0,
            [0, 1, 2, 3, 4, 5],
            [],
            [0, 1, 2, 3, 4, 5],
            [0, 1, 2, 3, 4, 5, 6],
            [0, 1, 2, 3, 4, 5, 6],
        ),
        (
            run.NsightCaptureWindow(1, 2),
            True,
            0,
            [0, 1, 2, 3, 4, 5],
            [1, 2],
            [0, 1, 1, 2, 2, 3, 4, 5],
            [0, 1, 2, 3, 4, 5, 6],
            [0, 1, 2, 3, 4, 5, 6],
        ),
        (
            run.NsightCaptureWindow(1, 2),
            True,
            3,
            [3, 4, 5, 6, 7, 8],
            [4, 5],
            [3, 4, 4, 5, 5, 6, 7, 8],
            [4, 5, 6, 7, 8, 9],
            [4, 5, 6, 7, 8, 9],
        ),
        (
            run.JaxProfilerTrace(2),
            True,
            0,
            [0, 1, 2, 3],
            [2, 3],
            [0, 0, 1, 1, 2, 3],
            [0, 1, 2],
            [1, 2],
        ),
        (
            run.JaxProfilerTrace(2),
            False,
            0,
            [0, 1, 2, 3],
            [2, 3],
            [0, 0, 1, 1, 2, 3],
            [0, 1, 2],
            [1, 2],
        ),
        (
            run.JaxProfilerTrace(2),
            True,
            3,
            [3, 4, 5, 6],
            [5, 6],
            [3, 3, 4, 4, 5, 6],
            [4, 5],
            [4, 5],
        ),
    ],
    ids=["training", "nsight", "nsight-resume", "jax-main", "jax-worker", "jax-resume"],
)
def test_profiling_preserves_training_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profiling: run.ProfilingMode | None,
    is_main: bool,
    start_step: int,
    executed: list[int],
    annotated: list[int],
    blocked: list[int],
    emitted: list[int],
    saved: list[int],
):
    optimizer_config = AdamWOptimizerConfig(lr_schedule=ScheduleConfig.constant(0.001))
    pd = PDConfigBase(
        components_optimizer=optimizer_config,
        ci_fn_optimizer=optimizer_config,
        steps=start_step + 6,
        batch_size=1,
    )
    optimizer = _adamw_optimizer(optimizer_config, pd.steps)
    manager = Mock(spec=ocp.CheckpointManager)
    resources = run._RunResources(
        opt_vu=optimizer,
        opt_ci=optimizer,
        init_key=jax.random.PRNGKey(0),
        src_key=jax.random.PRNGKey(1),
        run_key=jax.random.PRNGKey(2),
        mesh=hsdp_mesh(1, 1, 1),
        saver=run._PeriodicSaver(manager, 1),
        run_dir=tmp_path,
        start=run.FreshTraining() if start_step == 0 else run.ResumeTraining(manager, start_step),
        sigterm_consensus=lambda: False,
    )
    state = _state(_decomposition(), {})
    state = replace(state, training=replace(state.training, step=jnp.asarray(start_step)))
    events: list[tuple[str, int]] = []

    def run_step(state: PDState[object], step: int) -> tuple[PDState[object], dict[str, jax.Array]]:
        assert int(state.training.step) == step
        events.append(("step", step))
        return replace(state, training=replace(state.training, step=state.training.step + 1)), {
            "total": jnp.asarray(step, dtype=jnp.float32),
            "grad_norms/summary/components": jnp.asarray(1.0),
        }

    @contextmanager
    def jax_annotation(name: str, *, step_num: int) -> Iterator[None]:
        assert name == "param_decomp.profile_step"
        events.append(("jax-enter", step_num))
        yield
        events.append(("jax-exit", step_num))

    @contextmanager
    def nsight_annotation(name: str, *, domain: str, payload: int) -> Iterator[None]:
        assert (name, domain) == ("param_decomp.profile_step", "param_decomp")
        events.append(("nsight-enter", payload))
        yield
        events.append(("nsight-exit", payload))

    block_until_ready = jax.block_until_ready

    def block(
        value: jax.Array | tuple[PDState[object], dict[str, jax.Array]],
    ) -> jax.Array | tuple[PDState[object], dict[str, jax.Array]]:
        loss = value[1]["total"] if isinstance(value, tuple) else value
        events.append(("block", int(loss)))
        return block_until_ready(value)

    def evaluate(invocation: run.EvalInvocation[object]) -> int:
        events.append(("eval", invocation.now_step))
        return invocation.now_step

    sink = Mock(spec=run.MetricsSink)
    save = Mock()
    start_trace = Mock(side_effect=lambda *args, **kwargs: events.append(("start-trace", -1)))
    stop_trace = Mock(side_effect=lambda: events.append(("stop-trace", -1)))
    monkeypatch.setattr(run, "save_state", save)
    monkeypatch.setattr(jax, "block_until_ready", block)
    monkeypatch.setattr(jax.profiler, "TraceAnnotation", jax_annotation)
    monkeypatch.setattr(jax.profiler, "start_trace", start_trace)
    monkeypatch.setattr(jax.profiler, "stop_trace", stop_trace)
    monkeypatch.setattr(nvtx, "annotate", nsight_annotation)

    run._run_loop(
        pd,
        Cadence(
            train_log_every=2,
            checkpointing=PeriodicCheckpointing(save_every=1, retention=KeepAllCheckpoints()),
        ),
        lambda _run_key: run.EvaluationPlan(
            lambda _example: run.Evaluation(
                operations=(
                    run.StandaloneOperation(
                        schedule=FirstThenEvery(first=0, steps=1),
                        run=lambda step: {"eval/step": float(step)},
                    ),
                ),
                make_pass=evaluate,
                batch_contexts=run.no_batch_contexts,
            )
        ),
        sink,
        resources,
        state,
        is_main,
        run_step,
        StepCost(flops_per_step=1.0, n_devices=1, peak_flops_per_device=None),
        profiling,
        mfu_accounting=None,
    )

    assert [step for event, step in events if event == "step"] == executed
    assert [step for event, step in events if event == "block"] == blocked
    assert [step for event, step in events if event == "eval"] == emitted
    assert [call.args[0] for call in sink.log.call_args_list] == emitted
    assert [call.args[1] for call in save.call_args_list] == saved
    for call in save.call_args_list:
        assert int(call.args[2].training.step) == call.args[1]
    for call in sink.log.call_args_list:
        step, record = call.args
        assert ("total" in record) == (step > 0 and (step % 2 == 0 or step == pd.steps))

    match profiling:
        case run.JaxProfilerTrace():
            assert [step for event, step in events if event == "jax-enter"] == annotated
            assert not any(event.startswith("nsight") for event, _ in events)
            if is_main:
                start_trace.assert_called_once()
                assert start_trace.call_args.args == (str(tmp_path / "profile"),)
                stop_trace.assert_called_once_with()
                assert events.index(("start-trace", -1)) < events.index(("jax-enter", annotated[0]))
                assert events.index(("stop-trace", -1)) > events.index(("jax-exit", annotated[-1]))
            else:
                start_trace.assert_not_called()
                stop_trace.assert_not_called()
            annotation_kind = "jax"
            with pytest.raises(AssertionError, match="only 3 steps remain"):
                run._run_loop(
                    pd.model_copy(update={"steps": start_step + 3}),
                    Cadence(
                        train_log_every=2,
                        checkpointing=PeriodicCheckpointing(
                            save_every=1, retention=KeepAllCheckpoints()
                        ),
                    ),
                    None,
                    sink,
                    resources,
                    state,
                    is_main,
                    run_step,
                    StepCost(flops_per_step=1.0, n_devices=1, peak_flops_per_device=None),
                    profiling,
                    mfu_accounting=None,
                )
            assert [step for event, step in events if event == "step"] == executed
        case run.NsightCaptureWindow() | None:
            assert [step for event, step in events if event == "nsight-enter"] == annotated
            assert not any(event.startswith("jax") for event, _ in events)
            start_trace.assert_not_called()
            stop_trace.assert_not_called()
            annotation_kind = "nsight"
    for step in annotated:
        first = events.index((f"{annotation_kind}-enter", step))
        assert events[first : first + 4] == [
            (f"{annotation_kind}-enter", step),
            ("step", step),
            ("block", step),
            (f"{annotation_kind}-exit", step),
        ]


@pytest.mark.parametrize("is_main", [True, False], ids=["main", "worker"])
def test_failed_jax_trace_stops_without_reporting_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    is_main: bool,
):
    start_trace, stop_trace = Mock(), Mock()
    monkeypatch.setattr(jax.profiler, "start_trace", start_trace)
    monkeypatch.setattr(jax.profiler, "stop_trace", stop_trace)
    failure = RuntimeError("training failed")
    with pytest.raises(RuntimeError) as caught, run._jax_trace(tmp_path, range(2, 4), is_main):
        raise failure
    assert caught.value is failure
    assert "profile written" not in capsys.readouterr().out
    if is_main:
        start_trace.assert_called_once()
        stop_trace.assert_called_once_with()
    else:
        start_trace.assert_not_called()
        stop_trace.assert_not_called()
