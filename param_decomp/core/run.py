"""The generic VPD decomposition-training ENGINE — the one train loop every target
(LM, TMS, ResidMLP, …) runs through.

`run_decomposition_training(pd, cadence, run, model, ci_fn_initializer, positions,
remat_recon_forwards, sample_batch, build_evaluation)` owns
the generic machinery: init / restore / fine-tune init / faith warmup
(`_start_training`), the recon-plan traversal, orbax checkpointing, schedules,
metrics fan-out (`MetricsSink`), the figure-tier background renderer (`BackgroundRenderer`), and
SIGTERM-triggered save before restart. It reads the pydantic `PDConfig` / `Cadence` DIRECTLY; the
target injects two seams: the data source (`sample_batch`) and domain-bound evaluation
(`build_evaluation`).

This module is a pure library — it has NO `main()` and reads no YAML. The per-domain
composition root (read the run YAML → build the target / data loader / `BuiltRun` → call
this engine) lives lab-side: `param_decomp/experiments/lm/run.py` for the LM,
`param_decomp/experiments/{tms,resid_mlp}/run.py` for the toys.
"""

import atexit
import dataclasses
import io
import json
import math
import signal
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from time import monotonic
from types import FrameType, ModuleType
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import nvtx
import optax
import orbax.checkpoint as ocp
import yaml
from jax import random
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import PRNGKeyArray

from param_decomp.core.adversary import SourceStacks
from param_decomp.core.built_run import RunInstance
from param_decomp.core.checkpoint import (
    make_checkpoint_manager,
    make_read_only_checkpoint_manager,
    restore_decomposition,
    restore_destination,
    restore_step,
    save_state,
)
from param_decomp.core.components import ComponentStacks
from param_decomp.core.configs import (
    Cadence,
    Checkpointing,
    NoCheckpointing,
    NontargetConfig,
    PDConfig,
    PDConfigBase,
    PeriodicCheckpointing,
    ResumeProvenance,
    TargetedPDConfig,
    flatten_typed_lists,
)
from param_decomp.core.dict_utils import dict_safe_update_
from param_decomp.core.eval_schedule import EvalSchedule, eval_due
from param_decomp.core.faithfulness import FaithfulnessLossFn, faithfulness_loss_for
from param_decomp.core.hardware_utilization import StepCost
from param_decomp.core.init_placed import (
    CIFnInitializer,
    ComponentInitializer,
    padded_component_initializer,
    random_component_initializer,
)
from param_decomp.core.metrics import BarChart, LineChart, LogRecord, MetricValue, PNGImage
from param_decomp.core.model import ComponentActivations, PlacedModel, PositionAxis
from param_decomp.core.optimizer import ScheduledOptimizer
from param_decomp.core.placement import component_stacks_audit
from param_decomp.core.pytree import ShapeTree
from param_decomp.core.run_files import LAUNCH_CONFIG_FILENAME
from param_decomp.core.run_state import (
    build_optimizers,
    init_decomposition,
    init_pd_training,
    init_targeted_pd_training,
)
from param_decomp.core.sharding import target_shardings_audit
from param_decomp.core.train import (
    Decomposition,
    FaithWarmupStep,
    ForwardSubstrate,
    PDState,
    TargetedPDState,
    TrainingProgress,
    TrainState,
    make_faith_warmup_step,
    make_targeted_train_step,
    make_train_step,
    uv_norm_ratio_metrics,
)
from param_decomp.core.training_performance import MfuAccounting, PerformanceTracker
from param_decomp.metric_schema import MetricNames, MetricSchema, validate_resume_schema

_PROFILE_WARMUP_EXECUTIONS = 2


@dataclasses.dataclass(frozen=True)
class JaxProfilerTrace:
    """Trace `steps` executions after `_PROFILE_WARMUP_EXECUTIONS` warmup executions.

    Warmup retains normal logging, evaluation, and periodic checkpointing. Traced
    executions skip that work, and the loop returns when the capture finishes.
    The initial step-0 checkpoint is omitted; traces go to `<run_dir>/profile`.
    """

    steps: int

    def __post_init__(self) -> None:
        assert self.steps > 0, f"a profile trace needs a positive step count, got {self.steps}"


@dataclasses.dataclass(frozen=True)
class NsightCaptureWindow:
    """The engine-side half of an external Nsight Systems capture: nvtx-annotate the steps
    in `[start + warmup_steps, start + warmup_steps + capture_steps)` so an external
    `nsys --capture-range=nvtx` gate fires; training itself proceeds normally."""

    warmup_steps: int
    capture_steps: int

    def __post_init__(self) -> None:
        assert self.warmup_steps >= 0 and self.capture_steps > 0, (
            f"invalid Nsight capture window: {self}"
        )

    def contains(self, start_step: int, step: int) -> bool:
        first = start_step + self.warmup_steps
        return first <= step < first + self.capture_steps


ProfilingMode = JaxProfilerTrace | NsightCaptureWindow


@dataclasses.dataclass(frozen=True)
class EvalInvocation[Conditioning]:
    """The current decomposition and persistent sources needed to evaluate it."""

    decomposition: Decomposition[Conditioning]
    persistent_sources: dict[str, SourceStacks]
    now_step: int


@dataclasses.dataclass(frozen=True)
class StandaloneOperation[PassT]:
    """Own its evaluation computation without consuming shared forward results."""

    schedule: EvalSchedule
    run: Callable[[PassT], LogRecord]


@dataclasses.dataclass(frozen=True)
class SharedForwardOperation[PassT, ContextT]:
    """Accumulate results over shared batch forwards, then produce one log record.

    State is erased to `Any` so operations with different accumulator types can coexist;
    The typed constructor `shared_forward_operation` keeps its callbacks' state types aligned.
    """

    schedule: EvalSchedule
    init: Callable[[], Any]
    update: Callable[[Any, ContextT], Any]
    finish: Callable[[PassT, Any], LogRecord]


def shared_forward_operation[PassT, ContextT, S](
    schedule: EvalSchedule,
    init: Callable[[], S],
    update: Callable[[S, ContextT], S],
    finish: Callable[[PassT, S], LogRecord],
) -> SharedForwardOperation[PassT, ContextT]:
    """Type-checked constructor: `init`/`update`/`finish` must agree on one state type."""
    return SharedForwardOperation(schedule, init, update, finish)


type EvalOperation[PassT, ContextT] = (
    StandaloneOperation[PassT] | SharedForwardOperation[PassT, ContextT]
)


@dataclasses.dataclass(frozen=True)
class Evaluation[Conditioning, PassT, ContextT]:
    """Scheduled operations and their inputs for one evaluation pass.

    Shared batch contexts are produced only when a shared-forward operation is due.
    """

    operations: tuple[EvalOperation[PassT, ContextT], ...]
    make_pass: Callable[[EvalInvocation[Conditioning]], PassT]
    batch_contexts: Callable[[PassT], Iterable[ContextT]]

    def __post_init__(self) -> None:
        assert self.operations, "evaluation needs at least one operation"


@dataclasses.dataclass(frozen=True)
class StandaloneOperationPlan[PassT]:
    """Prepare an operation that owns its evaluation computation."""

    prepare: Callable[[PassT], StandaloneOperation[PassT]]


@dataclasses.dataclass(frozen=True)
class SharedForwardOperationPlan[PassT, ContextT]:
    """Prepare an operation using the shapes of shared forward results."""

    prepare: Callable[[PassT, ContextT], SharedForwardOperation[PassT, ContextT]]


type EvalOperationPlan[PassT, ContextT] = (
    StandaloneOperationPlan[PassT] | SharedForwardOperationPlan[PassT, ContextT]
)


@dataclasses.dataclass(frozen=True)
class EvaluationPlan[Conditioning, PassT, ContextT]:
    """Bind every configured operation to executable kernels before timing begins."""

    prepare: Callable[[EvalInvocation[Conditioning]], Evaluation[Conditioning, PassT, ContextT]]


type EvaluationBuilder[Conditioning, PassT, ContextT] = Callable[
    [PRNGKeyArray], EvaluationPlan[Conditioning, PassT, ContextT]
]
"""An evaluation plan drawing its randomness from the run key, which the engine alone
derives from `pd.seed`."""


def no_batch_contexts(eval_pass: object) -> tuple[()]:
    """No shared forwards are needed when every operation is standalone."""
    del eval_pass
    return ()


def _combine_step_records(
    train: LogRecord | None, evaluation: LogRecord | None
) -> LogRecord | None:
    """One model step becomes one committed transport record.

    W&B's `_step` is a monotonic row cursor, not a namespace: committing train and eval
    separately at the same model step silently drops the second record. Keeping the two
    producers independent and combining only at the sink boundary preserves both semantics.
    """
    match train, evaluation:
        case None, None:
            return None
        case (record, None) | (None, record):
            return record
        case train_record, eval_record:
            overlap = train_record.keys() & eval_record.keys()
            assert not overlap, f"train/eval records emitted colliding keys: {sorted(overlap)}"
            return {**train_record, **eval_record}


def _with_uv_norm_ratios[Conditioning](
    eval_record: LogRecord,
    decomposition: Decomposition[Conditioning],
    compute_norm_ratios: Callable[[ComponentStacks], dict[str, jax.Array]],
) -> LogRecord:
    factor_record = {
        f"eval/{key}": float(value)
        for key, value in compute_norm_ratios(decomposition.components).items()
    }
    overlap = eval_record.keys() & factor_record.keys()
    assert not overlap, f"U/V norm-ratio metrics collided with eval keys: {sorted(overlap)}"
    return {**eval_record, **factor_record}


def _run_due_evaluation[Conditioning, Training: TrainingProgress, PassT, ContextT](
    evaluation: Evaluation[Conditioning, PassT, ContextT],
    state: TrainState[Conditioning, Training],
    now_step: int,
) -> LogRecord | None:
    due_operations = tuple(
        operation for operation in evaluation.operations if eval_due(operation.schedule, now_step)
    )
    if not due_operations:
        return None
    eval_pass = evaluation.make_pass(
        EvalInvocation(
            decomposition=state.decomposition,
            persistent_sources={
                name: adversary.sources for name, adversary in state.training.adversaries.items()
            },
            now_step=now_step,
        )
    )
    states: dict[int, Any] = {
        index: operation.init()
        for index, operation in enumerate(due_operations)
        if isinstance(operation, SharedForwardOperation)
    }
    if states:
        for context in evaluation.batch_contexts(eval_pass):
            for index in states:
                operation = due_operations[index]
                assert isinstance(operation, SharedForwardOperation)
                states[index] = operation.update(states[index], context)
    record: dict[str, MetricValue] = {}
    for index, operation in enumerate(due_operations):
        match operation:
            case StandaloneOperation(run=run):
                values = run(eval_pass)
            case SharedForwardOperation(finish=finish):
                values = finish(eval_pass, states[index])
        overlap = record.keys() & values.keys()
        assert not overlap, f"eval operations emitted colliding keys: {sorted(overlap)}"
        record.update(values)
    return record


@dataclasses.dataclass(frozen=True)
class DeferredMediaRecord:
    """Pure-renderer output. ``media`` contains encoded images; the metrics sink alone
    translates them into W&B objects and assigns their semantic experiment step."""

    step_key: str
    step: int
    media: Mapping[str, bytes]


_sigterm_received = False


def install_sigterm_flag() -> None:
    """Install the SIGTERM handler the engine's save-on-preempt logic reads. Called by the
    composition root (which owns process setup) before `run_decomposition_training`."""

    def handler(_signum: int, _frame: FrameType | None) -> None:
        global _sigterm_received
        _sigterm_received = True

    signal.signal(signal.SIGTERM, handler)


def _any_sigterm_received(flags: jax.Array) -> jax.Array:
    return jnp.any(flags)


def _prepare_sigterm_consensus(training_mesh: Mesh) -> Callable[[], bool]:
    """Cross-rank-agreed SIGTERM flag. A process manager may deliver SIGTERM per process with no
    simultaneity guarantee, so reading the per-process flag independently at a collective gate
    (faith-warmup exit, eval entry, orbax save) can diverge ranks and hang. OR-reduce it across
    processes; callers read it once into a local the handler can't mutate mid-step. The reduce
    is a collective plus a blocking device->host readback, so the train loop takes it only at
    train-log steps (`Cadence.train_log_every`, denser under `dense_log_phase`, and the final
    step) — the log step already blocks on the step's metrics, and every rank derives the log
    step from the step count alone, so all enter the collective together and it adds no device
    sync of its own. Worst-case added latency to a requeue save: `train_log_every - 1` further
    training steps after the signal lands; the lead time between SIGTERM
    and SIGKILL must cover that plus the save. No-op when not distributed."""
    if jax.process_count() == 1:
        return lambda: _sigterm_received

    mesh = Mesh(training_mesh.devices.reshape(-1), ("device",))
    sharding = NamedSharding(mesh, P("device"))
    abstract_flags = jax.ShapeDtypeStruct((mesh.size,), np.bool_, sharding=sharding)
    with jax.set_mesh(mesh):
        any_received = jax.jit(_any_sigterm_received).lower(abstract_flags).compile()

    def consensus() -> bool:
        local_flags = np.full(len(mesh.local_devices), _sigterm_received, dtype=np.bool_)
        flags = jax.make_array_from_process_local_data(sharding, local_flags, (mesh.size,))
        return bool(any_received(flags))

    return consensus


def log_wandb_safe(
    wandb_module: "ModuleType",
    payload: Mapping[str, object],
    step: int | None,
    commit: bool,
    what_for_log: str,
) -> None:
    """`wandb.log` swallowing `CommError` only — a transient wandb-server outage must not
    kill a multi-day run, while genuine misuse (e.g. a non-dict record) still raises. The
    soft-fail is deliberate (drops the failed record, keeps training)."""
    import wandb.errors

    try:
        match step, commit:
            case int(), True:
                wandb_module.log(payload, step=step, commit=True)
            case None, False:
                wandb_module.log(payload, commit=False)
            case _:
                raise AssertionError((step, commit))
    except wandb.errors.CommError as e:
        print(f"wandb communication error, skipping {what_for_log}: {e}", flush=True)


def _ensure_global[T](tree: T, mesh: Mesh) -> T:
    """Re-materialize the NON-mesh array leaves (eagerly created scalars: step
    counters, Adam counts) as well-formed GLOBAL replicated arrays via an identity
    jit. Multi-controller orbax can only save global arrays — and an eager
    `device_put(local, replicated-NamedSharding)` yields arrays whose
    `addressable_shards` raise (jax 0.10 multi-process), while jit outputs with the
    same sharding are well-formed.

    Leaves that already carry a NamedSharding pass through UNTOUCHED."""
    repl = NamedSharding(mesh, P())

    def is_mesh_placed(a: object) -> bool:
        return eqx.is_array(a) and isinstance(a.sharding, NamedSharding)  # pyright: ignore[reportAttributeAccessIssue]

    mesh_placed, stragglers = eqx.partition(tree, is_mesh_placed)
    straggler_shardings = jax.tree.map(lambda _a: repl, stragglers)
    fixed = jax.jit(lambda t: t, out_shardings=straggler_shardings)(stragglers)
    return eqx.combine(mesh_placed, fixed)


# wandb keys match the torch trainer's (`train_step.py` emits `loss/<instance_key>`,
# `optimize.py` prefixes `train/`) so a torch-vs-jax run pair overlays on one panel.
# Recon-term keys arrive from the step already shaped (`loss/<instance_key>`) and are
# train/-prefixed by the sink; this table maps only the step's fixed scalar keys.
_METRIC_KEYS = {
    "total": "train/loss/total",
    "faith": "train/loss/FaithfulnessLoss",
    "imp": "train/loss/ImportanceMinimalityLoss",
    "freq": "train/loss/FrequencyMinimalityLoss",
    "freq_batch": "train/loss/FrequencyMinimalityLoss_batch",
    "gamma_imp": "train/schedules/gamma_imp",
    "nonlinearity_relative_threshold": "train/schedules/nonlinearity_relative_threshold",
    "src_lr": "train/schedules/lr/src",
    "step_time_s": "train/perf/step_time_s",
    "hfu": "train/perf/hfu",
    "flops_per_step": "train/perf/flops_per_step",
    "elapsed_s": "train/perf/elapsed_s",
    "eta_s": "train/perf/eta_s",
}


def _duration_for_log(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _is_verbose(key: str) -> bool:
    """Per-item metric families that belong in wandb but would swamp the console line — one
    scalar per trainable leaf, one per hidden-activation reconstruction point. Their aggregates
    (`grad_norms/summary/*`, `loss/<name>/hidden_acts_reconstruction`) stay."""
    if key.startswith("train/grad_norms/"):
        return not key.startswith("train/grad_norms/summary/")
    return "/hidden_acts_reconstruction/" in key


def _grad_norm_summary_window_stats(window: list[dict[str, jax.Array]]) -> dict[str, float]:
    """Window min/max/median for each `grad_norms/summary/*` scalar over every step since the
    last log. The per-step values are accumulated as device handles (appending one is async),
    so the whole window reduces in a single host transfer here — the loop stays unsynced
    between logs rather than subsampling grad norms at the log step."""
    assert window, "grad-norm summary window is empty at a log boundary"
    keys = list(window[0].keys())
    host_window = jax.device_get(window)
    stacked = np.asarray([[snapshot[key] for snapshot in host_window] for key in keys])
    mins = np.min(stacked, axis=1)
    maxs = np.max(stacked, axis=1)
    medians = np.median(stacked, axis=1)
    out: dict[str, float] = {}
    for i, key in enumerate(keys):
        out[f"{key}/min"] = float(mins[i])
        out[f"{key}/max"] = float(maxs[i])
        out[f"{key}/median"] = float(medians[i])
    return out


class MetricsSink:
    """Process-0 metrics fan-out: jsonl always, wandb when configured.

    Construct via `for_run(run, is_main)` — it takes the rank flag and resolves BOTH cases
    (main rank: open the run's `metrics.jsonl`, plus wandb when the run configures it;
    non-main rank: the no-op handle). Every rank calls it; none picks a constructor by
    rank. `silent()` is for tests and throwaway interactive runs only. Not the
    resolved-channel `__init__` directly."""

    def __init__(
        self,
        jsonl: io.TextIOWrapper | None,
        wandb_module: ModuleType | None,
        metric_schema: MetricSchema,
    ):
        self._jsonl = jsonl
        self._wandb = wandb_module
        self._metric_names = MetricNames(metric_schema)
        self._wandb_lock = threading.Lock()
        self._defined_deferred_metrics: set[tuple[str, str]] = set()
        self._deferred_media_keys: set[tuple[str, int, str]] = set()
        self._last_committed_step: int | None = None
        self._renderers: list[BackgroundRenderer] = []

    @classmethod
    def silent(cls) -> "MetricsSink":
        """NO metrics anywhere — `log` drops the record before it reaches either channel, so
        the run writes no `metrics.jsonl` at all. This is not "wandb off": a run without a
        tracker still wants its jsonl, and gets it from `for_run` (`wandb: null` in the
        config is what turns wandb off)."""
        return cls(jsonl=None, wandb_module=None, metric_schema="legacy")

    @classmethod
    def for_run(cls, run: RunInstance, is_main: bool) -> "MetricsSink":
        if not is_main:
            return cls.silent()
        metrics_path = run.run_dir / "metrics.jsonl"
        if run.wandb is None:
            return cls(jsonl=metrics_path.open("a"), wandb_module=None, metric_schema="legacy")
        import wandb

        # wandb.config is the pinned launch config verbatim — the run's ONE self-contained
        # yaml (the same bytes resume byte-compares), so programmatic config access works.
        # The metric lists flatten into the same flat keys torch logged (E14) so cross-impl
        # wandb config queries line up. Nothing else rides along: the logical mesh is pinned
        # in the config and its realized world is asserted at process bring-up.
        launch_config = run.run_dir / LAUNCH_CONFIG_FILENAME
        assert launch_config.exists(), launch_config
        tracker = wandb.init(
            project=run.wandb.project,
            entity=run.wandb.entity,
            name=run.run_name,
            id=run.run_id,
            group=run.wandb.group,
            tags=list(run.wandb.tags),
            resume="allow",
        )
        if tracker.resumed:
            validate_resume_schema(tracker.config.as_dict(), run.wandb.metric_schema)
        else:
            tracker.config.update(
                flatten_typed_lists(yaml.safe_load(launch_config.read_text())),
                allow_val_change=False,
            )
        match run.wandb.metric_schema:
            case "legacy":
                pass
            case "grouped":
                wandb.define_metric("axes/*", hidden=True)
        # Also save the pin as a downloadable wandb run file, alongside (not in place of)
        # the wandb.config dict.
        wandb.save(str(launch_config), base_path=str(run.run_dir), policy="now")
        return cls(
            jsonl=metrics_path.open("a"), wandb_module=wandb, metric_schema=run.wandb.metric_schema
        )

    def register_renderer(self, renderer: "BackgroundRenderer") -> None:
        self._renderers.append(renderer)

    def wait_for_renderers(self) -> None:
        for renderer in self._renderers:
            renderer.join()

    def log(self, step: int, record: "LogRecord") -> None:
        if self._jsonl is None:
            return
        assert self._last_committed_step is None or step > self._last_committed_step, (
            f"metrics steps must be strictly increasing: "
            f"previous={self._last_committed_step}, next={step}"
        )
        self._last_committed_step = step
        record = {
            _METRIC_KEYS.get(
                k, f"train/{k}" if k.startswith(("grad_norms/", "loss/", "schedules/")) else k
            ): v
            for k, v in record.items()
        }  # keys already starting "train/" or "eval/" pass through verbatim
        # wandb-only viz objects (e.g. the CI_L0 bar chart) ride alongside the scalars to
        # wandb but are not jsonl/console serializable; split them off.
        scalars = {k: v for k, v in record.items() if isinstance(v, float)}
        self._jsonl.write(json.dumps({"step": step, **scalars}) + "\n")
        self._jsonl.flush()
        # The console line drops the per-param grad norms — the full breakdown still rides to
        # wandb + jsonl.
        console = {k: v for k, v in scalars.items() if not _is_verbose(k)}
        head = f"[step {step}]"
        if "train/perf/eta_s" in console:  # train logs carry the paired timing; eval logs don't
            elapsed, eta = console.pop("train/perf/elapsed_s"), console.pop("train/perf/eta_s")
            head += f" {_duration_for_log(elapsed)}<{_duration_for_log(eta)}"
        print(head + " " + " ".join(f"{k}={v:.4g}" for k, v in console.items()), flush=True)
        if self._wandb is not None:
            with self._wandb_lock:
                named_record = self._metric_names.record(record)
            wandb_record: dict[str, object] = {}
            for key, value in named_record.items():
                match value:
                    case float() | int():
                        wandb_record[key] = float(value)
                    case BarChart(rows, x_label, y_label, title):
                        wandb_record[key] = self._wandb.plot.bar(
                            self._wandb.Table(
                                columns=[x_label, y_label],
                                data=[list(row) for row in rows],
                            ),
                            x_label,
                            y_label,
                            title=title,
                        )
                    case LineChart(xs, series, x_label, title):
                        # The run-media copy has a lower row cap than the artifact copy.
                        assert xs.size * len(series) <= self._wandb.Table.MAX_ROWS, (
                            "line chart exceeds W&B's run-media row limit; refusing to truncate"
                        )
                        wandb_record[key] = self._wandb.plot.line_series(
                            xs=xs.tolist(),
                            ys=[ys.tolist() for _, ys in series],
                            keys=[name for name, _ in series],
                            title=title,
                            xname=x_label,
                        )
                    case PNGImage(encoded):
                        import io

                        from PIL import Image

                        wandb_record[key] = self._wandb.Image(Image.open(io.BytesIO(encoded)))
            with self._wandb_lock:
                log_wandb_safe(self._wandb, wandb_record, step, True, "log")

    @property
    def accepts_deferred_media(self) -> bool:
        return self._wandb is not None

    def log_deferred_media(self, record: DeferredMediaRecord) -> None:
        """Serialize a pure renderer's encoded images onto their semantic step axis.

        Deferred records deliberately omit W&B's monotonic ``_step``: they may arrive after
        synchronous training has advanced it. The dedicated axis preserves the model step
        that produced the snapshot without allowing renderer threads to own W&B transport.
        """
        if self._wandb is None:
            return
        import io

        from PIL import Image

        with self._wandb_lock:
            step_key = self._metric_names.axis(record.step_key)
            media = self._metric_names.record(record.media)
            semantic_keys = {(record.step_key, record.step, key) for key in record.media}
            overlap = self._deferred_media_keys & semantic_keys
            assert not overlap, (
                "deferred renderers emitted colliding semantic keys: "
                f"{sorted(key for _, _, key in overlap)} at step {record.step}"
            )
            self._deferred_media_keys.update(semantic_keys)
            payload: dict[str, object] = {step_key: float(record.step)}
            for key, encoded in media.items():
                registration = (key, step_key)
                if registration not in self._defined_deferred_metrics:
                    self._wandb.define_metric(key, step_metric=step_key)
                    self._defined_deferred_metrics.add(registration)
                payload[key] = self._wandb.Image(Image.open(io.BytesIO(encoded)))
            log_wandb_safe(self._wandb, payload, None, False, "deferred media")


class BackgroundRenderer:
    """Background thread for a figure tier's pure host-rendering tail.

    The slow/plot tier and the LM arithmetic tier each hold one. A pure
    renderer returns ``DeferredMediaRecord``; the shared ``MetricsSink`` performs the
    serialized W&B write.

    The collective part of a figure eval (the jitted forwards + the device->host pulls)
    runs in lockstep on ALL ranks inside the eval pass. `submit` takes a `render` closure
    over ONLY the materialized numpy results and runs it on a background thread, so the
    main train loop on every rank proceeds immediately (near-zero cross-rank divergence).
    The closure must touch ZERO jax/device state.

    One render in flight at a time: a `submit` while a render is still running blocks
    briefly on `join()` first, so renders can't pile up (figure tiers are forward-only and
    coarse, so this effectively never blocks). Deferred figures carry their eval step as a
    dedicated W&B metric axis (`slow_eval/figure_step` or `eval/arithmetic/figure_step`)
    rather than writing an old `_step`: rendering may finish after synchronous scalar logs
    have advanced `_step`, and W&B correctly rejects out-of-order writes. The loop joins
    remaining renderers before its final timing observation. An `atexit` join also flushes
    on early exit (the trainer never calls `wandb.finish`). The atexit handler is
    registered on the FIRST submit, not in `__init__` — the first submit happens after
    `MetricsSink`'s `wandb.init` (eval comes after sink construction in the loop), so
    atexit's LIFO order runs our join BEFORE wandb's own atexit flush, and the figures
    land.

    A render that raises is re-raised on the owning thread at the next `join` (the next
    `submit`, or the atexit flush). Nothing here may fail soft: a worker thread that dies
    quietly turns "the figure tier stopped producing figures" into a run that looks healthy
    for days. Transient W&B outages are already absorbed downstream by `log_wandb_safe`, so
    anything reaching here is a genuine defect."""

    def __init__(self, sink: MetricsSink):
        self._sink = sink
        self._thread: threading.Thread | None = None
        self._failure: BaseException | None = None
        self._atexit_registered = False
        sink.register_renderer(self)

    def join(self) -> None:
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._failure is not None:
            failure, self._failure = self._failure, None
            raise RuntimeError("background figure render failed") from failure

    def _render_and_log(self, render: Callable[[], DeferredMediaRecord]) -> None:
        try:
            self._sink.log_deferred_media(render())
        except BaseException as failure:  # carried across the thread boundary, re-raised in join
            self._failure = failure

    def submit(self, render: Callable[[], DeferredMediaRecord]) -> None:
        if not self._sink.accepts_deferred_media:
            return
        if not self._atexit_registered:
            atexit.register(self.join)
            self._atexit_registered = True
        self.join()  # cap to one in-flight render
        self._thread = threading.Thread(target=lambda: self._render_and_log(render), daemon=True)
        self._thread.start()


@dataclasses.dataclass(frozen=True)
class FaithfulnessWarmup:
    """Faithfulness-only warmup before the main loop, for plain PD.

    Targeted runs omit this phase because they have no faithfulness objective."""

    steps: int
    lr: float
    weight_decay: float
    loss: FaithfulnessLossFn


@dataclasses.dataclass(frozen=True)
class Interrupted:
    """SIGTERM ended the phase before it produced a usable training state."""


def _warmup_pd[TargetIn, Out, PreparedT: ComponentActivations, Conditioning, PreparedMaskingT](
    state: PDState[Conditioning],
    config: FaithfulnessWarmup,
    model: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
    opt_vu: ScheduledOptimizer,
    mesh: Mesh,
    compiler_options: dict[str, bool | int | str],
    is_main: bool,
    sigterm_consensus: Callable[[], bool],
) -> PDState[Conditioning] | Interrupted:
    faith_warmup_optimizer = optax.adamw(config.lr, weight_decay=config.weight_decay)
    faith_warmup_opt_state = faith_warmup_optimizer.init(
        eqx.filter(state.decomposition.components, eqx.is_array)
    )
    faith_warmup_step: FaithWarmupStep[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT] = (
        make_faith_warmup_step(faith_warmup_optimizer, config.loss)
    )
    faith_warmup_step = (
        jax.jit(faith_warmup_step, compiler_options=compiler_options)
        .lower(model, state.decomposition.components, faith_warmup_opt_state)
        .compile()
    )
    warmed_components = state.decomposition.components
    t0 = time.time()
    faith_warmup_loss = None
    for _ in range(config.steps):
        warmed_components, faith_warmup_opt_state, faith_warmup_loss = faith_warmup_step(
            model, warmed_components, faith_warmup_opt_state
        )
        if sigterm_consensus():
            # No valid checkpoint exists yet (the step-0 save happens only after warmup
            # completes, and resume skips warmup whenever a checkpoint is present — a
            # partially-warmed step-0 save would resume as if fully warmed). Exit
            # cleanly; the restarted process redoes warmup from scratch.
            if is_main:
                print("SIGTERM during faith warmup: exiting for requeue", flush=True)
            return Interrupted()
    assert faith_warmup_loss is not None
    jax.block_until_ready(faith_warmup_loss)
    new_opt_vu = _ensure_global(opt_vu.init(eqx.filter(warmed_components, eqx.is_array)), mesh)
    state = dataclasses.replace(
        state,
        decomposition=dataclasses.replace(state.decomposition, components=warmed_components),
        training=dataclasses.replace(state.training, components_opt_state=new_opt_vu),
    )
    if is_main:
        print(
            f"faith warmup: {config.steps} steps in {time.time() - t0:.0f}s, "
            f"final faith {float(faith_warmup_loss):.3e}",
            flush=True,
        )
    return state


@dataclasses.dataclass(frozen=True)
class FreshTraining:
    """A new decomposition and fresh training history."""


@dataclasses.dataclass(frozen=True)
class ResumeTraining:
    manager: ocp.CheckpointManager
    step: int


@dataclasses.dataclass(frozen=True)
class FineTune:
    parent: ResumeProvenance


type TrainingStart = FreshTraining | ResumeTraining | FineTune


def _resolve_training_start(
    run: RunInstance, manager: ocp.CheckpointManager | None, total_steps: int
) -> TrainingStart:
    if manager is not None:
        checkpoint_step = manager.latest_step()
        if checkpoint_step is not None:
            assert checkpoint_step < total_steps, (
                f"run {run.run_id} is already trained to step {checkpoint_step} of pd.steps={total_steps}: "
                "nothing left to run. There is no way to extend this trajectory: raising `pd.steps` "
                "on this id is refused by the pinned-config byte-compare, a new run id trains from "
                "scratch, and a new run id with `resume_provenance` inherits the decomposition only "
                "— fresh optimizer state, fresh adversaries, step 0, schedules re-annealed."
            )
            return ResumeTraining(manager, checkpoint_step)
    match run.resume_provenance:
        case None:
            return FreshTraining()
        case ResumeProvenance() as parent:
            return FineTune(parent)


def _start_training[Conditioning, Training: TrainingProgress](
    start: TrainingStart,
    *,
    initialize_decomposition: Callable[[], Decomposition[Conditioning]],
    initialize: Callable[[Decomposition[Conditioning]], TrainState[Conditioning, Training]],
    destination: ShapeTree[TrainState[Conditioning, Training]],
    mesh: Mesh,
    is_main: bool,
) -> TrainState[Conditioning, Training]:
    match start:
        case ResumeTraining(manager=manager, step=step):
            state = restore_step(manager, destination, step)
            assert int(state.training.step) == step, (int(state.training.step), step)
            if is_main:
                print(f"resumed from checkpoint step {step}", flush=True)
            return state
        case FineTune(parent=parent):
            with make_read_only_checkpoint_manager(parent.parent_run_dir / "ckpts") as manager:
                decomposition = restore_decomposition(
                    manager, parent.parent_step, destination.decomposition
                )
            state = _ensure_global(initialize(decomposition), mesh)
            if is_main:
                print(
                    f"fine-tune: initialized V/U + ci_fn from {parent.parent_run_dir} "
                    f"step {parent.parent_step}; training fresh from step 0",
                    flush=True,
                )
            return state
        case FreshTraining():
            return _ensure_global(initialize(initialize_decomposition()), mesh)


@dataclasses.dataclass(frozen=True)
class _PeriodicSaver:
    """`PeriodicCheckpointing`'s runtime pair: the orbax manager plus its rhythm. Bundled
    so a `NoCheckpointing` run (`saver is None`) cannot carry half of it."""

    manager: ocp.CheckpointManager
    save_every: int


_NO_CHECKPOINT_ENTRY_MARKER = "entered-without-checkpoints"


def _make_saver(
    checkpointing: Checkpointing, run_dir: Path, is_main: bool
) -> _PeriodicSaver | None:
    match checkpointing:
        case PeriodicCheckpointing(save_every=save_every, retention=retention):
            return _PeriodicSaver(make_checkpoint_manager(run_dir / "ckpts", retention), save_every)
        case NoCheckpointing():
            # A no-checkpoint run id is single-entry: nothing is ever written to resume
            # from, so a re-entry (process restart or a --run-id rerun) would silently
            # retrain from step 0. The marker turns that into an enumerated refusal.
            # Checked on the main process only — a non-main rank racing the fresh write
            # must not misread a first entry as a re-entry.
            marker = run_dir / _NO_CHECKPOINT_ENTRY_MARKER
            if is_main:
                assert not marker.exists(), (
                    f"run dir {run_dir} was already entered without checkpointing "
                    "(cadence.checkpointing: none): there is nothing to resume from — "
                    "launch a fresh run id instead of re-entering this one"
                )
                marker.write_text(
                    "this run trains with cadence.checkpointing: none — it writes no "
                    "checkpoints and cannot be resumed; this marker refuses re-entry\n"
                )
            return None


@dataclasses.dataclass(frozen=True)
class _RunResources:
    """Optimizers, random keys, placement, and checkpoint storage for one run."""

    opt_vu: ScheduledOptimizer
    opt_ci: ScheduledOptimizer
    init_key: PRNGKeyArray
    src_key: PRNGKeyArray
    run_key: PRNGKeyArray
    mesh: Mesh
    saver: _PeriodicSaver | None
    run_dir: Path
    start: TrainingStart
    sigterm_consensus: Callable[[], bool]

    @property
    def start_step(self) -> int:
        match self.start:
            case ResumeTraining(step=step):
                return step
            case FreshTraining() | FineTune():
                return 0


def _training_mesh[
    TargetIn,
    Out,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](model: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT]) -> Mesh:
    """The rules' own mesh — the engine never receives a second copy to desync. A training
    run executes forwards, so the abstract (spec-check) arm of `PlacementRules.mesh` is
    refused here. The engine activates it around the whole run so bare-PartitionSpec
    `reshard`s inside the forward resolve (the attn q/k/v batch-sharding pin in
    `FrozenAttn.core`, needed for cuDNN flash attention under the scan+cond masked forward)."""
    assert model.placement is not None, "the engine trains placed models only"
    mesh = model.placement.mesh
    assert isinstance(mesh, Mesh), type(mesh)
    return mesh


def _prepare_run[
    TargetIn,
    Out,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    *,
    pd: PDConfigBase,
    cadence: Cadence,
    run: RunInstance,
    model: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
    mesh: Mesh,
    is_main: bool,
    component_initializer: ComponentInitializer[
        TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT
    ],
) -> _RunResources:
    """Prepare the run's resources and placement before allocating training state."""
    rules = model.placement
    assert rules is not None, "the engine trains placed models only"
    run.run_dir.mkdir(parents=True, exist_ok=True)
    opt_vu, opt_ci = build_optimizers(pd, rules, model.model.sites)

    key = random.PRNGKey(pd.seed)
    init_key, src_key, run_key = random.split(key, 3)

    saver = _make_saver(cadence.checkpointing, run.run_dir, is_main)
    start = _resolve_training_start(run, saver.manager if saver is not None else None, pd.steps)
    if is_main:
        audit = component_stacks_audit(
            eqx.filter_eval_shape(
                padded_component_initializer(rules, component_initializer), model.model, init_key
            ),
            rules,
        )
        print(
            rules.description_for_log(
                tensors=audit,
                sharded_tensors=target_shardings_audit(model),
                not_audited=("ci_fn", "persistent sources", "opt state"),
            ),
            flush=True,
        )
    return _RunResources(
        opt_vu=opt_vu,
        opt_ci=opt_ci,
        init_key=init_key,
        src_key=src_key,
        run_key=run_key,
        mesh=mesh,
        saver=saver,
        run_dir=run.run_dir,
        start=start,
        sigterm_consensus=_prepare_sigterm_consensus(mesh),
    )


def run_decomposition_training[
    TargetIn,
    Out,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
    EvalPassT,
    EvalContextT,
](
    pd: PDConfig,
    cadence: Cadence,
    run: RunInstance,
    model: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
    ci_fn_initializer: CIFnInitializer[Conditioning],
    positions: PositionAxis,
    remat_recon_forwards: bool,
    remat_ci_fn: bool,
    compiler_options: dict[str, bool | int | str],
    sample_batch: Callable[[int], TargetIn],
    build_evaluation: EvaluationBuilder[Conditioning, EvalPassT, EvalContextT] | None,
    sink: MetricsSink,
    profiling: ProfilingMode | None,
    mfu_accounting: MfuAccounting | None,
    component_initializer: ComponentInitializer[
        TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT
    ] = random_component_initializer,
) -> None:
    """The generic VPD decomposition-training engine — the ONE train loop every target
    (LM, TMS, ResidMLP, …) runs through.

    Reads the pydantic algorithm config DIRECTLY: `pd` (seed / steps / optimizers / loss
    metrics / faith warmup), `cadence` (log rhythm + the checkpointing arm),
    `run` (the run identity + wandb lineage). The lab-built objects ride alongside:
    the decomposed `model` (an `eqx.Module` carrying the frozen target weights as
    fields — threaded into the jitted step as a pytree arg, never closed over), the CI fn's
    `ci_fn_initializer` (called only on a fresh start; a resume or fine-tune traces it for
    shapes alone), the run's waist geometry (`positions`: `Positioned(seq_len)` for an LM,
    `Positionless()` for a toy), and the `remat_recon_forwards` compute knob.

    The target supplies only its three injectable seams:

    - `sample_batch(step) -> batch`: the opaque per-step model input (a pure function of
      `step`, for O(1) resume). The model interprets it (an LM's token ids `[B, T]` → embed;
      a toy's feature vector, which already is the `[*leading, d]` waist). The engine only
      assumes axis 0 is the batch/`dp` axis (for sharding); it never names tokens or `d`.
    - `build_evaluation(run_key)`: a fixed tuple of domain-bound operations plus the typed
      context factory they share. The engine alone schedules due operations, constructs one
      context, merges disjoint records, and logs the result. `None` disables evaluation.

    `profiling` is threaded as data from the composition root (the engine reads no ambient
    environment): `None` is a normal training run, `JaxProfilerTrace` turns the run into an
    in-process profile (trace then return), `NsightCaptureWindow` nvtx-annotates the
    caller-declared capture steps of an otherwise-normal run.

    Everything generic — state initialization, fine-tune init, faith warmup, the recon-grid
    step factory, orbax checkpointing, schedules, SIGTERM-save — lives here. The step
    numerics are identical across targets; only the data source and the eval metric differ.

    The targeted (tPD) twin is `run_targeted_decomposition_training`; both entries compose
    the same `_prepare_run` / `_run_loop` core.
    """
    is_main = jax.process_index() == 0
    faithfulness = faithfulness_loss_for(model)
    faith_warmup = (
        FaithfulnessWarmup(
            steps=pd.faithfulness_warmup_steps,
            lr=pd.faithfulness_warmup_lr,
            weight_decay=pd.faithfulness_warmup_weight_decay,
            loss=faithfulness,
        )
        if pd.faithfulness_warmup_steps > 0
        else None
    )

    mesh = _training_mesh(model)
    with jax.set_mesh(mesh):
        prepared = _prepare_run(
            pd=pd,
            cadence=cadence,
            run=run,
            model=model,
            mesh=mesh,
            is_main=is_main,
            component_initializer=component_initializer,
        )

        def initialize_decomposition() -> Decomposition[Conditioning]:
            return init_decomposition(
                model, ci_fn_initializer, prepared.init_key, component_initializer
            )

        def initialize(decomposition: Decomposition[Conditioning]) -> PDState[Conditioning]:
            return TrainState(
                decomposition=decomposition,
                training=init_pd_training(
                    pd,
                    model,
                    positions,
                    prepared.opt_vu,
                    prepared.opt_ci,
                    decomposition,
                    prepared.src_key,
                ),
            )

        shape = eqx.filter_eval_shape(lambda: initialize(initialize_decomposition()))
        if is_main:
            print(shape.decomposition.ci_fn.placement_for_log(), flush=True)
        substrate = ForwardSubstrate.of(
            model,
            remat_recon_forwards=remat_recon_forwards,
            remat_ci_fn=remat_ci_fn,
            ci_capture_keys=shape.decomposition.ci_fn.capture_keys,
        )
        step_fn = make_train_step(
            model_static=model,
            substrate=substrate,
            components_optimizer=prepared.opt_vu,
            ci_fn_optimizer=prepared.opt_ci,
            total_steps=pd.steps,
            faithfulness=faithfulness,
        )

        def training_step(
            model: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
            state: PDState[Conditioning],
            batch: TargetIn,
            run_key: PRNGKeyArray,
        ) -> tuple[PDState[Conditioning], dict[str, jax.Array]]:
            return step_fn(model, state, batch, random.fold_in(run_key, state.training.step))

        def run_step(
            state: PDState[Conditioning], step: int
        ) -> tuple[PDState[Conditioning], dict[str, jax.Array]]:
            return compiled(model, state, sample_batch(step), prepared.run_key)

        compiled = (
            jax.jit(training_step, donate_argnums=(1, 2), compiler_options=compiler_options)
            .lower(
                model,
                shape,
                sample_batch(prepared.start_step),
                prepared.run_key,
            )
            .compile()
        )

        state = _start_training(
            prepared.start,
            initialize_decomposition=initialize_decomposition,
            initialize=initialize,
            destination=restore_destination(shape, compiled.input_formats[0][1], prepared.mesh),
            mesh=prepared.mesh,
            is_main=is_main,
        )
        match prepared.start:
            case FreshTraining():
                if faith_warmup is not None:
                    warmed = _warmup_pd(
                        state,
                        faith_warmup,
                        model,
                        prepared.opt_vu,
                        prepared.mesh,
                        compiler_options,
                        is_main,
                        prepared.sigterm_consensus,
                    )
                    match warmed:
                        case Interrupted():
                            return
                        case TrainState():
                            state = warmed
            case ResumeTraining() | FineTune():
                pass
        step_cost = StepCost.of(compiled, jax.devices())
        jax.block_until_ready((model, state))
        _run_loop(
            pd,
            cadence,
            build_evaluation,
            sink,
            prepared,
            state,
            is_main,
            run_step,
            step_cost,
            profiling,
            mfu_accounting,
        )


def run_targeted_decomposition_training[
    TargetIn,
    Out,
    PreparedT: ComponentActivations,
    Conditioning,
    EvalPassT,
    EvalContextT,
    PreparedMaskingT,
](
    pd: TargetedPDConfig,
    nontarget: NontargetConfig,
    cadence: Cadence,
    run: RunInstance,
    model: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
    ci_fn_initializer: CIFnInitializer[Conditioning],
    positions: PositionAxis,
    remat_recon_forwards: bool,
    remat_ci_fn: bool,
    compiler_options: dict[str, bool | int | str],
    sample_target_batch: Callable[[int], TargetIn],
    sample_nontarget_batch: Callable[[int], TargetIn],
    build_evaluation: EvaluationBuilder[Conditioning, EvalPassT, EvalContextT] | None,
    sink: MetricsSink,
    profiling: ProfilingMode | None,
    mfu_accounting: MfuAccounting | None,
    component_initializer: ComponentInitializer[
        TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT
    ] = random_component_initializer,
) -> None:
    """The targeted-PD (tPD) engine entry — `run_decomposition_training`'s twin
    over the same `_prepare_run` / `_run_loop` core, stepping the two-pass
    `make_targeted_train_step`.

    Two data seams instead of one: `sample_target_batch(step)` feeds the narrow TARGET
    stream (global batch `pd.batch_size` — the pass the persistent adversaries and every
    other decomposition loss run on), and `sample_nontarget_batch(step)` the broad
    NON-TARGET stream (global batch `nontarget.batch_size`, delta pinned fully on,
    except in the unmasked-no-delta term). `positions` is the TARGET stream's waist
    geometry — persistent sources live in the target pass; each stream runs at its own
    natural sequence length.

    tPD has no faithfulness role: `TargetedPDConfig` admits no faithfulness loss
    member and carries no warmup fields, so neither exists to refuse here."""
    is_main = jax.process_index() == 0

    mesh = _training_mesh(model)
    with jax.set_mesh(mesh):
        prepared = _prepare_run(
            pd=pd,
            cadence=cadence,
            run=run,
            model=model,
            mesh=mesh,
            is_main=is_main,
            component_initializer=component_initializer,
        )

        def initialize_decomposition() -> Decomposition[Conditioning]:
            return init_decomposition(
                model, ci_fn_initializer, prepared.init_key, component_initializer
            )

        def initialize(decomposition: Decomposition[Conditioning]) -> TargetedPDState[Conditioning]:
            return TrainState(
                decomposition=decomposition,
                training=init_targeted_pd_training(
                    pd,
                    model,
                    positions,
                    prepared.opt_vu,
                    prepared.opt_ci,
                    decomposition,
                    prepared.src_key,
                    nontarget,
                ),
            )

        shape = eqx.filter_eval_shape(lambda: initialize(initialize_decomposition()))
        if is_main:
            print(shape.decomposition.ci_fn.placement_for_log(), flush=True)
        substrate = ForwardSubstrate.of(
            model,
            remat_recon_forwards=remat_recon_forwards,
            remat_ci_fn=remat_ci_fn,
            ci_capture_keys=shape.decomposition.ci_fn.capture_keys,
        )
        step_fn = make_targeted_train_step(
            model_static=model,
            substrate=substrate,
            components_optimizer=prepared.opt_vu,
            ci_fn_optimizer=prepared.opt_ci,
            total_steps=pd.steps,
        )

        def training_step(
            model: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
            state: TargetedPDState[Conditioning],
            target_batch: TargetIn,
            nontarget_batch: TargetIn,
            run_key: PRNGKeyArray,
        ) -> tuple[TargetedPDState[Conditioning], dict[str, jax.Array]]:
            return step_fn(
                model,
                state,
                target_batch,
                nontarget_batch,
                random.fold_in(run_key, state.training.step),
            )

        def run_step(
            state: TargetedPDState[Conditioning], step: int
        ) -> tuple[TargetedPDState[Conditioning], dict[str, jax.Array]]:
            return compiled(
                model,
                state,
                sample_target_batch(step),
                sample_nontarget_batch(step),
                prepared.run_key,
            )

        compiled = (
            jax.jit(training_step, donate_argnums=(1, 2, 3), compiler_options=compiler_options)
            .lower(
                model,
                shape,
                sample_target_batch(prepared.start_step),
                sample_nontarget_batch(prepared.start_step),
                prepared.run_key,
            )
            .compile()
        )
        state = _start_training(
            prepared.start,
            initialize_decomposition=initialize_decomposition,
            initialize=initialize,
            destination=restore_destination(shape, compiled.input_formats[0][1], prepared.mesh),
            mesh=prepared.mesh,
            is_main=is_main,
        )
        step_cost = StepCost.of(compiled, jax.devices())
        jax.block_until_ready((model, state))
        _run_loop(
            pd,
            cadence,
            build_evaluation,
            sink,
            prepared,
            state,
            is_main,
            run_step,
            step_cost,
            profiling,
            mfu_accounting,
        )


@contextmanager
def _jax_trace(run_dir: Path, steps: range, is_main: bool) -> Iterator[None]:
    assert steps, "a profile trace needs at least one execution"
    profile_dir = str(run_dir / "profile")
    if is_main:
        options = jax.profiler.ProfileOptions()
        options.host_tracer_level = 1
        options.device_tracer_level = 1
        options.python_tracer_level = 0
        options.advanced_configuration = {"gpu_max_activity_api_events": 2_000_000}
        jax.profiler.start_trace(
            profile_dir,
            create_perfetto_trace=True,
            profiler_options=options,
        )
    try:
        if is_main:
            print(f"profiling steps {steps[0]}..{steps[-1]}", flush=True)
        yield
    finally:
        if is_main:
            jax.profiler.stop_trace()
    if is_main:
        print(f"profile written to {profile_dir}", flush=True)


def _training_steps[Conditioning, Training: TrainingProgress](
    state: TrainState[Conditioning, Training],
    run_step: Callable[
        [TrainState[Conditioning, Training], int],
        tuple[TrainState[Conditioning, Training], dict[str, jax.Array]],
    ],
    start_step: int,
    total_steps: int,
    profiling: ProfilingMode | None,
    run_dir: Path,
    is_main: bool,
) -> Iterator[tuple[int, TrainState[Conditioning, Training], dict[str, jax.Array]]]:
    """Execute steps, yielding those due for training bookkeeping.

    JAX warmup yields normally; the traced tail completes without yielding so logging,
    evaluation, and checkpointing stay outside the capture. Annotation scopes close
    before each yield.
    """
    match profiling:
        case JaxProfilerTrace(steps=steps):
            profile_start = start_step + _PROFILE_WARMUP_EXECUTIONS
            profile_stop = profile_start + steps
            assert profile_stop <= total_steps, (
                f"profiling requires {_PROFILE_WARMUP_EXECUTIONS} warmup executions plus "
                f"{steps} marked executions, but only {total_steps - start_step} steps remain"
            )
            for step in range(start_step, profile_start):
                state, metrics = run_step(state, step)
                jax.block_until_ready(metrics["total"])
                yield step, state, metrics
            trace_steps = range(profile_start, profile_stop)
            with _jax_trace(run_dir, trace_steps, is_main):
                for step in trace_steps:
                    with jax.profiler.TraceAnnotation("param_decomp.profile_step", step_num=step):
                        state, metrics = run_step(state, step)
                        jax.block_until_ready(metrics["total"])
        case NsightCaptureWindow() as window:
            for step in range(start_step, total_steps):
                if window.contains(start_step, step):
                    with nvtx.annotate(
                        "param_decomp.profile_step", domain="param_decomp", payload=step
                    ):
                        state, metrics = run_step(state, step)
                        jax.block_until_ready(metrics["total"])
                else:
                    state, metrics = run_step(state, step)
                yield step, state, metrics
        case None:
            for step in range(start_step, total_steps):
                state, metrics = run_step(state, step)
                yield step, state, metrics


def _run_loop[Conditioning, Training: TrainingProgress, EvalPassT, EvalContextT](
    pd: PDConfigBase,
    cadence: Cadence,
    build_evaluation: EvaluationBuilder[Conditioning, EvalPassT, EvalContextT] | None,
    sink: MetricsSink,
    prepared: _RunResources,
    state: TrainState[Conditioning, Training],
    is_main: bool,
    run_step: Callable[
        [TrainState[Conditioning, Training], int],
        tuple[TrainState[Conditioning, Training], dict[str, jax.Array]],
    ],
    step_cost: StepCost,
    profiling: ProfilingMode | None,
    mfu_accounting: MfuAccounting | None,
) -> None:
    compiled_evaluation = (
        build_evaluation(prepared.run_key).prepare(
            EvalInvocation(
                decomposition=state.decomposition,
                persistent_sources={
                    name: adversary.sources
                    for name, adversary in state.training.adversaries.items()
                },
                now_step=prepared.start_step,
            )
        )
        if build_evaluation is not None
        else None
    )
    compute_norm_ratios = (
        jax.jit(uv_norm_ratio_metrics).lower(state.decomposition.components).compile()
    )
    saver = prepared.saver
    match profiling:
        case JaxProfilerTrace():
            pass
        case NsightCaptureWindow() | None:
            match prepared.start:
                case FreshTraining() | FineTune():
                    if saver is not None:
                        save_state(saver.manager, 0, state)
                case ResumeTraining():
                    pass

    _run_timed_loop(
        pd,
        cadence,
        compiled_evaluation,
        sink,
        prepared,
        state,
        is_main,
        run_step,
        step_cost,
        profiling,
        mfu_accounting,
        compute_norm_ratios,
    )


def _run_timed_loop[Conditioning, Training: TrainingProgress, EvalPassT, EvalContextT](
    pd: PDConfigBase,
    cadence: Cadence,
    evaluation: Evaluation[Conditioning, EvalPassT, EvalContextT] | None,
    sink: MetricsSink,
    prepared: _RunResources,
    state: TrainState[Conditioning, Training],
    is_main: bool,
    run_step: Callable[
        [TrainState[Conditioning, Training], int],
        tuple[TrainState[Conditioning, Training], dict[str, jax.Array]],
    ],
    step_cost: StepCost,
    profiling: ProfilingMode | None,
    mfu_accounting: MfuAccounting | None,
    compute_norm_ratios: Callable[[ComponentStacks], dict[str, jax.Array]],
) -> None:
    """Training windows exclude evaluation and checkpointing; the loop clock includes both."""
    saver = prepared.saver
    start_step = prepared.start_step

    performance = PerformanceTracker(monotonic(), mfu_accounting)

    if evaluation is not None and start_step == 0:
        baseline = _run_due_evaluation(evaluation, state, 0)
        if baseline is not None:
            baseline_record = dict(
                _with_uv_norm_ratios(baseline, state.decomposition, compute_norm_ratios)
            )
            dict_safe_update_(baseline_record, performance.metrics(monotonic()))
            sink.log(0, baseline_record)

    grad_norm_summary_window: list[dict[str, jax.Array]] = []
    training_steps = _training_steps(
        state, run_step, start_step, pd.steps, profiling, prepared.run_dir, is_main
    )
    performance.training_started(monotonic())
    for step, state, metrics in training_steps:
        performance.record_step(step)
        grad_norm_summary_window.append(
            {k: v for k, v in metrics.items() if k.startswith("grad_norms/summary/")}
        )

        now_step = step + 1
        dense = cadence.dense_log_phase
        train_record: LogRecord | None = None
        log_now = (
            now_step % cadence.train_log_every == 0
            or now_step == pd.steps
            or (dense is not None and now_step <= dense.until_step and now_step % dense.every == 0)
        )
        sigterm = log_now and prepared.sigterm_consensus()
        evaluate_now = (
            evaluation is not None
            and not sigterm
            and any(eval_due(operation.schedule, now_step) for operation in evaluation.operations)
        )
        save_now = saver is not None and (
            now_step % saver.save_every == 0 or now_step == pd.steps or sigterm
        )
        pause_training = log_now or evaluate_now or save_now
        if pause_training:
            jax.block_until_ready((state, metrics))
            performance.training_paused(monotonic())
        if log_now:
            timing = performance.metrics(monotonic())
            per_step = timing["train/perf/step_time_s"]
            record = {
                k: float(v) for k, v in metrics.items() if not k.startswith("grad_norms/summary/")
            }
            dict_safe_update_(record, _grad_norm_summary_window_stats(grad_norm_summary_window))
            grad_norm_summary_window.clear()
            nonfinite = {key: value for key, value in record.items() if not math.isfinite(value)}
            nonfinite_losses = {
                key: value
                for key, value in nonfinite.items()
                if key == "total" or key.startswith("loss/")
            }
            nonfinite_other = tuple(key for key in nonfinite if key not in nonfinite_losses)
            assert not nonfinite, (
                f"non-finite metrics at step {now_step}: losses={nonfinite_losses}; "
                f"other_count={len(nonfinite_other)}; other_first={nonfinite_other[:20]}"
            )
            record["flops_per_step"] = step_cost.flops_per_step
            if (hfu := step_cost.hfu(per_step)) is not None:
                record["hfu"] = hfu
            record["eta_s"] = (pd.steps - now_step) * per_step
            record["train/schedules/lr/components"] = float(
                state.training.components_opt_state.applied_learning_rate
            )
            record["train/schedules/lr/ci_fn"] = float(
                state.training.ci_fn_opt_state.applied_learning_rate
            )
            mem_stats = jax.local_devices()[0].memory_stats()
            if mem_stats is not None:
                record["train/mem/peak_gb_per_rank"] = mem_stats["peak_bytes_in_use"] / 1e9
                # Device 0 is also the default device, so its high-water mark carries
                # init-time weight staging (full-model assembly before resharding) and
                # can mask step demand for the whole run; the min across local devices
                # is staging-free — the step-and-eval high-water the fit check's
                # verdict is actually about.
                record["train/mem/peak_gb_min_device"] = (
                    min(
                        stats["peak_bytes_in_use"]
                        for device in jax.local_devices()
                        if (stats := device.memory_stats()) is not None
                    )
                    / 1e9
                )
            train_record = record

        eval_record = (
            _run_due_evaluation(evaluation, state, now_step)
            if evaluation is not None and evaluate_now
            else None
        )
        if eval_record is not None:
            eval_record = _with_uv_norm_ratios(
                eval_record, state.decomposition, compute_norm_ratios
            )
        if saver is not None and save_now:
            save_state(saver.manager, now_step, state)
            if is_main:
                print(f"checkpoint saved @ step {now_step}", flush=True)
        if now_step == pd.steps or sigterm:
            sink.wait_for_renderers()
        step_record = _combine_step_records(train_record, eval_record)
        if step_record is not None:
            timing = performance.metrics(monotonic())
            record_with_performance = dict(step_record)
            dict_safe_update_(record_with_performance, timing)
            if log_now:
                record_with_performance["elapsed_s"] = timing["train/perf/runtime_s"]
            sink.log(now_step, record_with_performance)
            if log_now:
                performance.reset_window(monotonic())
        if pause_training and now_step < pd.steps and not sigterm:
            performance.training_started(monotonic())
        if sigterm:
            if is_main:
                print(
                    "SIGTERM: checkpoint saved, exiting for requeue"
                    if saver is not None
                    else "SIGTERM: no checkpoint (cadence.checkpointing: none), exiting",
                    flush=True,
                )
            break
