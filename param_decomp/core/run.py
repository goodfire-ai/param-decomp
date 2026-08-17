"""The generic VPD decomposition-training ENGINE — the one train loop every target
(LM, TMS, ResidMLP, …) runs through.

`run_decomposition_training(pd, cadence, run, model, ci_fn, positions,
remat_recon_forwards, sample_batch, evaluation, mesh)` owns
the generic machinery: init / restore / fine-tune init / faith warmup
(`_init_or_restore_state`), the recon-plan traversal, orbax checkpointing, schedules,
metrics fan-out (`MetricsSink`), the figure-tier background renderer (`BackgroundRenderer`), and
SIGTERM-save for SLURM requeue. It reads the pydantic `PDConfig` / `Cadence` DIRECTLY; the
target injects two seams: the data source (`sample_batch`) and domain-bound `evaluation`.

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
from collections.abc import Callable, Mapping
from functools import partial as _partial
from types import FrameType, ModuleType
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import yaml
from jax import random
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import PRNGKeyArray

from param_decomp.core.built_run import LAUNCH_CONFIG_FILENAME, RunInstance
from param_decomp.core.checkpoint import (
    init_from_parent,
    make_checkpoint_manager,
    restore_latest,
    save_state,
)
from param_decomp.core.ci_fn import CIFnArch
from param_decomp.core.components import init_component_stacks
from param_decomp.core.configs import (
    AnyPDConfig,
    Cadence,
    NontargetConfig,
    PDConfig,
    PDConfigBase,
    TargetedPDConfig,
    flatten_typed_lists,
)
from param_decomp.core.eval_schedule import EvalSchedule, eval_due
from param_decomp.core.metrics import BarChart, LogRecord, PNGImage
from param_decomp.core.model import DecomposedModel, PositionAxis
from param_decomp.core.objective import build_objective, build_targeted_objective
from param_decomp.core.placement import PlacementRules, component_stacks_audit
from param_decomp.core.run_state import build_optimizers, init_train_state
from param_decomp.core.train import (
    CIScaledWeightDecay,
    TrainState,
    make_faith_warmup_step,
    make_targeted_train_step,
    make_train_step,
    uv_norm_ratio_metrics,
)


@dataclasses.dataclass(frozen=True)
class EvalInvocation:
    state: TrainState
    now_step: int


@dataclasses.dataclass(frozen=True)
class EvalOperation[ContextT]:
    schedule: EvalSchedule
    run: Callable[[ContextT], LogRecord]


@dataclasses.dataclass(frozen=True)
class Evaluation[ContextT]:
    operations: tuple[EvalOperation[ContextT], ...]
    make_context: Callable[[TrainState, int], ContextT]

    def __post_init__(self) -> None:
        assert self.operations, "evaluation needs at least one operation"


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


def _run_due_evaluation[ContextT](
    evaluation: Evaluation[ContextT], state: TrainState, now_step: int
) -> LogRecord | None:
    due_operations = tuple(
        operation for operation in evaluation.operations if eval_due(operation.schedule, now_step)
    )
    if not due_operations:
        return None
    context = evaluation.make_context(state, now_step)
    record: dict[str, float | BarChart | PNGImage] = {}
    for operation in due_operations:
        values = operation.run(context)
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


def _sigterm_consensus() -> bool:
    """Cross-rank-agreed SIGTERM flag. SLURM delivers SIGTERM per task with no simultaneity
    guarantee, so reading the per-process flag independently at a collective gate (faith-warmup
    exit, eval entry, orbax save) can diverge ranks and hang. OR-reduce it across processes;
    callers read it once per step into a local the handler can't mutate mid-step. No-op when not
    distributed."""
    if jax.process_count() == 1:
        return _sigterm_received
    import jax.experimental.multihost_utils as mhu

    return bool(np.asarray(mhu.process_allgather(np.asarray(_sigterm_received))).any())


def log_wandb_safe(
    wandb_module: "ModuleType",
    payload: Mapping[str, object],
    step: int | None,
    commit: bool,
    what: str,
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
        print(f"wandb communication error, skipping {what}: {e}", flush=True)


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
    "imp_smooth_l0": "train/loss/SmoothL0ImportanceMinimalityLoss",
    "freq": "train/loss/FrequencyMinimalityLoss",
    "p_imp": "train/schedules/p_imp",
    "gamma_imp": "train/schedules/gamma_imp",
    "src_lr": "train/schedules/lr/src",
    "step_time_s": "train/perf/step_time_s",
    "elapsed_s": "train/perf/elapsed_s",
    "eta_s": "train/perf/eta_s",
}


def _fmt_duration(seconds: float) -> str:
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
    stacked = jnp.stack([jnp.stack([snap[k] for snap in window]) for k in keys])  # [keys, steps]
    mins = np.asarray(jnp.min(stacked, axis=1))
    maxs = np.asarray(jnp.max(stacked, axis=1))
    medians = np.asarray(jnp.median(stacked, axis=1))
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

    def __init__(self, jsonl: io.TextIOWrapper | None, wandb_module: ModuleType | None):
        self._jsonl = jsonl
        self._wandb = wandb_module
        self._wandb_lock = threading.Lock()
        self._defined_deferred_metrics: set[tuple[str, str]] = set()
        self._deferred_media_keys: set[tuple[str, int, str]] = set()
        self._last_committed_step: int | None = None

    @classmethod
    def silent(cls) -> "MetricsSink":
        """NO metrics anywhere — `log` drops the record before it reaches either channel, so
        the run writes no `metrics.jsonl` at all. This is not "wandb off": a run without a
        tracker still wants its jsonl, and gets it from `for_run` (`wandb: null` in the
        config is what turns wandb off)."""
        return cls(jsonl=None, wandb_module=None)

    @classmethod
    def for_run(cls, run: RunInstance, is_main: bool) -> "MetricsSink":
        if not is_main:
            return cls.silent()
        jsonl = (run.run_dir / "metrics.jsonl").open("a")
        if run.wandb is None:
            return cls(jsonl=jsonl, wandb_module=None)
        import wandb

        # wandb.config is the pinned launch config verbatim — the run's ONE self-contained
        # yaml (the same bytes resume byte-compares), so programmatic config access works.
        # The metric lists flatten into the same flat keys torch logged (E14) so cross-impl
        # wandb config queries line up. Nothing else rides along: topology is config-driven
        # (`runtime.dp`, asserted in `init_distributed`), so the config IS the topology.
        launch_config = run.run_dir / LAUNCH_CONFIG_FILENAME
        assert launch_config.exists(), launch_config
        wandb.init(
            project=run.wandb.project,
            entity=run.wandb.entity,
            name=run.run_name,
            id=run.run_id,
            group=run.wandb.group,
            tags=list(run.wandb.tags),
            resume="allow",
            config=flatten_typed_lists(yaml.safe_load(launch_config.read_text())),
        )
        # Also save the pin as a downloadable wandb run file, alongside (not in place of)
        # the wandb.config dict.
        wandb.save(str(launch_config), base_path=str(run.run_dir), policy="now")
        return cls(jsonl=jsonl, wandb_module=wandb)

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
                k,
                f"train/{k}"
                if k.startswith(("grad_norms/", "loss/", "schedules/", "nontarget_data/"))
                else k,
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
            head += f" {_fmt_duration(elapsed)}<{_fmt_duration(eta)}"
        print(head + " " + " ".join(f"{k}={v:.4g}" for k, v in console.items()), flush=True)
        if self._wandb is not None:
            wandb_record: dict[str, object] = {}
            for key, value in record.items():
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
            semantic_keys = {(record.step_key, record.step, key) for key in record.media}
            overlap = self._deferred_media_keys & semantic_keys
            assert not overlap, (
                "deferred renderers emitted colliding semantic keys: "
                f"{sorted(key for _, _, key in overlap)} at step {record.step}"
            )
            self._deferred_media_keys.update(semantic_keys)
            payload: dict[str, object] = {record.step_key: float(record.step)}
            for key, encoded in record.media.items():
                registration = (key, record.step_key)
                if registration not in self._defined_deferred_metrics:
                    self._wandb.define_metric(key, step_metric=record.step_key)
                    self._defined_deferred_metrics.add(registration)
                payload[key] = self._wandb.Image(Image.open(io.BytesIO(encoded)))
            log_wandb_safe(self._wandb, payload, None, False, "deferred media")


class BackgroundRenderer:
    """Background thread for a figure tier's pure host-rendering tail.

    The slow/plot tier (SPEC S28/S29) and the LM arithmetic tier each hold one. A pure
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
    have advanced `_step`, and W&B correctly rejects out-of-order writes. An `atexit` join
    flushes the last render
    before process exit (the trainer never calls `wandb.finish`). The atexit handler is
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
    """SPEC S21's warmup phase as the engine consumes it — built by the PLAIN entry from
    `PDConfig`'s fields. A targeted run has no faithfulness role to warm (T3): its config
    shape carries no warmup fields, and its entry passes no warmup by construction."""

    steps: int
    lr: float
    weight_decay: float


def _init_or_restore_state(
    pd: AnyPDConfig,
    ci_fn_arch: CIFnArch,
    positions: PositionAxis,
    run: RunInstance,
    model: DecomposedModel,
    opt_vu: optax.GradientTransformation,
    opt_ci: optax.GradientTransformation,
    init_key: PRNGKeyArray,
    src_key: PRNGKeyArray,
    mesh: Mesh,
    rules: PlacementRules,
    checkpoint_manager: ocp.CheckpointManager,
    is_main: bool,
    compiler_options: dict[str, bool | int | str],
    faith_warmup: FaithfulnessWarmup | None,
) -> tuple[TrainState, int] | None:
    """The shared init/restore/finetune/faith-warmup phase (SPEC S21/S22/S33).

    Returns `(state, start_step)`, or `None` when a SIGTERM landed mid-warmup (the caller
    must exit cleanly for requeue — no valid checkpoint exists pre-step-0)."""
    state = _ensure_global(
        init_train_state(
            pd, model, ci_fn_arch, positions, opt_vu, opt_ci, init_key, src_key, mesh, rules
        ),
        mesh,
    )

    restored = restore_latest(checkpoint_manager, state)
    if restored is not None:
        state, ckpt_step = restored
        assert int(state.training.step) == ckpt_step, (int(state.training.step), ckpt_step)
        # A mid-training restore is the requeue path and proceeds; a restore at (or past)
        # the configured horizon has nothing to run, and exiting 0 there is indistinguishable
        # from a run that trained. Raising `pd.steps` on this id is not an option either —
        # the pinned launch config byte-compares on every re-entry.
        assert ckpt_step < pd.steps, (
            f"run {run.run_id} is already trained to step {ckpt_step} of pd.steps={pd.steps}: "
            "nothing left to run. There is no way to extend this trajectory: raising `pd.steps` "
            "on this id is refused by the pinned-config byte-compare, a new run id trains from "
            "scratch, and a new run id with `resume_provenance` inherits the decomposition only "
            "— fresh optimizer state, fresh adversaries, step 0, schedules re-annealed."
        )
        if is_main:
            print(f"resumed from checkpoint step {ckpt_step}", flush=True)
        return state, ckpt_step

    if run.resume_provenance is not None:
        # Fine-tune init (SPEC S33): own ckpts/ is empty, so this is the FIRST entry, not a
        # requeue — load the parent's trained V/U + ci_fn onto the fresh reference, start a
        # clean schedule from step 0 (fresh optimizer / sources, no faith warmup). The
        # parent↔new structural-compat check (sites + ci-fn arch) runs lab-side in the LM
        # composition root before this engine is entered.
        prov = run.resume_provenance
        state = init_from_parent(prov.parent_run_dir / "ckpts", prov.parent_step, state)
        save_state(checkpoint_manager, 0, state)
        if is_main:
            print(
                f"fine-tune: initialized V/U + ci_fn from {prov.parent_run_dir} "
                f"step {prov.parent_step}; training fresh from step 0",
                flush=True,
            )
        return state, 0

    if faith_warmup is not None:
        faith_warmup_optimizer = optax.adamw(
            faith_warmup.lr, weight_decay=faith_warmup.weight_decay
        )
        faith_warmup_opt_state = faith_warmup_optimizer.init(
            eqx.filter(state.decomposition.components, eqx.is_array)
        )
        faith_warmup_step = make_faith_warmup_step(faith_warmup_optimizer, compiler_options)
        warmed_components = state.decomposition.components
        t0 = time.time()
        faith_warmup_loss = None
        for _ in range(faith_warmup.steps):
            warmed_components, faith_warmup_opt_state, faith_warmup_loss = faith_warmup_step(
                model, warmed_components, faith_warmup_opt_state
            )
            if _sigterm_consensus():
                # No valid checkpoint exists yet (the step-0 save happens only after warmup
                # completes, and resume skips warmup whenever a checkpoint is present — a
                # partially-warmed step-0 save would resume as if fully warmed). Exit
                # cleanly; the SLURM requeue redoes warmup from scratch.
                if is_main:
                    print("SIGTERM during faith warmup: exiting for requeue", flush=True)
                return None
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
                f"faith warmup: {faith_warmup.steps} steps in {time.time() - t0:.0f}s, "
                f"final faith {float(faith_warmup_loss):.3e}",
                flush=True,
            )
    save_state(checkpoint_manager, 0, state)
    return state, 0


@dataclasses.dataclass(frozen=True)
class _PreparedRun:
    """The generic pre-loop phase's outputs, shared by both engine entries: the built
    optimizer pair + their schedule fns (re-read at log time so the reported LR is the
    applied one), the per-step batch/RNG key root, the checkpoint manager, and the
    initialized-or-restored state."""

    opt_vu: optax.GradientTransformation
    opt_ci: optax.GradientTransformation
    sched_vu: Callable[[Any], jax.Array]
    sched_ci: Callable[[Any], jax.Array]
    run_key: PRNGKeyArray
    checkpoint_manager: ocp.CheckpointManager
    state: TrainState
    start_step: int


def _prepare_run(
    pd: AnyPDConfig,
    cadence: Cadence,
    run: RunInstance,
    model: DecomposedModel,
    ci_fn: CIFnArch,
    positions: PositionAxis,
    compiler_options: dict[str, bool | int | str],
    mesh: Mesh,
    placement_rules: PlacementRules,
    is_main: bool,
    faith_warmup: FaithfulnessWarmup | None,
) -> _PreparedRun | None:
    """Everything before the train loop: mesh activation, optimizers, keys, checkpoint
    manager, the placement audit, and init/restore/finetune/faith-warmup. Returns `None`
    when a SIGTERM landed mid-warmup (the caller exits cleanly for requeue)."""
    # Activate the mesh so bare-PartitionSpec `with_sharding_constraint`s inside the forward
    # resolve (the attn q/k/v batch-sharding pin in `FrozenAttn.core`, needed for cuDNN
    # flash attention under the scan+cond masked forward). Explicit NamedShardings elsewhere
    # are unaffected.
    jax.set_mesh(mesh)
    run.run_dir.mkdir(parents=True, exist_ok=True)
    opt_vu, opt_ci, (sched_vu, sched_ci) = build_optimizers(pd, ci_fn, mesh)

    key = random.PRNGKey(pd.seed)
    init_key, src_key, run_key = random.split(key, 3)

    checkpoint_manager = make_checkpoint_manager(
        run.run_dir / "ckpts", cadence.checkpoint_retention
    )
    rules = placement_rules
    if is_main:
        audit = component_stacks_audit(
            eqx.filter_eval_shape(_partial(init_component_stacks, model.sites), init_key), rules
        )
        print(
            rules.describe(
                tensors=audit,
                not_audited=("ci_fn", "frozen target", "persistent sources", "opt state"),
            ),
            flush=True,
        )
    init = _init_or_restore_state(
        pd, ci_fn, positions, run, model, opt_vu, opt_ci, init_key, src_key, mesh, rules,
        checkpoint_manager, is_main, compiler_options, faith_warmup,
    )  # fmt: skip
    if init is None:
        return None  # SIGTERM mid-warmup: clean exit for requeue
    state, start_step = init
    return _PreparedRun(
        opt_vu=opt_vu,
        opt_ci=opt_ci,
        sched_vu=sched_vu,
        sched_ci=sched_ci,
        run_key=run_key,
        checkpoint_manager=checkpoint_manager,
        state=state,
        start_step=start_step,
    )


def run_decomposition_training[EvalContextT](
    pd: PDConfig,
    cadence: Cadence,
    run: RunInstance,
    model: DecomposedModel,
    ci_fn: CIFnArch,
    positions: PositionAxis,
    remat_recon_forwards: bool,
    remat_ci_fn: bool,
    ascend_replicate: bool,
    compiler_options: dict[str, bool | int | str],
    sample_batch: Callable[[int], Any],
    evaluation: Evaluation[EvalContextT] | None,
    sink: MetricsSink,
    mesh: Mesh,
    placement_rules: PlacementRules,
) -> None:
    """The generic VPD decomposition-training engine — the ONE train loop every target
    (LM, TMS, ResidMLP, …) runs through.

    Reads the pydantic algorithm config DIRECTLY: `pd` (seed / steps / optimizers / loss
    metrics / faith warmup), `cadence` (log / save / checkpoint-retention
    rhythm), `run` (the run identity + wandb lineage). The lab-built objects ride alongside:
    the decomposed `model` (an `eqx.Module` carrying the frozen target weights as
    fields — threaded into the jitted step as a pytree arg, never closed over), the CI-fn
    arch `ci_fn`, the run's waist geometry (`positions`: `Positioned(seq_len)` for an LM,
    `Positionless()` for a toy), and the `remat_recon_forwards` compute knob.

    The target supplies only its three injectable seams:

    - `sample_batch(step) -> batch`: the opaque per-step model input (a pure function of
      `step`, for O(1) resume). The model interprets it (an LM's token ids `[B, T]` → embed;
      a toy's feature vector, which already is the `[*leading, d]` waist). The engine only
      assumes axis 0 is the batch/`dp` axis (for sharding); it never names tokens or `d`.
    - `evaluation`: a fixed tuple of domain-bound operations plus the typed context factory
      they share. The engine alone schedules due operations, constructs one context, merges
      disjoint records, and logs the result. `None` disables evaluation.

    Everything generic — `init_train_state`, fine-tune init, faith warmup, the recon-grid
    step factory, orbax checkpointing, schedules, SIGTERM-save — lives here. The step
    numerics are identical across targets; only the data source and the eval metric differ.

    The targeted (tPD) twin is `run_targeted_decomposition_training`; both entries compose
    the same `_prepare_run` / `_run_loop` core.
    """
    is_main = jax.process_index() == 0
    faith_warmup = (
        FaithfulnessWarmup(
            steps=pd.faithfulness_warmup_steps,
            lr=pd.faithfulness_warmup_lr,
            weight_decay=pd.faithfulness_warmup_weight_decay,
        )
        if pd.faithfulness_warmup_steps > 0
        else None
    )
    prepared = _prepare_run(
        pd, cadence, run, model, ci_fn, positions, compiler_options, mesh, placement_rules,
        is_main, faith_warmup,
    )  # fmt: skip
    if prepared is None:
        return

    step_fn = make_train_step(
        model_static=model,
        losses=build_objective(pd.loss_metrics, model.site_names),
        components_optimizer=prepared.opt_vu,
        ci_fn_optimizer=prepared.opt_ci,
        total_steps=pd.steps,
        remat_recon_forwards=remat_recon_forwards,
        remat_ci_fn=remat_ci_fn,
        ci_capture_keys=prepared.state.decomposition.ci_fn.capture_keys,
        ascend_replicate=ascend_replicate,
        compiler_options=compiler_options,
        mesh=mesh,
    )

    def run_step(state: TrainState, step: int) -> tuple[TrainState, dict[str, jax.Array]]:
        return step_fn(model, state, sample_batch(step), random.fold_in(prepared.run_key, step))

    _run_loop(pd, cadence, evaluation, sink, prepared, is_main, run_step)


def run_targeted_decomposition_training[EvalContextT](
    pd: TargetedPDConfig,
    nontarget: NontargetConfig,
    cadence: Cadence,
    run: RunInstance,
    model: DecomposedModel,
    ci_fn: CIFnArch,
    positions: PositionAxis,
    remat_recon_forwards: bool,
    remat_ci_fn: bool,
    ascend_replicate: bool,
    compiler_options: dict[str, bool | int | str],
    sample_target_batch: Callable[[int], Any],
    sample_nontarget_batch: Callable[[int], Any],
    evaluation: Evaluation[EvalContextT] | None,
    sink: MetricsSink,
    mesh: Mesh,
    placement_rules: PlacementRules,
) -> None:
    """The targeted-PD (tPD) engine entry (SPEC §11) — `run_decomposition_training`'s twin
    over the same `_prepare_run` / `_run_loop` core, stepping the two-pass
    `make_targeted_train_step`.

    Two data seams instead of one: `sample_target_batch(step)` feeds the narrow TARGET
    stream (global batch `pd.batch_size` — the pass the persistent adversaries and every
    other decomposition loss run on), and `sample_nontarget_batch(step)` the broad
    NON-TARGET stream (global batch `nontarget.batch_size`, delta pinned fully on save
    T4's one unmasked-no-delta exception). `positions` is the TARGET stream's waist
    geometry — persistent sources live in the target pass; each stream runs at its own
    natural sequence length (SPEC T2/T8).

    tPD has no faithfulness role (T3): `TargetedPDConfig` admits no faithfulness loss
    member and carries no warmup fields, so neither exists to refuse here."""
    is_main = jax.process_index() == 0
    prepared = _prepare_run(
        pd, cadence, run, model, ci_fn, positions, compiler_options, mesh, placement_rules,
        is_main, None,
    )  # fmt: skip
    if prepared is None:
        return

    step_fn = make_targeted_train_step(
        model_static=model,
        objective=build_targeted_objective(pd.loss_metrics, nontarget, model.site_names),
        ci_scaled_weight_decay=(
            CIScaledWeightDecay(pd.ci_scaled_weight_decay, prepared.sched_vu)
            if pd.ci_scaled_weight_decay is not None
            else None
        ),
        components_optimizer=prepared.opt_vu,
        ci_fn_optimizer=prepared.opt_ci,
        total_steps=pd.steps,
        remat_recon_forwards=remat_recon_forwards,
        remat_ci_fn=remat_ci_fn,
        ci_capture_keys=prepared.state.decomposition.ci_fn.capture_keys,
        ascend_replicate=ascend_replicate,
        compiler_options=compiler_options,
        mesh=mesh,
    )

    def run_step(state: TrainState, step: int) -> tuple[TrainState, dict[str, jax.Array]]:
        return step_fn(
            model,
            state,
            sample_target_batch(step),
            sample_nontarget_batch(step),
            random.fold_in(prepared.run_key, step),
        )

    _run_loop(pd, cadence, evaluation, sink, prepared, is_main, run_step)


def _run_loop[EvalContextT](
    pd: PDConfigBase,
    cadence: Cadence,
    evaluation: Evaluation[EvalContextT] | None,
    sink: MetricsSink,
    prepared: _PreparedRun,
    is_main: bool,
    run_step: Callable[[TrainState, int], tuple[TrainState, dict[str, jax.Array]]],
) -> None:
    """The generic train loop over one already-built `run_step(state, step)`: log cadence,
    eval scheduling, checkpointing, and SIGTERM-save — identical for both engine entries."""
    save_every = cadence.save_every
    checkpoint_manager = prepared.checkpoint_manager
    sched_vu, sched_ci = prepared.sched_vu, prepared.sched_ci
    state, start_step = prepared.state, prepared.start_step

    window_t0 = loop_t0 = time.time()
    last_logged = start_step
    grad_norm_summary_window: list[dict[str, jax.Array]] = []

    for step in range(start_step, pd.steps):
        state, metrics = run_step(state, step)

        grad_norm_summary_window.append(
            {k: v for k, v in metrics.items() if k.startswith("grad_norms/summary/")}
        )

        now_step = step + 1
        sigterm = _sigterm_consensus()
        dense = cadence.dense_log_phase
        train_record: LogRecord | None = None
        log_now = (
            now_step % cadence.train_log_every == 0
            or now_step == pd.steps
            or (dense is not None and now_step <= dense.until_step and now_step % dense.every == 0)
        )
        if log_now:
            jax.block_until_ready(metrics["total"])
            dt = time.time() - window_t0
            per_step = dt / max(now_step - last_logged, 1)
            last_logged = now_step
            record = {
                k: float(v) for k, v in metrics.items() if not k.startswith("grad_norms/summary/")
            }
            record.update(_grad_norm_summary_window_stats(grad_norm_summary_window))
            grad_norm_summary_window.clear()
            for loss_name in ("total", *(k for k in record if k.startswith("loss/"))):
                assert math.isfinite(record[loss_name]), (
                    f"non-finite loss {loss_name!r} at step {now_step}: {record[loss_name]}"
                )
            record["step_time_s"] = per_step
            record["elapsed_s"] = time.time() - loop_t0
            record["eta_s"] = (pd.steps - now_step) * per_step
            # the LR this step applied (optax count is the pre-increment `step` == now_step - 1)
            record["train/schedules/lr/components"] = float(jnp.asarray(sched_vu(now_step - 1)))
            record["train/schedules/lr/ci_fn"] = float(jnp.asarray(sched_ci(now_step - 1)))
            mem_stats = jax.local_devices()[0].memory_stats()
            if mem_stats is not None:
                record["train/mem/peak_gb_per_rank"] = mem_stats["peak_bytes_in_use"] / 1e9
            train_record = record

        eval_record = (
            _run_due_evaluation(evaluation, state, now_step)
            if evaluation is not None and not sigterm
            else None
        )
        if eval_record is not None:
            factor_record = {
                f"eval/{key}": float(value)
                for key, value in uv_norm_ratio_metrics(state.decomposition.components).items()
            }
            overlap = eval_record.keys() & factor_record.keys()
            assert not overlap, f"U/V norm-ratio metrics collided with eval keys: {sorted(overlap)}"
            eval_record = {**eval_record, **factor_record}
        step_record = _combine_step_records(train_record, eval_record)
        if step_record is not None:
            sink.log(now_step, step_record)
            window_t0 = time.time()

        if now_step % save_every == 0 or now_step == pd.steps or sigterm:
            save_state(checkpoint_manager, now_step, state)
            if is_main:
                print(f"checkpoint saved @ step {now_step}", flush=True)
            window_t0 = time.time()
        if sigterm:
            if is_main:
                print("SIGTERM: checkpoint saved, exiting for requeue", flush=True)
            break
