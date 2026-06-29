"""The generic VPD decomposition-training ENGINE — the one train loop every target
(LM, TMS, ResidMLP, …) runs through.

`run_decomposition_training(pd, cadence, run, lm, ci_fn, data,
remat_recon_forwards, sample_batch, eval_fn, eval_every, mesh)` owns
the generic machinery: init / restore / fine-tune init / faith warmup
(`_init_or_restore_state`), the recon-grid step factory, orbax checkpointing, schedules,
metrics fan-out (`MetricsSink`), the in-loop slow/plot renderer (`SlowEvalRenderer`), and
SIGTERM-save for SLURM requeue. It reads the pydantic `PDConfig` / `Cadence` DIRECTLY; the
target injects two seams: the data source (`sample_batch`) and the eval metric (`eval_fn`).

This module is a pure library — it has NO `main()` and reads no YAML. The per-domain
composition root (read the run YAML → build the target / data loader / `BuiltRun` → call
this engine) lives lab-side: `param_decomp_lab/experiments/lm/run.py` for the LM,
`param_decomp_lab/experiments/{tms,resid_mlp}/run.py` for the toys.
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
from types import FrameType, ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import wandb

    LogRecord = Mapping[str, float | wandb.plot.CustomChart]

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from jax import random
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import PRNGKeyArray

from param_decomp.built_run import DataConfig, RunInstance
from param_decomp.checkpoint import (
    init_from_parent,
    make_checkpoint_manager,
    restore_latest,
    save_state,
)
from param_decomp.ci_fn import CIFnArch
from param_decomp.configs import Cadence, PDConfig, flatten_typed_lists
from param_decomp.lm import DecomposedModel
from param_decomp.recon import build_loss_terms
from param_decomp.run_state import build_optimizers, init_train_state
from param_decomp.slow_eval import (
    PermutationMetricSpec,
    PositionCI,
    SiteReduction,
    render_permutation_figures,
    render_slow_eval_figures,
)
from param_decomp.train import TrainState, make_faith_warmup_step, make_train_step

_sigterm_received = False


def install_sigterm_flag() -> None:
    """Install the SIGTERM handler the engine's save-on-preempt logic reads. Called by the
    composition root (which owns process setup) before `run_decomposition_training`."""

    def handler(_signum: int, _frame: FrameType | None) -> None:
        global _sigterm_received
        _sigterm_received = True

    signal.signal(signal.SIGTERM, handler)


def sigterm_received() -> bool:
    """Whether a SIGTERM has landed. A composition root's eval pass reads this to abandon
    a partial eval cleanly (the engine's save block then checkpoints + exits for requeue)."""
    return _sigterm_received


def _log_wandb_safe(wandb_module: "ModuleType", payload: "LogRecord", step: int, what: str) -> None:
    """`wandb.log` swallowing `CommError` only — a transient wandb-server outage must not
    kill a multi-day run, while genuine misuse (e.g. a non-dict record) still raises. The
    soft-fail is deliberate (drops the failed record, keeps training)."""
    import wandb.errors

    try:
        wandb_module.log(payload, step=step)
    except wandb.errors.CommError as e:
        print(f"wandb communication error, skipping {what}: {e}", flush=True)


def _ensure_global[T](tree: T, mesh: Mesh) -> T:
    """Re-materialize the NON-mesh array leaves (eagerly created scalars: step
    counters, Adam counts) as well-formed GLOBAL replicated arrays via an identity
    jit. Multi-controller orbax can only save global arrays — and an eager
    `device_put(local, replicated-NamedSharding)` yields arrays whose
    `addressable_shards` raise (jax 0.10 multi-process), while jit outputs with the
    same sharding are well-formed.

    Leaves that already carry a NamedSharding pass through UNTOUCHED: routing them
    through the identity jit re-materializes the whole state in one executable,
    which OOM'd at the multi-chunk config's ~110 GB global state (job 50458,
    168 GiB alloc in jit__identity_fn)."""
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
    "p_imp": "train/schedules/p_imp",
    "src_lr": "train/schedules/lr/src",
    "step_time_s": "train/perf/step_time_s",
    "elapsed_s": "train/perf/elapsed_s",
    "eta_s": "train/perf/eta_s",
}


def _fmt_duration(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _is_verbose_grad_norm(key: str) -> bool:
    return key.startswith("train/grad_norms/") and not key.startswith("train/grad_norms/summary/")


class MetricsSink:
    """Process-0 metrics fan-out: jsonl always, wandb when configured.

    Construct via `for_run` (main rank, opens jsonl + maybe wandb) or `silent` (the no-op
    handle for non-main DP ranks, tests, and quick interactive runs) — not the resolved-
    channel `__init__` directly."""

    def __init__(self, jsonl: io.TextIOWrapper | None, wandb_module: ModuleType | None):
        self._jsonl = jsonl
        self._wandb = wandb_module

    @classmethod
    def silent(cls) -> "MetricsSink":
        return cls(jsonl=None, wandb_module=None)

    @classmethod
    def for_run(
        cls, run: RunInstance, wandb_config: dict[str, object], is_main: bool
    ) -> "MetricsSink":
        if not is_main:
            return cls.silent()
        jsonl = (run.run_dir / "metrics.jsonl").open("a")
        if run.wandb is None:
            return cls(jsonl=jsonl, wandb_module=None)
        import wandb

        wandb.init(
            project=run.wandb.project,
            entity=run.wandb.entity,
            name=run.run_name,
            id=run.run_id,
            group=run.wandb.group,
            tags=list(run.wandb.tags),
            resume="allow",
            config=wandb_config,
        )
        # Persist the run's pinned config.yaml as a downloadable wandb run file
        # (parity with the torch trainer's init_pd_run -> wandb.save), not just the
        # flattened wandb.config dict. Pinned to run_dir before train() / wandb.init.
        config_yaml = run.run_dir / "config.yaml"
        assert config_yaml.exists(), config_yaml
        wandb.save(str(config_yaml), base_path=str(run.run_dir), policy="now")
        # The in-loop slow tier (`SlowEvalRenderer`) logs `slow_eval/*` on the live
        # `_step` axis at the eval step (SPEC S28/S29), so NO dedicated `slow_eval/step`
        # metric is defined here. Slow eval is in-loop only (no offline CLI).
        return cls(jsonl=jsonl, wandb_module=wandb)

    def log(self, step: int, record: "LogRecord") -> None:
        if self._jsonl is None:
            return
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
        console = {k: v for k, v in scalars.items() if not _is_verbose_grad_norm(k)}
        head = f"[step {step}]"
        if "train/perf/eta_s" in console:  # train logs carry the paired timing; eval logs don't
            elapsed, eta = console.pop("train/perf/elapsed_s"), console.pop("train/perf/eta_s")
            head += f" {_fmt_duration(elapsed)}<{_fmt_duration(eta)}"
        print(head + " " + " ".join(f"{k}={v:.4g}" for k, v in console.items()), flush=True)
        if self._wandb is not None:
            _log_wandb_safe(self._wandb, record, step, "log")


class SlowEvalRenderer:
    """Rank-0 background renderer for the in-loop slow/plot tier (SPEC S28/S29).

    The collective part of slow eval (the jitted forward + the device->host pull whose
    `np.asarray` triggers the C-shard all-gather) runs in lockstep on ALL ranks inside the
    eval pass. This renderer takes ONLY the materialized numpy reductions (the per-site
    `SiteReduction` plot inputs; when the config names a CI-heatmap/permutation metric, the
    batch-mean `(T, C)` position CI; and when the config names `UVPlots`, the host-gathered
    V/U `components`) and does the pure-host part — matplotlib + `wandb.log` — on a
    background thread, so the main train loop on every rank proceeds immediately (near-zero
    cross-rank divergence). The thread touches ZERO jax/device state. `UVPlots` is a NAIVE
    full host gather of the C-sharded V/U: cheap small-scale, OOMs / breaks at production C
    BY DESIGN (per Oli) — no special handling, the gather (collective, on the eval pass) is
    the cost. The `IdentityCIError` SCALARS are computed synchronously on the collective path
    (cheap, and `_step`-monotonic), not on this thread.

    One render in flight at a time: a `submit` while a render is still running blocks
    briefly on `join()` first, so renders can't pile up (slow eval is forward-only and
    coarse, so this effectively never blocks). The figures log on the live `_step` axis at
    `step=now_step` — the slow tier lands on a fast-eval step, so the sink has just opened
    `now_step` and the background `wandb.log(..., step=now_step)` merges into the same open
    step. A render that lands AFTER the next train-log advances the head is dropped by
    wandb's monotonic-`_step` rule (a benign one-figure-set miss, warned not raised; the
    next slow eval renders fine) — slow eval is forward-only seconds against a coarse
    `slow_every`, so this is not expected to fire. An `atexit` join flushes the last render
    before process exit (the trainer never calls `wandb.finish`). The atexit handler is
    registered on the FIRST submit, not in `__init__` — the first submit happens after
    `MetricsSink`'s `wandb.init` (eval comes after sink construction in the loop), so
    atexit's LIFO order runs our join BEFORE wandb's own atexit flush, and the figures
    land."""

    def __init__(self, is_main: bool):
        self._is_main = is_main
        self._thread: threading.Thread | None = None
        self._atexit_registered = False

    def join(self) -> None:
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def submit(
        self,
        reductions: dict[str, SiteReduction],
        perm_spec: PermutationMetricSpec,
        position_ci: dict[str, PositionCI] | None,
        components: dict[str, tuple[np.ndarray, np.ndarray]] | None,
        now_step: int,
    ) -> None:
        if not self._is_main:
            return
        if not self._atexit_registered:
            atexit.register(self.join)
            self._atexit_registered = True
        self.join()  # cap to one in-flight render
        self._thread = threading.Thread(
            target=_render_and_log_slow_eval,
            args=(reductions, perm_spec, position_ci, components, now_step),
            daemon=True,
        )
        self._thread.start()


def slow_eval_due(now_step: int, every: int, slow_every: int, slow_on_first_step: bool) -> bool:
    """The slow/plot tier cadence (SPEC S28). `now_step` is always a fast-eval step (the
    engine only calls the eval pass on `every`), and `slow_every` is a multiple of `every`,
    so a `slow_every` multiple coincides with an eval step. `slow_on_first_step` additionally
    fires the tier once at the first eval step (`now_step == every`), matching torch."""
    return now_step % slow_every == 0 or (slow_on_first_step and now_step == every)


def _render_and_log_slow_eval(
    reductions: dict[str, SiteReduction],
    perm_spec: PermutationMetricSpec,
    position_ci: dict[str, PositionCI] | None,
    components: dict[str, tuple[np.ndarray, np.ndarray]] | None,
    now_step: int,
) -> None:
    """Pure-host: render the slow figures (the base plot set plus, when `position_ci` is
    materialized, the config-driven CI-heatmap/permutation figures, and when `components` is
    the host-gathered V/U, the `UVPlots` heatmaps) and log them to wandb on the live `_step`
    axis at `now_step`. No jax/device access — safe off the train loop."""
    import wandb
    from PIL import Image

    figures = render_slow_eval_figures(reductions)
    if position_ci is not None:
        figures |= render_permutation_figures(perm_spec, position_ci, components)
    payload: dict[str, Any] = {
        f"slow_eval/{k}": wandb.Image(Image.open(io.BytesIO(v))) for k, v in figures.items()
    }
    _log_wandb_safe(wandb, payload, now_step, "slow-eval figures")


def _init_or_restore_state(
    pd: PDConfig,
    ci_fn_arch: CIFnArch,
    data: DataConfig | None,
    run: RunInstance,
    lm: DecomposedModel,
    opt_vu: optax.GradientTransformation,
    opt_ci: optax.GradientTransformation,
    init_key: PRNGKeyArray,
    src_key: PRNGKeyArray,
    mesh: Mesh,
    checkpoint_manager: ocp.CheckpointManager,
    is_main: bool,
) -> tuple[TrainState, int] | None:
    """The shared init/restore/finetune/faith-warmup phase (SPEC S21/S22/S33).

    Returns `(state, start_step)`, or `None` when a SIGTERM landed mid-warmup (the caller
    must exit cleanly for requeue — no valid checkpoint exists pre-step-0)."""
    state = _ensure_global(
        init_train_state(pd, lm, ci_fn_arch, data, opt_vu, opt_ci, init_key, src_key, mesh), mesh
    )

    restored = restore_latest(checkpoint_manager, state)
    if restored is not None:
        state, ckpt_step = restored
        assert int(state.step) == ckpt_step, (int(state.step), ckpt_step)
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

    if pd.faithfulness_warmup_steps > 0:
        faith_warmup_optimizer = optax.adamw(pd.faithfulness_warmup_lr, weight_decay=0.0)
        faith_warmup_opt_state = faith_warmup_optimizer.init(
            eqx.filter(state.components, eqx.is_array)
        )
        faith_warmup_step = make_faith_warmup_step(faith_warmup_optimizer)
        warmed_components = state.components
        t0 = time.time()
        faith_warmup_loss = None
        for _ in range(pd.faithfulness_warmup_steps):
            warmed_components, faith_warmup_opt_state, faith_warmup_loss = faith_warmup_step(
                lm, warmed_components, faith_warmup_opt_state
            )
            if _sigterm_received:
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
            state, components=warmed_components, components_opt_state=new_opt_vu
        )
        if is_main:
            print(
                f"faith warmup: {pd.faithfulness_warmup_steps} steps in {time.time() - t0:.0f}s, "
                f"final faith {float(faith_warmup_loss):.3e}",
                flush=True,
            )
    save_state(checkpoint_manager, 0, state)
    return state, 0


def run_decomposition_training(
    pd: PDConfig,
    cadence: Cadence,
    run: RunInstance,
    lm: DecomposedModel,
    ci_fn: CIFnArch,
    data: DataConfig | None,
    remat_recon_forwards: bool,
    remat_ci_fn: bool,
    sample_batch: Callable[[int], Any],
    eval_fn: "Callable[[TrainState, int], LogRecord] | None",
    eval_every: int,
    mesh: Mesh,
) -> None:
    """The generic VPD decomposition-training engine — the ONE train loop every target
    (LM, TMS, ResidMLP, …) runs through.

    Reads the pydantic algorithm config DIRECTLY: `pd` (seed / steps / optimizers / loss
    metrics / faith warmup), `cadence` (log / save / checkpoint-retention
    rhythm), `run` (the run identity + wandb lineage). The lab-built objects ride alongside:
    the decomposed model `lm` (an `eqx.Module` carrying the frozen target weights as
    fields — threaded into the jitted step as a pytree arg, never closed over), the CI-fn
    arch `ci_fn`, the data source `data` (None for a toy), and the `remat_recon_forwards`
    compute knob.

    The target supplies only its three injectable seams:

    - `sample_batch(step) -> batch`: the opaque per-step model input (a pure function of
      `step`, for O(1) resume). The model interprets it (an LM's token ids `[B, T]` → embed;
      a toy's feature vector, which already is the `[*leading, d]` waist). The engine only
      assumes axis 0 is the batch/`dp` axis (for sharding); it never names tokens or `d`.
    - `eval_fn(state, now_step) -> dict[str, float]`: an in-loop eval pass run every
      `eval_every` completed steps, its record logged under that step. `None` disables it.
    - `eval_every`: the eval cadence. For an LM this is `eval.every`; a toy folds its
      cheap target-CI eval onto the `train_log_every` cadence.

    Everything generic — `init_train_state`, fine-tune init, faith warmup, the recon-grid
    step factory, orbax checkpointing, schedules, SIGTERM-save — lives here. The step
    numerics are identical across targets; only the data source and the eval metric differ.
    """
    is_main = jax.process_index() == 0
    ndev = mesh.devices.size
    # Activate the mesh so bare-PartitionSpec `with_sharding_constraint`s inside the forward
    # resolve (the attn q/k/v batch-sharding pin in `FrozenAttn.core`, needed for cuDNN
    # flash attention under the scan+cond masked forward). Explicit NamedShardings elsewhere
    # are unaffected.
    jax.set_mesh(mesh)
    assert cadence.save_every is not None and cadence.keep_last_n_checkpoints is not None, cadence
    save_every = cadence.save_every

    run.run_dir.mkdir(parents=True, exist_ok=True)
    opt_vu, opt_ci, (sched_vu, sched_ci) = build_optimizers(pd)

    key = random.PRNGKey(pd.seed)
    init_key, src_key, run_key = random.split(key, 3)

    checkpoint_manager = make_checkpoint_manager(
        run.run_dir / "ckpts", cadence.keep_last_n_checkpoints
    )
    init = _init_or_restore_state(
        pd, ci_fn, data, run, lm, opt_vu, opt_ci, init_key, src_key, mesh,
        checkpoint_manager, is_main,
    )  # fmt: skip
    if init is None:
        return  # SIGTERM mid-warmup: clean exit for requeue
    state, start_step = init

    step_fn = make_train_step(
        lm=lm,
        loss_terms=build_loss_terms(pd.loss_metrics, lm.site_names),
        components_optimizer=opt_vu,
        ci_fn_optimizer=opt_ci,
        total_steps=pd.steps,
        remat_recon_forwards=remat_recon_forwards,
        remat_ci_fn=remat_ci_fn,
        mesh=mesh,
    )

    # record what this run actually executes on so wandb never lies about topology.
    # flatten the metric lists into the same flat keys torch logs (E14) so cross-impl
    # wandb config queries line up.
    wandb_config = flatten_typed_lists(
        dict(
            jax_runtime={
                "n_devices": ndev,
                "n_processes": jax.process_count(),
                "remat_recon_forwards": remat_recon_forwards,
                "remat_ci_fn": remat_ci_fn,
                "run_id": run.run_id,
                "run_dir": str(run.run_dir),
            },
        )
    )
    sink = MetricsSink.for_run(run, wandb_config, is_main)
    window_t0 = loop_t0 = time.time()
    last_logged = start_step

    for step in range(start_step, pd.steps):
        batch = sample_batch(step)
        state, metrics = step_fn(lm, state, batch, random.fold_in(run_key, step))

        now_step = step + 1
        dense = cadence.dense_log_phase
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
            record = {k: float(v) for k, v in metrics.items()}
            for loss_name in ("total", *(k for k in record if k.startswith("loss/"))):
                assert math.isfinite(record[loss_name]), (
                    f"non-finite loss {loss_name!r} at step {now_step}: {record[loss_name]}"
                )
            record["step_time_s"] = per_step
            record["elapsed_s"] = time.time() - loop_t0
            record["eta_s"] = (pd.steps - now_step) * per_step
            record["train/schedules/lr/components"] = float(jnp.asarray(sched_vu(now_step)))
            record["train/schedules/lr/ci_fn"] = float(jnp.asarray(sched_ci(now_step)))
            mem_stats = jax.local_devices()[0].memory_stats()
            if mem_stats is not None:
                record["train/mem/peak_gb_per_rank"] = mem_stats["peak_bytes_in_use"] / 1e9
            sink.log(now_step, record)
            window_t0 = time.time()

        if eval_fn is not None and now_step % eval_every == 0 and not _sigterm_received:
            eval_record = eval_fn(state, now_step)
            # A SIGTERM raised DURING the eval pass abandons its partial record unlogged and
            # falls through to the save block (synchronous save of the completed `now_step`).
            if not _sigterm_received:
                sink.log(now_step, eval_record)
                window_t0 = time.time()

        if now_step % save_every == 0 or now_step == pd.steps or _sigterm_received:
            save_state(checkpoint_manager, now_step, state)
            if is_main:
                print(f"checkpoint saved @ step {now_step}", flush=True)
            window_t0 = time.time()
        if _sigterm_received:
            if is_main:
                print("SIGTERM: checkpoint saved, exiting for requeue", flush=True)
            break
