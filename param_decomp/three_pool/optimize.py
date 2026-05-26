"""``ThreePoolTrainer`` and ``optimize_three_pool`` — 3-pool sibling of
:class:`param_decomp.optimize.Trainer` / :func:`param_decomp.optimize.optimize`.

Mirrors the single-pool call shape: caller hands in ``target_model``,
dataloader, configs, sink. Internal validation, per-pool wiring, cross-pool
comms, and the layerwise streaming loss strategy are all hidden behind the
class boundary.

  * **CI pool** trains the CI fn (replicated across ranks; DP-sharded across
    batch). Holds CI fn + AdamW state. Each step: target_fwd → CI fn fwd →
    broadcast CI to LW + PPGD → dead-time prefetch H_{T+1} → fused backward
    seeded by imp_min + per-site g_CI from LW + PPGD → in-pool all-reduce →
    AdamW. See :mod:`param_decomp.three_pool.step_ci`.

  * **Layerwise pool** trains V/U (block-DDP within group; sharded across
    sites). Recv CI → faithfulness + layerwise stoch recon → send g_CI back
    → recv g_VU from PPGD → combine → in-block all-reduce → AdamW → async
    ship updated V/U → PPGD. See :mod:`param_decomp.three_pool.step_layerwise`.

  * **PPGD pool** is a stateless full V/U replica. Recv CI → PPGD warmup +
    final recon → backward seeds V/U + CI grads (no outer source step;
    warmup inner loop owns sources) → sum-reduce V/U within PPGD pool →
    send g_VU to LW + g_CI to CI → recv updated V/U. See
    :mod:`param_decomp.three_pool.step_ppgd`.

Data-handling contract
----------------------
Every rank must read the FULL global batch on every step (i.e. each batch
tensor from ``train_loader`` has shape ``[batch_global, ...]``). The runner
asserts this. Callers wiring up the loader should pass ``dist_state=None`` so
the data path replicates the batch across ranks instead of sharding it.

Each step function then slices to its own per-pool batch shard via the
layout's ``my_batch_slice_*`` helpers. The 3-way batch routing in
``layout.py`` (see "Batch-split routing" in its docstring) assumes this
sliced-from-global pattern.
"""

import itertools
import os
import time
from contextlib import nullcontext
from typing import Any, Self

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from param_decomp._trace import dump_memory_stats, trace
from param_decomp.batch_and_loss_fns import ReconstructionLoss, RunBatch
from param_decomp.component_model import ComponentModel
from param_decomp.configs import Cadence, PDConfig, RuntimeConfig
from param_decomp.decomposition_targets import (
    DecompositionTarget,
    resolve_decomposition_targets,
)
from param_decomp.distributed import seed_all_ranks, seed_per_rank
from param_decomp.masks import AllLayersRouter
from param_decomp.metrics.base import LossMetricConfig
from param_decomp.metrics.importance_minimality import ImportanceMinimalityLossConfig
from param_decomp.metrics.persistent_pgd_recon import (
    PersistentPGDReconLossConfig,
    validate_pgd_scope,
)
from param_decomp.metrics.persistent_pgd_state import PersistentPGDState
from param_decomp.run_sink import RunSink
from param_decomp.schedule import get_scheduled_value
from param_decomp.three_pool.checkpoint import gather_full_state_dict_to_rank0
from param_decomp.three_pool.config import ThreePoolConfig
from param_decomp.three_pool.layout import (
    LayerwiseBlockGroup,
    ThreePoolLayout,
    build_world,
)
from param_decomp.three_pool.profiler import PhaseProfiler
from param_decomp.three_pool.reductions import (
    aggregate_losses_to_rank0,
    aggregate_max_memory_to_rank0,
)
from param_decomp.three_pool.runtime import _ThreePoolRuntime
from param_decomp.three_pool.step_ci import step_ci
from param_decomp.three_pool.step_layerwise import (
    finalize_layerwise_async_drain,
    run_faithfulness_warmup_layerwise,
    step_layerwise,
)
from param_decomp.three_pool.step_ppgd import finalize_ppgd_async_drain, step_ppgd
from param_decomp.torch_helpers import loop_dataloader
from param_decomp.trainer_snapshot import TrainerSnapshot
from param_decomp.two_pool.loss_strategy import LayerwiseLossStrategy

# Loss-metric type discriminators required for the 3-pool training path.
# Same set as two_pool — three-pool reuses the same loss-metric vocabulary.
REQUIRED_LOSS_METRIC_TYPES: frozenset[str] = frozenset(
    {
        "FaithfulnessLoss",
        "ImportanceMinimalityLoss",
        "StochasticReconLayerwiseLoss",
        "PersistentPGDReconLoss",
    }
)

FORBIDDEN_LOSS_METRIC_TYPES: frozenset[str] = frozenset(
    {
        "StochasticReconLoss",
        "StochasticReconSubsetLoss",
        "StochasticReconSubsetCEAndKL",
        "PersistentPGDReconSubsetLoss",
        "CIMaskedReconLoss",
        "CIMaskedReconLayerwiseLoss",
        "CIMaskedReconSubsetLoss",
        "UnmaskedReconLoss",
        "PGDReconLoss",
        "PGDReconLayerwiseLoss",
        "PGDReconSubsetLoss",
        "CIMaskedAttnPatternsReconLoss",
        "StochasticAttnPatternsReconLoss",
        "StochasticHiddenActsReconLoss",
        "CIHiddenActsReconLoss",
    }
)


class ThreePoolTrainer:
    """Stateful 3-pool trainer.

    Construction wires up the runtime bundle, world layout, ComponentModel,
    layerwise loss strategy, and the per-pool optimizer (CI / LW have one;
    PPGD has none — see module docstring). The PPGD state itself is built on
    the first batch of :meth:`run` because its source tensor shapes depend on
    the data's sequence dims.

    Resume support is rank-local: :meth:`snapshot` produces a self-contained
    :class:`~param_decomp.trainer_snapshot.TrainerSnapshot` whose ``resume``
    half carries this rank's slice, and :meth:`from_snapshot` reconstructs
    one from it. The lab's resume loader composes per-rank shards across
    ranks. The snapshot's ``consumable`` half is the gathered full state
    (rank 0 only; ``None`` elsewhere) — the gather happens inside
    :meth:`snapshot`, so all ranks must call it in sync.
    """

    pd_config: PDConfig
    runtime_config: RuntimeConfig
    three_pool_config: ThreePoolConfig
    reconstruction_loss: ReconstructionLoss
    component_model: ComponentModel
    layout: ThreePoolLayout
    strategy: LayerwiseLossStrategy
    optimizer: torch.optim.Optimizer | None
    ppgd_state: PersistentPGDState | None
    step: int

    def __init__(
        self,
        *,
        target_model: nn.Module,
        run_batch: RunBatch,
        reconstruction_loss: ReconstructionLoss,
        pd_config: PDConfig,
        runtime_config: RuntimeConfig,
        three_pool_config: ThreePoolConfig,
    ) -> None:
        assert dist.is_initialized(), (
            "init the distributed process group before constructing ThreePoolTrainer"
        )
        self.pd_config = pd_config
        self.runtime_config = runtime_config
        self.three_pool_config = three_pool_config
        self.reconstruction_loss = reconstruction_loss
        self._run_batch = run_batch
        self._target_model = target_model
        self.step = 0

        trace("ThreePoolTrainer.__init__: enter")
        _validate_pd_config_for_three_pool(pd_config, three_pool_config)
        # PPGD runs only on PPGD pool; the relevant per-rank batch is batch // n_ppgd.
        validate_pgd_scope(
            pd_config.loss_metrics,
            batch_size=pd_config.batch_size,
            world_size=len(three_pool_config.ppgd_ranks),
        )

        trace("ThreePoolTrainer.__init__: building runtime")
        self.runtime = _build_runtime(
            target_model=target_model,
            pd_config=pd_config,
            runtime_config=runtime_config,
            three_pool_config=three_pool_config,
            run_batch=run_batch,
            reconstruction_loss=reconstruction_loss,
        )

        torch.set_float32_matmul_precision("high")

        self._device = torch.device(runtime_config.device)
        block_groups = [
            LayerwiseBlockGroup(ranks=tuple(bg.ranks), owned_sites=tuple(bg.owned_sites))
            for bg in three_pool_config.layerwise_block_groups
        ]
        trace("ThreePoolTrainer.__init__: build_world: enter")
        world = build_world(
            ci_ranks=list(three_pool_config.ci_ranks),
            layerwise_block_groups=block_groups,
            ppgd_ranks=list(three_pool_config.ppgd_ranks),
            batch_global=self.runtime.batch_global,
            device=self._device,
        )
        trace("ThreePoolTrainer.__init__: build_world: done")
        self.layout = ThreePoolLayout.from_world(world, dist.get_rank())
        decomposition_targets = _decomposition_targets_for_pool(
            self.layout, self.runtime.c_per_site
        )
        trace(
            f"ThreePoolTrainer.__init__: my_pool={self.layout.my_pool} "
            f"n_decomp_targets={len(decomposition_targets)}"
        )

        target_model.requires_grad_(False)
        # Resync RNG across ranks before V/U + CI fn init — see
        # ``two_pool.optimize.TwoPoolTrainer.__init__`` for rationale.
        seed_all_ranks(pd_config.seed)
        trace("ThreePoolTrainer.__init__: ComponentModel ctor: enter")
        self.component_model = ComponentModel(
            target_model=target_model,
            run_batch=run_batch,
            decomposition_targets=decomposition_targets,
            ci_config=pd_config.ci_config,
            sigmoid_type=pd_config.sigmoid_type,
        )
        trace("ThreePoolTrainer.__init__: ComponentModel ctor: done")
        # Drop pool-irrelevant params before moving to GPU. RNG draws used to
        # init them already happened (in the ctor above), so equivalence with
        # single-pool / 2-pool is preserved.
        match self.layout.my_pool:
            case "layerwise" | "ppgd":
                self.component_model.drop_ci_fn()
                trace(f"ThreePoolTrainer.__init__: dropped ci_fn ({self.layout.my_pool} pool)")
            case "ci":
                self.component_model.drop_components()
                trace("ThreePoolTrainer.__init__: dropped V/U components (ci pool)")
        trace("ThreePoolTrainer.__init__: ComponentModel.to(device): enter")
        self.component_model = self.component_model.to(self._device)
        trace("ThreePoolTrainer.__init__: ComponentModel.to(device): done")
        dump_memory_stats("after ComponentModel.to(device)")
        # CI pool: optionally torch.compile the CI fn. Eats compile time on
        # step 0 / step 1 (first fwd + first bwd) but should cut ``ci/8_fused_bwd``
        # substantially — that backward through the 2.64B-param CI fn dominates
        # the critical path (70% of CI step at batch=48).
        if self.layout.my_pool == "ci" and os.environ.get("PD_COMPILE_CI_FN", "").strip() in (
            "1",
            "true",
            "yes",
        ):
            assert self.component_model.ci_fn is not None
            trace("ThreePoolTrainer.__init__: torch.compile(ci_fn)")
            self.component_model.ci_fn = torch.compile(self.component_model.ci_fn)  # pyright: ignore[reportAttributeAccessIssue]
        # Diverge stochastic RNG per rank for mask sampling.
        seed_per_rank(pd_config.seed)

        trace("ThreePoolTrainer.__init__: building LayerwiseLossStrategy")
        self.strategy = LayerwiseLossStrategy.from_cfg(
            target_model,
            use_fused_kl=three_pool_config.use_fused_kl,
            unfused_recon=reconstruction_loss,
        )
        trace("ThreePoolTrainer.__init__: LayerwiseLossStrategy: done")

        self.optimizer = None
        self._all_params: list[nn.Parameter] = []
        self._ci_fn_params: list[nn.Parameter] = []
        self._component_params: list[nn.Parameter] = []
        self.ppgd_state = None
        self._pending_ppgd_resume_state: dict[str, Any] | None = None

        trace(f"ThreePoolTrainer.__init__: optimizer build: enter (pool={self.layout.my_pool})")
        match self.layout.my_pool:
            case "ci":
                assert self.component_model.ci_fn is not None, "CI pool must keep its CI fn"
                self._ci_fn_params = list(self.component_model.ci_fn.parameters())
                n_params = sum(p.numel() for p in self._ci_fn_params)
                trace(f"ThreePoolTrainer.__init__: CI fn params={n_params / 1e9:.3f}B")
                self.optimizer = torch.optim.AdamW(
                    [
                        {
                            "params": self._ci_fn_params,
                            "lr": pd_config.ci_fn_optimizer.lr_schedule.start_val,
                        }
                    ],
                    weight_decay=0.0,
                    fused=True,
                )
            case "layerwise":
                for name in self.layout.my_owned_sites:
                    self._component_params.extend(
                        self.component_model.components[name].parameters()
                    )
                self._all_params = self._component_params
                self.optimizer = torch.optim.AdamW(
                    [
                        {
                            "params": self._component_params,
                            "lr": pd_config.components_optimizer.lr_schedule.start_val,
                        }
                    ],
                    weight_decay=0.0,
                    fused=True,
                )
            case "ppgd":
                pass  # ppgd_state constructed lazily from first batch in run()
        trace("ThreePoolTrainer.__init__: optimizer build: done")
        dump_memory_stats("after optimizer build")
        trace("ThreePoolTrainer.__init__: exit")

    # ============================ Atomic cfg + state ============================

    def snapshot(self) -> TrainerSnapshot:
        """Atomic point-in-time view: rank-local resume + (rank-0-only) consumable.

        All ranks participate in the gather (P2P sends/recvs) that produces the
        consumable half; rank 0 receives the gathered full state dict, other
        ranks get ``consumable=None``. The resume half is populated on every rank
        (its own slice of the model + own optimizer/PPGD state, plus the layout
        fingerprint for sanity checks on load).
        """
        gathered_consumable = gather_full_state_dict_to_rank0(
            layout=self.layout,
            component_model=self.component_model,
            target_model=self._target_model,
            run_batch=self._run_batch,
            ci_config=self.pd_config.ci_config,
            sigmoid_type=self.pd_config.sigmoid_type,
            c_per_site=self.runtime.c_per_site,
            device=self._device,
        )

        state: dict[str, Any] = {
            "step": self.step,
            "component_model": self.component_model.state_dict(),
            "pool": self.layout.my_pool,
        }
        if self.optimizer is not None:
            state["optimizer"] = self.optimizer.state_dict()
        if self.ppgd_state is not None:
            state["ppgd"] = self.ppgd_state.state_dict()
        return TrainerSnapshot(
            step=self.step,
            resume={
                "pd_config": self.pd_config.model_dump(),
                "runtime_config": self.runtime_config.model_dump(),
                "three_pool_config": self.three_pool_config.model_dump(),
                "layout_fingerprint": _layout_fingerprint(self.layout),
                "state": state,
            },
            consumable=gathered_consumable,  # rank 0 only; None elsewhere
        )

    @classmethod
    def from_snapshot(
        cls,
        snapshot: TrainerSnapshot,
        *,
        target_model: nn.Module,
        run_batch: RunBatch,
        reconstruction_loss: ReconstructionLoss,
        cfg_overrides: dict[str, Any] | None = None,
    ) -> Self:
        resume = snapshot.resume
        pd_dict = resume["pd_config"]
        if cfg_overrides is not None:
            pd_dict = {**pd_dict, **cfg_overrides}
        pd_config = PDConfig.model_validate(pd_dict)
        runtime_config = RuntimeConfig.model_validate(resume["runtime_config"])
        three_pool_config = ThreePoolConfig.model_validate(resume["three_pool_config"])

        trainer = cls(
            target_model=target_model,
            run_batch=run_batch,
            reconstruction_loss=reconstruction_loss,
            pd_config=pd_config,
            runtime_config=runtime_config,
            three_pool_config=three_pool_config,
        )
        saved_fp = resume["layout_fingerprint"]
        current_fp = _layout_fingerprint(trainer.layout)
        assert saved_fp == current_fp, (
            f"3-pool layout fingerprint mismatch on resume:\n"
            f"  saved:   {saved_fp}\n"
            f"  current: {current_fp}\n"
        )
        trainer._load_state(resume["state"])
        return trainer

    def _load_state(self, state: dict[str, Any]) -> None:
        self.step = state["step"]
        assert state["pool"] == self.layout.my_pool, (
            f"pool mismatch on resume: rank {self.layout.my_rank} was in pool "
            f"{state['pool']!r} when saved, now {self.layout.my_pool!r}"
        )
        self.component_model.load_state_dict(state["component_model"])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(state["optimizer"])
        if self.layout.my_pool == "ppgd":
            # ppgd_state is constructed lazily in run() — defer load until then.
            self._pending_ppgd_resume_state = state.get("ppgd")

    # ============================ Training loop ============================

    def run(
        self,
        train_loader: DataLoader[Any],
        sink: RunSink,
        cadence: Cadence,
        profiler: PhaseProfiler | None = None,
    ) -> None:
        """Advance training from ``self.step`` to ``self.pd_config.steps``."""
        trace("Trainer.run: enter")
        pd_config = self.pd_config
        layout = self.layout
        runtime = self.runtime
        n_steps = pd_config.steps
        defer_vu_opt = self.three_pool_config.defer_vu_opt
        device = self._device

        train_iterator = loop_dataloader(train_loader)
        # Loader skip-replay (resumed mid-trajectory).
        for _ in range(self.step):
            next(train_iterator)

        # Peek first batch (after any skip) for PPGD source-shape sizing.
        trace("Trainer.run: first_batch peek: enter")
        first_batch = next(train_iterator)
        trace("Trainer.run: first_batch peek: done")
        train_iterator = itertools.chain([first_batch], train_iterator)
        _assert_full_global_batch(first_batch, runtime.batch_global)

        if layout.my_pool == "ppgd" and self.ppgd_state is None:
            trace("Trainer.run: PPGDState ctor: enter")
            ppgd_cfg = runtime.ppgd_cfg
            self.ppgd_state = PersistentPGDState(
                module_to_c=runtime.c_per_site,
                batch_dims=(layout.world.batch_local_ppgd, *_seq_dims_from_batch(first_batch)),
                device=device,
                use_delta_component=True,
                optimizer_cfg=ppgd_cfg.optimizer,
                scope=ppgd_cfg.scope,
                use_sigmoid_parameterization=ppgd_cfg.use_sigmoid_parameterization,
                n_warmup_steps=ppgd_cfg.n_warmup_steps,
                n_samples=ppgd_cfg.n_samples,
                router=AllLayersRouter(),
                reconstruction_loss=self.strategy.recon_loss,
            )
            if self._pending_ppgd_resume_state is not None:
                self.ppgd_state.load_state_dict(self._pending_ppgd_resume_state)
                self._pending_ppgd_resume_state = None
            trace("Trainer.run: PPGDState ctor: done")

        if (
            self.step == 0
            and layout.my_pool == "layerwise"
            and pd_config.faithfulness_warmup_steps > 0
        ):
            trace(
                f"Trainer.run: faithfulness warmup: enter ({pd_config.faithfulness_warmup_steps} steps)"
            )
            run_faithfulness_warmup_layerwise(
                component_model=self.component_model,
                component_params=self._component_params,
                n_steps=pd_config.faithfulness_warmup_steps,
                lr=pd_config.faithfulness_warmup_lr,
                weight_decay=pd_config.faithfulness_warmup_weight_decay,
                numel_global=self.runtime.numel_global,
            )
            trace("Trainer.run: faithfulness warmup: done")

        components_lr_schedule = pd_config.components_optimizer.lr_schedule
        ci_fn_lr_schedule = pd_config.ci_fn_optimizer.lr_schedule

        profiler_ctx = profiler if profiler is not None else nullcontext()
        h_cache_ci: dict[str, Tensor] | None = None
        # Async-pipeline state threaded across iterations on LW + PPGD pools.
        pending_all_reduce_lw: list[tuple[list[Tensor], Tensor, dist.Work]] | None = None
        pending_recv_vu_ppgd: list[tuple[Any, Tensor, dist.Work]] | None = None

        def _to_device(b: Any) -> Any:
            """Move a batch yielded by the train loader to this rank's GPU.

            3-pool's step functions assume the batch is already on-device
            (mirroring 2-pool's `_extract_batch_tensor`). The loader produces
            CPU tensors; moving here keeps the step functions thin.
            """
            if b is None:
                return None
            if isinstance(b, Tensor):
                return b.to(device)
            if isinstance(b, dict) and "input_ids" in b:
                return {**b, "input_ids": b["input_ids"].to(device)}
            if isinstance(b, list | tuple) and len(b) > 0 and isinstance(b[0], Tensor):
                return type(b)([b[0].to(device), *b[1:]])
            raise TypeError(f"Unsupported batch type from DataLoader: {type(b).__name__}")

        with profiler_ctx:
            # 2-batch peek window: batch_T is the step's batch, batch_T_plus_1 is the
            # next-step batch peeked early for CI pool's dead-time prefetch.
            trace("Trainer.run: pre-loop batch peek: enter")
            batch_T = _to_device(next(train_iterator))
            batch_T_plus_1 = _to_device(next(train_iterator, None))
            trace("Trainer.run: pre-loop batch peek: done, entering training loop")

            for step in range(self.step, n_steps):
                self.step = step
                _assert_full_global_batch(batch_T, runtime.batch_global)
                trace(f"Trainer.run: step {step}: start (pool={layout.my_pool})")

                if self.optimizer is not None:
                    # CI pool: one param group (CI fn); LW pool: one (components).
                    # PPGD pool has no optimizer.
                    if layout.my_pool == "ci":
                        self.optimizer.param_groups[0]["lr"] = get_scheduled_value(
                            step, n_steps, ci_fn_lr_schedule
                        )
                    elif layout.my_pool == "layerwise":
                        lr_step = max(step - 1, 0) if defer_vu_opt else step
                        self.optimizer.param_groups[0]["lr"] = get_scheduled_value(
                            lr_step, n_steps, components_lr_schedule
                        )

                if profiler is not None:
                    dist.barrier()

                torch.cuda.synchronize(device)
                step_start = time.perf_counter()

                # batch_T should already be on this rank's device (placed by _to_device).
                if isinstance(batch_T, Tensor):
                    assert batch_T.device == device, (
                        f"3-pool batch device drift at step {step}: {batch_T.device} vs {device}"
                    )
                match layout.my_pool:
                    case "ci":
                        assert self.optimizer is not None, (
                            f"CI rank {layout.my_rank} missing optimizer"
                        )
                        assert len(self._ci_fn_params) > 0, (
                            f"CI rank {layout.my_rank} has no ci_fn params to optimize"
                        )
                        next_batch_for_prefetch = batch_T_plus_1 if step < n_steps - 1 else None
                        metrics, h_cache_ci = step_ci(
                            layout,
                            self.component_model,
                            self.optimizer,
                            self._ci_fn_params,
                            batch_T=batch_T,
                            batch_T_plus_1=next_batch_for_prefetch,
                            h_cache_T=h_cache_ci,
                            cfg=runtime,
                            current_frac_of_training=step / n_steps if n_steps > 0 else 0.0,
                            profiler=profiler,
                        )
                    case "layerwise":
                        assert self.optimizer is not None, (
                            f"LW rank {layout.my_rank} missing optimizer"
                        )
                        assert layout.my_owned_sites, (
                            f"LW rank {layout.my_rank} has no owned_sites — empty block"
                        )
                        metrics, pending_all_reduce_lw = step_layerwise(
                            layout,
                            self.component_model,
                            self.optimizer,
                            self._all_params,
                            batch_T,
                            runtime,
                            self.strategy,
                            defer_vu_opt=defer_vu_opt,
                            prev_pending_all_reduce=pending_all_reduce_lw,
                            profiler=profiler,
                        )
                    case "ppgd":
                        assert self.ppgd_state is not None, (
                            f"PPGD rank {layout.my_rank} has no ppgd_state — lazy init failed"
                        )
                        metrics, pending_recv_vu_ppgd = step_ppgd(
                            layout,
                            self.component_model,
                            self.ppgd_state,
                            batch_T,
                            runtime,
                            self.strategy,
                            step=step,
                            n_steps=n_steps,
                            defer_vu_opt=defer_vu_opt,
                            prev_pending_recv_vu=pending_recv_vu_ppgd,
                            profiler=profiler,
                        )
                # Catch silent NaN propagation early. Covers both the per-rank
                # display scalars (``loss/*``) and the raw aggregation
                # ingredients (``_raw/*``) the logger sums across the pool.
                for k, v in metrics.items():
                    if k.startswith("loss/") or k.startswith("_raw/"):
                        assert v == v, f"NaN in metrics[{k!r}] at step {step}"  # NaN != NaN

                # current_stream().synchronize() — not torch.cuda.synchronize() — so we
                # don't wait for *all* CUDA streams. In ``defer_vu_opt=True`` mode, PPGD
                # has a pending async dist.broadcast (irecv side) on a NCCL stream that
                # only completes when LW step N+1 phase B4 fires its matching broadcast.
                # Waiting for that here would deadlock against ``_log_train_metrics``
                # (LW rank 0 blocks on ``dist.recv`` from PPGD leader → PPGD leader
                # blocks here on the irecv → LW can't reach phase B4). The default
                # stream carries the step's compute, which is all we need for an
                # accurate ``step_ms`` measurement.
                torch.cuda.current_stream(device).synchronize()
                step_ms = (time.perf_counter() - step_start) * 1000.0
                trace(f"Trainer.run: step {step}: done in {step_ms:.1f}ms")
                if step % cadence.train_log_every == 0:
                    dump_memory_stats(f"step {step} done")

                if step % cadence.train_log_every == 0:
                    _log_train_metrics(
                        metrics=metrics,
                        layout=layout,
                        device=device,
                        step=step,
                        step_ms=step_ms,
                        runtime=runtime,
                        optimizer=self.optimizer,
                        sink=sink,
                    )

                if cadence.should_save(step):
                    sink.checkpoint(self.snapshot())

                batch_T = (
                    batch_T_plus_1
                    if batch_T_plus_1 is not None
                    else _to_device(next(train_iterator))
                )
                batch_T_plus_1 = _to_device(next(train_iterator, None))

                if profiler is not None:
                    profiler.step()

            # Drain the final iter's deferred opt (async mode only). Without this,
            # the saved checkpoint would be missing the last iter's update.
            if defer_vu_opt:
                match layout.my_pool:
                    case "layerwise":
                        assert self.optimizer is not None
                        if pending_all_reduce_lw is not None:
                            self.optimizer.param_groups[0]["lr"] = get_scheduled_value(
                                n_steps - 1, n_steps, components_lr_schedule
                            )
                            finalize_layerwise_async_drain(
                                layout,
                                self.component_model,
                                self.optimizer,
                                self._all_params,
                                pending_all_reduce_lw,
                                runtime.grad_clip_norm_components,
                            )
                            pending_all_reduce_lw = None
                    case "ppgd":
                        if pending_recv_vu_ppgd is not None:
                            finalize_ppgd_async_drain(
                                layout,
                                self.component_model,
                                pending_recv_vu_ppgd,  # type: ignore[arg-type]
                            )
                            pending_recv_vu_ppgd = None
                    case "ci":
                        pass  # CI pool doesn't defer

            self.step = n_steps
            sink.checkpoint(self.snapshot())


def optimize_three_pool(
    target_model: nn.Module,
    train_loader: DataLoader[Any],
    *,
    run_batch: RunBatch,
    reconstruction_loss: ReconstructionLoss,
    pd_config: PDConfig,
    runtime_config: RuntimeConfig,
    three_pool_config: ThreePoolConfig,
    cadence: Cadence,
    sink: RunSink,
    profiler: PhaseProfiler | None = None,
) -> None:
    """Train a ComponentModel under the 3-pool strategy.

    Thin wrapper over :class:`ThreePoolTrainer` for callers that don't need
    resumption or interactive control.
    """
    trainer = ThreePoolTrainer(
        target_model=target_model,
        run_batch=run_batch,
        reconstruction_loss=reconstruction_loss,
        pd_config=pd_config,
        runtime_config=runtime_config,
        three_pool_config=three_pool_config,
    )
    trainer.run(train_loader, sink, cadence, profiler=profiler)


def _layout_fingerprint(layout: ThreePoolLayout) -> dict[str, Any]:
    """Compact summary of the 3-pool world layout. Compared at resume time."""
    return {
        "world_size": layout.world.world_size,
        "ci_ranks": list(layout.world.ci_ranks),
        "ppgd_ranks": list(layout.world.ppgd_ranks),
        "n_layerwise_blocks": len(layout.world.layerwise_block_groups),
        "my_rank": layout.my_rank,
        "my_pool": layout.my_pool,
        "owned_sites": list(layout.my_owned_sites) if layout.my_pool == "layerwise" else [],
    }


def _validate_pd_config_for_three_pool(
    pd_config: PDConfig,
    three_pool_config: ThreePoolConfig,
) -> None:
    """Fail loudly on any PDConfig the 3-pool path can't honour."""
    by_type: dict[str, LossMetricConfig] = {m.type: m for m in pd_config.loss_metrics}

    missing = sorted(REQUIRED_LOSS_METRIC_TYPES - set(by_type))
    assert not missing, (
        f"3-pool requires these loss metrics: {sorted(REQUIRED_LOSS_METRIC_TYPES)}.\n"
        f"Missing: {missing}. Got: {sorted(by_type)}."
    )
    illegal = sorted(FORBIDDEN_LOSS_METRIC_TYPES & set(by_type))
    assert not illegal, (
        f"3-pool does not implement these loss metrics (they would be silently ignored): "
        f"{illegal}. Remove them or extend the 3-pool path."
    )

    for name in REQUIRED_LOSS_METRIC_TYPES:
        assert by_type[name].coeff is not None, (
            f"pd_config.loss_metrics[{name!r}].coeff is required for 3-pool training"
        )

    n_per_block = len(three_pool_config.layerwise_block_groups[0].ranks)
    n_ci = len(three_pool_config.ci_ranks)
    n_ppgd = len(three_pool_config.ppgd_ranks)
    bs = pd_config.batch_size
    assert bs % n_per_block == 0, (
        f"pd_config.batch_size ({bs}) must be divisible by N_per_block ({n_per_block}) "
        f"= len(layerwise_block_groups[0].ranks)"
    )
    assert bs % n_ci == 0, (
        f"pd_config.batch_size ({bs}) must be divisible by N_ci ({n_ci}) = len(ci_ranks)"
    )
    assert bs % n_ppgd == 0, (
        f"pd_config.batch_size ({bs}) must be divisible by N_ppgd ({n_ppgd}) = len(ppgd_ranks)"
    )

    assert pd_config.use_delta_component, (
        "3-pool requires pd_config.use_delta_component=True (hardcoded in LW's "
        "layerwise stoch recon + PPGD's PPGD warmup)."
    )

    assert pd_config.sampling == "continuous", (
        "3-pool hardcodes `sampling='continuous'` in CI pool's CI computation; "
        f"got pd_config.sampling={pd_config.sampling!r}."
    )
    assert pd_config.n_mask_samples == 1, (
        "3-pool draws exactly one stochastic mask per site per step in LW; "
        f"got pd_config.n_mask_samples={pd_config.n_mask_samples}."
    )

    assert pd_config.identity_decomposition_targets is None, (
        "3-pool path does not call `insert_identity_operations_`; "
        "`identity_decomposition_targets` would be silently ignored."
    )

    # Convention: rank 0 must be the Layerwise pool's block 0 leader.
    assert three_pool_config.layerwise_block_groups[0].ranks[0] == 0, (
        "Convention: rank 0 must be the LW pool's block 0 leader (so reductions "
        "can ship CI/PPGD pool losses to rank 0). Reorder layerwise_block_groups "
        "so the first group starts with rank 0."
    )

    ppgd_cfg = by_type["PersistentPGDReconLoss"]
    assert isinstance(ppgd_cfg, PersistentPGDReconLossConfig)
    assert ppgd_cfg.start_frac == 0.0, (
        "3-pool path does not implement PersistentPGDReconLoss.start_frac > 0; "
        "PPGD always runs from step 0."
    )


def _build_runtime(
    target_model: nn.Module,
    pd_config: PDConfig,
    runtime_config: RuntimeConfig,
    three_pool_config: ThreePoolConfig,
    run_batch: RunBatch,
    reconstruction_loss: ReconstructionLoss,
) -> _ThreePoolRuntime:
    """Assemble the step-context bundle from configs + target."""
    targets = resolve_decomposition_targets(target_model, pd_config.decomposition_targets)
    c_per_site = {t.module_path: t.C for t in targets}
    numel_global = 0
    for t in targets:
        w = target_model.get_submodule(t.module_path).weight
        assert isinstance(w, Tensor)
        numel_global += w.numel()

    for bg in three_pool_config.layerwise_block_groups:
        for site in bg.owned_sites:
            assert site in c_per_site, (
                f"site '{site}' in layerwise block group but not in "
                f"pd_config.decomposition_targets after pattern expansion. "
                f"Available: {sorted(c_per_site)[:5]}..."
            )

    by_type: dict[str, LossMetricConfig] = {m.type: m for m in pd_config.loss_metrics}
    ppgd_cfg = by_type["PersistentPGDReconLoss"]
    imp_min_cfg = by_type["ImportanceMinimalityLoss"]
    assert isinstance(ppgd_cfg, PersistentPGDReconLossConfig)
    assert isinstance(imp_min_cfg, ImportanceMinimalityLossConfig)

    def _coeff(name: str) -> float:
        c = by_type[name].coeff
        assert c is not None
        return float(c)

    block_groups = tuple(
        LayerwiseBlockGroup(ranks=tuple(bg.ranks), owned_sites=tuple(bg.owned_sites))
        for bg in three_pool_config.layerwise_block_groups
    )

    return _ThreePoolRuntime(
        ci_ranks=tuple(three_pool_config.ci_ranks),
        layerwise_block_groups=block_groups,
        ppgd_ranks=tuple(three_pool_config.ppgd_ranks),
        batch_global=pd_config.batch_size,
        c_per_site=c_per_site,
        ci_config=pd_config.ci_config,
        sigmoid_type=pd_config.sigmoid_type,
        run_batch=run_batch,
        reconstruction_loss=reconstruction_loss,
        ppgd_cfg=ppgd_cfg,
        coeff_faith=_coeff("FaithfulnessLoss"),
        coeff_imp=_coeff("ImportanceMinimalityLoss"),
        coeff_stoch=_coeff("StochasticReconLayerwiseLoss"),
        coeff_ppgd=_coeff("PersistentPGDReconLoss"),
        imp_min_pnorm=imp_min_cfg.pnorm,
        imp_min_beta=imp_min_cfg.beta,
        imp_min_eps=imp_min_cfg.eps,
        imp_min_p_anneal_start_frac=imp_min_cfg.p_anneal_start_frac,
        imp_min_p_anneal_final_p=imp_min_cfg.p_anneal_final_p,
        imp_min_p_anneal_end_frac=imp_min_cfg.p_anneal_end_frac,
        lr_components=pd_config.components_optimizer.lr_schedule.start_val,
        lr_ci_fn=pd_config.ci_fn_optimizer.lr_schedule.start_val,
        grad_clip_norm_components=pd_config.components_optimizer.grad_clip_norm,
        grad_clip_norm_ci_fn=pd_config.ci_fn_optimizer.grad_clip_norm,
        numel_global=numel_global,
        bf16_autocast=runtime_config.autocast_bf16,
        use_fused_kl=three_pool_config.use_fused_kl,
    )


def _decomposition_targets_for_pool(
    layout: ThreePoolLayout, c_per_site: dict[str, int]
) -> list[DecompositionTarget]:
    """CI/PPGD pools: every site (full CI fn / full V/U replica).
    Layerwise pool: only this rank's owned sites."""
    match layout.my_pool:
        case "ci" | "ppgd":
            sites = layout.world.all_sites
        case "layerwise":
            sites = layout.my_owned_sites
    return [DecompositionTarget(module_path=s, C=c_per_site[s]) for s in sites]


def _assert_full_global_batch(batch: Any, batch_global: int) -> None:
    """The 3-pool data contract: every rank reads the FULL global batch on
    every step so each pool can slice to its own DP shard.
    """
    if isinstance(batch, Tensor):
        actual = batch.shape[0]
    elif isinstance(batch, dict) and "input_ids" in batch:
        actual = batch["input_ids"].shape[0]
    elif isinstance(batch, list | tuple) and len(batch) > 0 and isinstance(batch[0], Tensor):
        actual = batch[0].shape[0]
    else:
        raise TypeError(f"Unsupported batch type from DataLoader: {type(batch).__name__}")
    assert actual == batch_global, (
        f"3-pool requires each rank to read the FULL global batch; got batch with "
        f"leading-dim {actual}, expected {batch_global}. Most likely cause: the data "
        f"loader was built with a non-None dist_state (which shards the batch). "
        f"For 3-pool, pass dist_state=None to build_loader."
    )


def _seq_dims_from_batch(batch: Any) -> tuple[int, ...]:
    """Sequence dims (everything past the leading batch dim) of a sample batch."""
    if isinstance(batch, Tensor):
        return tuple(batch.shape[1:])
    if isinstance(batch, dict) and "input_ids" in batch:
        return tuple(batch["input_ids"].shape[1:])
    if isinstance(batch, list | tuple) and len(batch) > 0 and isinstance(batch[0], Tensor):
        return tuple(batch[0].shape[1:])
    raise TypeError(f"Cannot infer seq dims from batch of type {type(batch).__name__}")


def _log_train_metrics(
    *,
    metrics: dict[str, float],
    layout: ThreePoolLayout,
    device: torch.device,
    step: int,
    step_ms: float,
    runtime: _ThreePoolRuntime,
    optimizer: torch.optim.Optimizer | None,
    sink: RunSink,
) -> None:
    """Aggregate per-pool metrics to rank 0 and dispatch to ``sink``."""
    combined = aggregate_losses_to_rank0(metrics, layout, device)
    mem_combined = aggregate_max_memory_to_rank0(layout, device)

    # Reduce step_ms with MAX across LW pool (slowest LW rank is the wall-clock floor).
    step_ms_t = torch.tensor([step_ms], device=device)
    if layout.my_pool == "layerwise":
        dist.all_reduce(step_ms_t, op=dist.ReduceOp.MAX, group=layout.world.layerwise_pool_group)

    if layout.my_rank != 0 or combined is None:
        return

    if mem_combined is not None:
        combined.update(mem_combined)
    combined["perf/step_ms"] = step_ms_t.item()
    combined["loss/total"] = (
        runtime.coeff_faith * combined["loss/faith"]
        + runtime.coeff_imp * combined["loss/imp"]
        + runtime.coeff_stoch * combined["loss/stoch"]
        + runtime.coeff_ppgd * combined["loss/ppgd"]
    )
    assert layout.my_pool == "layerwise", "rank 0 must be in LW pool (per validator)"
    assert optimizer is not None
    combined["schedules/lr/components"] = optimizer.param_groups[0]["lr"]
    sink.console(
        f"--- Step {step} ---",
        *(f"train/{name}: {value:.6g}" for name, value in combined.items()),
    )
    sink.log({f"train/{k}": v for k, v in combined.items()}, step=step)
