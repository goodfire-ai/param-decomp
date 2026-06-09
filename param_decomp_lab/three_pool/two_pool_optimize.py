"""``TwoPoolTrainer`` and ``optimize_two_pool`` — 2-pool sibling of
``ThreePoolTrainer``.

The 2-pool variant merges the 3-pool CI and PPGD pools into a single **Pool A**:
each Pool A rank holds the replicated CI fn (DDP across Pool A) AND a full V/U
replica + persistent PPGD sources, and runs the CI forward + the adversary on the
SAME batch slice. The chunkwise pool (**Pool B**) is unchanged. Deleting the CI↔PPGD
edge removes the cross-pool mask send + g_CI return entirely (the adversary's g_CI is
the local ``.grad`` of the CI forward's own ``lower_leaky``) — which is also the edge
that deadlocked seq-2048 runs, so this is a structural fix.

Reuses the 3-pool's ``ThreePoolConstrainedPDConfig`` (same four-loss set + frozen
algorithm scalars) and ``_ThreePoolRuntime`` (with ``ci_ranks == ppgd_ranks ==
pool_a_ranks``), the chunkwise step + portals verbatim, and the merged Pool A step
(``step_pool_a``). See ``two_pool_layout.py`` for why ``ci_ranks == ppgd_ranks`` makes
all of that reuse correct.

Checkpoint/resume reuses the 3-pool machinery: each rank writes a self-contained partial
on the loop (chunk leaders → V/U; the Pool A leader → CI fn + ci-fn optimizer), and Pool A
ranks write their data-shaped PPGD sources straight to ``ppgd_<S>/rank_<r>.pth`` in
parallel. The async job consolidates the partials into ``model_<S>.pth`` + a
``ThreePoolTrainingState`` (whose slots line up for the 2-pool), and ``from_snapshot``
restores the step + all owned state (reading the PPGD shard per-rank).
"""

import itertools
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Self, cast

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.profiler
from torch import Tensor
from torch.utils.data import DataLoader

from param_decomp._trace import dump_memory_stats, trace
from param_decomp.batch_and_loss_fns import ReconstructionLoss, RunBatch
from param_decomp.ci_fns import GlobalSharedTransformerCiFn
from param_decomp.component_model import ComponentModel
from param_decomp.configs import Cadence, RuntimeConfig
from param_decomp.decomposition_targets import DecompositionTarget
from param_decomp.distributed import seed_all_ranks, seed_per_rank
from param_decomp.masks import AllLayersRouter
from param_decomp.metrics.persistent_pgd_recon import validate_pgd_scope
from param_decomp.metrics.persistent_pgd_state import (
    BroadcastAcrossBatchScope,
    PerBatchPerPositionScope,
    PersistentPGDState,
    scope_needs_replica_sync,
)
from param_decomp.optimize import (
    EvalLoop,
    load_optimizer_state_by_name,
    optimizer_state_by_name,
)
from param_decomp.run_sink import ThreePoolRunSink
from param_decomp.schedule import get_scheduled_value
from param_decomp.sdpa_strict import verify_flash_attention_available
from param_decomp.torch_helpers import loop_dataloader
from param_decomp.training_state import ThreePoolTrainingState
from param_decomp_lab.experiments.lm.vendored.component_model import LMComponentModel
from param_decomp_lab.infra.run_files import save_file
from param_decomp_lab.three_pool.checkpoint import (
    ci_fn_state_keys,
    owned_model_state_keys,
)
from param_decomp_lab.three_pool.config import PooledRuntimeConfig
from param_decomp_lab.three_pool.consolidate import (
    CONSOLIDATE_META_FILENAME,
    load_ppgd_shard,
    ppgd_shard_dirname,
    prune_old_scratch,
    step_scratch_dir,
)
from param_decomp_lab.three_pool.context import ChunkContext
from param_decomp_lab.three_pool.layout import Chunk, flush_nccl_event_timings
from param_decomp_lab.three_pool.optimize import (
    _ci_attn_shape_or_none,
    _rank_invariant_fingerprint_core,
    _resolve_pg_timeout,
    _seq_dims_from_batch,
)
from param_decomp_lab.three_pool.pd_config import ThreePoolConstrainedPDConfig
from param_decomp_lab.three_pool.recon_loss_strategy import ReconLossStrategy
from param_decomp_lab.three_pool.runtime import _ThreePoolRuntime
from param_decomp_lab.three_pool.step_chunkwise import (
    run_faithfulness_warmup_chunkwise,
    step_chunkwise,
)
from param_decomp_lab.three_pool.step_pool_a import step_pool_a
from param_decomp_lab.three_pool.two_pool_config import TwoPoolTopology
from param_decomp_lab.three_pool.two_pool_context import (
    PoolAContext,
    TwoPoolContext,
    build_two_pool_context,
)
from param_decomp_lab.three_pool.two_pool_layout import build_two_world
from param_decomp_lab.three_pool.two_pool_reductions import (
    aggregate_component_grad_by_loss_to_rank0,
    aggregate_grad_norms_to_rank0,
    aggregate_losses_to_rank0,
    aggregate_max_memory_to_rank0,
)


class TwoPoolTrainer:
    """Stateful 2-pool trainer. Construction wires up the runtime bundle, world,
    ComponentModel, recon strategy, and the per-pool optimizer (Pool A → CI fn;
    chunkwise → V/U). The PPGD state is built on the first batch of :meth:`run`."""

    pd_config: ThreePoolConstrainedPDConfig
    runtime_config: PooledRuntimeConfig
    two_pool_config: TwoPoolTopology
    reconstruction_loss: ReconstructionLoss
    component_model: LMComponentModel
    ctx: TwoPoolContext
    strategy: ReconLossStrategy
    optimizer: torch.optim.Optimizer | None
    ppgd_state: PersistentPGDState | None
    step: int

    def __init__(
        self,
        *,
        target_model: nn.Module,
        run_batch: RunBatch,
        reconstruction_loss: ReconstructionLoss,
        pd_config: ThreePoolConstrainedPDConfig,
        runtime_config: PooledRuntimeConfig,
        two_pool_config: TwoPoolTopology,
    ) -> None:
        assert dist.is_initialized(), (
            "init the distributed process group before constructing TwoPoolTrainer"
        )
        self.pd_config = pd_config
        self.runtime_config = runtime_config
        self.two_pool_config = two_pool_config
        self.reconstruction_loss = reconstruction_loss
        self.step = 0

        ci_attn = _ci_attn_shape_or_none(pd_config)
        if ci_attn is not None:
            d_model, n_heads = ci_attn
            verify_flash_attention_available(head_dim=d_model // n_heads)

        self.runtime = _build_two_pool_runtime(
            target_model=target_model,
            pd_config=pd_config,
            runtime_config=runtime_config,
            two_pool_config=two_pool_config,
            run_batch=run_batch,
            reconstruction_loss=reconstruction_loss,
        )

        # The adversary runs on Pool A; its per-rank batch is batch // n_a.
        validate_pgd_scope(
            [pd_config.losses.ppgd],
            batch_size=pd_config.batch_size,
            world_size=len(self.runtime.ppgd_ranks),
        )

        torch.set_float32_matmul_precision("high")

        self._device = torch.device(runtime_config.device)
        world = build_two_world(
            pool_a_ranks=list(self.runtime.ci_ranks),
            chunks=list(self.runtime.chunks),
            batch_global=self.runtime.batch_global,
            pg_timeout=_resolve_pg_timeout(
                compiling=runtime_config.compile_chunkwise
                or runtime_config.compile_ci_fn
                or runtime_config.compile_ppgd
            ),
            device=self._device,
        )
        self.ctx = build_two_pool_context(world, dist.get_rank())
        decomposition_targets = _decomposition_targets_for_pool(self.ctx, self.runtime.c_per_site)

        target_model.requires_grad_(False)
        seed_all_ranks(pd_config.seed)
        self.component_model = LMComponentModel.build(
            target_model=target_model,
            decomposition_targets=decomposition_targets,
            ci_config=pd_config.ci_config,
            sigmoid_type=pd_config.sigmoid_type,
        )
        # Pool A keeps BOTH the CI fn (to train) and the full V/U replica (the
        # adversary). The chunkwise pool drops the CI fn (it only owns V/U).
        if isinstance(self.ctx, ChunkContext):
            self.component_model.drop_ci_fn()

        # Activation checkpointing of the recon/target model is chunkwise-only:
        # Pool A's adversary differentiates via autograd.grad, whose recompute is
        # non-deterministic under checkpoint (same constraint the 3-pool PPGD pool has).
        if not isinstance(self.ctx, ChunkContext):
            self.component_model.model._use_activation_checkpointing = False  # type: ignore[attr-defined]
        self.component_model = self.component_model.to(self._device)

        is_pool_a = isinstance(self.ctx, PoolAContext)
        if is_pool_a:
            assert self.component_model.ci_fn is not None
            if runtime_config.checkpoint_ci_fn:
                for m in self.component_model.ci_fn.modules():
                    if isinstance(m, GlobalSharedTransformerCiFn):
                        m.enable_activation_checkpointing()
            # Pool A holds both the CI fn (trained) and the adversary's masked forward, so
            # both compiled artifacts live on this rank. Per-rank inductor/triton caches guard
            # against shared-cache contention across the concurrent compilers (set once before
            # either compile).
            if runtime_config.compile_ci_fn or runtime_config.compile_ppgd:
                user = os.environ.get("USER", "u")
                rank = dist.get_rank()
                os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/torchinductor_{user}_r{rank}"
                os.environ["TRITON_CACHE_DIR"] = f"/tmp/triton_{user}_r{rank}"
            if runtime_config.compile_ci_fn:
                self.component_model.ci_fn.compile()
            # PPGD: compile the masked model forward the adversary runs (warmup PGD inner loop
            # + final recon). Same compiled artifact + fused-autograd.grad validation as the
            # 3-pool PPGD pool. Pool A's model has activation checkpointing off (autograd.grad
            # recompute is nondeterministic under ckpt — see above), so this is a plain forward
            # compile, no checkpointed-RNG-op partitioner concern.
            if runtime_config.compile_ppgd:
                self.component_model.model.compile()
        if isinstance(self.ctx, ChunkContext) and runtime_config.compile_chunkwise:
            user = os.environ.get("USER", "u")
            rank = dist.get_rank()
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/torchinductor_{user}_r{rank}"
            os.environ["TRITON_CACHE_DIR"] = f"/tmp/triton_{user}_r{rank}"
            self.component_model.model.compile()
        seed_per_rank(pd_config.seed)

        self.strategy = ReconLossStrategy.from_cfg(
            self.component_model,
            use_fused_kl=two_pool_config.use_fused_kl,
            unfused_recon=reconstruction_loss,
        )

        self.optimizer = None
        self._all_params: list[nn.Parameter] = []
        self._ci_fn_params: list[nn.Parameter] = []
        self._component_params: list[nn.Parameter] = []
        self.ppgd_state = None
        self._pending_ppgd_resume_state: dict[str, Any] | None = None

        match self.ctx:
            case PoolAContext():
                assert self.component_model.ci_fn is not None
                self._ci_fn_params = list(self.component_model.ci_fn.parameters())
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
            case ChunkContext():
                for name in self.ctx.role.sites:
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

    # ============================ Checkpoint / resume ============================

    def _named_params_for_my_optimizer(self) -> list[tuple[str, nn.Parameter]]:
        """The ``(name, param)`` pairs in the order they were added to this rank's
        optimizer. Pool A returns ``ci_fn.*`` pairs; chunkwise returns
        ``components.<site>.*`` pairs for this rank's chunk sites."""
        match self.ctx:
            case PoolAContext():
                assert self.component_model.ci_fn is not None
                return [(f"ci_fn.{n}", p) for n, p in self.component_model.ci_fn.named_parameters()]
            case ChunkContext():
                out: list[tuple[str, nn.Parameter]] = []
                for site in self.ctx.role.sites:
                    for n, p in self.component_model.components[site].named_parameters():
                        out.append((f"components.{site}.{n}", p))
                return out

    def _owned_model_params(self) -> dict[str, Tensor]:
        """The slice of this rank's model state_dict it's responsible for saving.

        Only leaders contribute: chunk leaders own their sites' V/U, the Pool A
        leader owns the CI fn. Every other rank holds a replica, so it contributes
        nothing — the union across leaders covers the full model.
        """
        sd = self.component_model.state_dict()
        keys = set(sd.keys())
        match self.ctx:
            case ChunkContext() if self.ctx.role.is_chunk_leader:
                owned = owned_model_state_keys(keys, sites=self.ctx.role.sites)
            case PoolAContext() if self.ctx.role.is_pool_leader:
                owned = ci_fn_state_keys(keys)
            case _:
                owned = set()
        return {k: sd[k].cpu() for k in owned}

    def _build_my_partial(self) -> dict[str, Any]:
        my_named_params = self._named_params_for_my_optimizer()
        my_optimizer_by_name: dict[str, dict[str, Any]] = (
            optimizer_state_by_name(self.optimizer, my_named_params)
            if self.optimizer is not None
            else {}
        )
        return {
            "pool": self.ctx.kind,
            "model_params": self._owned_model_params(),
            "optimizer_by_name": my_optimizer_by_name,
        }

    def _my_ppgd_state_dict(self) -> dict[str, Any] | None:
        """This rank's PPGD adversarial sources to shard out, or ``None`` if it owns none.

        Only Pool A ranks own sources: the live state once built, else the pending
        resume state for a resumed-but-not-yet-stepped rank.
        """
        if self.ppgd_state is not None:
            return self.ppgd_state.state_dict()
        return self._pending_ppgd_resume_state

    def _build_meta(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "world_size": self.ctx.world.world_size,
            "all_sites": list(self.ctx.world.all_sites),
            "c_per_site": dict(self.runtime.c_per_site),
            "pd_config": self.pd_config.model_dump(),
            "runtime_config": self.runtime_config.model_dump(),
            "three_pool_config": self.two_pool_config.model_dump(),
            "layout_fingerprint": _two_pool_layout_fingerprint(self.ctx),
        }

    def snapshot(self, scratch_dir: Path) -> None:
        """Write this rank's self-contained checkpoint partial; then return.

        Mirrors ``ThreePoolTrainer.snapshot``: each rank writes its owned model
        params (chunk leaders → V/U; Pool A leader → CI fn) and its optimizer
        state to ``scratch_dir/step_<S>/rank_<r>.pth``. Pool A ranks ALSO write
        their data-shaped PPGD sources, in parallel, straight to the stable shard
        ``out_dir / ppgd_<S> / rank_<r>.pth`` (``out_dir == scratch_dir.parent``)
        — not into the partial. Rank 0 also writes ``meta.pth``. One pre-write +
        one post-write barrier; no rank-0 read, no model assembly. The async
        consolidation job assembles the canonical artifacts off the critical path
        by streaming only the small partials.
        """
        partial = self._build_my_partial()
        my_ppgd = self._my_ppgd_state_dict()

        p2p_group = self.ctx.world.cross_pool_p2p_group
        step_dir = step_scratch_dir(scratch_dir, self.step)
        ppgd_shard_path = (
            scratch_dir.parent / ppgd_shard_dirname(self.step) / f"rank_{self.ctx.role.rank}.pth"
        )
        if self.ctx.role.rank == 0:
            step_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self._build_meta(), step_dir / CONSOLIDATE_META_FILENAME)

        trace("snapshot: enter barrier (pre-write)")
        dist.barrier(group=p2p_group)
        trace("snapshot: barrier (pre-write) done")

        partial_path = step_dir / f"rank_{self.ctx.role.rank}.pth"
        trace(f"snapshot: writing partial {partial_path.name}")
        torch.save(partial, partial_path)
        trace("snapshot: partial write done")

        if my_ppgd is not None:
            trace(f"snapshot: writing PPGD shard {ppgd_shard_path.name}")
            save_file(my_ppgd, ppgd_shard_path)
            trace("snapshot: PPGD shard write done")

        if os.environ.get("PD_3POOL_DISABLE_REJOIN_BARRIER", "").strip() in ("1", "true"):
            trace("snapshot: REJOIN BARRIER DISABLED (repro mode)")
            return
        trace("snapshot: enter barrier (post-write rejoin)")
        dist.barrier(group=p2p_group)
        trace("snapshot: barrier (post-write rejoin) done")

        # Backstop the async consolidation that normally clears scratch: if its job
        # stalls or never runs, this keeps partials from growing without bound.
        if self.ctx.role.rank == 0:
            prune_old_scratch(scratch_dir)

    @classmethod
    def from_snapshot(
        cls,
        snapshot: ThreePoolTrainingState,
        *,
        target_model: nn.Module,
        run_batch: RunBatch,
        reconstruction_loss: ReconstructionLoss,
        ppgd_shard_dir: Path | None,
    ) -> Self:
        """Reconstruct a 2-pool trainer from a canonical snapshot.

        The 2-pool persists the SAME ``ThreePoolTrainingState`` shape (the owned-state
        keys line up: chunkwise → ``components_optimizer``, Pool A's ci-fn →
        ``ci_fn_optimizer``). The ``three_pool_config`` slot holds the ``TwoPoolTopology``
        dump. ``ppgd_shard_dir`` is the run's ``ppgd_<step>/`` dir holding the per-rank
        adversarial source shards (``None`` ⇒ the adversary re-warms from scratch).
        """
        pd_config = ThreePoolConstrainedPDConfig.model_validate(snapshot.pd_config)
        # `topology` is reconstructed separately into `two_pool_config`; the rest of the dump
        # (base scalars + compile flags) revalidates as `PooledRuntimeConfig` so the flags
        # survive resume without importing the lab-side `TwoPoolRuntimeConfig` (which would cycle).
        runtime_dict = {k: v for k, v in snapshot.runtime_config.items() if k != "topology"}
        runtime_config = PooledRuntimeConfig.model_validate(runtime_dict)
        two_pool_config = TwoPoolTopology.model_validate(snapshot.three_pool_config)

        trainer = cls(
            target_model=target_model,
            run_batch=run_batch,
            reconstruction_loss=reconstruction_loss,
            pd_config=pd_config,
            runtime_config=runtime_config,
            two_pool_config=two_pool_config,
        )
        saved_fp = _rank_invariant_fingerprint_core(snapshot.layout_fingerprint)
        current_fp = _rank_invariant_fingerprint_core(_two_pool_layout_fingerprint(trainer.ctx))
        assert saved_fp == current_fp, (
            f"2-pool layout topology mismatch on resume:\n"
            f"  saved:   {saved_fp}\n"
            f"  current: {current_fp}\n"
        )
        trainer._load_canonical_state(snapshot, ppgd_shard_dir)
        return trainer

    def _load_canonical_state(
        self, state: ThreePoolTrainingState, ppgd_shard_dir: Path | None
    ) -> None:
        """Each rank extracts the slice of the canonical state it owns."""
        self.step = state.step
        local_model_keys = set(self.component_model.state_dict().keys())
        local_slice = {k: v for k, v in state.component_model.items() if k in local_model_keys}
        self.component_model.load_state_dict(local_slice, strict=False)

        if self.optimizer is not None:
            named_params = self._named_params_for_my_optimizer()
            match self.ctx:
                case ChunkContext():
                    by_name = state.components_optimizer
                case PoolAContext():
                    by_name = state.ci_fn_optimizer
            load_optimizer_state_by_name(self.optimizer, named_params, by_name)
        if isinstance(self.ctx, PoolAContext):
            self._pending_ppgd_resume_state = load_ppgd_shard(ppgd_shard_dir, self.ctx.role.rank)

    def run(
        self,
        train_loader: DataLoader[Any],
        sink: ThreePoolRunSink,
        cadence: Cadence,
        scratch_dir: Path,
        eval_loop: EvalLoop | None = None,
        profiler: torch.profiler.profile | None = None,
    ) -> None:
        """Advance training from ``self.step`` to ``self.pd_config.steps``."""
        pd_config = self.pd_config
        ctx = self.ctx
        world = ctx.world
        runtime = self.runtime
        n_steps = pd_config.steps
        device = self._device

        train_iterator = loop_dataloader(train_loader)
        for _ in range(self.step):
            next(train_iterator)

        eval_iterator = loop_dataloader(eval_loop.loader) if eval_loop is not None else None
        if eval_loop is not None and isinstance(ctx, PoolAContext):
            for m in eval_loop.metrics:
                m.bind(
                    model=cast(ComponentModel, cast(object, self.component_model)),
                    device=str(device),
                )

        first_batch = next(train_iterator)
        train_iterator = itertools.chain([first_batch], train_iterator)
        _assert_full_global_batch(first_batch, runtime.batch_global)

        if isinstance(ctx, PoolAContext) and self.ppgd_state is None:
            ppgd_cfg = runtime.ppgd_cfg
            assert isinstance(
                ppgd_cfg.scope, PerBatchPerPositionScope | BroadcastAcrossBatchScope
            ), (
                f"2-pool supports PerBatchPerPositionScope and BroadcastAcrossBatchScope "
                f"PPGD sources; got {type(ppgd_cfg.scope).__name__}."
            )
            # A broadcast (whole-global-batch) source is replicated across the Pool A
            # data-parallel ranks: the state broadcast-inits it from the group leader and
            # AVG-reduces its grads over the group, so all replicas step in lockstep.
            replica_sync_group = (
                world.ci_pool_group if scope_needs_replica_sync(ppgd_cfg.scope) else None
            )
            self.ppgd_state = PersistentPGDState(
                module_to_c=runtime.c_per_site,
                batch_dims=(world.batch_local_ppgd, *_seq_dims_from_batch(first_batch)),
                device=device,
                use_delta_component=True,
                optimizer_cfg=ppgd_cfg.optimizer,
                scope=ppgd_cfg.scope,
                use_sigmoid_parameterization=ppgd_cfg.use_sigmoid_parameterization,
                n_warmup_steps=ppgd_cfg.n_warmup_steps,
                n_samples=ppgd_cfg.n_samples,
                router=AllLayersRouter(),
                reconstruction_loss=self.strategy.recon_loss,
                replica_sync_group=replica_sync_group,
            )
            if self._pending_ppgd_resume_state is not None:
                self.ppgd_state.load_state_dict(self._pending_ppgd_resume_state)
                self._pending_ppgd_resume_state = None

        if (
            self.step == 0
            and isinstance(ctx, ChunkContext)
            and pd_config.faithfulness_warmup_steps > 0
        ):
            run_faithfulness_warmup_chunkwise(
                component_model=self.component_model,
                component_params=self._component_params,
                n_steps=pd_config.faithfulness_warmup_steps,
                lr=pd_config.faithfulness_warmup_lr,
                weight_decay=pd_config.faithfulness_warmup_weight_decay,
                numel_global=self.runtime.numel_global,
            )

        components_lr_schedule = pd_config.components_optimizer.lr_schedule
        ci_fn_lr_schedule = pd_config.ci_fn_optimizer.lr_schedule

        profiler_ctx = profiler if profiler is not None else nullcontext()

        def _to_device(b: Any) -> Any:
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
            batch_T = _to_device(next(train_iterator))

            for step in range(self.step, n_steps):
                self.step = step
                _assert_full_global_batch(batch_T, runtime.batch_global)
                step_start = time.perf_counter()
                should_log = cadence.should_log_train(step)

                match ctx:
                    case PoolAContext():
                        assert self.optimizer is not None and self.ppgd_state is not None
                        self.optimizer.param_groups[0]["lr"] = get_scheduled_value(
                            step, n_steps, ci_fn_lr_schedule
                        )
                        metrics = step_pool_a(
                            ctx,
                            self.component_model,
                            self.optimizer,
                            self._ci_fn_params,
                            self.ppgd_state,
                            batch_T,
                            cfg=runtime,
                            strategy=self.strategy,
                            step=step,
                            n_steps=n_steps,
                            current_frac_of_training=step / n_steps if n_steps > 0 else 0.0,
                            should_log=should_log,
                        )
                    case ChunkContext():
                        assert self.optimizer is not None
                        self.optimizer.param_groups[0]["lr"] = get_scheduled_value(
                            step, n_steps, components_lr_schedule
                        )
                        metrics = step_chunkwise(
                            ctx,
                            self.component_model,
                            self.optimizer,
                            self._all_params,
                            batch_T,
                            runtime,
                            self.strategy,
                            should_log=should_log,
                        )

                if should_log:
                    for k, v in metrics.items():
                        if k.startswith("loss/") or k.startswith("_raw/"):
                            assert v == v, f"NaN in metrics[{k!r}] at step {step}"  # NaN != NaN

                step_ms = (time.perf_counter() - step_start) * 1000.0
                trace(f"TwoPoolTrainer.run: step {step}: dispatched in {step_ms:.1f}ms")
                flush_nccl_event_timings()
                if should_log:
                    dump_memory_stats(f"step {step} done")
                    _log_train_metrics(
                        metrics=metrics,
                        ctx=ctx,
                        device=device,
                        step=step,
                        step_ms=step_ms,
                        ci_fn_lr=get_scheduled_value(step, n_steps, ci_fn_lr_schedule),
                        runtime=runtime,
                        optimizer=self.optimizer,
                        sink=sink,
                    )

                if eval_loop is not None and eval_loop.should_eval(step):
                    from param_decomp_lab.three_pool.two_pool_eval_step import (
                        run_two_pool_eval_step,
                    )

                    assert eval_iterator is not None
                    run_two_pool_eval_step(
                        eval_iterator,
                        n_steps=eval_loop.n_steps,
                        slow_step=eval_loop.should_run_slow_eval(step),
                        metrics=list(eval_loop.metrics),
                        ctx=ctx,
                        step=step,
                        device=str(device),
                        component_model=self.component_model,
                        config=pd_config,
                        runtime_config=self.runtime_config,
                        reconstruction_loss=runtime.reconstruction_loss,
                        sink=sink,
                    )

                if cadence.should_save(step):
                    self.snapshot(scratch_dir)
                    sink.checkpoint_written(step, final=False)

                batch_T = _to_device(next(train_iterator))
                if profiler is not None:
                    profiler.step()

            self.step = n_steps
            self.snapshot(scratch_dir)
            sink.checkpoint_written(self.step, final=True)


def optimize_two_pool(
    target_model: nn.Module,
    train_loader: DataLoader[Any],
    *,
    run_batch: RunBatch,
    reconstruction_loss: ReconstructionLoss,
    pd_config: ThreePoolConstrainedPDConfig,
    runtime_config: PooledRuntimeConfig,
    two_pool_config: TwoPoolTopology,
    cadence: Cadence,
    sink: ThreePoolRunSink,
    scratch_dir: Path,
    eval_loop: EvalLoop | None = None,
    profiler: torch.profiler.profile | None = None,
) -> None:
    trainer = TwoPoolTrainer(
        target_model=target_model,
        run_batch=run_batch,
        reconstruction_loss=reconstruction_loss,
        pd_config=pd_config,
        runtime_config=runtime_config,
        two_pool_config=two_pool_config,
    )
    trainer.run(
        train_loader, sink, cadence, scratch_dir=scratch_dir, eval_loop=eval_loop, profiler=profiler
    )


def _two_pool_layout_fingerprint(ctx: TwoPoolContext) -> dict[str, Any]:
    """Rank-invariant summary of the 2-pool world topology, compared at resume.

    Same key structure as the 3-pool ``_layout_fingerprint`` (so the shared
    ``_rank_invariant_fingerprint_core`` reduces both): ``ci_ranks == ppgd_ranks ==
    pool_a_ranks`` for the 2-pool world.
    """
    world = ctx.world
    return {
        "world_size": world.world_size,
        "ci_ranks": list(world.ci_ranks),
        "ppgd_ranks": list(world.ppgd_ranks),
        "chunks": [{"ranks": list(c.ranks), "sites": list(c.sites)} for c in world.chunks],
    }


def _required[T](value: T | None) -> T:
    assert value is not None
    return value


def _build_two_pool_runtime(
    target_model: nn.Module,
    pd_config: ThreePoolConstrainedPDConfig,
    runtime_config: RuntimeConfig,
    two_pool_config: TwoPoolTopology,
    run_batch: RunBatch,
    reconstruction_loss: ReconstructionLoss,
) -> _ThreePoolRuntime:
    """Assemble the step-context bundle. Reuses ``_ThreePoolRuntime`` with
    ``ci_ranks == ppgd_ranks == pool_a_ranks`` (Pool A is both)."""
    from param_decomp.decomposition_targets import resolve_decomposition_targets

    targets = resolve_decomposition_targets(target_model, pd_config.decomposition_targets)
    c_per_site = {t.module_path: t.C for t in targets}
    numel_global = 0
    for t in targets:
        w = target_model.get_submodule(t.module_path).weight
        assert isinstance(w, Tensor)
        numel_global += w.numel()

    ordered_sites = [t.module_path for t in targets]
    layout = two_pool_config.resolve(ordered_sites, pd_config.batch_size)
    chunks = tuple(Chunk(ranks=ranks, sites=sites) for ranks, sites in layout.chunks)
    for chunk in chunks:
        for site in chunk.sites:
            assert site in c_per_site, (
                f"site '{site}' in a chunk but not in pd_config.decomposition_targets"
            )

    losses = pd_config.losses
    recon_plan = losses.recon_plan
    n_est = sum(recon_plan.n_forwards(chunk.sites) for chunk in chunks)
    imp_min_cfg = losses.imp

    return _ThreePoolRuntime(
        ci_ranks=layout.pool_a_ranks,
        chunks=chunks,
        ppgd_ranks=layout.pool_a_ranks,
        batch_global=pd_config.batch_size,
        c_per_site=c_per_site,
        ci_config=pd_config.ci_config,
        sigmoid_type=pd_config.sigmoid_type,
        run_batch=run_batch,
        reconstruction_loss=reconstruction_loss,
        ppgd_cfg=losses.ppgd,
        recon_plan=recon_plan,
        n_est=n_est,
        coeff_faith=float(_required(losses.faith.coeff)),
        coeff_imp=float(_required(losses.imp.coeff)),
        coeff_stoch=float(_required(losses.stoch.coeff)),
        coeff_ppgd=float(_required(losses.ppgd.coeff)),
        log_name_faith=losses.faith.type,
        log_name_imp=losses.imp.type,
        log_name_stoch=losses.stoch.type,
        log_name_ppgd=losses.ppgd.type,
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
        use_fused_kl=two_pool_config.use_fused_kl,
    )


def _decomposition_targets_for_pool(
    ctx: TwoPoolContext, c_per_site: dict[str, int]
) -> list[DecompositionTarget]:
    """Pool A: every site (full CI fn + full V/U replica). Chunkwise: this rank's sites."""
    match ctx:
        case PoolAContext():
            sites = ctx.world.all_sites
        case ChunkContext():
            sites = ctx.role.sites
    return [DecompositionTarget(module_path=s, C=c_per_site[s]) for s in sites]


def _assert_full_global_batch(batch: Any, batch_global: int) -> None:
    if isinstance(batch, Tensor):
        actual = batch.shape[0]
    elif isinstance(batch, dict) and "input_ids" in batch:
        actual = batch["input_ids"].shape[0]
    elif isinstance(batch, list | tuple) and len(batch) > 0 and isinstance(batch[0], Tensor):
        actual = batch[0].shape[0]
    else:
        raise TypeError(f"Unsupported batch type from DataLoader: {type(batch).__name__}")
    assert actual == batch_global, (
        f"2-pool requires each rank to read the FULL global batch; got leading-dim "
        f"{actual}, expected {batch_global}. Pass dist_state=None to build_loader."
    )


def _log_train_metrics(
    *,
    metrics: dict[str, float],
    ctx: TwoPoolContext,
    device: torch.device,
    step: int,
    step_ms: float,
    ci_fn_lr: float,
    runtime: _ThreePoolRuntime,
    optimizer: torch.optim.Optimizer | None,
    sink: ThreePoolRunSink,
) -> None:
    """Aggregate per-pool metrics to rank 0 and dispatch to ``sink`` (same ``train/``
    key families as the 3-pool / single-pool paths).

    Times the cross-pool aggregation as ``perf/log_ms`` (rank 0). The leading
    ``synchronize`` drains the step's still-in-flight GPU tail so the timer measures
    only the log-path comm, not the step's compute; the ``aggregate_*_to_rank0`` calls
    each end in a host read (``.item()``), so the elapsed time captures real collective
    completion despite CUDA's async dispatch.
    """
    torch.cuda.synchronize()
    log_start = time.perf_counter()
    combined = aggregate_losses_to_rank0(metrics, ctx, device)
    mem_combined = aggregate_max_memory_to_rank0(ctx, device)
    grad_norms = aggregate_grad_norms_to_rank0(metrics, ctx, device)
    comp_grad_by_loss = aggregate_component_grad_by_loss_to_rank0(metrics, ctx, device)

    step_ms_t = torch.tensor([step_ms], device=device)
    if isinstance(ctx, ChunkContext):
        dist.all_reduce(step_ms_t, op=dist.ReduceOp.MAX, group=ctx.world.chunkwise_pool_group)

    if ctx.role.rank != 0 or combined is None:
        return

    if mem_combined is not None:
        combined.update(mem_combined)
    combined["perf/step_ms"] = step_ms_t.item()
    combined["perf/log_ms"] = (time.perf_counter() - log_start) * 1000.0
    combined["loss/total"] = (
        runtime.coeff_faith * combined["loss/faith"]
        + runtime.coeff_imp * combined["loss/imp"]
        + runtime.coeff_stoch * combined["loss/stoch"]
        + runtime.coeff_ppgd * combined["loss/ppgd"]
    )
    for short, class_name in (
        ("faith", runtime.log_name_faith),
        ("imp", runtime.log_name_imp),
        ("stoch", runtime.log_name_stoch),
        ("ppgd", runtime.log_name_ppgd),
    ):
        combined[f"loss/{class_name}"] = combined.pop(f"loss/{short}")
    assert isinstance(ctx, ChunkContext), "rank 0 must be in chunkwise pool (canonical order)"
    assert optimizer is not None
    assert grad_norms is not None
    assert comp_grad_by_loss is not None
    combined.update(grad_norms)
    for short, class_name in (
        ("faith", runtime.log_name_faith),
        ("stoch", runtime.log_name_stoch),
        ("ppgd", runtime.log_name_ppgd),
    ):
        combined[f"grad_norms/components/by_loss/{class_name}"] = comp_grad_by_loss[
            f"grad_norms/by_loss/{short}/components"
        ]
    combined["schedules/lr/components"] = optimizer.param_groups[0]["lr"]
    combined["schedules/lr/ci_fn"] = ci_fn_lr
    sink.console(
        f"--- Step {step} ---",
        *(f"train/{name}: {value:.6g}" for name, value in combined.items()),
    )
    sink.log({f"train/{k}": v for k, v in combined.items()}, step=step)
