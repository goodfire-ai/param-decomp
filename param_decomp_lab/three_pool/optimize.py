"""``ThreePoolTrainer`` and ``optimize_three_pool`` — 3-pool sibling of
:class:`param_decomp.optimize.Trainer` / :func:`param_decomp.optimize.optimize`.

Mirrors the single-pool call shape: caller hands in ``target_model``,
dataloader, configs, sink. Internal validation, per-pool wiring, cross-pool
comms, and the chunkwise streaming recon strategy are all hidden behind the
class boundary.

  * **CI pool** trains the CI fn (replicated across ranks; DP-sharded across
    batch). Holds CI fn + AdamW state. Each step: target_fwd → CI fn fwd →
    broadcast CI to chunkwise + PPGD → dead-time prefetch H_{T+1} → fused backward
    seeded by imp_min + per-site g_CI from chunkwise + PPGD → in-pool all-reduce →
    AdamW. See :mod:`param_decomp_lab.three_pool.step_ci`.

  * **Chunkwise pool** trains V/U (chunk-DDP within chunk; sharded across
    sites). Recv CI → faithfulness + chunkwise stoch recon → send g_CI back
    → recv g_VU from PPGD → combine → in-chunk all-reduce → AdamW → async
    ship updated V/U → PPGD. See :mod:`param_decomp_lab.three_pool.step_chunkwise`.

  * **PPGD pool** is a stateless full V/U replica. Recv CI → PPGD warmup +
    final recon → backward seeds V/U + CI grads (no outer source step;
    warmup inner loop owns sources) → sum-reduce V/U within PPGD pool →
    send g_VU to chunkwise + g_CI to CI → recv updated V/U. See
    :mod:`param_decomp_lab.three_pool.step_ppgd`.

Data-handling contract
----------------------
Every rank must read the FULL global batch on every step (i.e. each batch
tensor from ``train_loader`` has shape ``[batch_global, ...]``). The runner
asserts this. Callers wiring up the loader should pass ``dist_state=None`` so
the data path replicates the batch across ranks instead of sharding it.

Each step function then slices to its own per-pool batch shard via its
``role.batch_slice(...)`` (see ``role.py``). The 3-way batch routing in
``layout.py`` (see "Batch-split routing" in its docstring) assumes this
sliced-from-global pattern.
"""

import datetime
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
from param_decomp.configs import Cadence, PDConfig, RuntimeConfig
from param_decomp.decomposition_targets import (
    DecompositionTarget,
    resolve_decomposition_targets,
)
from param_decomp.distributed import seed_all_ranks, seed_per_rank
from param_decomp.masks import AllLayersRouter
from param_decomp.metrics.persistent_pgd_recon import validate_pgd_scope
from param_decomp.metrics.persistent_pgd_state import (
    PerBatchPerPositionScope,
    PersistentPGDState,
)
from param_decomp.optimize import EvalLoop, load_optimizer_state_by_name, optimizer_state_by_name
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
from param_decomp_lab.three_pool.config import PooledRuntimeConfig, ThreePoolTopology
from param_decomp_lab.three_pool.consolidate import (
    CONSOLIDATE_META_FILENAME,
    load_ppgd_shard,
    ppgd_shard_dirname,
    prune_old_scratch,
    step_scratch_dir,
)
from param_decomp_lab.three_pool.context import (
    ChunkContext,
    CIContext,
    PoolContext,
    PPGDContext,
    build_pool_context,
)
from param_decomp_lab.three_pool.layout import (
    Chunk,
    build_world,
    flush_nccl_event_timings,
)
from param_decomp_lab.three_pool.pd_config import ThreePoolConstrainedPDConfig
from param_decomp_lab.three_pool.recon_loss_strategy import ReconLossStrategy
from param_decomp_lab.three_pool.reductions import (
    aggregate_component_grad_by_loss_to_rank0,
    aggregate_grad_norms_to_rank0,
    aggregate_losses_to_rank0,
    aggregate_max_memory_to_rank0,
)
from param_decomp_lab.three_pool.runtime import _ThreePoolRuntime
from param_decomp_lab.three_pool.step_chunkwise import (
    run_faithfulness_warmup_chunkwise,
    step_chunkwise,
)
from param_decomp_lab.three_pool.step_ci import step_ci
from param_decomp_lab.three_pool.step_ppgd import step_ppgd

# Collective timeout for every 3-pool subgroup. Now that consolidation is async
# (off the train loop), the longest gap between collectives on the loop is the
# in-train (fast) eval pass plus a checkpoint partial-write barrier — minutes,
# not the old ~10-min rank-0 read of ~100 GB. 10 min covers that worst case with
# generous margin while still being far below "hang forever". Env-overridable
# (seconds) so the watchdog-safe-at-low-timeout test can force a tight bound.
_DEFAULT_PG_TIMEOUT = datetime.timedelta(minutes=10)
# With torch.compile on, step 0 pays a one-time ~minutes compilation while the other pools
# wait at the first cross-pool collective; widen the timeout so that wait can't trip the
# watchdog. Steady-state collectives are sub-second, so the looser bound only delays
# detection of a genuine first-step hang.
_COMPILE_PG_TIMEOUT = datetime.timedelta(minutes=20)


def _resolve_pg_timeout(*, compiling: bool = False) -> datetime.timedelta:
    override_s = os.environ.get("PD_3POOL_PG_TIMEOUT_S", "").strip()
    if override_s:
        return datetime.timedelta(seconds=float(override_s))
    return _COMPILE_PG_TIMEOUT if compiling else _DEFAULT_PG_TIMEOUT


class ThreePoolTrainer:
    """Stateful 3-pool trainer.

    Construction wires up the runtime bundle, world layout, ComponentModel,
    recon loss strategy, and the per-pool optimizer (CI / chunkwise have one;
    PPGD has none — see module docstring). The PPGD state itself is built on
    the first batch of :meth:`run` because its source tensor shapes depend on
    the data's sequence dims.

    Checkpointing is split across two phases. :meth:`snapshot` runs on the train
    loop: each rank writes a self-contained partial (its owned model params +
    optimizer state + PPGD sources) to the shared-FS scratch dir, with a single
    barrier — no rank-0 read, no model assembly. The async consolidation job
    (:mod:`param_decomp_lab.three_pool.consolidate`) later reads every partial,
    assembles the canonical :class:`~param_decomp.training_state.ThreePoolTrainingState`
    + ``model_<step>.pth``, and runs the slow eval. :meth:`from_snapshot`
    reconstructs a trainer from a consolidated ``ThreePoolTrainingState`` (each
    rank loads its own slice).
    """

    pd_config: ThreePoolConstrainedPDConfig
    runtime_config: PooledRuntimeConfig
    three_pool_config: ThreePoolTopology
    reconstruction_loss: ReconstructionLoss
    component_model: LMComponentModel
    ctx: PoolContext
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
        three_pool_config: ThreePoolTopology,
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
        # Verify FA can dispatch on the CI fn's largest SDPA shape. If it
        # can't (head_dim > 128, missing kernel, etc.), error here rather
        # than silently fall back to a 5-10x slower math kernel during
        # training. Skipping the global-toggle approach to stay compatible
        # with torch.compile's fake-tensor trace.
        ci_attn = _ci_attn_shape_or_none(pd_config)
        if ci_attn is not None:
            d_model, n_heads = ci_attn
            verify_flash_attention_available(head_dim=d_model // n_heads)

        trace("ThreePoolTrainer.__init__: building runtime")
        self.runtime = _build_runtime(
            target_model=target_model,
            pd_config=pd_config,
            runtime_config=runtime_config,
            three_pool_config=three_pool_config,
            run_batch=run_batch,
            reconstruction_loss=reconstruction_loss,
        )

        # PPGD runs only on PPGD pool; the relevant per-rank batch is batch // n_ppgd.
        validate_pgd_scope(
            [pd_config.losses.ppgd],
            batch_size=pd_config.batch_size,
            world_size=len(self.runtime.ppgd_ranks),
        )

        torch.set_float32_matmul_precision("high")

        self._device = torch.device(runtime_config.device)
        trace("ThreePoolTrainer.__init__: build_world: enter")
        world = build_world(
            ci_ranks=list(self.runtime.ci_ranks),
            chunks=list(self.runtime.chunks),
            ppgd_ranks=list(self.runtime.ppgd_ranks),
            batch_global=self.runtime.batch_global,
            pg_timeout=_resolve_pg_timeout(
                compiling=runtime_config.compile_chunkwise
                or runtime_config.compile_ci_fn
                or runtime_config.compile_ppgd
            ),
            device=self._device,
        )
        trace("ThreePoolTrainer.__init__: build_world: done")
        self.ctx = build_pool_context(world, dist.get_rank())
        decomposition_targets = _decomposition_targets_for_pool(self.ctx, self.runtime.c_per_site)
        trace(
            f"ThreePoolTrainer.__init__: my_pool={self.ctx.kind} "
            f"n_decomp_targets={len(decomposition_targets)}"
        )

        target_model.requires_grad_(False)
        # Resync RNG across ranks before V/U + CI fn init. DDP partners within a
        # chunk must start with identical params, but anything between
        # ``set_seed(pd.seed)`` in _fresh_main and here (loader build, distributed
        # init, etc.) can advance the RNG by rank-dependent amounts. Without this,
        # partners initialize different V/U and the in-chunk grad all-reduce can't
        # bring them back into sync.
        seed_all_ranks(pd_config.seed)
        trace("ThreePoolTrainer.__init__: LMComponentModel build: enter")
        # Target-type dispatch + validation lives in LMComponentModel.build → _componentize
        # (single source of truth); an unsupported target raises there.
        self.component_model = LMComponentModel.build(
            target_model=target_model,
            decomposition_targets=decomposition_targets,
            ci_config=pd_config.ci_config,
            sigmoid_type=pd_config.sigmoid_type,
        )
        trace("ThreePoolTrainer.__init__: LMComponentModel build: done")
        # Drop pool-irrelevant params before moving to GPU. RNG draws used to
        # init them already happened (in the ctor above), so equivalence with
        # single-pool is preserved.
        match self.ctx:
            case ChunkContext() | PPGDContext():
                self.component_model.drop_ci_fn()
                trace(f"ThreePoolTrainer.__init__: dropped ci_fn ({self.ctx.kind} pool)")
            case CIContext():
                self.component_model.drop_components()
                trace("ThreePoolTrainer.__init__: dropped V/U components (ci pool)")
        # Activation checkpointing is chunkwise-only. The chunkwise pool carries the full per-rank
        # batch (bl_chunk = batch_size; no within-chunk DP at chunk_dp=1) so it needs ckpt to fit;
        # PPGD/CI don't (PPGD fits plain at its small bl_pp; CI's target fwd is no_grad). PPGD's
        # autograd.grad recompute is also non-deterministic under ckpt (nested mask_infos through
        # checkpoint) — tracked as a follow-up; chunkwise's .backward path is checkpoint-clean.
        if not isinstance(self.ctx, ChunkContext):
            self.component_model.model._use_activation_checkpointing = False
        trace("ThreePoolTrainer.__init__: ComponentModel.to(device): enter")
        self.component_model = self.component_model.to(self._device)
        trace("ThreePoolTrainer.__init__: ComponentModel.to(device): done")
        dump_memory_stats("after ComponentModel.to(device)")
        # CI pool: activation-checkpoint the CI-fn transformer blocks, then
        # torch.compile the whole CI-fn forward (the checkpoint loop sits inside the
        # compiled region so the compiler optimizes the recompute). Checkpoint saves
        # ~15 GB of block-activation high-water (the wide MLP/attn intermediates are
        # recomputed in backward); compile pays the recompute back and then some — the
        # 1-GPU CI probe at bl=8/seq=1024 measured baseline 227ms → +ckpt 256ms (+13%)
        # → +ckpt+compile(whole) 206ms (-9% vs baseline). The CI pool is the compute-idle
        # pool (PPGD is the long pole), so even the bare +13% would be free on the
        # critical path. Whole-forward compile + checkpoint + flash-SDPA composes cleanly
        # on torch>=2.11 (its AOT min-cut partitioner guards the DCE'd checkpointed
        # flash-SDPA RNG op via functionalize_rng_ops; <=2.10 KeyError'd at distributed
        # scale). Both default-on (runtime_config.checkpoint_ci_fn / .compile_ci_fn).
        is_ci = isinstance(self.ctx, CIContext)
        if is_ci:
            assert self.component_model.ci_fn is not None
            if runtime_config.checkpoint_ci_fn:
                trace("ThreePoolTrainer.__init__: enable CI-fn activation checkpointing")
                for m in self.component_model.ci_fn.modules():
                    if isinstance(m, GlobalSharedTransformerCiFn):
                        m.enable_activation_checkpointing()
            if runtime_config.compile_ci_fn:
                user = os.environ.get("USER", "u")
                rank = dist.get_rank()
                os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/torchinductor_{user}_r{rank}"
                os.environ["TRITON_CACHE_DIR"] = f"/tmp/triton_{user}_r{rank}"
                trace("ThreePoolTrainer.__init__: torch.compile(ci_fn) [whole forward]")
                self.component_model.ci_fn.compile()
        # Chunkwise pool: torch.compile the (target + masked) model forward — a validated 2.74× on
        # the chunkwise step single-GPU (the throughput pole). The vendored mask-arg forward traces
        # cleanly (0 graph breaks) and attention uses F.sdpa directly. Default-on
        # (runtime_config.compile_chunkwise). The one-time first-step compilation is absorbed by the
        # widened PG timeout (see _resolve_pg_timeout(compiling=...)). The chunkwise forward is only
        # called in step_chunkwise (target=None + masked dict, both validated); eval barriers it
        # through, so no recompile.
        if isinstance(self.ctx, ChunkContext) and runtime_config.compile_chunkwise:
            # Whole-model compile: the block-loop checkpoint(block, ...) sits INSIDE the compiled
            # region. Requires torch >= 2.11 — its AOT min-cut partitioner guards the DCE'd
            # checkpointed flash-SDPA RNG op in `functionalize_rng_ops` (partitioners.py) instead of
            # `KeyError '_scaled_dot_product_flash_attention'`-ing on it as <=2.10 did (only at
            # distributed scale; not reproducible single-GPU). ~2.74x single-GPU. Per-rank
            # inductor/triton cache dirs are defensive against shared-cache contention across the 160
            # concurrent compilers (/tmp is shared here); set before the first compile.
            user = os.environ.get("USER", "u")
            rank = dist.get_rank()
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/torchinductor_{user}_r{rank}"
            os.environ["TRITON_CACHE_DIR"] = f"/tmp/triton_{user}_r{rank}"
            trace("ThreePoolTrainer.__init__: torch.compile(whole model) [per-rank cache]")
            self.component_model.model.compile()
        # PPGD pool: compile the SAME masked model forward (the warmup PGD inner loop + the final
        # recon forward both run it). The forward-at-scale is already proven by the chunkwise pool
        # above (identical compiled artifact); the PPGD-specific fused autograd.grad over
        # V/U + CI + sources was 1-GPU-validated correct (isolated fp32 grad rel-err 8e-7). Default-on
        # (runtime_config.compile_ppgd). Per-rank inductor/triton caches as above.
        if isinstance(self.ctx, PPGDContext) and runtime_config.compile_ppgd:
            user = os.environ.get("USER", "u")
            rank = dist.get_rank()
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/torchinductor_{user}_r{rank}"
            os.environ["TRITON_CACHE_DIR"] = f"/tmp/triton_{user}_r{rank}"
            trace("ThreePoolTrainer.__init__: torch.compile(ppgd model) [per-rank cache]")
            self.component_model.model.compile()
        # Diverge stochastic RNG per rank for mask sampling.
        seed_per_rank(pd_config.seed)

        trace("ThreePoolTrainer.__init__: building ReconLossStrategy")
        self.strategy = ReconLossStrategy.from_cfg(
            self.component_model,
            use_fused_kl=three_pool_config.use_fused_kl,
            unfused_recon=reconstruction_loss,
        )
        trace("ThreePoolTrainer.__init__: ReconLossStrategy: done")

        self.optimizer = None
        self._all_params: list[nn.Parameter] = []
        self._ci_fn_params: list[nn.Parameter] = []
        self._component_params: list[nn.Parameter] = []
        self.ppgd_state = None
        self._pending_ppgd_resume_state: dict[str, Any] | None = None

        trace(f"ThreePoolTrainer.__init__: optimizer build: enter (pool={self.ctx.kind})")
        match self.ctx:
            case CIContext():
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
            case PPGDContext():
                pass  # ppgd_state constructed lazily from first batch in run()
        trace("ThreePoolTrainer.__init__: optimizer build: done")
        dump_memory_stats("after optimizer build")
        trace("ThreePoolTrainer.__init__: exit")

    # ============================ Atomic cfg + state ============================

    def _named_params_for_my_optimizer(self) -> list[tuple[str, nn.Parameter]]:
        """The ``(name, param)`` pairs in the order they were added to this rank's
        optimizer. ``CI`` pool returns ``ci_fn.*`` pairs; ``chunkwise`` pool returns
        ``components.<site>.*`` pairs for this rank's chunk sites; ``PPGD`` has
        no optimizer and returns ``[]``."""
        match self.ctx:
            case CIContext():
                assert self.component_model.ci_fn is not None
                return [(f"ci_fn.{n}", p) for n, p in self.component_model.ci_fn.named_parameters()]
            case ChunkContext():
                out: list[tuple[str, nn.Parameter]] = []
                for site in self.ctx.role.sites:
                    for n, p in self.component_model.components[site].named_parameters():
                        out.append((f"components.{site}.{n}", p))
                return out
            case PPGDContext():
                return []

    def snapshot(self, scratch_dir: Path) -> None:
        """Write this rank's self-contained checkpoint partial; then return.

        Each rank writes everything needed to reconstruct the checkpoint for
        its own slice to ``scratch_dir / f"step_{S}" / f"rank_{rank}.pth"``:

        * ``model_params``: a slice of this rank's ``component_model.state_dict()`` —
          chunk leaders contribute their sites' ``_components.<site>.*``,
          the CI pool leader contributes ``ci_fn.*``, everyone else contributes
          nothing (their V/U / CI fn are replicas of a leader's).
        * ``optimizer_by_name``: this rank's optimizer state, name-keyed.

        PPGD ranks ALSO write their data-shaped adversarial sources, in parallel,
        straight to the stable shard ``out_dir / ppgd_<S> / rank_<rank>.pth``
        (``out_dir == scratch_dir.parent``) — not into the partial. Each rank
        writes its own shard, so this is free (the same bytes the partial used to
        carry, just split into the file resume reads directly). Consolidation
        then streams only the small parameter-shaped partials.

        Rank 0 also writes ``meta.pth`` (configs + layout fingerprint + the
        ``c_per_site`` / ``all_sites`` needed to rebuild the full model). A
        single barrier ensures every partial is on the shared FS before the
        train loop continues — there is NO rank-0 read and NO model assembly
        here. The async consolidation job
        (:func:`param_decomp_lab.three_pool.consolidate.consolidate_step`)
        reads these partials off the critical path.
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

        # The mkdir + meta write race: rank 0 must finish them before any other
        # rank writes its partial into step_dir.
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

        # One rejoining barrier so all ranks leave snapshot together. This is
        # cheap now (just the partial write, no 100 GB read), so it cannot
        # overrun the collective watchdog. PD_3POOL_DISABLE_REJOIN_BARRIER skips
        # it for the legacy race repro. The old PD_3POOL_SNAPSHOT_RANK0_SLEEP_S
        # fault injection now lives in the async consolidate job (off the loop),
        # since that is where the slow read moved — the train loop has nothing
        # slow left to sleep through.
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

        Only PPGD ranks own sources: the live state once built, else the pending
        resume state for a resumed-but-not-yet-stepped rank.
        """
        if self.ppgd_state is not None:
            return self.ppgd_state.state_dict()
        return self._pending_ppgd_resume_state

    def _owned_model_params(self) -> dict[str, Tensor]:
        """The slice of this rank's model state_dict it's responsible for saving.

        Only leaders contribute: chunk leaders own their sites' V/U, the CI
        pool leader owns the CI fn. Every other rank holds a replica, so it
        contributes nothing — the union across leaders covers the full model.
        """
        sd = self.component_model.state_dict()
        keys = set(sd.keys())
        match self.ctx:
            case ChunkContext() if self.ctx.role.is_chunk_leader:
                owned = owned_model_state_keys(keys, sites=self.ctx.role.sites)
            case CIContext() if self.ctx.role.is_pool_leader:
                owned = ci_fn_state_keys(keys)
            case _:
                owned = set()
        return {k: sd[k].cpu() for k in owned}

    def _build_meta(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "world_size": self.ctx.world.world_size,
            "all_sites": list(self.ctx.world.all_sites),
            "c_per_site": dict(self.runtime.c_per_site),
            "pd_config": self.pd_config.model_dump(),
            "runtime_config": self.runtime_config.model_dump(),
            "three_pool_config": self.three_pool_config.model_dump(),
            "layout_fingerprint": _layout_fingerprint(self.ctx),
        }

    @classmethod
    def from_snapshot(
        cls,
        snapshot: ThreePoolTrainingState,
        *,
        target_model: nn.Module,
        run_batch: RunBatch,
        reconstruction_loss: ReconstructionLoss,
        ppgd_shard_dir: Path | None,
        cfg_overrides: dict[str, Any] | None = None,
    ) -> Self:
        """Reconstruct a 3-pool trainer from a canonical snapshot.

        Each rank must arrive with the canonical state populated (typically by
        reading ``training_<step>.pth`` on every rank from a shared filesystem).
        Each rank extracts its own slice of the canonical state by rank id and
        loads it into its locally-owned sub-state. ``ppgd_shard_dir`` is the run's
        ``ppgd_<step>/`` dir holding the per-rank adversarial source shards (``None``
        ⇒ the adversary re-warms from scratch).
        """
        pd_dict = snapshot.pd_config
        if cfg_overrides is not None:
            pd_dict = {**pd_dict, **cfg_overrides}
        pd_config = ThreePoolConstrainedPDConfig.model_validate(pd_dict)
        # `topology` is reconstructed separately into `three_pool_config`; the rest of the dump
        # (base scalars + compile flags) revalidates as `PooledRuntimeConfig` so the flags
        # survive resume without importing the lab-side `ThreePoolRuntimeConfig` (which would cycle).
        runtime_dict = {k: v for k, v in snapshot.runtime_config.items() if k != "topology"}
        runtime_config = PooledRuntimeConfig.model_validate(runtime_dict)
        three_pool_config = ThreePoolTopology.model_validate(snapshot.three_pool_config)

        trainer = cls(
            target_model=target_model,
            run_batch=run_batch,
            reconstruction_loss=reconstruction_loss,
            pd_config=pd_config,
            runtime_config=runtime_config,
            three_pool_config=three_pool_config,
        )
        saved_fp = _rank_invariant_fingerprint_core(snapshot.layout_fingerprint)
        current_fp = _rank_invariant_fingerprint_core(_layout_fingerprint(trainer.ctx))
        assert saved_fp == current_fp, (
            f"3-pool layout topology mismatch on resume:\n"
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
        # The full gathered model state has every site's V/U + the CI fn. Each
        # rank's locally-constructed ComponentModel only has the keys it owns,
        # so we filter to the local subset before loading. `strict=False` lets
        # us drop the canonical entries this rank doesn't hold.
        local_model_keys = set(self.component_model.state_dict().keys())
        local_slice = {k: v for k, v in state.component_model.items() if k in local_model_keys}
        self.component_model.load_state_dict(local_slice, strict=False)

        if self.optimizer is not None:
            named_params = self._named_params_for_my_optimizer()
            match self.ctx:
                case ChunkContext():
                    by_name = state.components_optimizer
                case CIContext():
                    by_name = state.ci_fn_optimizer
                case PPGDContext():
                    by_name = {}
            load_optimizer_state_by_name(self.optimizer, named_params, by_name)
        if isinstance(self.ctx, PPGDContext):
            self._pending_ppgd_resume_state = load_ppgd_shard(ppgd_shard_dir, self.ctx.role.rank)

    # ============================ Training loop ============================

    def run(
        self,
        train_loader: DataLoader[Any],
        sink: ThreePoolRunSink,
        cadence: Cadence,
        scratch_dir: Path,
        eval_loop: EvalLoop | None = None,
        profiler: torch.profiler.profile | None = None,
    ) -> None:
        """Advance training from ``self.step`` to ``self.pd_config.steps``.

        ``scratch_dir`` is a shared-filesystem directory used by
        :meth:`snapshot` for cross-rank file-based gather (replaces
        ``dist.gather_object`` which scales poorly at XL payload sizes).

        When ``eval_loop`` is non-None, runs a 3-pool eval pass on cadence:
        CI pool ships full ``CIOutputs`` to PPGD; PPGD assembles a
        ``MetricContext`` and runs every ``eval_loop.metric``; chunkwise pool
        barriers through. Metric reductions are scoped to the PPGD pool
        subgroup via :func:`use_reduction_group` so non-PPGD pools don't
        block on them. See :mod:`param_decomp_lab.three_pool.eval_step`.
        """
        trace("Trainer.run: enter")
        pd_config = self.pd_config
        ctx = self.ctx
        world = ctx.world
        runtime = self.runtime
        n_steps = pd_config.steps
        device = self._device

        train_iterator = loop_dataloader(train_loader)
        # Loader skip-replay (resumed mid-trajectory).
        for _ in range(self.step):
            next(train_iterator)

        # Eval setup. Only the PPGD pool runs eval metrics; bind there only —
        # CI / chunkwise pools never touch metric state. The eval iterator runs on
        # every rank so all pools advance the loader in lock-step (CI + PPGD
        # each consume the full eval batch and slice internally, mirroring
        # the train data-handling contract; chunkwise reads but discards).
        eval_iterator = loop_dataloader(eval_loop.loader) if eval_loop is not None else None
        if eval_loop is not None and isinstance(ctx, PPGDContext):
            for m in eval_loop.metrics:
                # LMComponentModel exposes the metric-facing surface the eval metrics use;
                # it shares no base with ComponentModel, so cast through object.
                m.bind(
                    model=cast(ComponentModel, cast(object, self.component_model)),
                    device=str(device),
                )

        # Peek first batch (after any skip) for PPGD source-shape sizing.
        trace("Trainer.run: first_batch peek: enter")
        first_batch = next(train_iterator)
        trace("Trainer.run: first_batch peek: done")
        train_iterator = itertools.chain([first_batch], train_iterator)
        _assert_full_global_batch(first_batch, runtime.batch_global)

        if isinstance(ctx, PPGDContext) and self.ppgd_state is None:
            trace("Trainer.run: PPGDState ctor: enter")
            ppgd_cfg = runtime.ppgd_cfg
            # The 3-pool currently only supports per-batch-per-position sources:
            # they're independent per batch element, so a PPGD-pool batch split is
            # just slicing — no cross-rank source sync needed. Any replicated scope
            # would require broadcast-init + grad-reduce over ppgd_pool_group, which
            # we don't implement here. Add it if/when another arrangement is wanted.
            assert isinstance(ppgd_cfg.scope, PerBatchPerPositionScope), (
                f"3-pool supports only PerBatchPerPositionScope PPGD sources; got "
                f"{type(ppgd_cfg.scope).__name__}. Replicated scopes need cross-pool "
                f"source-replica sync, not implemented in the 3-pool."
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
                replica_sync_group=None,
            )
            if self._pending_ppgd_resume_state is not None:
                self.ppgd_state.load_state_dict(self._pending_ppgd_resume_state)
                self._pending_ppgd_resume_state = None
            trace("Trainer.run: PPGDState ctor: done")

        if (
            self.step == 0
            and isinstance(ctx, ChunkContext)
            and pd_config.faithfulness_warmup_steps > 0
        ):
            trace(
                f"Trainer.run: faithfulness warmup: enter ({pd_config.faithfulness_warmup_steps} steps)"
            )
            run_faithfulness_warmup_chunkwise(
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

        def _to_device(b: Any) -> Any:
            """Move a batch yielded by the train loader to this rank's GPU.

            3-pool's step functions assume the batch is already on-device. The
            loader produces CPU tensors; moving here keeps the step functions thin.
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
                trace(f"Trainer.run: step {step}: start (pool={ctx.kind})")

                step_start = time.perf_counter()
                should_log = cadence.should_log_train(step)

                # batch_T should already be on this rank's device (placed by _to_device).
                if isinstance(batch_T, Tensor):
                    assert batch_T.device == device, (
                        f"3-pool batch device drift at step {step}: {batch_T.device} vs {device}"
                    )
                match ctx:
                    case CIContext():
                        assert self.optimizer is not None, (
                            f"CI rank {ctx.role.rank} missing optimizer"
                        )
                        assert len(self._ci_fn_params) > 0, (
                            f"CI rank {ctx.role.rank} has no ci_fn params to optimize"
                        )
                        self.optimizer.param_groups[0]["lr"] = get_scheduled_value(
                            step, n_steps, ci_fn_lr_schedule
                        )
                        next_batch_for_prefetch = batch_T_plus_1 if step < n_steps - 1 else None
                        metrics, h_cache_ci = step_ci(
                            ctx,
                            self.component_model,
                            self.optimizer,
                            self._ci_fn_params,
                            batch_T=batch_T,
                            batch_T_plus_1=next_batch_for_prefetch,
                            h_cache_T=h_cache_ci,
                            cfg=runtime,
                            current_frac_of_training=step / n_steps if n_steps > 0 else 0.0,
                            should_log=should_log,
                        )
                    case ChunkContext():
                        assert self.optimizer is not None, (
                            f"chunkwise rank {ctx.role.rank} missing optimizer"
                        )
                        assert ctx.role.sites, (
                            f"chunkwise rank {ctx.role.rank} has no sites — empty chunk"
                        )
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
                    case PPGDContext():
                        assert self.ppgd_state is not None, (
                            f"PPGD rank {ctx.role.rank} has no ppgd_state — lazy init failed"
                        )
                        metrics = step_ppgd(
                            ctx,
                            self.component_model,
                            self.ppgd_state,
                            batch_T,
                            runtime,
                            self.strategy,
                            step=step,
                            n_steps=n_steps,
                            should_log=should_log,
                        )
                # NaN check + .item()-bearing metrics only fire on log steps —
                # the step fns return empty ``metrics={}`` otherwise to avoid
                # CPU↔GPU syncs on the per-step critical path. NaN won't catch
                # until the next log step, but train_log_every is small enough
                # that the propagation distance is bounded.
                if should_log:
                    for k, v in metrics.items():
                        if k.startswith("loss/") or k.startswith("_raw/"):
                            assert v == v, f"NaN in metrics[{k!r}] at step {step}"  # NaN != NaN

                # step_ms is CPU wall-clock only — the kernels enqueued during this
                # step are still running on the GPU. So this number is "time to
                # dispatch + Python overhead", not GPU time. For real per-step or
                # per-kernel timing, run a profile (PD_TORCH_PROFILE_RANKS) or set
                # CUDA_LAUNCH_BLOCKING=1 for a serialized re-run.
                step_ms = (time.perf_counter() - step_start) * 1000.0
                trace(f"Trainer.run: step {step}: dispatched in {step_ms:.1f}ms")
                flush_nccl_event_timings()
                if should_log:
                    dump_memory_stats(f"step {step} done")

                if should_log:
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
                    from param_decomp_lab.three_pool.eval_step import run_eval_step

                    assert eval_iterator is not None
                    run_eval_step(
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
                        c_per_site=runtime.c_per_site,
                        sink=sink,
                    )

                if cadence.should_save(step):
                    self.snapshot(scratch_dir)
                    sink.checkpoint_written(step, final=False)

                batch_T = (
                    batch_T_plus_1
                    if batch_T_plus_1 is not None
                    else _to_device(next(train_iterator))
                )
                batch_T_plus_1 = _to_device(next(train_iterator, None))

                if profiler is not None:
                    profiler.step()

            self.step = n_steps
            self.snapshot(scratch_dir)
            sink.checkpoint_written(self.step, final=True)


def optimize_three_pool(
    target_model: nn.Module,
    train_loader: DataLoader[Any],
    *,
    run_batch: RunBatch,
    reconstruction_loss: ReconstructionLoss,
    pd_config: ThreePoolConstrainedPDConfig,
    runtime_config: PooledRuntimeConfig,
    three_pool_config: ThreePoolTopology,
    cadence: Cadence,
    sink: ThreePoolRunSink,
    scratch_dir: Path,
    eval_loop: EvalLoop | None = None,
    profiler: torch.profiler.profile | None = None,
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
    trainer.run(
        train_loader,
        sink,
        cadence,
        scratch_dir=scratch_dir,
        eval_loop=eval_loop,
        profiler=profiler,
    )


def _layout_fingerprint(ctx: PoolContext) -> dict[str, Any]:
    """Rank-invariant summary of the 3-pool world topology, compared at resume.

    Must NOT include this rank's local view (``role.rank`` / ``kind`` /
    ``sites``): the snapshot stores the fingerprint computed on rank 0,
    but ``from_snapshot`` compares it on EVERY rank, so a rank-local fingerprint
    would mismatch on every non-rank-0 rank. The full chunk→ranks→sites mapping
    below is identical on all ranks and fully captures the topology that
    ``_load_canonical_state`` relies on.
    """
    world = ctx.world
    return {
        "world_size": world.world_size,
        "ci_ranks": list(world.ci_ranks),
        "ppgd_ranks": list(world.ppgd_ranks),
        "chunks": [{"ranks": list(c.ranks), "sites": list(c.sites)} for c in world.chunks],
    }


def _rank_invariant_fingerprint_core(fp: dict[str, Any]) -> dict[str, Any]:
    """Reduce a saved-or-current layout fingerprint to its rank-invariant core.

    Validating the topology on resume must not depend on which rank wrote the
    snapshot. The fingerprint carries ``world_size`` / ``ci_ranks`` / ``ppgd_ranks``
    plus the chunk count (``len(chunks)``), which together pin down the pool
    partition. The per-chunk ranks→sites mapping is fully re-derived from
    ``three_pool_config`` (also stored in the snapshot and used to rebuild the
    trainer), so comparing the core here is sufficient.
    """
    return {
        "world_size": fp["world_size"],
        "ci_ranks": list(fp["ci_ranks"]),
        "ppgd_ranks": list(fp["ppgd_ranks"]),
        "n_chunks": len(fp["chunks"]),
    }


def _required[T](value: T | None) -> T:
    """Narrow a value the `ThreePoolLosses` validator already guarantees non-None."""
    assert value is not None
    return value


def _build_runtime(
    target_model: nn.Module,
    pd_config: ThreePoolConstrainedPDConfig,
    runtime_config: RuntimeConfig,
    three_pool_config: ThreePoolTopology,
    run_batch: RunBatch,
    reconstruction_loss: ReconstructionLoss,
) -> _ThreePoolRuntime:
    """Assemble the step-context bundle from configs + target.

    Expands ``decomposition_targets`` → the ordered site list, resolves the topology
    into the canonical rank/chunk assignment (needs the expanded sites + batch), and
    builds the runtime ``Chunk`` objects. The per-chunk site coverage is automatic now
    (chunks ARE the expanded sites), but we keep an assert that every resolved chunk
    site is in ``c_per_site`` as a tripwire against a resolver/expansion drift.
    """
    targets = resolve_decomposition_targets(target_model, pd_config.decomposition_targets)
    c_per_site = {t.module_path: t.C for t in targets}
    numel_global = 0
    for t in targets:
        w = target_model.get_submodule(t.module_path).weight
        assert isinstance(w, Tensor)
        numel_global += w.numel()

    ordered_sites = [t.module_path for t in targets]
    layout = three_pool_config.resolve(ordered_sites, pd_config.batch_size)

    chunks = tuple(Chunk(ranks=ranks, sites=sites) for ranks, sites in layout.chunks)
    for chunk in chunks:
        for site in chunk.sites:
            assert site in c_per_site, (
                f"site '{site}' in a chunk but not in pd_config.decomposition_targets "
                f"after pattern expansion. Available: {sorted(c_per_site)[:5]}..."
            )

    losses = pd_config.losses
    ppgd_cfg = losses.ppgd
    imp_min_cfg = losses.imp

    recon_plan = losses.recon_plan
    n_est = sum(recon_plan.n_forwards(chunk.sites) for chunk in chunks)

    return _ThreePoolRuntime(
        ci_ranks=layout.ci_ranks,
        chunks=chunks,
        ppgd_ranks=layout.ppgd_ranks,
        batch_global=pd_config.batch_size,
        c_per_site=c_per_site,
        ci_config=pd_config.ci_config,
        sigmoid_type=pd_config.sigmoid_type,
        run_batch=run_batch,
        reconstruction_loss=reconstruction_loss,
        ppgd_cfg=ppgd_cfg,
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
        use_fused_kl=three_pool_config.use_fused_kl,
    )


def _ci_attn_shape_or_none(pd_config: PDConfig) -> tuple[int, int] | None:
    """Pull ``(d_model, n_heads)`` from the CI fn's attn config if the CI fn
    has one (transformer variants), else ``None``. Used to derive the SDPA
    shape for FA startup verification.
    """
    ci_cfg = pd_config.ci_config
    # Walk the nested config tree without a hard dependency on the CI types
    # (mode/fn_type discriminators bury the attn config differently per variant).
    for attr in ("simple_transformer_ci_cfg", "transformer_cfg"):
        nested = getattr(ci_cfg, attr, None)
        if nested is None:
            continue
        attn = getattr(nested, "attn_config", None)
        d_model = getattr(nested, "d_model", None)
        if attn is not None and d_model is not None:
            return d_model, attn.n_heads
    return None


def _decomposition_targets_for_pool(
    ctx: PoolContext, c_per_site: dict[str, int]
) -> list[DecompositionTarget]:
    """CI/PPGD pools: every site (full CI fn / full V/U replica).
    Chunkwise pool: only this rank's chunk sites."""
    match ctx:
        case CIContext() | PPGDContext():
            sites = ctx.world.all_sites
        case ChunkContext():
            sites = ctx.role.sites
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
    ctx: PoolContext,
    device: torch.device,
    step: int,
    step_ms: float,
    ci_fn_lr: float,
    runtime: _ThreePoolRuntime,
    optimizer: torch.optim.Optimizer | None,
    sink: ThreePoolRunSink,
) -> None:
    """Aggregate per-pool metrics to rank 0 and dispatch to ``sink``.

    Emits the same `train/` key families as the single-pool `Trainer`
    (`param_decomp.optimize`): the four losses keyed by their metric class name,
    `loss/total`, `grad_norms/*` (per-param + summaries, gathered cross-pool),
    and both LR schedules. Plus 3-pool-only `perf/step_ms` and `mem/*_peak_gb`.

    `metrics` carries each pool's pre-clip per-parameter grad norms under
    `grad_norms/...` (stashed by the step fns); they're gathered to rank 0 here.

    Times the cross-pool aggregation as `perf/log_ms` (rank 0). The leading
    `synchronize` drains the step's still-in-flight GPU tail so the timer measures only
    the log-path comm, not the step's compute; the `aggregate_*_to_rank0` calls each end
    in a host read (`.item()`), so the elapsed time captures real collective completion
    despite CUDA's async dispatch.
    """
    torch.cuda.synchronize()
    log_start = time.perf_counter()
    combined = aggregate_losses_to_rank0(metrics, ctx, device)
    mem_combined = aggregate_max_memory_to_rank0(ctx, device)
    grad_norms = aggregate_grad_norms_to_rank0(metrics, ctx, device)
    comp_grad_by_loss = aggregate_component_grad_by_loss_to_rank0(metrics, ctx, device)

    # Reduce step_ms with MAX across chunkwise pool (slowest chunk rank is the wall-clock floor).
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
    # Rename the four short loss keys to their metric class name so the wandb
    # panel keys match the single-pool path (`train/loss/<ClassName>`).
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
    # Per-loss component grad norms: faith/stoch/ppgd contribution to the V/U grad,
    # keyed by metric class name (so `train/grad_norms/components/by_loss/<ClassName>`).
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
