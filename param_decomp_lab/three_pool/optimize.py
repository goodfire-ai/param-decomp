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
    AdamW. See :mod:`param_decomp_lab.three_pool.step_ci`.

  * **Layerwise pool** trains V/U (block-DDP within group; sharded across
    sites). Recv CI → faithfulness + layerwise stoch recon → send g_CI back
    → recv g_VU from PPGD → combine → in-block all-reduce → AdamW → async
    ship updated V/U → PPGD. See :mod:`param_decomp_lab.three_pool.step_layerwise`.

  * **PPGD pool** is a stateless full V/U replica. Recv CI → PPGD warmup +
    final recon → backward seeds V/U + CI grads (no outer source step;
    warmup inner loop owns sources) → sum-reduce V/U within PPGD pool →
    send g_VU to LW + g_CI to CI → recv updated V/U. See
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
from param_decomp_lab.experiments.lm.pretrain.models.gpt2_simple import GPT2Simple
from param_decomp_lab.experiments.lm.three_pool_pd import ThreePoolConstrainedPDConfig
from param_decomp_lab.experiments.lm.vendored.component_model import LMComponentModel
from param_decomp_lab.three_pool.checkpoint import (
    ci_fn_state_keys,
    owned_model_state_keys,
)
from param_decomp_lab.three_pool.config import ThreePoolConfig
from param_decomp_lab.three_pool.consolidate import (
    CONSOLIDATE_META_FILENAME,
    step_scratch_dir,
)
from param_decomp_lab.three_pool.context import (
    CIContext,
    LWContext,
    PoolContext,
    PPGDContext,
    build_pool_context,
)
from param_decomp_lab.three_pool.layout import (
    LayerwiseBlockGroup,
    build_world,
    flush_nccl_event_timings,
)
from param_decomp_lab.three_pool.loss_strategy import LayerwiseLossStrategy
from param_decomp_lab.three_pool.reductions import (
    aggregate_losses_to_rank0,
    aggregate_max_memory_to_rank0,
)
from param_decomp_lab.three_pool.runtime import _ThreePoolRuntime
from param_decomp_lab.three_pool.step_ci import step_ci
from param_decomp_lab.three_pool.step_layerwise import (
    run_faithfulness_warmup_layerwise,
    step_layerwise,
)
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


def _lw_compile_enabled() -> bool:
    """torch.compile the LW pool's model forward — default on; disable for repro/debug.

    Global (the launcher exports env to every rank, so this is identical everywhere), which
    is required because it also widens the collective PG timeout uniformly across ranks.
    """
    return os.environ.get("PD_DISABLE_LW_COMPILE", "").strip() not in ("1", "true", "yes")


def _resolve_pg_timeout(*, compiling: bool) -> datetime.timedelta:
    override_s = os.environ.get("PD_3POOL_PG_TIMEOUT_S", "").strip()
    if override_s:
        return datetime.timedelta(seconds=float(override_s))
    return _COMPILE_PG_TIMEOUT if compiling else _DEFAULT_PG_TIMEOUT


class ThreePoolTrainer:
    """Stateful 3-pool trainer.

    Construction wires up the runtime bundle, world layout, ComponentModel,
    layerwise loss strategy, and the per-pool optimizer (CI / LW have one;
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
    runtime_config: RuntimeConfig
    three_pool_config: ThreePoolConfig
    reconstruction_loss: ReconstructionLoss
    component_model: LMComponentModel
    ctx: PoolContext
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
        pd_config: ThreePoolConstrainedPDConfig,
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
        # Verify FA can dispatch on the CI fn's largest SDPA shape. If it
        # can't (head_dim > 128, missing kernel, etc.), error here rather
        # than silently fall back to a 5-10x slower math kernel during
        # training. Skipping the global-toggle approach to stay compatible
        # with torch.compile's fake-tensor trace.
        ci_attn = _ci_attn_shape_or_none(pd_config)
        if ci_attn is not None:
            d_model, n_heads = ci_attn
            verify_flash_attention_available(head_dim=d_model // n_heads)
        # PPGD runs only on PPGD pool; the relevant per-rank batch is batch // n_ppgd.
        validate_pgd_scope(
            [pd_config.losses.ppgd],
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

        # PD_SYNC_DEBUG: when set, ask PyTorch to flag every implicit CPU↔GPU
        # sync (.item(), .tolist(), bool(tensor), .cpu(), etc.) — these stall
        # the GPU and are easy to introduce by accident. ``warn`` logs every
        # one; ``error`` crashes on the first one (useful when you want a
        # stack trace pinpointing the culprit). Off by default.
        _sync_debug = os.environ.get("PD_SYNC_DEBUG", "").strip()
        if _sync_debug in ("1", "warn", "true"):
            torch.cuda.set_sync_debug_mode("warn")
        elif _sync_debug == "error":
            torch.cuda.set_sync_debug_mode("error")

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
            pg_timeout=_resolve_pg_timeout(compiling=_lw_compile_enabled()),
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
        # block must start with identical params, but anything between
        # ``set_seed(pd.seed)`` in _fresh_main and here (loader build, distributed
        # init, etc.) can advance the RNG by rank-dependent amounts. Without this,
        # partners initialize different V/U and the in-block grad all-reduce can't
        # bring them back into sync.
        seed_all_ranks(pd_config.seed)
        trace("ThreePoolTrainer.__init__: LMComponentModel build: enter")
        assert isinstance(target_model, GPT2Simple), (
            f"3-pool LMComponentModel requires a GPT2Simple target; got {type(target_model).__name__}"
        )
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
            case LWContext() | PPGDContext():
                self.component_model.drop_ci_fn()
                trace(f"ThreePoolTrainer.__init__: dropped ci_fn ({self.ctx.kind} pool)")
            case CIContext():
                self.component_model.drop_components()
                trace("ThreePoolTrainer.__init__: dropped V/U components (ci pool)")
        # Activation checkpointing is LW-only. The LW pool carries the full per-rank batch
        # (bl_lw = batch_size; no within-block DP at GPUs_per_block=1) so it needs ckpt to fit;
        # PPGD/CI don't (PPGD fits plain at its small bl_pp; CI's target fwd is no_grad). PPGD's
        # autograd.grad recompute is also non-deterministic under ckpt (nested mask_infos through
        # checkpoint) — tracked as a follow-up; LW's .backward path is checkpoint-clean.
        if not isinstance(self.ctx, LWContext):
            self.component_model.model._use_activation_checkpointing = False
        trace("ThreePoolTrainer.__init__: ComponentModel.to(device): enter")
        self.component_model = self.component_model.to(self._device)
        trace("ThreePoolTrainer.__init__: ComponentModel.to(device): done")
        dump_memory_stats("after ComponentModel.to(device)")
        # CI pool: optionally torch.compile the CI fn. Eats compile time on
        # step 0 / step 1 (first fwd + first bwd) but should cut ``ci/8_fused_bwd``
        # substantially — that backward through the 2.64B-param CI fn dominates
        # the critical path (70% of CI step at batch=48).
        is_ci = isinstance(self.ctx, CIContext)
        if is_ci and os.environ.get("PD_COMPILE_CI_FN", "").strip() in (
            "1",
            "true",
            "yes",
        ):
            assert self.component_model.ci_fn is not None
            trace("ThreePoolTrainer.__init__: torch.compile(ci_fn)")
            self.component_model.ci_fn = torch.compile(self.component_model.ci_fn)  # pyright: ignore[reportAttributeAccessIssue]
        if is_ci and os.environ.get("PD_CI_FN_BWD_PROFILE", "").strip() in (
            "1",
            "true",
            "yes",
        ):
            from param_decomp.ci_fns import GlobalSharedTransformerCiFn

            assert self.component_model.ci_fn is not None
            for m in self.component_model.ci_fn.modules():
                if isinstance(m, GlobalSharedTransformerCiFn):
                    m.enable_bwd_profile()
                    trace("ThreePoolTrainer.__init__: enabled CI fn bwd-stage profile")
                    break
        # LW pool: torch.compile the (target + masked) model forward — a validated 2.61× on
        # the LW step, which is the throughput pole (≈the whole step). The vendored mask-arg
        # forward traces cleanly (0 graph breaks) and attention uses F.sdpa directly (no
        # sdpa_kernel context, the thing that made compile a wash on the CI fn). LW-only:
        # PPGD/CI have slack so compiling them wouldn't move the wall, and PPGD's autograd.grad
        # path is unvalidated under compile. Default on; PD_DISABLE_LW_COMPILE=1 to disable.
        # The one-time first-step compilation is absorbed by the widened PG timeout (see
        # _resolve_pg_timeout(compiling=...)). LW's forward is only called in step_layerwise
        # (target=None + masked dict, both validated); eval barriers LW through, so no recompile.
        if isinstance(self.ctx, LWContext) and _lw_compile_enabled():
            trace("ThreePoolTrainer.__init__: torch.compile(LW model forward)")
            self.component_model.model.compile()
        # Diverge stochastic RNG per rank for mask sampling.
        seed_per_rank(pd_config.seed)

        trace("ThreePoolTrainer.__init__: building LayerwiseLossStrategy")
        self.strategy = LayerwiseLossStrategy.from_cfg(
            self.component_model,
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
            case LWContext():
                for name in self.ctx.role.owned_sites:
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
        optimizer. ``CI`` pool returns ``ci_fn.*`` pairs; ``LW`` pool returns
        ``components.<site>.*`` pairs for this rank's owned sites; ``PPGD`` has
        no optimizer and returns ``[]``."""
        match self.ctx:
            case CIContext():
                assert self.component_model.ci_fn is not None
                return [(f"ci_fn.{n}", p) for n, p in self.component_model.ci_fn.named_parameters()]
            case LWContext():
                out: list[tuple[str, nn.Parameter]] = []
                for site in self.ctx.role.owned_sites:
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
          LW block leaders contribute their owned sites' ``_components.<site>.*``,
          the CI pool leader contributes ``ci_fn.*``, everyone else contributes
          nothing (their V/U / CI fn are replicas of a leader's).
        * ``optimizer_by_name``: this rank's optimizer state, name-keyed.
        * ``ppgd``: PPGD adversarial sources (PPGD ranks only).

        Rank 0 also writes ``meta.pth`` (configs + layout fingerprint + the
        ``c_per_site`` / ``all_sites`` needed to rebuild the full model). A
        single barrier ensures every partial is on the shared FS before the
        train loop continues — there is NO rank-0 read and NO model assembly
        here. The async consolidation job
        (:func:`param_decomp_lab.three_pool.consolidate.consolidate_step`)
        reads these partials off the critical path.
        """
        partial = self._build_my_partial()

        p2p_group = self.ctx.world.cross_pool_p2p_group
        step_dir = step_scratch_dir(scratch_dir, self.step)
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

    def _build_my_partial(self) -> dict[str, Any]:
        my_named_params = self._named_params_for_my_optimizer()
        my_optimizer_by_name: dict[str, dict[str, Any]] = (
            optimizer_state_by_name(self.optimizer, my_named_params)
            if self.optimizer is not None
            else {}
        )
        partial: dict[str, Any] = {
            "pool": self.ctx.kind,
            "model_params": self._owned_model_params(),
            "optimizer_by_name": my_optimizer_by_name,
        }
        if self.ppgd_state is not None:
            partial["ppgd"] = self.ppgd_state.state_dict()
        elif self._pending_ppgd_resume_state is not None:
            partial["ppgd"] = self._pending_ppgd_resume_state
        return partial

    def _owned_model_params(self) -> dict[str, Tensor]:
        """The slice of this rank's model state_dict it's responsible for saving.

        Only leaders contribute: LW block leaders own their sites' V/U, the CI
        pool leader owns the CI fn. Every other rank holds a replica, so it
        contributes nothing — the union across leaders covers the full model.
        """
        sd = self.component_model.state_dict()
        keys = set(sd.keys())
        match self.ctx:
            case LWContext() if self.ctx.role.is_block_leader:
                owned = owned_model_state_keys(keys, owned_sites=self.ctx.role.owned_sites)
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
        cfg_overrides: dict[str, Any] | None = None,
    ) -> Self:
        """Reconstruct a 3-pool trainer from a canonical snapshot.

        Each rank must arrive with the canonical state populated (typically by
        reading ``training_<step>.pth`` on every rank from a shared filesystem).
        Each rank extracts its own slice of the canonical state by rank id and
        loads it into its locally-owned sub-state.
        """
        pd_dict = snapshot.pd_config
        if cfg_overrides is not None:
            pd_dict = {**pd_dict, **cfg_overrides}
        pd_config = ThreePoolConstrainedPDConfig.model_validate(pd_dict)
        # The saved runtime_config is a `ThreePoolRuntimeConfig` dump (base scalars +
        # `topology`). The trainer takes the base `RuntimeConfig` scalars here and the
        # topology via `three_pool_config` (the source of truth, identical to the dumped
        # `topology`), so drop the duplicate `topology` key before validating as base.
        runtime_dict = {k: v for k, v in snapshot.runtime_config.items() if k != "topology"}
        runtime_config = RuntimeConfig.model_validate(runtime_dict)
        three_pool_config = ThreePoolConfig.model_validate(snapshot.three_pool_config)

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
        trainer._load_canonical_state(snapshot)
        return trainer

    def _load_canonical_state(self, state: ThreePoolTrainingState) -> None:
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
                case LWContext():
                    by_name = state.components_optimizer
                case CIContext():
                    by_name = state.ci_fn_optimizer
                case PPGDContext():
                    by_name = {}
            load_optimizer_state_by_name(self.optimizer, named_params, by_name)
        if isinstance(self.ctx, PPGDContext):
            self._pending_ppgd_resume_state = state.ppgd_state_by_rank.get(self.ctx.role.rank)

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
        ``MetricContext`` and runs every ``eval_loop.metric``; LW pool
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
        # CI / LW pools never touch metric state. The eval iterator runs on
        # every rank so all pools advance the loader in lock-step (CI + PPGD
        # each consume the full eval batch and slice internally, mirroring
        # the train data-handling contract; LW reads but discards).
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
            )
            if self._pending_ppgd_resume_state is not None:
                self.ppgd_state.load_state_dict(self._pending_ppgd_resume_state)
                self._pending_ppgd_resume_state = None
            trace("Trainer.run: PPGDState ctor: done")

        if (
            self.step == 0
            and isinstance(ctx, LWContext)
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
                should_log = step % cadence.train_log_every == 0

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
                    case LWContext():
                        assert self.optimizer is not None, (
                            f"LW rank {ctx.role.rank} missing optimizer"
                        )
                        assert ctx.role.owned_sites, (
                            f"LW rank {ctx.role.rank} has no owned_sites — empty block"
                        )
                        self.optimizer.param_groups[0]["lr"] = get_scheduled_value(
                            step, n_steps, components_lr_schedule
                        )
                        metrics = step_layerwise(
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
                if step % cadence.train_log_every == 0:
                    dump_memory_stats(f"step {step} done")

                if step % cadence.train_log_every == 0:
                    _log_train_metrics(
                        metrics=metrics,
                        ctx=ctx,
                        device=device,
                        step=step,
                        step_ms=step_ms,
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
    runtime_config: RuntimeConfig,
    three_pool_config: ThreePoolConfig,
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
    ``owned_sites``): the snapshot stores the fingerprint computed on rank 0,
    but ``from_snapshot`` compares it on EVERY rank, so a rank-local fingerprint
    would mismatch on every non-rank-0 rank. The full block→ranks→sites mapping
    below is identical on all ranks and fully captures the topology that
    ``_load_canonical_state`` relies on.
    """
    world = ctx.world
    return {
        "world_size": world.world_size,
        "ci_ranks": list(world.ci_ranks),
        "ppgd_ranks": list(world.ppgd_ranks),
        "layerwise_blocks": [
            {"ranks": list(bg.ranks), "owned_sites": list(bg.owned_sites)}
            for bg in world.layerwise_block_groups
        ],
    }


def _rank_invariant_fingerprint_core(fp: dict[str, Any]) -> dict[str, Any]:
    """Reduce a saved-or-current layout fingerprint to its rank-invariant core.

    Validating the topology on resume must not depend on which rank wrote the
    snapshot. Earlier checkpoints (e.g. the 144-GPU p-a5b667e9 run) stored
    rank-0's local view (``my_rank`` / ``my_pool`` / ``owned_sites``) alongside
    the topology fields; current snapshots store the full block mapping. Both
    carry ``world_size`` / ``ci_ranks`` / ``ppgd_ranks`` plus the block count
    (as ``n_layerwise_blocks`` in the old format or ``len(layerwise_blocks)`` in
    the new), which together pin down the pool partition. The per-block
    ranks→sites mapping is fully re-derived from ``three_pool_config`` (also
    stored in the snapshot and used to rebuild the trainer), so comparing the
    core here is sufficient.
    """
    # TODO(remove once p-df5b9fbd is past step 10000): the `n_layerwise_blocks`
    # fallback exists only to read the pre-#536 rank-local fingerprint baked into
    # p-a5b667e9's training_5000.pth. Once the resumed run has written a
    # new-format checkpoint, no old-format snapshot is in use — drop this branch
    # and the old-format clause in the docstring, keeping only the new-format read.
    n_blocks = fp.get("n_layerwise_blocks")
    if n_blocks is None:
        n_blocks = len(fp["layerwise_blocks"])
    return {
        "world_size": fp["world_size"],
        "ci_ranks": list(fp["ci_ranks"]),
        "ppgd_ranks": list(fp["ppgd_ranks"]),
        "n_layerwise_blocks": n_blocks,
    }


def _required[T](value: T | None) -> T:
    """Narrow a value the `ThreePoolLosses` validator already guarantees non-None."""
    assert value is not None
    return value


def _build_runtime(
    target_model: nn.Module,
    pd_config: ThreePoolConstrainedPDConfig,
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

    losses = pd_config.losses
    ppgd_cfg = losses.ppgd
    imp_min_cfg = losses.imp

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
        coeff_faith=float(_required(losses.faith.coeff)),
        coeff_imp=float(_required(losses.imp.coeff)),
        coeff_stoch=float(_required(losses.stoch.coeff)),
        coeff_ppgd=float(_required(losses.ppgd.coeff)),
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
    Layerwise pool: only this rank's owned sites."""
    match ctx:
        case CIContext() | PPGDContext():
            sites = ctx.world.all_sites
        case LWContext():
            sites = ctx.role.owned_sites
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
    runtime: _ThreePoolRuntime,
    optimizer: torch.optim.Optimizer | None,
    sink: ThreePoolRunSink,
) -> None:
    """Aggregate per-pool metrics to rank 0 and dispatch to ``sink``."""
    combined = aggregate_losses_to_rank0(metrics, ctx, device)
    mem_combined = aggregate_max_memory_to_rank0(ctx, device)

    # Reduce step_ms with MAX across LW pool (slowest LW rank is the wall-clock floor).
    step_ms_t = torch.tensor([step_ms], device=device)
    if isinstance(ctx, LWContext):
        dist.all_reduce(step_ms_t, op=dist.ReduceOp.MAX, group=ctx.world.layerwise_pool_group)

    if ctx.role.rank != 0 or combined is None:
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
    assert isinstance(ctx, LWContext), "rank 0 must be in LW pool (per validator)"
    assert optimizer is not None
    combined["schedules/lr/components"] = optimizer.param_groups[0]["lr"]
    sink.console(
        f"--- Step {step} ---",
        *(f"train/{name}: {value:.6g}" for name, value in combined.items()),
    )
    sink.log({f"train/{k}": v for k, v in combined.items()}, step=step)
