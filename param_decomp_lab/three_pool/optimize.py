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

Each step function then slices to its own per-pool batch shard via the
layout's ``my_batch_slice_*`` helpers. The 3-way batch routing in
``layout.py`` (see "Batch-split routing" in its docstring) assumes this
sliced-from-global pattern.
"""

import itertools
import os
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Self

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
from param_decomp.metrics.base import LossMetricConfig
from param_decomp.metrics.importance_minimality import ImportanceMinimalityLossConfig
from param_decomp.metrics.persistent_pgd_recon import (
    PersistentPGDReconLossConfig,
    validate_pgd_scope,
)
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
from param_decomp_lab.three_pool.checkpoint import gather_full_state_dict_to_rank0
from param_decomp_lab.three_pool.config import ThreePoolConfig
from param_decomp_lab.three_pool.layout import (
    LayerwiseBlockGroup,
    ThreePoolLayout,
    build_world,
    flush_nccl_event_timings,
)
from param_decomp_lab.three_pool.loss_strategy import LayerwiseLossStrategy
from param_decomp_lab.three_pool.pool_state import CIState, LWState, PoolState, PPGDState
from param_decomp_lab.three_pool.portals import Portals
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

# Loss-metric type discriminators required for the 3-pool training path.
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
    portals: Portals
    strategy: LayerwiseLossStrategy
    pool_state: PoolState
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
        # Verify FA can dispatch on the CI fn's largest SDPA shape. If it
        # can't (head_dim > 128, missing kernel, etc.), error here rather
        # than silently fall back to a 5-10x slower math kernel during
        # training. Skipping the global-toggle approach to stay compatible
        # with torch.compile's fake-tensor trace.
        ci_attn = _ci_attn_shape_or_none(pd_config)
        if ci_attn is not None:
            d_model, n_heads = ci_attn
            verify_flash_attention_available(head_dim=d_model // n_heads)
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
            device=self._device,
        )
        trace("ThreePoolTrainer.__init__: build_world: done")
        self.layout = ThreePoolLayout.from_world(world, dist.get_rank())
        self.portals = Portals.from_world(world)
        decomposition_targets = _decomposition_targets_for_pool(
            self.layout, self.runtime.c_per_site
        )
        trace(
            f"ThreePoolTrainer.__init__: my_pool={self.layout.my_pool} "
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
        # single-pool is preserved.
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
        if self.layout.my_pool == "ci" and os.environ.get("PD_CI_FN_BWD_PROFILE", "").strip() in (
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
        # Diverge stochastic RNG per rank for mask sampling.
        seed_per_rank(pd_config.seed)

        trace("ThreePoolTrainer.__init__: building LayerwiseLossStrategy")
        self.strategy = LayerwiseLossStrategy.from_cfg(
            target_model,
            use_fused_kl=three_pool_config.use_fused_kl,
            unfused_recon=reconstruction_loss,
        )
        trace("ThreePoolTrainer.__init__: LayerwiseLossStrategy: done")

        trace(f"ThreePoolTrainer.__init__: pool state build: enter (pool={self.layout.my_pool})")
        match self.layout.my_pool:
            case "ci":
                assert self.component_model.ci_fn is not None, "CI pool must keep its CI fn"
                ci_fn_params = list(self.component_model.ci_fn.parameters())
                n_params = sum(p.numel() for p in ci_fn_params)
                trace(f"ThreePoolTrainer.__init__: CI fn params={n_params / 1e9:.3f}B")
                ci_optimizer = torch.optim.AdamW(
                    [
                        {
                            "params": ci_fn_params,
                            "lr": pd_config.ci_fn_optimizer.lr_schedule.start_val,
                        }
                    ],
                    weight_decay=0.0,
                    fused=True,
                )
                self.pool_state = CIState(optimizer=ci_optimizer, ci_fn_params=ci_fn_params)
            case "layerwise":
                component_params: list[nn.Parameter] = []
                for name in self.layout.my_owned_sites:
                    component_params.extend(self.component_model.components[name].parameters())
                lw_optimizer = torch.optim.AdamW(
                    [
                        {
                            "params": component_params,
                            "lr": pd_config.components_optimizer.lr_schedule.start_val,
                        }
                    ],
                    weight_decay=0.0,
                    fused=True,
                )
                self.pool_state = LWState(optimizer=lw_optimizer, component_params=component_params)
            case "ppgd":
                # ppgd_state constructed lazily from first batch in run().
                self.pool_state = PPGDState()
        trace("ThreePoolTrainer.__init__: pool state build: done")
        dump_memory_stats("after optimizer build")
        trace("ThreePoolTrainer.__init__: exit")

    # ============================ Atomic cfg + state ============================

    def _named_params_for_my_optimizer(self) -> list[tuple[str, nn.Parameter]]:
        """The ``(name, param)`` pairs in the order they were added to this rank's
        optimizer. ``CI`` pool returns ``ci_fn.*`` pairs; ``LW`` pool returns
        ``components.<site>.*`` pairs for this rank's owned sites; ``PPGD`` has
        no optimizer and returns ``[]``."""
        match self.layout.my_pool:
            case "ci":
                assert self.component_model.ci_fn is not None
                return [(f"ci_fn.{n}", p) for n, p in self.component_model.ci_fn.named_parameters()]
            case "layerwise":
                out: list[tuple[str, nn.Parameter]] = []
                for site in self.layout.my_owned_sites:
                    for n, p in self.component_model.components[site].named_parameters():
                        out.append((f"components.{site}.{n}", p))
                return out
            case "ppgd":
                return []

    def snapshot(self, scratch_dir: Path) -> ThreePoolTrainingState | None:
        """Canonical point-in-time view of the 3-pool trainer.

        On rank 0 (the only rank a sink consumes from), assembles a
        topology-free :class:`ThreePoolTrainingState`:

        * ``component_model``: the full gathered model state (every site's V/U
          plus the CI fn) from :func:`gather_full_state_dict_to_rank0`.
        * ``components_optimizer`` / ``ci_fn_optimizer``: per-parameter
          optimizer state keyed by param name, merged across all ranks.
          ``components_optimizer`` comes from the layerwise pool (each LW
          rank contributes its owned sites); ``ci_fn_optimizer`` comes from
          the CI pool (all CI ranks have the same state via in-pool
          all-reduce, so any suffices).
        * ``ppgd_state_by_rank``: PPGD adversarial sources, keyed by PPGD
          rank id. Genuinely rank-coupled for ``PerBatchPerPositionScope``
          sources (sized by the rank-local batch slice).

        Returns ``None`` on non-rank-0 — the lab sink is silent on those
        ranks anyway, and the trainer never needs their canonical state.

        Args:
            scratch_dir: Shared-filesystem directory all ranks can write
                to / read from. Used as the rendezvous for the per-rank
                contributions (optimizer state, PPGD sources). Each
                snapshot at step ``S`` writes / reads under
                ``scratch_dir / f"step_{S}"`` and cleans it up on rank 0.
        """
        trace("snapshot: enter gather_full_state_dict_to_rank0")
        gathered_model = gather_full_state_dict_to_rank0(
            layout=self.layout,
            component_model=self.component_model,
            target_model=self._target_model,
            run_batch=self._run_batch,
            ci_config=self.pd_config.ci_config,
            sigmoid_type=self.pd_config.sigmoid_type,
            c_per_site=self.runtime.c_per_site,
            device=self._device,
        )
        trace("snapshot: gather_full_state_dict_to_rank0 done")

        my_named_params = self._named_params_for_my_optimizer()
        my_contribution: dict[str, Any] = {
            "pool": self.layout.my_pool,
            "optimizer_by_name": {},
        }
        match self.pool_state:
            case CIState() | LWState():
                my_contribution["optimizer_by_name"] = optimizer_state_by_name(
                    self.pool_state.optimizer, my_named_params
                )
            case PPGDState(ppgd_state=ppgd) if ppgd is not None:
                my_contribution["ppgd"] = ppgd.state_dict()
            case PPGDState(pending_resume_state=pending) if pending is not None:
                my_contribution["ppgd"] = pending
            case PPGDState():
                pass

        # File-based gather rather than dist.gather_object: at XL the aggregate
        # pickle payload (LW optimizer state + PPGD sources) is ~8 GB and
        # gather_object scales poorly. Each rank writes its contribution to a
        # shared-FS path, a barrier ensures all writes are visible, then rank 0
        # reads them all.
        p2p_group = self.layout.world.cross_pool_p2p_group
        world_size = self.layout.world.world_size
        step_dir = scratch_dir / f"step_{self.step}"
        if self.layout.my_rank == 0:
            step_dir.mkdir(parents=True, exist_ok=True)

        trace("snapshot: enter barrier (pre-write)")
        dist.barrier(group=p2p_group)
        trace("snapshot: barrier (pre-write) done")

        partial_path = step_dir / f"rank_{self.layout.my_rank}.pth"
        trace(f"snapshot: writing partial {partial_path.name}")
        torch.save(my_contribution, partial_path)
        trace("snapshot: partial write done")

        trace("snapshot: enter barrier (post-write)")
        dist.barrier(group=p2p_group)
        trace("snapshot: barrier (post-write) done")

        if self.layout.my_rank != 0:
            return None

        gathered: list[dict[str, Any]] = []
        for r in range(world_size):
            gathered.append(torch.load(step_dir / f"rank_{r}.pth", weights_only=False))
        trace("snapshot: rank-0 read all partials")
        shutil.rmtree(step_dir, ignore_errors=True)
        components_by_name: dict[str, dict[str, Any]] = {}
        ci_fn_by_name: dict[str, dict[str, Any]] = {}
        ppgd_by_rank: dict[int, dict[str, Any]] = {}
        for r, c in enumerate(gathered):
            pool: str = c["pool"]
            match pool:
                case "layerwise":
                    components_by_name.update(c["optimizer_by_name"])
                case "ci":
                    ci_fn_by_name.update(c["optimizer_by_name"])
                case "ppgd":
                    if "ppgd" in c:
                        ppgd_by_rank[r] = c["ppgd"]
                case _:
                    raise AssertionError(f"unknown pool {pool!r} in rank-{r} contribution")
        assert gathered_model is not None  # rank 0 always has the gathered model
        return ThreePoolTrainingState(
            step=self.step,
            pd_config=self.pd_config.model_dump(),
            runtime_config=self.runtime_config.model_dump(),
            three_pool_config=self.three_pool_config.model_dump(),
            layout_fingerprint=_layout_fingerprint(self.layout),
            component_model=gathered_model,
            components_optimizer=components_by_name,
            ci_fn_optimizer=ci_fn_by_name,
            ppgd_state_by_rank=ppgd_by_rank,
        )

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
        pd_config = PDConfig.model_validate(pd_dict)
        runtime_config = RuntimeConfig.model_validate(snapshot.runtime_config)
        three_pool_config = ThreePoolConfig.model_validate(snapshot.three_pool_config)

        trainer = cls(
            target_model=target_model,
            run_batch=run_batch,
            reconstruction_loss=reconstruction_loss,
            pd_config=pd_config,
            runtime_config=runtime_config,
            three_pool_config=three_pool_config,
        )
        saved_fp = snapshot.layout_fingerprint
        current_fp = _layout_fingerprint(trainer.layout)
        assert saved_fp == current_fp, (
            f"3-pool layout fingerprint mismatch on resume:\n"
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

        named_params = self._named_params_for_my_optimizer()
        match self.pool_state:
            case CIState():
                load_optimizer_state_by_name(
                    self.pool_state.optimizer, named_params, state.ci_fn_optimizer
                )
            case LWState():
                load_optimizer_state_by_name(
                    self.pool_state.optimizer, named_params, state.components_optimizer
                )
            case PPGDState():
                self.pool_state.pending_resume_state = state.ppgd_state_by_rank.get(
                    self.layout.my_rank
                )

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
        layout = self.layout
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
        if eval_loop is not None and layout.my_pool == "ppgd":
            for m in eval_loop.metrics:
                m.bind(model=self.component_model, device=str(device))

        # Peek first batch (after any skip) for PPGD source-shape sizing.
        trace("Trainer.run: first_batch peek: enter")
        first_batch = next(train_iterator)
        trace("Trainer.run: first_batch peek: done")
        train_iterator = itertools.chain([first_batch], train_iterator)
        _assert_full_global_batch(first_batch, runtime.batch_global)

        if isinstance(self.pool_state, PPGDState) and self.pool_state.ppgd_state is None:
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
            self.pool_state.ppgd_state = PersistentPGDState(
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
            if self.pool_state.pending_resume_state is not None:
                self.pool_state.ppgd_state.load_state_dict(self.pool_state.pending_resume_state)
                self.pool_state.pending_resume_state = None
            trace("Trainer.run: PPGDState ctor: done")

        if (
            self.step == 0
            and isinstance(self.pool_state, LWState)
            and pd_config.faithfulness_warmup_steps > 0
        ):
            trace(
                f"Trainer.run: faithfulness warmup: enter ({pd_config.faithfulness_warmup_steps} steps)"
            )
            run_faithfulness_warmup_layerwise(
                component_model=self.component_model,
                component_params=self.pool_state.component_params,
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
                trace(f"Trainer.run: step {step}: start (pool={layout.my_pool})")

                # CI pool: one param group (CI fn); LW pool: one (components).
                # PPGD pool has no optimizer to schedule.
                match self.pool_state:
                    case CIState():
                        self.pool_state.optimizer.param_groups[0]["lr"] = get_scheduled_value(
                            step, n_steps, ci_fn_lr_schedule
                        )
                    case LWState():
                        self.pool_state.optimizer.param_groups[0]["lr"] = get_scheduled_value(
                            step, n_steps, components_lr_schedule
                        )
                    case PPGDState():
                        pass

                step_start = time.perf_counter()
                should_log = step % cadence.train_log_every == 0

                # batch_T should already be on this rank's device (placed by _to_device).
                if isinstance(batch_T, Tensor):
                    assert batch_T.device == device, (
                        f"3-pool batch device drift at step {step}: {batch_T.device} vs {device}"
                    )
                match self.pool_state:
                    case CIState(optimizer=opt, ci_fn_params=ci_fn_params):
                        assert len(ci_fn_params) > 0, (
                            f"CI rank {layout.my_rank} has no ci_fn params to optimize"
                        )
                        next_batch_for_prefetch = batch_T_plus_1 if step < n_steps - 1 else None
                        metrics, h_cache_ci = step_ci(
                            layout,
                            self.portals,
                            self.component_model,
                            opt,
                            ci_fn_params,
                            batch_T=batch_T,
                            batch_T_plus_1=next_batch_for_prefetch,
                            h_cache_T=h_cache_ci,
                            cfg=runtime,
                            current_frac_of_training=step / n_steps if n_steps > 0 else 0.0,
                            should_log=should_log,
                        )
                    case LWState(optimizer=opt, component_params=component_params):
                        assert layout.my_owned_sites, (
                            f"LW rank {layout.my_rank} has no owned_sites — empty block"
                        )
                        metrics = step_layerwise(
                            layout,
                            self.portals,
                            self.component_model,
                            opt,
                            component_params,
                            batch_T,
                            runtime,
                            self.strategy,
                            should_log=should_log,
                        )
                    case PPGDState(ppgd_state=ppgd):
                        assert ppgd is not None, (
                            f"PPGD rank {layout.my_rank} has no ppgd_state — lazy init failed"
                        )
                        metrics = step_ppgd(
                            layout,
                            self.portals,
                            self.component_model,
                            ppgd,
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
                    lw_optimizer = (
                        self.pool_state.optimizer if isinstance(self.pool_state, LWState) else None
                    )
                    _log_train_metrics(
                        metrics=metrics,
                        layout=layout,
                        device=device,
                        step=step,
                        step_ms=step_ms,
                        runtime=runtime,
                        optimizer=lw_optimizer,
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
                        layout=layout,
                        portals=self.portals,
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
                    snap = self.snapshot(scratch_dir)
                    if snap is not None:
                        sink.checkpoint(snap, final=False)

                batch_T = (
                    batch_T_plus_1
                    if batch_T_plus_1 is not None
                    else _to_device(next(train_iterator))
                )
                batch_T_plus_1 = _to_device(next(train_iterator, None))

                if profiler is not None:
                    profiler.step()

            self.step = n_steps
            snap = self.snapshot(scratch_dir)
            if snap is not None:
                sink.checkpoint(snap, final=True)


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
    sink: ThreePoolRunSink,
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
