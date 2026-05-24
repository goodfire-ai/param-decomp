"""``optimize_three_pool`` — 3-pool sibling of :func:`param_decomp.optimize.optimize`.

Mirrors the call shape of the single-process entrypoint. Internal validation,
per-pool wiring, cross-pool comms, and the layerwise streaming loss strategy
are all hidden behind the function boundary.

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
import time
from contextlib import nullcontext
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from param_decomp.batch_and_loss_fns import ReconstructionLoss, RunBatch
from param_decomp.component_model import ComponentModel
from param_decomp.configs import Cadence, PDConfig, RuntimeConfig
from param_decomp.decomposition_targets import (
    DecompositionTarget,
    resolve_decomposition_targets,
)
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

    Sibling of :func:`param_decomp.optimize.optimize` with the same call shape
    plus the explicit ``three_pool_config``. ``dist.init_process_group`` must
    already be set up; the per-rank device is read from ``runtime_config.device``.
    """
    assert dist.is_initialized(), (
        "init the distributed process group before calling optimize_three_pool"
    )

    _validate_pd_config_for_three_pool(pd_config, three_pool_config)
    # PPGD runs only on PPGD pool; the relevant per-rank batch is batch // n_ppgd.
    validate_pgd_scope(
        pd_config.loss_metrics,
        batch_size=pd_config.batch_size,
        world_size=len(three_pool_config.ppgd_ranks),
    )

    runtime = _build_runtime(
        target_model=target_model,
        pd_config=pd_config,
        runtime_config=runtime_config,
        three_pool_config=three_pool_config,
        run_batch=run_batch,
        reconstruction_loss=reconstruction_loss,
    )

    torch.set_float32_matmul_precision("high")

    device = torch.device(runtime_config.device)
    block_groups = [
        LayerwiseBlockGroup(ranks=tuple(bg.ranks), owned_sites=tuple(bg.owned_sites))
        for bg in three_pool_config.layerwise_block_groups
    ]
    world = build_world(
        ci_ranks=list(three_pool_config.ci_ranks),
        layerwise_block_groups=block_groups,
        ppgd_ranks=list(three_pool_config.ppgd_ranks),
        batch_global=runtime.batch_global,
    )
    layout = ThreePoolLayout.from_world(world, dist.get_rank())
    decomposition_targets = _decomposition_targets_for_pool(layout, runtime.c_per_site)

    target_model.requires_grad_(False)
    component_model = ComponentModel(
        target_model=target_model,
        run_batch=run_batch,
        decomposition_targets=decomposition_targets,
        ci_config=pd_config.ci_config,
        sigmoid_type=pd_config.sigmoid_type,
    ).to(device)

    # Built once; consumed by LW + PPGD step functions. CI pool doesn't need it
    # (target_fwd on CI pool runs through the full model with LM head — we
    # discard the output and only use cached pre-weight acts for the CI fn).
    strategy = LayerwiseLossStrategy.from_cfg(
        target_model,
        use_fused_kl=three_pool_config.use_fused_kl,
        unfused_recon=reconstruction_loss,
    )

    # Peek the first batch so PPGD can size its source tensors; all pools
    # synchronise here so the batch-shape contract is honoured.
    train_iterator = loop_dataloader(train_loader)
    first_batch = next(train_iterator)
    train_iterator = itertools.chain([first_batch], train_iterator)
    _assert_full_global_batch(first_batch, runtime.batch_global)

    components_lr_schedule = pd_config.components_optimizer.lr_schedule
    ci_fn_lr_schedule = pd_config.ci_fn_optimizer.lr_schedule

    optimizer: torch.optim.Optimizer | None = None
    all_params: list[nn.Parameter] = []
    ci_fn_params: list[nn.Parameter] = []
    ppgd_state: PersistentPGDState | None = None

    match layout.my_pool:
        case "ci":
            ci_fn_params = list(component_model.ci_fn.parameters())
            optimizer = torch.optim.AdamW(
                [{"params": ci_fn_params, "lr": ci_fn_lr_schedule.start_val}],
                weight_decay=0.0,
                fused=True,
            )
        case "layerwise":
            component_params: list[nn.Parameter] = []
            for name in layout.my_owned_sites:
                component_params.extend(component_model.components[name].parameters())
            all_params = component_params
            optimizer = torch.optim.AdamW(
                [{"params": component_params, "lr": components_lr_schedule.start_val}],
                weight_decay=0.0,
                fused=True,
            )
            if pd_config.faithfulness_warmup_steps > 0:
                run_faithfulness_warmup_layerwise(
                    component_model=component_model,
                    component_params=component_params,
                    n_steps=pd_config.faithfulness_warmup_steps,
                    lr=pd_config.faithfulness_warmup_lr,
                    weight_decay=pd_config.faithfulness_warmup_weight_decay,
                )
        case "ppgd":
            ppgd_cfg = runtime.ppgd_cfg
            ppgd_state = PersistentPGDState(
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
                reconstruction_loss=strategy.recon_loss,
            )

    n_steps = pd_config.steps
    profiler_ctx = profiler if profiler is not None else nullcontext()
    h_cache_ci: dict[str, Tensor] | None = None  # CI pool's H_T cache (threaded across steps)
    # Async-pipeline state threaded across iterations on LW + PPGD pools.
    # In sync mode both stay None forever; in async mode they hold the previous
    # iter's pending async work (in-block all_reduce on LW; V/U recv broadcast
    # on PPGD), which the next iter's "finalize prev" phase consumes.
    pending_all_reduce_lw: list[tuple[list[Tensor], Tensor, dist.Work]] | None = None
    pending_recv_vu_ppgd: list[tuple[Any, Tensor, dist.Work]] | None = None
    defer_vu_opt = three_pool_config.defer_vu_opt

    with profiler_ctx:
        # Pre-fetch the next batch each step so CI pool can prefetch its
        # target_fwd. batch_T is the current step's batch; batch_T_plus_1 is
        # peeked so CI's dead-time prefetch has something to chew on.
        batch_T = next(train_iterator)
        batch_T_plus_1 = next(train_iterator, None)

        for step in range(n_steps):
            _assert_full_global_batch(batch_T, runtime.batch_global)

            if optimizer is not None:
                # LR schedules. CI pool has one param group (CI fn); LW pool has
                # one (components). PPGD pool has no optimizer.
                # In async mode on LW, the opt step inside step_layerwise's
                # "finalize prev" block uses iter-(T-1)'s grads — so we set the
                # LR to the schedule value at step-1 (clamped to 0 on iter 0).
                # CI pool's opt is not deferred, so it always uses step's LR.
                if layout.my_pool == "ci":
                    optimizer.param_groups[0]["lr"] = get_scheduled_value(
                        step, n_steps, ci_fn_lr_schedule
                    )
                elif layout.my_pool == "layerwise":
                    lr_step = max(step - 1, 0) if defer_vu_opt else step
                    optimizer.param_groups[0]["lr"] = get_scheduled_value(
                        lr_step, n_steps, components_lr_schedule
                    )

            # When profiling, barrier ranks at step boundary so all pools share
            # a common time origin in the trace.
            if profiler is not None:
                dist.barrier()

            torch.cuda.synchronize(device)
            step_start = time.perf_counter()

            match layout.my_pool:
                case "ci":
                    assert optimizer is not None
                    next_batch_for_prefetch = batch_T_plus_1 if step < n_steps - 1 else None
                    metrics, h_cache_ci = step_ci(
                        layout,
                        component_model,
                        optimizer,
                        ci_fn_params,
                        batch_T=batch_T,
                        batch_T_plus_1=next_batch_for_prefetch,
                        h_cache_T=h_cache_ci,
                        cfg=runtime,
                        current_frac_of_training=step / n_steps if n_steps > 0 else 0.0,
                        profiler=profiler,
                    )
                case "layerwise":
                    assert optimizer is not None
                    metrics, pending_all_reduce_lw = step_layerwise(
                        layout,
                        component_model,
                        optimizer,
                        all_params,
                        batch_T,
                        runtime,
                        strategy,
                        defer_vu_opt=defer_vu_opt,
                        prev_pending_all_reduce=pending_all_reduce_lw,
                        profiler=profiler,
                    )
                case "ppgd":
                    assert ppgd_state is not None
                    metrics, pending_recv_vu_ppgd = step_ppgd(
                        layout,
                        component_model,
                        ppgd_state,
                        batch_T,
                        runtime,
                        strategy,
                        step=step,
                        n_steps=n_steps,
                        defer_vu_opt=defer_vu_opt,
                        prev_pending_recv_vu=pending_recv_vu_ppgd,
                        profiler=profiler,
                    )

            torch.cuda.synchronize(device)
            step_ms = (time.perf_counter() - step_start) * 1000.0

            if step % cadence.train_log_every == 0:
                _log_train_metrics(
                    metrics=metrics,
                    layout=layout,
                    device=device,
                    step=step,
                    step_ms=step_ms,
                    runtime=runtime,
                    optimizer=optimizer,
                    sink=sink,
                )

            if cadence.should_save(step):
                _gather_and_save(
                    layout=layout,
                    component_model=component_model,
                    target_model=target_model,
                    run_batch=run_batch,
                    pd_config=pd_config,
                    runtime=runtime,
                    device=device,
                    sink=sink,
                    step=step,
                )

            # Advance the 2-batch peek window for the next step.
            batch_T = batch_T_plus_1 if batch_T_plus_1 is not None else next(train_iterator)
            batch_T_plus_1 = next(train_iterator, None)

            if profiler is not None:
                profiler.step()

        # Drain the final iter's deferred opt (async mode only). Without this,
        # the saved checkpoint would be missing the last iter's update.
        if defer_vu_opt:
            match layout.my_pool:
                case "layerwise":
                    assert optimizer is not None
                    if pending_all_reduce_lw is not None:
                        optimizer.param_groups[0]["lr"] = get_scheduled_value(
                            n_steps - 1, n_steps, components_lr_schedule
                        )
                        finalize_layerwise_async_drain(
                            layout,
                            component_model,
                            optimizer,
                            pending_all_reduce_lw,
                        )
                        pending_all_reduce_lw = None
                case "ppgd":
                    if pending_recv_vu_ppgd is not None:
                        finalize_ppgd_async_drain(
                            layout,
                            component_model,
                            pending_recv_vu_ppgd,  # type: ignore[arg-type]
                        )
                        pending_recv_vu_ppgd = None
                case "ci":
                    pass  # CI pool doesn't defer

        # Final save after the loop (matches single-pool optimize() behavior).
        _gather_and_save(
            layout=layout,
            component_model=component_model,
            target_model=target_model,
            run_batch=run_batch,
            pd_config=pd_config,
            runtime=runtime,
            device=device,
            sink=sink,
            step=n_steps,
        )


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

    # 3-pool allows either layerwise OR global CI fns. The whole point of the
    # CI-pool split is to make global_shared_transformer physically realizable
    # — but smaller targets can still use layerwise (validated by pydantic
    # discriminated union; we just don't restrict here).
    # No additional CI-mode assertion needed.

    assert pd_config.sampling == "continuous", (
        "3-pool hardcodes `sampling='continuous'` in CI pool's CI computation; "
        f"got pd_config.sampling={pd_config.sampling!r}."
    )
    assert pd_config.n_mask_samples == 1, (
        "3-pool draws exactly one stochastic mask per site per step in LW; "
        f"got pd_config.n_mask_samples={pd_config.n_mask_samples}."
    )

    # insert_identity_operations_ isn't called from the 3-pool path, so any
    # identity-layer decomposition target would be silently ignored.
    assert pd_config.identity_decomposition_targets is None, (
        "3-pool path does not call `insert_identity_operations_`; "
        "`identity_decomposition_targets` would be silently ignored."
    )

    # Convention: rank 0 must be the Layerwise pool's block 0 leader. The
    # reductions module assumes this so it can collect ci + ppgd pool's
    # averaged losses on rank 0.
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
    every step so each pool can slice to its own DP shard. See
    ``optimize_three_pool`` docstring for setup notes.
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


def _gather_and_save(
    *,
    layout: ThreePoolLayout,
    component_model: ComponentModel,
    target_model: nn.Module,
    run_batch: RunBatch,
    pd_config: PDConfig,
    runtime: _ThreePoolRuntime,
    device: torch.device,
    sink: RunSink,
    step: int,
) -> None:
    """Gather full state to rank 0 and dispatch to ``sink.checkpoint``.

    All ranks must enter this in sync — the gather uses ordered P2P sends/recvs.
    Only rank 0 actually writes via the sink.
    """
    state_dict = gather_full_state_dict_to_rank0(
        layout=layout,
        component_model=component_model,
        target_model=target_model,
        run_batch=run_batch,
        ci_config=pd_config.ci_config,
        sigmoid_type=pd_config.sigmoid_type,
        c_per_site=runtime.c_per_site,
        device=device,
    )
    if layout.my_rank == 0:
        assert state_dict is not None
        sink.checkpoint(state_dict, step=step)


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

    # Reduce step_ms with MAX across LW pool (slowest LW rank is the wall-clock
    # floor; CI + PPGD should track it).
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
    # CI fn LR is set on a different rank's optimizer (CI pool rank 0); skip
    # logging it here for MVP — could be plumbed via the loss aggregation if
    # needed later.
    sink.console(
        f"--- Step {step} ---",
        *(f"train/{name}: {value:.6g}" for name, value in combined.items()),
    )
    sink.log({f"train/{k}": v for k, v in combined.items()}, step=step)
