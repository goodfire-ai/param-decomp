"""``optimize_two_pool`` — 2-pool sibling of :func:`param_decomp.optimize.optimize`.

Mirrors the call shape of the single-process entrypoint: caller hands in
``target_model``, dataloader, configs, sink. Internal validation, per-pool
wiring, cross-pool comms, and the layerwise streaming loss strategy are all
hidden behind the function boundary.

  - **Pool A** trains V/U + CI fn. Each pool-A rank holds the components for its
    owned sites and runs target+CI forward, per-site streaming layerwise loss,
    home losses (faithfulness, importance-minimality), combined backward seeded
    by pool B's ci grads, in-block all-reduce, AdamW step. See
    :mod:`param_decomp.two_pool.pool_a`.

  - **Pool B** is a stateless PPGD replica that holds full-target V/U replicas
    (received from pool A each step). Each pool-B rank does target forward,
    PPGD warmup + recon loss, sends V/U + CI grads back, receives updated V/U.
    See :mod:`param_decomp.two_pool.pool_b`.
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
from param_decomp.ci_fns import LayerwiseCiConfig
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
from param_decomp.torch_helpers import loop_dataloader
from param_decomp.two_pool.config import TwoPoolConfig
from param_decomp.two_pool.layout import BlockDDPLayout, BlockGroup, build_block_ddp_world
from param_decomp.two_pool.loss_strategy import LayerwiseLossStrategy
from param_decomp.two_pool.pool_a import run_faithfulness_warmup_pool_a, step_pool_a
from param_decomp.two_pool.pool_b import step_pool_b
from param_decomp.two_pool.profiler import PhaseProfiler
from param_decomp.two_pool.reductions import (
    aggregate_losses_to_rank0,
    aggregate_max_memory_to_rank0,
)
from param_decomp.two_pool.runtime import _TwoPoolRuntime

# Loss-metric type discriminators required for the 2-pool training path.
# Each MUST appear in ``pd_config.loss_metrics`` with a non-None ``coeff``.
REQUIRED_LOSS_METRIC_TYPES: frozenset[str] = frozenset(
    {
        "FaithfulnessLoss",
        "ImportanceMinimalityLoss",
        "StochasticReconLayerwiseLoss",
        "PersistentPGDReconLoss",
    }
)

# Loss metrics 2-pool does not implement; configuring them would be silent.
# Listed explicitly so misconfiguration is loud.
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


def optimize_two_pool(
    target_model: nn.Module,
    train_loader: DataLoader[Any],
    *,
    run_batch: RunBatch,
    reconstruction_loss: ReconstructionLoss,
    pd_config: PDConfig,
    runtime_config: RuntimeConfig,
    two_pool_config: TwoPoolConfig,
    cadence: Cadence,
    sink: RunSink,
    profiler: PhaseProfiler | None = None,
) -> None:
    """Train a ComponentModel under the 2-pool strategy.

    Sibling of :func:`param_decomp.optimize.optimize` with the same call shape.
    ``dist.init_process_group`` must already be set up; the per-rank device
    is read from ``runtime_config.device``.
    """
    assert dist.is_initialized(), (
        "init the distributed process group before calling optimize_two_pool"
    )

    _validate_pd_config_for_two_pool(pd_config, two_pool_config, cadence)
    # PPGD runs only on pool B; the relevant per-rank batch is batch // n_pool_b.
    validate_pgd_scope(
        pd_config.loss_metrics,
        batch_size=pd_config.batch_size,
        world_size=len(two_pool_config.pool_b_ranks),
    )

    runtime = _build_runtime(
        target_model=target_model,
        pd_config=pd_config,
        runtime_config=runtime_config,
        two_pool_config=two_pool_config,
        run_batch=run_batch,
        reconstruction_loss=reconstruction_loss,
    )

    # TF32 matmuls are ~2-3x faster on H200 with sub-ULP precision loss — fine
    # for SPD training where we already use fp32 throughout.
    torch.set_float32_matmul_precision("high")

    device = torch.device(runtime_config.device)
    world = build_block_ddp_world(
        block_groups=list(runtime.block_groups),
        pool_b_ranks=list(runtime.pool_b_ranks),
        batch_global=runtime.batch_global,
    )
    layout = BlockDDPLayout.from_world(world, dist.get_rank())
    decomposition_targets = _decomposition_targets_for_pool(layout, runtime.c_per_site)

    target_model.requires_grad_(False)
    component_model = ComponentModel(
        target_model=target_model,
        run_batch=run_batch,
        decomposition_targets=decomposition_targets,
        ci_config=pd_config.ci_config,
        sigmoid_type=pd_config.sigmoid_type,
    ).to(device)

    # Build the layerwise-loss strategy once. Both pools consume it; the rest
    # of the runner doesn't see `use_fused_kl` at all.
    strategy = LayerwiseLossStrategy.from_cfg(
        target_model,
        use_fused_kl=two_pool_config.use_fused_kl,
        unfused_recon=reconstruction_loss,
    )

    # Peek one batch so pool B can size its PPGD source tensors; both pools
    # synchronise here so the layout's batch-shape contract is honoured.
    train_iterator = loop_dataloader(train_loader)
    first_batch = next(train_iterator)
    train_iterator = itertools.chain([first_batch], train_iterator)

    components_lr_schedule = pd_config.components_optimizer.lr_schedule
    ci_fn_lr_schedule = pd_config.ci_fn_optimizer.lr_schedule

    optimizer: torch.optim.Optimizer | None = None
    all_params: list[nn.Parameter] = []
    ppgd_state: PersistentPGDState | None = None

    match layout.my_pool:
        case "a":
            component_params: list[nn.Parameter] = []
            for name in component_model.target_module_paths:
                component_params.extend(component_model.components[name].parameters())
            ci_fn_params = list(component_model.ci_fn.parameters())
            all_params = component_params + ci_fn_params
            optimizer = torch.optim.AdamW(
                [
                    {"params": component_params, "lr": components_lr_schedule.start_val},
                    {"params": ci_fn_params, "lr": ci_fn_lr_schedule.start_val},
                ],
                weight_decay=0.0,
                fused=True,
            )

            if pd_config.faithfulness_warmup_steps > 0:
                run_faithfulness_warmup_pool_a(
                    component_model=component_model,
                    component_params=component_params,
                    n_steps=pd_config.faithfulness_warmup_steps,
                    lr=pd_config.faithfulness_warmup_lr,
                    weight_decay=pd_config.faithfulness_warmup_weight_decay,
                )
        case "b":
            ppgd_cfg = runtime.ppgd_cfg
            ppgd_state = PersistentPGDState(
                module_to_c=runtime.c_per_site,
                batch_dims=(layout.world.batch_local_b, *_seq_dims_from_batch(first_batch)),
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
    with profiler_ctx:
        for step in range(n_steps):
            if layout.my_pool == "a" and optimizer is not None:
                optimizer.param_groups[0]["lr"] = get_scheduled_value(
                    step, n_steps, components_lr_schedule
                )
                optimizer.param_groups[1]["lr"] = get_scheduled_value(
                    step, n_steps, ci_fn_lr_schedule
                )

            # When profiling, barrier ranks at step boundary so both pools share a
            # common time origin in the trace.
            if profiler is not None:
                dist.barrier()

            batch = _extract_batch_tensor(next(train_iterator), device)

            torch.cuda.synchronize(device)
            step_start = time.perf_counter()
            match layout.my_pool:
                case "a":
                    assert optimizer is not None
                    metrics = step_pool_a(
                        layout,
                        component_model,
                        optimizer,
                        all_params,
                        batch,
                        runtime,
                        strategy,
                        current_frac_of_training=step / n_steps if n_steps > 0 else 0.0,
                        profiler=profiler,
                    )
                case "b":
                    assert ppgd_state is not None
                    metrics = step_pool_b(
                        layout,
                        component_model,
                        ppgd_state,
                        batch,
                        runtime,
                        strategy,
                        step=step,
                        n_steps=n_steps,
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

            if (
                cadence.save_every is not None
                and step > 0
                and step % cadence.save_every == 0
                and layout.my_rank == 0
            ):
                sink.checkpoint(component_model.state_dict(), step=step)

            if profiler is not None:
                profiler.step()


def _validate_pd_config_for_two_pool(
    pd_config: PDConfig, two_pool_config: TwoPoolConfig, cadence: Cadence
) -> None:
    """Fail loudly on any PDConfig the 2-pool path can't honour."""
    by_type: dict[str, LossMetricConfig] = {m.type: m for m in pd_config.loss_metrics}

    missing = sorted(REQUIRED_LOSS_METRIC_TYPES - set(by_type))
    assert not missing, (
        f"2-pool requires these loss metrics: {sorted(REQUIRED_LOSS_METRIC_TYPES)}.\n"
        f"Missing: {missing}. Got: {sorted(by_type)}."
    )
    illegal = sorted(FORBIDDEN_LOSS_METRIC_TYPES & set(by_type))
    assert not illegal, (
        f"2-pool does not implement these loss metrics (they would be silently ignored): "
        f"{illegal}. Remove them or extend the 2-pool path."
    )

    for name in REQUIRED_LOSS_METRIC_TYPES:
        assert by_type[name].coeff is not None, (
            f"pd_config.loss_metrics[{name!r}].coeff is required for 2-pool training"
        )

    n_per_block = len(two_pool_config.block_groups[0].ranks)
    n_pool_b = len(two_pool_config.pool_b_ranks)
    bs = pd_config.batch_size
    assert bs % n_per_block == 0, (
        f"pd_config.batch_size ({bs}) must be divisible by N_per_block ({n_per_block}) "
        f"= len(block_groups[0].ranks)"
    )
    assert bs % n_pool_b == 0, (
        f"pd_config.batch_size ({bs}) must be divisible by N_pool_b ({n_pool_b}) "
        f"= len(pool_b_ranks)"
    )

    assert pd_config.use_delta_component, (
        "2-pool requires pd_config.use_delta_component=True (hardcoded in pool A's "
        "layerwise stoch recon + pool B's PPGD)."
    )

    # Global CI fns can't span all sites under 2-pool because pool A shards
    # sites across ranks — each rank only sees its owned sites' activations.
    # Use `mode: layerwise` (with `fn_type: transformer` for the transformer variant).
    assert isinstance(pd_config.ci_config, LayerwiseCiConfig), (
        "2-pool requires `pd.ci_config.mode == 'layerwise'`. Global CI fns can't "
        "span all sites under 2-pool (sites are sharded across pool-A ranks). "
        f"Got mode={pd_config.ci_config.mode!r}."
    )

    # Pool A's layerwise streaming loop hardcodes continuous sampling and one
    # mask per site; let the user know if their YAML disagrees.
    assert pd_config.sampling == "continuous", (
        "2-pool hardcodes `sampling='continuous'` in pool A's CI computation; "
        f"got pd_config.sampling={pd_config.sampling!r}."
    )
    assert pd_config.n_mask_samples == 1, (
        "2-pool draws exactly one stochastic mask per site per step; "
        f"got pd_config.n_mask_samples={pd_config.n_mask_samples}."
    )

    # insert_identity_operations_ isn't called from the 2-pool path, so any
    # identity-layer decomposition target would be silently ignored.
    assert pd_config.identity_decomposition_targets is None, (
        "2-pool path does not call `insert_identity_operations_`; "
        "`identity_decomposition_targets` would be silently ignored."
    )

    # No distributed-aware checkpoint gather yet — rank 0 only holds its
    # block's V/U + CI fn, so saving would produce a structurally partial file.
    assert cadence.save_every is None, (
        "2-pool does not yet implement distributed checkpointing. "
        "rank 0 only holds its own block's V/U + CI fn, so `cadence.save_every` "
        "would produce a partial checkpoint that can't be reloaded. "
        "Set `cadence.save_every: null` for now."
    )

    ppgd_cfg = by_type["PersistentPGDReconLoss"]
    assert isinstance(ppgd_cfg, PersistentPGDReconLossConfig)
    assert ppgd_cfg.start_frac == 0.0, (
        "2-pool path does not implement PersistentPGDReconLoss.start_frac > 0; "
        "PPGD always runs from step 0."
    )


def _build_runtime(
    target_model: nn.Module,
    pd_config: PDConfig,
    runtime_config: RuntimeConfig,
    two_pool_config: TwoPoolConfig,
    run_batch: RunBatch,
    reconstruction_loss: ReconstructionLoss,
) -> _TwoPoolRuntime:
    """Assemble the step-context bundle from configs + target."""
    targets = resolve_decomposition_targets(target_model, pd_config.decomposition_targets)
    c_per_site = {t.module_path: t.C for t in targets}

    for bg in two_pool_config.block_groups:
        for site in bg.owned_sites:
            assert site in c_per_site, (
                f"site '{site}' in two-pool topology but not in pd_config.decomposition_targets "
                f"after pattern expansion. Available: {sorted(c_per_site)[:5]}..."
            )

    by_type: dict[str, LossMetricConfig] = {m.type: m for m in pd_config.loss_metrics}
    ppgd_cfg = by_type["PersistentPGDReconLoss"]
    imp_min_cfg = by_type["ImportanceMinimalityLoss"]
    assert isinstance(ppgd_cfg, PersistentPGDReconLossConfig)
    assert isinstance(imp_min_cfg, ImportanceMinimalityLossConfig)

    def _coeff(name: str) -> float:
        c = by_type[name].coeff
        assert c is not None  # validated above
        return float(c)

    block_groups = tuple(
        BlockGroup(ranks=tuple(bg.ranks), owned_sites=tuple(bg.owned_sites))
        for bg in two_pool_config.block_groups
    )

    return _TwoPoolRuntime(
        block_groups=block_groups,
        pool_b_ranks=tuple(two_pool_config.pool_b_ranks),
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
        use_fused_kl=two_pool_config.use_fused_kl,
    )


def _decomposition_targets_for_pool(
    layout: BlockDDPLayout, c_per_site: dict[str, int]
) -> list[DecompositionTarget]:
    """Pool A: only this rank's owned sites. Pool B: every site (replicated V/U)."""
    sites = layout.my_owned_sites if layout.my_pool == "a" else layout.world.all_sites
    return [DecompositionTarget(module_path=s, C=c_per_site[s]) for s in sites]


def _extract_batch_tensor(batch: Any, device: str | torch.device) -> Tensor:
    """Pull a single input tensor out of a dataloader yield and move to device.

    Supports the three common shapes: a bare Tensor, a dict with ``input_ids``,
    or a tuple/list whose first element is a Tensor.
    """
    if isinstance(batch, Tensor):
        return batch.to(device)
    if isinstance(batch, dict) and "input_ids" in batch:
        return batch["input_ids"].to(device)
    if isinstance(batch, list | tuple) and len(batch) > 0 and isinstance(batch[0], Tensor):
        return batch[0].to(device)
    raise TypeError(f"Unsupported batch type from DataLoader: {type(batch).__name__}")


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
    layout: BlockDDPLayout,
    device: torch.device,
    step: int,
    step_ms: float,
    runtime: _TwoPoolRuntime,
    optimizer: torch.optim.Optimizer | None,
    sink: RunSink,
) -> None:
    """Aggregate per-pool metrics to rank 0 and dispatch to ``sink``."""
    combined = aggregate_losses_to_rank0(metrics, layout, device)
    mem_combined = aggregate_max_memory_to_rank0(layout, device)

    # Reduce step_ms with MAX across pool A (slowest pool A rank is the wall-clock
    # floor; pool B should track it).
    step_ms_t = torch.tensor([step_ms], device=device)
    if layout.my_pool == "a":
        dist.all_reduce(step_ms_t, op=dist.ReduceOp.MAX, group=layout.world.pool_a_group)

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
    assert layout.my_pool == "a", "rank 0 must be in pool A"
    assert optimizer is not None
    combined["schedules/lr/components"] = optimizer.param_groups[0]["lr"]
    combined["schedules/lr/ci_fn"] = optimizer.param_groups[1]["lr"]
    sink.console(
        f"--- Step {step} ---",
        *(f"train/{name}: {value:.6g}" for name, value in combined.items()),
    )
    sink.log({f"train/{k}": v for k, v in combined.items()}, step=step)
