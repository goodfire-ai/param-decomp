"""``optimize_two_pool`` — parallel orchestrator for the 2-pool training strategy.

Sibling to ``run_pd.optimize`` that composes the same primitives (ComponentModel,
PersistentPGDState, ReconstructionLoss, AdamW) but wires them up differently:

  - **Pool A** trains V/U + CI fn. Each pool-A rank holds the components for its
    owned sites and runs target+CI forward, per-site streaming layerwise loss,
    home losses (faithfulness, importance-minimality), combined backward seeded
    by pool B's ci grads, in-block all-reduce, AdamW step. See
    :mod:`param_decomp.two_pool.pool_a`.

  - **Pool B** is a stateless PPGD replica that holds full-target V/U replicas
    (received from pool A each step). Each pool-B rank does target forward,
    PPGD warmup + recon loss, sends V/U + CI grads back, receives updated V/U.
    See :mod:`param_decomp.two_pool.pool_b`.

This module orchestrates the loop; the per-pool step logic lives in the pool_a
and pool_b modules; loss-strategy and runtime bundle live in their own files.
"""

import time
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from param_decomp.component_model import ComponentModel
from param_decomp.configs import Cadence
from param_decomp.masks import AllLayersRouter
from param_decomp.metrics.persistent_pgd_state import PersistentPGDState
from param_decomp.run_sink import RunSink
from param_decomp.schedule import ScheduleConfig, get_scheduled_value
from param_decomp.two_pool.install import (
    build_pool_a_module_path_info,
    build_pool_b_module_path_info,
)
from param_decomp.two_pool.layout import BlockDDPLayout
from param_decomp.two_pool.loss_strategy import LayerwiseLossStrategy
from param_decomp.two_pool.pool_a import run_faithfulness_warmup_pool_a, step_pool_a
from param_decomp.two_pool.pool_b import step_pool_b
from param_decomp.two_pool.profiler import PhaseProfiler
from param_decomp.two_pool.reductions import (
    aggregate_losses_to_rank0,
    aggregate_max_memory_to_rank0,
)
from param_decomp.two_pool.runtime import (
    _TwoPoolRuntime,
    build_two_pool_runtime,
    seq_dims_from_batch_iter,
)

# Re-exports for callers (driver_entry, benchmarks) that imported these from
# `param_decomp.two_pool.run` before the decomposition. Keeps the public API
# stable while the internals live in focused modules.
__all__ = [
    "PhaseProfiler",
    "_TwoPoolRuntime",
    "build_two_pool_runtime",
    "optimize_two_pool",
]


def optimize_two_pool(
    target_model: nn.Module,
    pool_config: _TwoPoolRuntime,
    device: torch.device,
    n_steps: int,
    batch_iter: Callable[[int], Any],
    *,
    on_step: Callable[[int, dict[str, float]], None] | None = None,
    enable_tf32: bool = True,
    fused_optimizer: bool = True,
    profiler: PhaseProfiler | None = None,
    components_lr_schedule: ScheduleConfig | None = None,
    ci_fn_lr_schedule: ScheduleConfig | None = None,
    faithfulness_warmup_steps: int = 0,
    faithfulness_warmup_lr: float = 0.0,
    faithfulness_warmup_weight_decay: float = 0.0,
    sink: RunSink | None = None,
    cadence: Cadence | None = None,
) -> tuple[ComponentModel, BlockDDPLayout, PhaseProfiler | None]:
    """Train a ComponentModel under the 2-pool strategy.

    Composes the same primitives that ``run_pd.optimize`` uses but orchestrates
    them under a ``BlockDDPLayout``. The ``LayerwiseLossStrategy`` constructed
    from ``pool_config.use_fused_kl`` is the only place the fused-vs-unfused
    choice lives — every step function consumes it without re-branching.

    Args:
        target_model: A frozen target whose decomposable modules' paths appear in
            ``pool_config.c_per_site``.
        pool_config: Topology + per-pool knobs (see ``TwoPoolConfig`` + builder).
        device: The CUDA device for this rank.
        n_steps: Number of training steps to run.
        batch_iter: Callable taking step idx → batch (anything ``run_batch`` accepts).
        on_step: Optional callback invoked after each step with (step, metrics).

    Returns:
        (component_model, layout, profiler) for caller introspection.

    The function assumes ``dist.init_process_group`` has already been called.
    """
    assert dist.is_initialized(), (
        "init the distributed process group before calling optimize_two_pool"
    )
    rank = dist.get_rank()

    # TF32 matmuls are ~2-3x faster on H200 with sub-ULP precision loss — fine
    # for SPD training where we already use fp32 throughout.
    if enable_tf32:
        torch.set_float32_matmul_precision("high")

    from param_decomp.two_pool.layout import build_block_ddp_world

    world = build_block_ddp_world(
        block_groups=list(pool_config.block_groups),
        pool_b_ranks=list(pool_config.pool_b_ranks),
        batch_global=pool_config.batch_global,
    )
    layout = BlockDDPLayout.from_world(world, rank)

    if layout.my_pool == "a":
        mpi = build_pool_a_module_path_info(layout, pool_config.c_per_site)
    else:
        mpi = build_pool_b_module_path_info(layout, pool_config.c_per_site)

    target_model.requires_grad_(False)
    component_model = ComponentModel(
        target_model=target_model,
        run_batch=pool_config.run_batch,
        decomposition_targets=mpi,
        ci_config=pool_config.ci_config,
        sigmoid_type=pool_config.sigmoid_type,
    ).to(device)

    # Build the layerwise-loss strategy once. Both pools consume it; the rest
    # of the runner doesn't see `use_fused_kl` at all.
    strategy = LayerwiseLossStrategy.from_cfg(
        target_model,
        use_fused_kl=pool_config.use_fused_kl,
        unfused_recon=pool_config.reconstruction_loss,
    )

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
                    {"params": component_params, "lr": pool_config.lr_components},
                    {"params": ci_fn_params, "lr": pool_config.lr_ci_fn},
                ],
                weight_decay=0.0,
                fused=fused_optimizer,
            )

            if faithfulness_warmup_steps > 0:
                run_faithfulness_warmup_pool_a(
                    component_model=component_model,
                    component_params=component_params,
                    n_steps=faithfulness_warmup_steps,
                    lr=faithfulness_warmup_lr,
                    weight_decay=faithfulness_warmup_weight_decay,
                )
        case "b":
            ppgd_cfg = pool_config.ppgd_cfg
            ppgd_state = PersistentPGDState(
                module_to_c=pool_config.c_per_site,
                batch_dims=(layout.world.batch_local_b, *seq_dims_from_batch_iter(batch_iter)),
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

    profiler_ctx = profiler if profiler is not None else nullcontext()
    with profiler_ctx:
        for step in range(n_steps):
            # LR schedule update (pool A only — pool B has no optimizer).
            if layout.my_pool == "a" and optimizer is not None:
                if components_lr_schedule is not None:
                    lr_c = get_scheduled_value(step, n_steps, components_lr_schedule)
                    optimizer.param_groups[0]["lr"] = lr_c
                if ci_fn_lr_schedule is not None:
                    lr_ci = get_scheduled_value(step, n_steps, ci_fn_lr_schedule)
                    optimizer.param_groups[1]["lr"] = lr_ci
            # When profiling, barrier ranks at step boundary so both pools share a
            # common time origin in the trace.
            if profiler is not None:
                dist.barrier()
            batch = batch_iter(step)
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
                        pool_config,
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
                        pool_config,
                        strategy,
                        step=step,
                        n_steps=n_steps,
                        profiler=profiler,
                    )
            torch.cuda.synchronize(device)
            step_ms = (time.perf_counter() - step_start) * 1000.0

            if on_step is not None:
                on_step(step, metrics)

            # --- Train logging (mirrors optimize()'s cadence) ---
            if sink is not None and cadence is not None:
                if step % cadence.train_log_every == 0:
                    combined = aggregate_losses_to_rank0(metrics, layout, device)
                    mem_combined = aggregate_max_memory_to_rank0(layout, device)
                    # Reduce step_ms with MAX across pool A (slowest pool A rank
                    # is the wall-clock floor; pool B should track it).
                    step_ms_t = torch.tensor([step_ms], device=device)
                    if layout.my_pool == "a":
                        dist.all_reduce(
                            step_ms_t, op=dist.ReduceOp.MAX, group=layout.world.pool_a_group
                        )
                    if layout.my_rank == 0 and combined is not None:
                        if mem_combined is not None:
                            combined.update(mem_combined)
                        combined["perf/step_ms"] = step_ms_t.item()
                        combined["loss/total"] = (
                            pool_config.coeff_faith * combined["loss/faith"]
                            + pool_config.coeff_imp * combined["loss/imp"]
                            + pool_config.coeff_stoch * combined["loss/stoch"]
                            + pool_config.coeff_ppgd * combined["loss/ppgd"]
                        )
                        assert layout.my_pool == "a", "rank 0 must be in pool A"
                        assert optimizer is not None
                        combined["schedules/lr/components"] = optimizer.param_groups[0]["lr"]
                        combined["schedules/lr/ci_fn"] = optimizer.param_groups[1]["lr"]
                        # grad_norms intentionally skipped — each pool-A rank only
                        # holds grads for its owned sites; a 2-pool variant of
                        # get_grad_norms_dict can land when needed.
                        sink.console(
                            f"--- Step {step} ---",
                            *(f"train/{name}: {value:.6g}" for name, value in combined.items()),
                        )
                        sink.log({f"train/{k}": v for k, v in combined.items()}, step=step)

                # --- Checkpoint (pool A leader only) ---
                if (
                    cadence.save_every is not None
                    and step > 0
                    and step % cadence.save_every == 0
                    and layout.my_rank == 0
                ):
                    sink.checkpoint(component_model.state_dict(), step=step)

            if profiler is not None:
                profiler.step()

    return component_model, layout, profiler
