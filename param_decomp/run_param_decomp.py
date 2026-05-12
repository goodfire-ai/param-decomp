"""Run PD on a model."""

import gc
import json
import os
import time
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.parallel
import wandb
from PIL import Image
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from param_decomp.base_config import BaseConfig
from param_decomp.configs import (
    Config,
    FaithfulnessLossConfig,
    LossMetricConfigType,
    MetricConfigType,
    PersistentPGDReconLossConfig,
    PersistentPGDReconSubsetLossConfig,
    PGDMultiBatchConfig,
    PGDMultiBatchReconLossConfig,
    PGDMultiBatchReconSubsetLossConfig,
)
from param_decomp.data import loop_dataloader
from param_decomp.eval import evaluate, evaluate_multibatch_pgd
from param_decomp.identity_insertion import insert_identity_operations_
from param_decomp.log import logger
from param_decomp.losses import compute_losses
from param_decomp.metrics import faithfulness_loss
from param_decomp.models.batch_and_loss_fns import ReconstructionLoss, RunBatch
from param_decomp.models.component_model import (
    ComponentModel,
    OutputWithCache,
    move_batch_to_device,
)
from param_decomp.persistent_pgd import PersistentPGDState
from param_decomp.settings import PARAM_DECOMP_OUT_DIR
from param_decomp.utils.component_utils import calc_ci_l_zero
from param_decomp.utils.distributed_utils import (
    avg_metrics_across_ranks,
    get_distributed_state,
    is_main_process,
    seed_per_rank,
    sync_across_processes,
)
from param_decomp.utils.general_utils import (
    bf16_autocast,
    dict_safe_update_,
    get_scheduled_value,
    save_pre_run_info,
)
from param_decomp.utils.logging_utils import get_grad_norms_dict, local_log
from param_decomp.utils.module_utils import expand_module_patterns
from param_decomp.utils.run_utils import generate_run_id, save_file
from param_decomp.utils.wandb_utils import init_wandb, try_wandb


def _log_param_breakdown(component_model: ComponentModel) -> None:
    """Log a parameter-count breakdown at run start. Used by the scaling investigation."""
    target_p = sum(p.numel() for p in component_model.target_model.parameters())
    component_p = sum(
        p.numel()
        for n in component_model.target_module_paths
        for p in component_model.components[n].parameters()
    )
    ci_p = sum(p.numel() for p in component_model.ci_fn.parameters())
    logger.info(f"target_params:    {target_p:>14,}  ({target_p * 4 / 1e9:.2f} GB fp32)")
    logger.info(f"component_params: {component_p:>14,}  ({component_p * 4 / 1e9:.2f} GB fp32)")
    logger.info(f"ci_fn_params:     {ci_p:>14,}  ({ci_p * 4 / 1e9:.2f} GB fp32)")
    logger.info(f"trainable total:  {component_p + ci_p:>14,}")


def run_faithfulness_warmup(
    component_model: ComponentModel,
    component_params: list[torch.nn.Parameter],
    config: Config,
) -> None:
    """Run faithfulness warmup phase to improve initialization."""
    logger.info("Starting faithfulness warmup phase...")

    assert component_params, "component_params is empty"

    faithfulness_warmup_optimizer = optim.AdamW(
        component_params,
        lr=config.faithfulness_warmup_lr,
        weight_decay=config.faithfulness_warmup_weight_decay,
    )

    for faithfulness_warmup_step in range(config.faithfulness_warmup_steps):
        faithfulness_warmup_optimizer.zero_grad()
        weight_deltas = component_model.calc_weight_deltas()
        loss = faithfulness_loss(weight_deltas)
        loss.backward()
        faithfulness_warmup_optimizer.step()

        if (
            faithfulness_warmup_step % 100 == 0
            or faithfulness_warmup_step == config.faithfulness_warmup_steps - 1
        ):
            logger.info(
                f"Faithfulness warmup step {faithfulness_warmup_step + 1} / {config.faithfulness_warmup_steps}; Faithfulness loss: {loss.item():.9f}"
            )
    del faithfulness_warmup_optimizer
    # TODO: we should reverse the order of these two calls
    torch.cuda.empty_cache()
    gc.collect()


def get_unique_metric_configs(
    loss_configs: list[LossMetricConfigType], eval_configs: list[MetricConfigType]
) -> list[MetricConfigType]:
    """If a metric appears in both loss and eval configs, only include the eval version."""
    eval_config_names = [type(cfg).__name__ for cfg in eval_configs]
    eval_metric_configs = eval_configs[:]
    for cfg in loss_configs:
        if type(cfg).__name__ not in eval_config_names:
            eval_metric_configs.append(cfg)
        else:
            logger.warning(
                f"{type(cfg).__name__} is in both loss and eval configs, only including eval config"
            )
    return eval_metric_configs


def optimize(
    target_model: nn.Module,
    config: Config,
    device: str,
    train_loader: DataLoader[Any],
    eval_loader: DataLoader[Any],
    run_batch: RunBatch,
    reconstruction_loss: ReconstructionLoss,
    out_dir: Path | None,
    tied_weights: list[tuple[str, str]] | None = None,
) -> None:
    """Run the optimization loop for LM decomposition."""

    train_iterator = loop_dataloader(train_loader)
    eval_iterator = loop_dataloader(eval_loader)

    def create_pgd_data_iter() -> Iterator[Any]:
        assert hasattr(train_loader, "generator") and train_loader.generator is not None
        train_loader.generator.manual_seed(config.seed)
        return iter(train_loader)

    if is_main_process():
        logger.info(f"Train+eval logs saved to directory: {out_dir}")

    if config.identity_module_info is not None:
        insert_identity_operations_(
            target_model,
            identity_module_info=config.identity_module_info,
        )

    target_model.requires_grad_(False)

    if config.target_gradient_checkpointing:
        assert hasattr(target_model, "config") and hasattr(
            target_model.config, "gradient_checkpointing"
        ), (
            "target_gradient_checkpointing=True requires target model to have "
            "config.gradient_checkpointing (currently only LlamaSimpleMLP supports this)"
        )
        target_cfg: Any = target_model.config  # type: ignore[attr-defined]
        target_model.config = target_cfg.model_copy(update={"gradient_checkpointing": True})  # type: ignore[attr-defined]

    module_path_info = expand_module_patterns(target_model, config.all_module_info)

    model = ComponentModel(
        target_model=target_model,
        run_batch=run_batch,
        module_path_info=module_path_info,
        ci_config=config.ci_config,
        sigmoid_type=config.sigmoid_type,
    )

    model.to(device)

    # Diverge global RNG per rank so stochastic masks/sources differ across DP workers.
    seed_per_rank(config.seed)

    # Wrap model with DDP or FSDP if distributed.
    dist_state = get_distributed_state()
    wrapped_model: nn.Module = model

    component_model: ComponentModel
    if dist_state is not None:
        match config.parallel_strategy:
            case "ddp":
                if dist_state.backend == "nccl":
                    device_id = dist_state.local_rank
                    wrapped_model = torch.nn.parallel.DistributedDataParallel(
                        model,
                        device_ids=[device_id],
                        output_device=device_id,
                    )
                else:
                    # For CPU, don't pass device_ids or output_device
                    wrapped_model = torch.nn.parallel.DistributedDataParallel(model)
                # Access the underlying module for component operations
                component_model = cast(ComponentModel, wrapped_model.module)
            case "fsdp":
                from param_decomp.utils.fsdp import fsdp_wrap

                assert dist_state.world_size > 1, "FSDP requires world_size > 1"
                assert dist_state.backend == "nccl", "FSDP requires NCCL backend"
                assert config.optimizer_strategy == "adamw", (
                    "FSDP shards optimizer state itself; set optimizer_strategy='adamw' "
                    "(not 'zero_adamw') when parallel_strategy='fsdp'."
                )
                wrapped_model = fsdp_wrap(
                    model,
                    device_id=dist_state.local_rank,
                    autocast_bf16=config.autocast_bf16,
                )
                # FSDP2 mutates the module in place rather than wrapping it — `wrapped_model`
                # and `model` are the same Python object after `fsdp_wrap`. We keep the
                # `component_model` / `wrapped_model` naming for symmetry with the DDP path
                # and to make sites where we'd want a "pre-FSDP" handle obvious.
                component_model = model
    else:
        component_model = model
    assert isinstance(component_model, ComponentModel), "component_model is not a ComponentModel"

    uses_faithfulness_loss = any(
        isinstance(cfg, FaithfulnessLossConfig) for cfg in config.loss_metric_configs
    )
    uses_faithfulness_metric = any(
        isinstance(cfg, FaithfulnessLossConfig)
        for cfg in [*config.loss_metric_configs, *config.eval_metric_configs]
    )
    if config.parallel_strategy == "fsdp":
        assert config.faithfulness_warmup_steps == 0, (
            "faithfulness_warmup_steps materializes full weight deltas and is not compatible "
            "with FSDP-scale site-local delta math."
        )
        assert not uses_faithfulness_metric, (
            "FaithfulnessLossConfig materializes full weight deltas and is not compatible "
            "with FSDP-scale site-local delta math."
        )

    if tied_weights is not None:
        # Tie component weights. Assume that the first element is a transpose of the second element
        # NOTE: Tying weights will make your training nondeterministic
        for src_name, tgt_name in tied_weights:
            tgt = component_model.components[tgt_name]
            src = component_model.components[src_name]
            assert tgt is not None and src is not None, (
                f"Cannot tie weights between {src_name} and {tgt_name} - one or both are None"
            )
            tgt.U.data = src.V.data.T
            tgt.V.data = src.U.data.T

    component_params: list[torch.nn.Parameter] = []
    for name in component_model.target_module_paths:
        # DecomposedLinear/Embedding sites also own the frozen wrapped target submodule;
        # filter on requires_grad so we get only V/U for the optimizer.
        component_params.extend(
            p for p in component_model.components[name].parameters() if p.requires_grad
        )

    ci_fn_params = list(component_model.ci_fn.parameters())

    assert len(component_params) > 0, "No parameters found in components to optimize"

    optimized_params = component_params + ci_fn_params
    optimizer: optim.Optimizer
    match config.optimizer_strategy:
        case "adamw":
            optimizer = optim.AdamW(
                optimized_params, lr=config.lr_schedule.start_val, weight_decay=0
            )
        case "zero_adamw":
            from torch.distributed.optim import ZeroRedundancyOptimizer

            assert dist_state is not None and dist_state.world_size > 1, (
                "optimizer_strategy='zero_adamw' requires a distributed run with world_size > 1"
            )
            optimizer = ZeroRedundancyOptimizer(
                optimized_params,
                optimizer_class=optim.AdamW,
                lr=config.lr_schedule.start_val,
                weight_decay=0,
            )

    if config.profile_memory and is_main_process():
        _log_param_breakdown(component_model)
        torch.cuda.memory._record_memory_history(max_entries=200_000)

    if config.faithfulness_warmup_steps > 0:
        run_faithfulness_warmup(component_model, component_params, config)

    persistent_pgd_configs: list[
        PersistentPGDReconLossConfig | PersistentPGDReconSubsetLossConfig
    ] = [
        cfg
        for cfg in config.loss_metric_configs
        if isinstance(cfg, PersistentPGDReconLossConfig | PersistentPGDReconSubsetLossConfig)
    ]

    eval_metric_configs = get_unique_metric_configs(
        loss_configs=config.loss_metric_configs, eval_configs=config.eval_metric_configs
    )

    multibatch_pgd_eval_configs: list[
        PGDMultiBatchReconLossConfig | PGDMultiBatchReconSubsetLossConfig
    ] = [cfg for cfg in eval_metric_configs if isinstance(cfg, PGDMultiBatchConfig)]

    eval_metric_configs = [
        cfg for cfg in eval_metric_configs if cfg not in multibatch_pgd_eval_configs
    ]

    # Route the sample forward through `wrapped_model` rather than the underlying `model` —
    # under FSDP, the bare `model` has only its local parameter shards, so forwarding through
    # it directly returns a 1-D FlatParameter where the embedding expected 2-D.
    sample_out = wrapped_model(next(train_iterator))
    batch_dims = sample_out.shape[:-1]
    ppgd_states: dict[
        PersistentPGDReconLossConfig | PersistentPGDReconSubsetLossConfig, PersistentPGDState
    ] = {
        ppgd_cfg: PersistentPGDState(
            module_to_c=model.module_to_c,
            batch_dims=batch_dims,
            device=device,
            use_delta_component=config.use_delta_component,
            cfg=ppgd_cfg,
            reconstruction_loss=reconstruction_loss,
        )
        for ppgd_cfg in persistent_pgd_configs
    }

    step_times: list[float] = []
    step_start = 0.0
    for step in tqdm(range(config.steps + 1), ncols=0, disable=not is_main_process()):
        if config.profile_memory:
            torch.cuda.synchronize()
            step_start = time.perf_counter()

        optimizer.zero_grad()

        step_lr = get_scheduled_value(
            step=step, total_steps=config.steps, config=config.lr_schedule
        )
        for group in optimizer.param_groups:
            group["lr"] = step_lr

        frac = step / config.steps
        active_ppgd_configs = [c for c in persistent_pgd_configs if frac >= c.start_frac]

        for ppgd_cfg in active_ppgd_configs:
            ppgd_states[ppgd_cfg].update_lr(step, config.steps)

        faithfulness_weight_deltas = (
            component_model.calc_weight_deltas() if uses_faithfulness_loss else None
        )

        batch_log_data: defaultdict[str, float] = defaultdict(float)

        batch = move_batch_to_device(next(train_iterator), device)
        # FSDP MixedPrecision (configured in fsdp_wrap) already casts activations to bf16; doing
        # autocast on top can leave dangling bf16 buffers across the gather/reshard boundary,
        # which manifested as CUDA illegal memory access during backward. Disable autocast
        # under FSDP and let MixedPrecision handle bf16 casts.
        autocast_active = config.autocast_bf16 and config.parallel_strategy != "fsdp"
        with bf16_autocast(enabled=autocast_active):
            # NOTE: we need to call the wrapped_model at least once each step in order to setup
            # the DDP gradient syncing for all parameters in the component model. Gradients will
            # sync regardless of whether the parameters are used in this call to wrapped_model.
            target_model_output: OutputWithCache = wrapped_model(batch, cache_type="input")

            # `GlobalSharedTransformerCiFn` is itself an FSDP unit (see fsdp.py auto-wrap
            # policy), so `ci_fn(layer_acts)` self-gathers its own params + lets nested
            # TransformerBlocks self-gather. No external summon_full_params needed here.
            ci = component_model.calc_causal_importances(
                pre_weight_acts=target_model_output.cache,
                detach_inputs=False,
                sampling=config.sampling,
            )

            # `wrapped_model` is the forward target for loss/PPGD code — under FSDP it's the
            # FSDP-wrapped ComponentModel (which gathers params on each forward), under DDP it
            # is the DDP-wrapped ComponentModel. `component_model` stays available for
            # attribute access that doesn't trigger a forward.
            forward_model = cast(ComponentModel, wrapped_model)

            for ppgd_cfg in active_ppgd_configs:
                ppgd_states[ppgd_cfg].warmup(
                    model=forward_model,
                    batch=batch,
                    target_out=target_model_output.output,
                    ci=ci.lower_leaky,
                )

            losses = compute_losses(
                loss_metric_configs=config.loss_metric_configs,
                model=forward_model,
                batch=batch,
                ci=ci,
                target_out=target_model_output.output,
                faithfulness_weight_deltas=faithfulness_weight_deltas,
                current_frac_of_training=step / config.steps,
                sampling=config.sampling,
                use_delta_component=config.use_delta_component,
                n_mask_samples=config.n_mask_samples,
                ppgd_states=ppgd_states,
                reconstruction_loss=reconstruction_loss,
            )

        total_loss = torch.tensor(0.0, device=device)
        for loss_cfg, loss_val in losses.items():
            assert loss_cfg.coeff is not None
            total_loss = total_loss + loss_cfg.coeff * loss_val
            batch_log_data[f"train/loss/{loss_cfg.classname}"] = loss_val.item()

        batch_log_data["train/loss/total"] = total_loss.item()

        ppgd_grads = {
            cfg: ppgd_states[cfg].get_grads(losses[cfg], retain_graph=True)
            for cfg in active_ppgd_configs
        }

        total_loss.backward()

        for ppgd_cfg in active_ppgd_configs:
            ppgd_states[ppgd_cfg].step(ppgd_grads[ppgd_cfg])

        for layer_name, layer_ci in ci.lower_leaky.items():
            l0_val = calc_ci_l_zero(layer_ci, config.ci_alive_threshold)
            batch_log_data[f"train/l0/{layer_name}"] = l0_val

        # --- Train Logging --- #
        if step % config.train_log_freq == 0:
            avg_metrics = avg_metrics_across_ranks(batch_log_data, device=device)
            batch_log_data = cast(defaultdict[str, float], avg_metrics)

            grad_norms = get_grad_norms_dict(component_model, device)
            dict_safe_update_(
                batch_log_data, {f"train/grad_norms/{k}": v for k, v in grad_norms.items()}
            )

            batch_log_data["train/schedules/lr"] = step_lr

            if is_main_process():
                assert out_dir is not None
                tqdm.write(f"--- Step {step} ---")
                tqdm.write(f"LR: {step_lr:.6f}")
                for name, value in batch_log_data.items():
                    tqdm.write(f"{name}: {value:.15f}")
                local_log(batch_log_data, step, out_dir)
                if config.wandb_project:
                    try_wandb(wandb.log, batch_log_data, step=step)

        # --- Evaluation --- #
        if step % config.eval_freq == 0:
            with torch.no_grad(), bf16_autocast(enabled=config.autocast_bf16):
                slow_step: bool = (
                    config.slow_eval_on_first_step
                    if step == 0
                    else step % config.slow_eval_freq == 0
                )

                multibatch_pgd_metrics = evaluate_multibatch_pgd(
                    multibatch_pgd_eval_configs=multibatch_pgd_eval_configs,
                    model=component_model,
                    create_data_iter=create_pgd_data_iter,
                    config=config,
                    device=device,
                    reconstruction_loss=reconstruction_loss,
                )

                metrics = evaluate(
                    eval_metric_configs=eval_metric_configs,
                    model=component_model,  # No backward passes so DDP wrapped_model not needed
                    eval_iterator=eval_iterator,
                    device=device,
                    run_config=config,
                    slow_step=slow_step,
                    n_eval_steps=config.n_eval_steps,
                    current_frac_of_training=step / config.steps,
                    reconstruction_loss=reconstruction_loss,
                    ppgd_states=ppgd_states,
                )

                dict_safe_update_(metrics, multibatch_pgd_metrics)

                if is_main_process():
                    assert out_dir is not None
                    for k, v in metrics.items():
                        tqdm.write(f"eval/{k}: {v}")
                    local_log(metrics, step, out_dir)
                    if config.wandb_project:
                        wandb_logs = {
                            f"eval/{k}": wandb.Image(v) if isinstance(v, Image.Image) else v
                            for k, v in metrics.items()
                        }
                        try_wandb(wandb.log, wandb_logs, step=step)

                del metrics
                # TODO: we should reverse the order of these two calls
                torch.cuda.empty_cache()
                gc.collect()

        # --- Saving Checkpoint --- #
        should_save = (
            config.save_freq is not None and step % config.save_freq == 0 and step > 0
        ) or step == config.steps
        if should_save:
            if config.parallel_strategy == "fsdp":
                # FSDP2 stores params as DTensors; a proper full-state-dict save needs
                # `torch.distributed.checkpoint.state_dict.get_model_state_dict` with
                # `StateDictOptions(full_state_dict=True, cpu_offload=True)`. Out of scope for
                # the current FSDP-fit derisk — assert save_freq is null so we don't silently
                # write a sharded state dict.
                assert config.save_freq is None, (
                    "FSDP2 checkpoint save is not implemented yet; set save_freq=null."
                )
            elif is_main_process():
                assert out_dir is not None
                # Save the state dict of the underlying module (not DDP wrapper)
                save_file(component_model.state_dict(), out_dir / f"model_{step}.pth")

            if is_main_process():
                assert out_dir is not None
                logger.info(f"Saved model, optimizer, and out_dir to {out_dir}")
                if config.wandb_project:
                    try_wandb(
                        wandb.save,
                        str(out_dir / f"model_{step}.pth"),
                        base_path=str(out_dir),
                        policy="now",
                    )

        # Skip gradient step if we are at the last step (last step just for plotting and logging)
        if step != config.steps:
            sync_across_processes()
            if config.parallel_strategy == "fsdp":
                # FSDP2 stores grads as DTensors; `torch.nn.utils.clip_grad_norm_` handles
                # them correctly (computes a cross-rank norm internally).
                assert config.grad_clip_norm_ci_fns is None, (
                    "grad_clip_norm_ci_fns is not supported under parallel_strategy='fsdp' — "
                    "the FSDP2 clip applies a single threshold to all trainable params."
                )
                if config.grad_clip_norm_components is not None:
                    clip_grad_norm_(
                        [p for p in wrapped_model.parameters() if p.requires_grad],
                        config.grad_clip_norm_components,
                    )
            else:
                if config.grad_clip_norm_components is not None:
                    clip_grad_norm_(component_params, config.grad_clip_norm_components)
                if config.grad_clip_norm_ci_fns is not None:
                    clip_grad_norm_(ci_fn_params, config.grad_clip_norm_ci_fns)
            optimizer.step()

        if config.profile_memory:
            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - step_start)

        if config.profile_memory and step == config.profile_memory_step and is_main_process():
            assert out_dir is not None
            snap_path = out_dir / "memory_snapshot.pickle"
            torch.cuda.memory._dump_snapshot(str(snap_path))
            torch.cuda.memory._record_memory_history(enabled=None)
            peak_gb = torch.cuda.max_memory_allocated() / 1e9
            logger.info(f"Memory snapshot dumped to {snap_path}")
            logger.info(f"Peak memory: {peak_gb:.2f} GB")
            logger.info(f"\n{torch.cuda.memory_summary(abbreviated=True)}")

    if config.profile_memory and is_main_process() and out_dir is not None:
        warmup_steps = 5
        warmed = step_times[warmup_steps:]
        avg_ms = 1000 * sum(warmed) / len(warmed) if warmed else 0.0
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        summary = {
            "step_times_ms": [t * 1000 for t in step_times],
            "avg_step_time_ms_post_warmup": avg_ms,
            "warmup_steps_skipped": warmup_steps,
            "peak_memory_gb": peak_gb,
            "world_size": dist_state.world_size if dist_state is not None else 1,
            "batch_size": config.batch_size,
        }
        (out_dir / "profile_summary.json").write_text(json.dumps(summary, indent=2))
        logger.info(f"Avg step time (post-warmup): {avg_ms:.1f} ms")
        logger.info(f"Peak memory: {peak_gb:.2f} GB")

    if is_main_process():
        logger.info("Finished training loop.")


def run_experiment(
    target_model: nn.Module,
    config: Config,
    device: str,
    train_loader: DataLoader[Any],
    eval_loader: DataLoader[Any],
    run_batch: RunBatch,
    reconstruction_loss: ReconstructionLoss,
    experiment_tag: str,
    run_id: str | None = None,
    launch_id: str | None = None,
    evals_id: str | None = None,
    sweep_params: dict[str, Any] | None = None,
    target_model_train_config: BaseConfig | None = None,
    tied_weights: list[tuple[str, str]] | None = None,
) -> None:
    """Run a full PD experiment: setup, optimize, cleanup.

    All ranks call this function. Only the main process does wandb/logging setup.
    """
    if is_main_process():
        run_id = run_id or generate_run_id("param_decomp")
        out_dir = PARAM_DECOMP_OUT_DIR / "decompositions" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Run ID: {run_id}")
        logger.info(f"Output directory: {out_dir}")

        tags = [str(i) for i in [experiment_tag, evals_id, launch_id] if i is not None]
        slurm_array_job_id = os.getenv("SLURM_ARRAY_JOB_ID")
        if slurm_array_job_id is not None:
            tags.append(f"slurm-array-job-id_{slurm_array_job_id}")

        if config.wandb_project:
            init_wandb(config, config.wandb_project, run_id, config.wandb_run_name, tags)

        logger.info(config)

        save_pre_run_info(
            save_to_wandb=config.wandb_project is not None,
            out_dir=out_dir,
            pd_config=config,
            sweep_params=sweep_params,
            target_model=target_model if target_model_train_config is not None else None,
            train_config=target_model_train_config,
            task_name=getattr(config.task_config, "task_name", None),
        )
    else:
        out_dir = None

    optimize(
        target_model=target_model,
        config=config,
        device=device,
        train_loader=train_loader,
        eval_loader=eval_loader,
        run_batch=run_batch,
        reconstruction_loss=reconstruction_loss,
        out_dir=out_dir,
        tied_weights=tied_weights,
    )

    if is_main_process() and config.wandb_project:
        wandb.finish()
