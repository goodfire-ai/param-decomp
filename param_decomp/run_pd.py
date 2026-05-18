"""Run PD on a model."""

import gc
import os
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

from param_decomp.configs import (
    LoggingConfig,
    MetricConfigType,
    PDConfig,
    PersistentPGDReconLossConfig,
    PersistentPGDReconSubsetLossConfig,
    PGDMultiBatchConfig,
    PGDMultiBatchReconLossConfig,
    PGDMultiBatchReconSubsetLossConfig,
    RepeatAcrossBatchScope,
    RuntimeConfig,
)
from param_decomp.eval import evaluate, evaluate_multibatch_pgd
from param_decomp.identity_insertion import insert_identity_operations_
from param_decomp.log import logger
from param_decomp.losses import compute_losses
from param_decomp.metrics import faithfulness_loss
from param_decomp.models.batch_and_loss_fns import (
    PDTarget,
    ReconstructionLoss,
    RunBatch,
    move_batch_to_device,
)
from param_decomp.models.component_model import ComponentModel, OutputWithCache
from param_decomp.persistent_pgd import PersistentPGDState
from param_decomp.run_metadata import RunMetadata
from param_decomp.settings import PARAM_DECOMP_OUT_DIR
from param_decomp.utils.component_utils import calc_ci_l_zero
from param_decomp.utils.data_utils import loop_dataloader
from param_decomp.utils.distributed_utils import (
    DistributedState,
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


def run_faithfulness_warmup(
    component_model: ComponentModel,
    component_params: list[torch.nn.Parameter],
    config: PDConfig,
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


def optimize(
    target_model: nn.Module,
    config: PDConfig,
    logging_config: LoggingConfig,
    runtime_config: RuntimeConfig,
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
        return (move_batch_to_device(batch, device) for batch in train_loader)

    if is_main_process():
        logger.info(f"Train+eval logs saved to directory: {out_dir}")

    if config.identity_module_info is not None:
        insert_identity_operations_(
            target_model,
            identity_module_info=config.identity_module_info,
        )

    target_model.requires_grad_(False)

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

    # Wrap model with DDP if distributed
    dist_state = get_distributed_state()
    wrapped_model: nn.Module = model

    component_model: ComponentModel
    if dist_state is not None:
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
    else:
        component_model = model
    assert isinstance(component_model, ComponentModel), "component_model is not a ComponentModel"

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
        component_params.extend(component_model.components[name].parameters())

    ci_fn_params = list(component_model.ci_fn.parameters())

    assert len(component_params) > 0, "No parameters found in components to optimize"

    components_optimizer = optim.AdamW(
        component_params,
        lr=config.components_optimizer.lr_schedule.start_val,
        betas=config.components_optimizer.betas,
        weight_decay=config.components_optimizer.weight_decay,
    )
    ci_fn_optimizer = optim.AdamW(
        ci_fn_params,
        lr=config.ci_fn_optimizer.lr_schedule.start_val,
        betas=config.ci_fn_optimizer.betas,
        weight_decay=config.ci_fn_optimizer.weight_decay,
    )

    if config.faithfulness_warmup_steps > 0:
        run_faithfulness_warmup(component_model, component_params, config)

    persistent_pgd_configs: list[
        PersistentPGDReconLossConfig | PersistentPGDReconSubsetLossConfig
    ] = [
        cfg
        for cfg in (
            config.loss_metrics.persistent_pgd_recon,
            config.loss_metrics.persistent_pgd_recon_subset,
        )
        if cfg is not None
    ]

    multibatch_pgd_eval_configs: list[
        PGDMultiBatchReconLossConfig | PGDMultiBatchReconSubsetLossConfig
    ] = [
        cfg
        for cfg in (
            config.eval_metrics.pgd_multibatch_recon,
            config.eval_metrics.pgd_multibatch_recon_subset,
        )
        if cfg is not None
    ]

    eval_metric_configs: list[MetricConfigType] = [
        cfg
        for cfg in config.loss_metrics.active() + config.eval_metrics.active()
        if not isinstance(cfg, PGDMultiBatchConfig)
    ]

    sample_out = model(move_batch_to_device(next(train_iterator), device))
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

    for step in tqdm(range(config.steps + 1), ncols=0, disable=not is_main_process()):
        components_optimizer.zero_grad()
        ci_fn_optimizer.zero_grad()

        components_lr = get_scheduled_value(
            step=step, total_steps=config.steps, config=config.components_optimizer.lr_schedule
        )
        ci_fn_lr = get_scheduled_value(
            step=step, total_steps=config.steps, config=config.ci_fn_optimizer.lr_schedule
        )
        for group in components_optimizer.param_groups:
            group["lr"] = components_lr
        for group in ci_fn_optimizer.param_groups:
            group["lr"] = ci_fn_lr

        frac = step / config.steps
        active_ppgd_configs = [c for c in persistent_pgd_configs if frac >= c.start_frac]

        for ppgd_cfg in active_ppgd_configs:
            ppgd_states[ppgd_cfg].update_lr(step, config.steps)

        weight_deltas = component_model.calc_weight_deltas()

        batch_log_data: defaultdict[str, float] = defaultdict(float)

        batch = move_batch_to_device(next(train_iterator), device)
        with bf16_autocast(enabled=runtime_config.autocast_bf16):
            # NOTE: we need to call the wrapped_model at least once each step in order to setup
            # the DDP gradient syncing for all parameters in the component model. Gradients will
            # sync regardless of whether the parameters are used in this call to wrapped_model.
            target_model_output: OutputWithCache = wrapped_model(batch, cache_type="input")

            ci = component_model.calc_causal_importances(
                pre_weight_acts=target_model_output.cache,
                detach_inputs=False,
                sampling=config.sampling,
            )

            for ppgd_cfg in active_ppgd_configs:
                ppgd_states[ppgd_cfg].warmup(
                    model=component_model,
                    batch=batch,
                    target_out=target_model_output.output,
                    ci=ci.lower_leaky,
                    weight_deltas=weight_deltas if config.use_delta_component else None,
                )

            losses = compute_losses(
                loss_metrics=config.loss_metrics,
                model=component_model,
                batch=batch,
                ci=ci,
                target_out=target_model_output.output,
                weight_deltas=weight_deltas,
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
            l0_val = calc_ci_l_zero(layer_ci, logging_config.ci_alive_threshold)
            batch_log_data[f"train/l0/{layer_name}"] = l0_val

        # --- Train Logging --- #
        if step % logging_config.train_log_freq == 0:
            avg_metrics = avg_metrics_across_ranks(batch_log_data, device=device)
            batch_log_data = cast(defaultdict[str, float], avg_metrics)

            grad_norms = get_grad_norms_dict(component_model, device)
            dict_safe_update_(
                batch_log_data, {f"train/grad_norms/{k}": v for k, v in grad_norms.items()}
            )

            batch_log_data["train/schedules/lr/components"] = components_lr
            batch_log_data["train/schedules/lr/ci_fn"] = ci_fn_lr

            if is_main_process():
                assert out_dir is not None
                tqdm.write(f"--- Step {step} ---")
                tqdm.write(f"LR[components]: {components_lr:.6f}")
                tqdm.write(f"LR[ci_fn]: {ci_fn_lr:.6f}")
                for name, value in batch_log_data.items():
                    tqdm.write(f"{name}: {value:.15f}")
                local_log(batch_log_data, step, out_dir)
                if wandb.run is not None:
                    try_wandb(wandb.log, batch_log_data, step=step)

        # --- Evaluation --- #
        if step % logging_config.eval_freq == 0:
            with torch.no_grad(), bf16_autocast(enabled=runtime_config.autocast_bf16):
                slow_step: bool = (
                    logging_config.slow_eval_on_first_step
                    if step == 0
                    else step % logging_config.slow_eval_freq == 0
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
                    ci_alive_threshold=logging_config.ci_alive_threshold,
                    slow_step=slow_step,
                    n_eval_steps=logging_config.n_eval_steps,
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
                    if wandb.run is not None:
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
        if (
            (
                logging_config.save_freq is not None
                and step % logging_config.save_freq == 0
                and step > 0
            )
            or step == config.steps
        ) and is_main_process():
            assert out_dir is not None
            # Save the state dict of the underlying module (not DDP wrapper)
            save_file(component_model.state_dict(), out_dir / f"model_{step}.pth")
            logger.info(f"Saved model, optimizer, and out_dir to {out_dir}")
            if wandb.run is not None:
                try_wandb(
                    wandb.save,
                    str(out_dir / f"model_{step}.pth"),
                    base_path=str(out_dir),
                    policy="now",
                )

        # Skip gradient step if we are at the last step (last step just for plotting and logging)
        if step != config.steps:
            sync_across_processes()
            if config.components_optimizer.grad_clip_norm is not None:
                clip_grad_norm_(component_params, config.components_optimizer.grad_clip_norm)
            if config.ci_fn_optimizer.grad_clip_norm is not None:
                clip_grad_norm_(ci_fn_params, config.ci_fn_optimizer.grad_clip_norm)
            components_optimizer.step()
            ci_fn_optimizer.step()

    if is_main_process():
        logger.info("Finished training loop.")


def _validate_pgd_scope(config: PDConfig, dist_state: DistributedState | None) -> None:
    """Assert that PGD `repeat_across_batch` divides the per-rank training batch size."""
    world_size = dist_state.world_size if dist_state is not None else 1
    assert config.batch_size % world_size == 0, (
        f"batch_size {config.batch_size} not divisible by world size {world_size}"
    )
    per_rank = config.batch_size // world_size
    for cfg in (
        config.loss_metrics.persistent_pgd_recon,
        config.loss_metrics.persistent_pgd_recon_subset,
    ):
        if cfg is not None and isinstance(cfg.scope, RepeatAcrossBatchScope):
            n = cfg.scope.n_sources
            assert per_rank % n == 0, (
                f"repeat_across_batch n_sources={n} must divide per-rank batch_size={per_rank}"
            )


def run_pd(
    config: PDConfig,
    logging_config: LoggingConfig,
    runtime_config: RuntimeConfig,
    target: PDTarget,
    train_loader: DataLoader[Any],
    eval_loader: DataLoader[Any],
    device: str,
    *,
    run_id: str | None = None,
    metadata: RunMetadata | None = None,
    artifacts: dict[str, Any] | None = None,
    wandb_tags: list[str] | None = None,
) -> Path | None:
    """Run a full PD decomposition: setup, optimize, cleanup.

    `metadata` is written to ``run_metadata.yaml``.  Driver-mediated callers
    (via ``experiments/runner.py``) pass a fully populated ``RunMetadata``;
    notebook callers can omit it and a minimal one is synthesized.

    All ranks call this function. Only the main process does wandb/logging setup.
    Returns the output directory on the main process and None on other ranks.
    """
    _validate_pgd_scope(config, get_distributed_state())

    out_dir: Path | None
    if is_main_process():
        run_id = run_id or generate_run_id("param_decomp")
        out_dir = PARAM_DECOMP_OUT_DIR / "decompositions" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Run ID: {run_id}")
        logger.info(f"Output directory: {out_dir}")

        artifacts = artifacts or {}
        if metadata is None:
            metadata = RunMetadata(
                driver=None,
                config={
                    "pd": config.model_dump(mode="json"),
                    "logging": logging_config.model_dump(mode="json"),
                    "runtime": runtime_config.model_dump(mode="json"),
                },
            )

        tags = list(wandb_tags or [])
        slurm_array_job_id = os.getenv("SLURM_ARRAY_JOB_ID")
        if slurm_array_job_id is not None:
            tags.append(f"slurm-array-job-id_{slurm_array_job_id}")

        if metadata.wandb_project:
            init_wandb(
                config,
                metadata.wandb_project,
                run_id,
                name=metadata.wandb_run_name,
                tags=tags,
                extra_configs={"logging": logging_config, "runtime": runtime_config},
                view_meta=metadata.view_meta,
            )

        logger.info(config)

        save_pre_run_info(
            save_to_wandb=metadata.wandb_project is not None,
            out_dir=out_dir,
            metadata=metadata,
            artifacts=artifacts,
        )
    else:
        out_dir = None

    optimize(
        target_model=target.model,
        config=config,
        logging_config=logging_config,
        runtime_config=runtime_config,
        device=device,
        train_loader=train_loader,
        eval_loader=eval_loader,
        run_batch=target.run_batch,
        reconstruction_loss=target.reconstruction_loss,
        out_dir=out_dir,
        tied_weights=target.tied_weights,
    )

    if is_main_process() and wandb.run is not None:
        wandb.finish()

    return out_dir
