"""TMS-specific PD training loop.

Stripped-down version of ``param_decomp.run_pd.optimize`` for TMS targets:

- No DDP wrapping, no ``DistributedState``, no cross-rank syncing or averaging.
- No autocast (the TMS model is small enough that bf16 doesn't help).
- Applies TMS-specific weight tying inline after building the ComponentModel.

The TMS driver dispatches here via ``Driver.optimize``. Notebook callers running
TMS decompositions directly can also call this function.
"""

import gc
from collections import defaultdict
from functools import partial
from typing import Any, cast

import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from param_decomp.configs import LoggingConfig, PDConfig, RuntimeConfig
from param_decomp.eval import evaluate
from param_decomp.identity_insertion import insert_identity_operations_
from param_decomp.log import logger
from param_decomp.metrics.base import LossMetricConfig
from param_decomp.models.batch_and_loss_fns import PDTarget
from param_decomp.models.component_model import ComponentModel
from param_decomp.run_pd import (
    _build_ctx,
    _build_metric_instances,
    compute_losses,
    run_faithfulness_warmup,
)
from param_decomp.run_sink import RunSink
from param_decomp.utils.data_utils import loop_dataloader
from param_decomp.utils.general_utils import (
    combine_nonoverlapping_dicts,
    get_scheduled_value,
    set_seed,
)
from param_decomp.utils.logging_utils import get_grad_norms_dict
from param_decomp.utils.module_utils import expand_module_patterns


def _tie_linear1_linear2(component_model: ComponentModel) -> None:
    tgt = component_model.components["linear2"]
    src = component_model.components["linear1"]
    tgt.U.data = src.V.data.T
    tgt.V.data = src.U.data.T


def tms_optimize(
    target: PDTarget,
    train_loader: DataLoader[Any],
    eval_loader: DataLoader[Any],
    *,
    pd_config: PDConfig,
    logging_config: LoggingConfig,
    runtime_config: RuntimeConfig,
    device: str,
    sink: RunSink,
    tied_weights: bool,
) -> None:
    """Run the TMS PD optimization loop.

    Single-process: no DDP, no rank guards, no cross-rank metric averaging.
    ``runtime_config`` is accepted for API symmetry with ``optimize`` but its
    DDP/autocast knobs don't apply to TMS.
    """
    del runtime_config  # TMS doesn't use autocast / DDP.

    train_iterator = loop_dataloader(train_loader)
    eval_iterator = loop_dataloader(eval_loader)

    if sink.out_dir is not None:
        logger.info(f"Train+eval logs saved to directory: {sink.out_dir}")

    target_model = target.model
    if pd_config.identity_module_info is not None:
        insert_identity_operations_(
            target_model,
            identity_module_info=pd_config.identity_module_info,
        )

    target_model.requires_grad_(False)
    module_path_info = expand_module_patterns(target_model, pd_config.all_module_info)

    component_model = ComponentModel(
        target_model=target_model,
        run_batch=target.run_batch,
        module_path_info=module_path_info,
        ci_config=pd_config.ci_config,
        sigmoid_type=pd_config.sigmoid_type,
    )
    component_model.to(device)

    set_seed(pd_config.seed)

    if tied_weights:
        _tie_linear1_linear2(component_model)

    component_params: list[torch.nn.Parameter] = []
    for name in component_model.target_module_paths:
        component_params.extend(component_model.components[name].parameters())
    ci_fn_params = list(component_model.ci_fn.parameters())
    assert len(component_params) > 0, "No parameters found in components to optimize"

    components_optimizer = optim.AdamW(
        component_params,
        lr=pd_config.components_optimizer.lr_schedule.start_val,
        betas=pd_config.components_optimizer.betas,
        weight_decay=pd_config.components_optimizer.weight_decay,
    )
    ci_fn_optimizer = optim.AdamW(
        ci_fn_params,
        lr=pd_config.ci_fn_optimizer.lr_schedule.start_val,
        betas=pd_config.ci_fn_optimizer.betas,
        weight_decay=pd_config.ci_fn_optimizer.weight_decay,
    )

    if pd_config.faithfulness_warmup_steps > 0:
        run_faithfulness_warmup(component_model, component_params, pd_config, device)

    loss_instances, eval_only_instances = _build_metric_instances(
        pd_config, logging_config, component_model, device
    )
    all_instances = {**loss_instances, **eval_only_instances}

    for step in tqdm(range(pd_config.steps + 1), ncols=0):
        components_optimizer.zero_grad()
        ci_fn_optimizer.zero_grad()

        components_lr = get_scheduled_value(
            step=step,
            total_steps=pd_config.steps,
            config=pd_config.components_optimizer.lr_schedule,
        )
        ci_fn_lr = get_scheduled_value(
            step=step, total_steps=pd_config.steps, config=pd_config.ci_fn_optimizer.lr_schedule
        )
        for group in components_optimizer.param_groups:
            group["lr"] = components_lr
        for group in ci_fn_optimizer.param_groups:
            group["lr"] = ci_fn_lr

        batch_log_data: defaultdict[str, float] = defaultdict(float)

        build_ctx = partial(
            _build_ctx,
            step=step,
            device=device,
            wrapped_model=component_model,
            component_model=component_model,
            config=pd_config,
            reconstruction_loss=target.reconstruction_loss,
        )

        ctx = build_ctx(next(train_iterator), is_eval=False)
        losses = compute_losses(loss_instances, ctx)

        total_loss = torch.zeros((), device=device)
        for metric_name, loss_val in losses.items():
            if loss_val is None:
                continue
            cfg = cast(LossMetricConfig, loss_instances[metric_name].cfg)
            assert cfg.coeff is not None
            total_loss = total_loss + cfg.coeff * loss_val
            batch_log_data[f"train/loss/{type(loss_instances[metric_name]).__name__}"] = (
                loss_val.item()
            )
        batch_log_data["train/loss/total"] = total_loss.item()

        for metric_name, m in loss_instances.items():
            m.before_backward(losses[metric_name])

        total_loss.backward()

        for m in loss_instances.values():
            m.after_backward()

        if step % logging_config.train_log_freq == 0:
            grad_norms = get_grad_norms_dict(component_model, device)
            combine_nonoverlapping_dicts(
                batch_log_data, {f"train/grad_norms/{k}": v for k, v in grad_norms.items()}
            )
            batch_log_data["train/schedules/lr/components"] = components_lr
            batch_log_data["train/schedules/lr/ci_fn"] = ci_fn_lr

            tqdm.write(f"--- Step {step} ---")
            tqdm.write(f"LR[components]: {components_lr:.6f}")
            tqdm.write(f"LR[ci_fn]: {ci_fn_lr:.6f}")
            for name, value in batch_log_data.items():
                tqdm.write(f"{name}: {value:.15f}")
            sink.log(batch_log_data, step=step)

        if step % logging_config.eval_freq == 0:
            with torch.no_grad():
                slow_step: bool = (
                    logging_config.slow_eval_on_first_step
                    if step == 0
                    else step % logging_config.slow_eval_freq == 0
                )

                metrics = evaluate(
                    instances=all_instances,
                    eval_iterator=eval_iterator,
                    ctx_builder=partial(build_ctx, is_eval=True),
                    n_eval_steps=logging_config.n_eval_steps,
                    slow_step=slow_step,
                )

                for k, v in metrics.items():
                    tqdm.write(f"eval/{k}: {v}")
                sink.log(metrics, step=step, section="eval")

                del metrics
                gc.collect()

        if (
            logging_config.save_freq is not None
            and step % logging_config.save_freq == 0
            and step > 0
        ) or step == pd_config.steps:
            sink.checkpoint(component_model.state_dict(), step=step)

        if step != pd_config.steps:
            if pd_config.components_optimizer.grad_clip_norm is not None:
                clip_grad_norm_(component_params, pd_config.components_optimizer.grad_clip_norm)
            if pd_config.ci_fn_optimizer.grad_clip_norm is not None:
                clip_grad_norm_(ci_fn_params, pd_config.ci_fn_optimizer.grad_clip_norm)
            components_optimizer.step()
            ci_fn_optimizer.step()

    logger.info("Finished training loop.")


def tie_tms_component_weights_(component_model: ComponentModel) -> None:
    """Tie the U/V parameters of ``linear2`` to those of ``linear1`` for a TMS model.

    Used after reloading a TMS PD checkpoint whose target model has ``tied_weights=True``.
    """
    _tie_linear1_linear2(component_model)
