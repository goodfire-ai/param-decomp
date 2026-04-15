from collections.abc import Callable, Iterator
from functools import partial
from typing import Literal

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import ReduceOp

from spd.configs import PGDConfig, PGDInitStrategy, PGDMultiBatchConfig, SamplingType
from spd.log import logger
from spd.models.component_model import ComponentModel, OutputWithCache
from spd.models.components import ComponentsMaskInfo, RoutingMasks, make_mask_infos
from spd.routing import Router
from spd.utils.distributed_utils import all_reduce, broadcast_tensor
from spd.utils.general_utils import calc_sum_recon_loss_lm, extract_batch_data


def _init_adv_sources(
    model: ComponentModel,
    batch_dims: tuple[int, ...],
    device: torch.device | str,
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    pgd_config: PGDConfig,
) -> dict[str, Float[Tensor, "*batch_dims mask_c"]]:
    adv_sources: dict[str, Float[Tensor, "*batch_dims mask_c"]] = {}
    for module_name in model.target_module_paths:
        module_c = model.module_to_c[module_name]
        mask_c = module_c if weight_deltas is None else module_c + 1
        match pgd_config.mask_scope:
            case "unique_per_datapoint":
                shape = torch.Size([*batch_dims, mask_c])
                source = get_pgd_init_tensor(pgd_config.init, shape, device)
            case "shared_across_batch":
                singleton_batch_dims = [1 for _ in batch_dims]
                shape = torch.Size([*singleton_batch_dims, mask_c])
                source = broadcast_tensor(get_pgd_init_tensor(pgd_config.init, shape, device))
        adv_sources[module_name] = source.requires_grad_(True)
    return adv_sources


def _run_pgd_loop(
    adv_sources: dict[str, Float[Tensor, "..."]],
    pgd_config: PGDConfig,
    fwd_fn: Callable[[], tuple[Float[Tensor, ""], int]],
) -> tuple[Float[Tensor, ""], int]:
    for _ in range(pgd_config.n_steps):
        assert all(adv.grad is None for adv in adv_sources.values())
        with torch.enable_grad():
            sum_loss, n_examples = fwd_fn()
            loss = sum_loss / n_examples
        grads = torch.autograd.grad(loss, list(adv_sources.values()))
        match pgd_config.mask_scope:
            case "shared_across_batch":
                adv_sources_grads = {
                    k: all_reduce(g, op=ReduceOp.AVG)
                    for k, g in zip(adv_sources.keys(), grads, strict=True)
                }
            case "unique_per_datapoint":
                adv_sources_grads = dict(zip(adv_sources.keys(), grads, strict=True))
        with torch.no_grad():
            for k in adv_sources:
                adv_sources[k].add_(pgd_config.step_size * adv_sources_grads[k].sign())
                adv_sources[k].clamp_(0.0, 1.0)

    return fwd_fn()


def _construct_mask_infos_from_adv_sources(
    adv_sources: dict[str, Float[Tensor, "*batch_dim_or_ones mask_c"]],
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    routing_masks: RoutingMasks,
    batch_dims: tuple[int, ...],
) -> dict[str, "ComponentsMaskInfo"]:
    expanded_adv_sources = {k: v.expand(*batch_dims, -1) for k, v in adv_sources.items()}
    adv_sources_components: dict[str, Float[Tensor, "*batch_dims C"]]
    match weight_deltas:
        case None:
            weight_deltas_and_masks = None
            adv_sources_components = expanded_adv_sources
        case dict():
            weight_deltas_and_masks = {
                k: (weight_deltas[k], expanded_adv_sources[k][..., -1]) for k in weight_deltas
            }
            adv_sources_components = {k: v[..., :-1] for k, v in expanded_adv_sources.items()}

    return make_mask_infos(
        component_masks=interpolate_pgd_mask(ci, adv_sources_components),
        weight_deltas_and_masks=weight_deltas_and_masks,
        routing_masks=routing_masks,
    )


def pgd_masked_recon_loss_update(
    model: ComponentModel,
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    target_out: Float[Tensor, "... vocab"],
    output_loss_type: Literal["mse", "kl"],
    router: Router,
    pgd_config: PGDConfig,
) -> tuple[Float[Tensor, ""], int]:
    """Central implementation of PGD masked reconstruction loss.

    Optimizes adversarial stochastic masks and optionally weight deltas for the given objective function.
    """
    batch_dims = next(iter(ci.values())).shape[:-1]
    routing_masks = router.get_masks(module_names=model.target_module_paths, mask_shape=batch_dims)
    adv_sources = _init_adv_sources(model, batch_dims, batch.device, weight_deltas, pgd_config)

    fwd_pass = partial(
        _forward_with_adv_sources,
        model=model,
        batch=batch,
        adv_sources=adv_sources,
        ci=ci,
        weight_deltas=weight_deltas,
        routing_masks=routing_masks,
        target_out=target_out,
        output_loss_type=output_loss_type,
        batch_dims=batch_dims,
    )

    return _run_pgd_loop(adv_sources, pgd_config, fwd_pass)


CreateDataIter = Callable[
    [],
    Iterator[Int[Tensor, "..."]] | Iterator[tuple[Float[Tensor, "..."], Float[Tensor, "..."]]],
]


def calc_multibatch_pgd_masked_recon_loss(
    pgd_config: PGDMultiBatchConfig,
    model: ComponentModel,
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    create_data_iter: CreateDataIter,
    output_loss_type: Literal["mse", "kl"],
    router: Router,
    sampling: SamplingType,
    use_delta_component: bool,
    batch_dims: tuple[int, ...],
    device: str,
) -> Float[Tensor, ""]:
    """PGD masked reconstruction loss with gradient accumulation over multiple batches.

    This function optimizes adversarial masks by accumulating gradients over pgd_config.n_batches
    batches before each PGD update step.

    Args:
        pgd_config: Multibatch PGD configuration
        model: The ComponentModel to evaluate
        create_data_iter: Function to create an iterator over batches. This function should return
            an iterator which behaves identically each time. Specifically in terms of data ordering
            and shuffling.
        output_loss_type: Loss type for reconstruction ("mse" or "kl")
        router: Router to use for routing masks
        sampling: Sampling mode for causal importance calculation
        use_delta_component: Whether to include weight delta component
        batch_dims: Dimensions of batch (e.g., (batch_size,) or (batch_size, seq_len))
    Returns:
        Final reconstruction loss after PGD optimization
    """
    singleton_batch_dims = [1 for _ in batch_dims]

    adv_sources: dict[str, Float[Tensor, "*ones mask_c"]] = {}
    for module_name in model.target_module_paths:
        module_c = model.module_to_c[module_name]
        mask_c = module_c if not use_delta_component else module_c + 1
        shape = torch.Size([*singleton_batch_dims, mask_c])
        adv_sources[module_name] = broadcast_tensor(
            get_pgd_init_tensor(pgd_config.init, shape, device)
        ).requires_grad_(True)

    fwd_bwd_fn = partial(
        _multibatch_pgd_fwd_bwd,
        adv_sources=adv_sources,
        pgd_config=pgd_config,
        model=model,
        weight_deltas=weight_deltas,
        device=device,
        output_loss_type=output_loss_type,
        sampling=sampling,
        router=router,
        batch_dims=batch_dims,
    )

    for _ in range(pgd_config.n_steps):
        assert all(adv.grad is None for adv in adv_sources.values())
        _, _, adv_sources_grads = fwd_bwd_fn(data_iter=create_data_iter())

        with torch.no_grad():
            for k in adv_sources:
                adv_sources[k].add_(pgd_config.step_size * adv_sources_grads[k].sign())
                adv_sources[k].clamp_(0.0, 1.0)

    final_loss, final_n_examples, _ = fwd_bwd_fn(data_iter=create_data_iter())
    return final_loss / final_n_examples


def _forward_with_adv_sources(
    model: ComponentModel,
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    adv_sources: dict[str, Float[Tensor, "*batch_dim_or_ones mask_c"]],
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    routing_masks: RoutingMasks,
    target_out: Float[Tensor, "... vocab"],
    output_loss_type: Literal["mse", "kl"],
    batch_dims: tuple[int, ...],
):
    mask_infos = _construct_mask_infos_from_adv_sources(
        adv_sources=adv_sources,
        ci=ci,
        weight_deltas=weight_deltas,
        routing_masks=routing_masks,
        batch_dims=batch_dims,
    )
    out = model(batch, mask_infos=mask_infos)

    sum_loss = calc_sum_recon_loss_lm(pred=out, target=target_out, loss_type=output_loss_type)

    n_examples = (
        target_out.shape.numel() if output_loss_type == "mse" else target_out.shape[:-1].numel()
    )

    return sum_loss, n_examples


def _multibatch_pgd_fwd_bwd(
    adv_sources: dict[str, Float[Tensor, "*ones mask_c"]],
    pgd_config: PGDMultiBatchConfig,
    model: ComponentModel,
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    data_iter: Iterator[Int[Tensor, "..."]]
    | Iterator[tuple[Float[Tensor, "..."], Float[Tensor, "..."]]],
    device: torch.device | str,
    output_loss_type: Literal["mse", "kl"],
    router: Router,
    sampling: SamplingType,
    batch_dims: tuple[int, ...],
) -> tuple[Float[Tensor, ""], int, dict[str, Float[Tensor, "*ones mask_c"]]]:
    """Perform a forward and backward pass over multiple batches with gradient accumulation.

    Returns:
        - The total loss for the PGD step (only used for the final step)
        - The number of examples used in the PGD step
        - The gradients of the adv_sources (dict keyed by module name)
    """
    pgd_step_accum_sum_loss = torch.tensor(0.0, device=device)
    pgd_step_accum_n_examples = 0
    pgd_step_accum_grads = {k: torch.zeros_like(v) for k, v in adv_sources.items()}

    for microbatch_idx in range(pgd_config.gradient_accumulation_steps):
        try:
            microbatch_item = next(data_iter)
        except StopIteration:
            logger.warning(f"Dataloader exhausted after {microbatch_idx} batches, ending PGD step.")
            break
        microbatch = extract_batch_data(microbatch_item).to(device)

        # NOTE: technically this is duplicated work across PGD steps, but that's the price we pay to
        # enable accumulating gradients over more microbatches than we'd be able to fit CI values in
        # memory for. In other words, you can't fit 100,000 microbatches worth of CI values in memory.
        target_model_output: OutputWithCache = model(microbatch, cache_type="input")
        ci = model.calc_causal_importances(
            pre_weight_acts=target_model_output.cache,
            sampling=sampling,
        ).lower_leaky

        # It's important that we call this every microbatch to ensure stochastic routing masks are
        # sampled independently for each example.
        routing_masks = router.get_masks(
            module_names=model.target_module_paths, mask_shape=batch_dims
        )

        batch_sum_loss, batch_n_examples = _forward_with_adv_sources(
            model=model,
            batch=microbatch,
            adv_sources=adv_sources,
            ci=ci,
            weight_deltas=weight_deltas,
            routing_masks=routing_masks,
            target_out=target_model_output.output,
            output_loss_type=output_loss_type,
            batch_dims=batch_dims,
        )

        pgd_step_accum_sum_loss += batch_sum_loss
        pgd_step_accum_n_examples += batch_n_examples

        # important: take gradient wrt the UNEXPANDED adv_sources, not the expanded ones
        grads = torch.autograd.grad(batch_sum_loss, list(adv_sources.values()))
        for k, g in zip(adv_sources.keys(), grads, strict=True):
            pgd_step_accum_grads[k] += all_reduce(g, op=ReduceOp.AVG).detach()

        del target_model_output, ci

    return pgd_step_accum_sum_loss, pgd_step_accum_n_examples, pgd_step_accum_grads


def get_pgd_init_tensor(
    init: PGDInitStrategy,
    shape: tuple[int, ...],
    device: torch.device | str,
) -> Float[Tensor, "... shape"]:
    """Create initial PGD source tensor (random, ones, or zeroes). Shared by training PGD and app eval PGD."""
    match init:
        case "random":
            return torch.rand(shape, device=device)
        case "ones":
            return torch.ones(shape, device=device)
        case "zeroes":
            return torch.zeros(shape, device=device)


def interpolate_pgd_mask(
    ci: dict[str, Float[Tensor, "*batch_dims C"]],
    adv_sources_components: dict[str, Float[Tensor, "*batch_dims C"]],
) -> dict[str, Float[Tensor, "*batch_dims C"]]:
    """Set the mask value to ci + (1 - ci) * adv_sources_components.

    NOTE: This is not ideal. Suppose ci is 0.2 and adv_sources_components is 0.8. Then the mask
    value will be 0.2 + (1 - 0.2) * 0.8 = 0.84. It would make more sense to set the mask value to
    0.8 directly, since this is the value output by PGD. If we wanted to instead use the more
    natural setup of mask = max(ci, adv_sources_components), we would need to work out how to
    maintain the gradient flow. We feel that this likely isn't a big problem as it is right now
    since the change would just give PGD more optimization power, and we already get a very bad
    loss value for it.
    """
    component_masks: dict[str, Float[Tensor, "*batch_dims C"]] = {}
    for module_name in ci:
        adv_source = adv_sources_components[module_name]
        assert ci[module_name].shape[-1] == adv_source.shape[-1]
        scaled_noise_to_add = (1 - ci[module_name]) * adv_source
        component_masks[module_name] = ci[module_name] + scaled_noise_to_add
    return component_masks
