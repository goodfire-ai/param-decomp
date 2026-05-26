from collections.abc import Callable
from functools import partial
from typing import Any, Literal

import torch
from jaxtyping import Float
from torch import Tensor
from torch.distributed import ReduceOp

from param_decomp.batch_and_loss_fns import ReconstructionLoss
from param_decomp.component_model import ComponentModel
from param_decomp.distributed import all_reduce, broadcast_tensor
from param_decomp.masks import (
    ComponentsMaskInfo,
    Router,
    RoutingMasks,
    interpolate_component_mask,
    make_mask_infos,
)
from param_decomp.metrics.base import LossMetricConfig

PGDInitStrategy = Literal["random", "ones", "zeroes"]
MaskScope = Literal["unique_per_datapoint", "shared_across_batch"]


class PGDConfig(LossMetricConfig):
    """Shared base for per-step PGD loss configs."""

    init: PGDInitStrategy
    step_size: float
    n_steps: int
    mask_scope: MaskScope


def get_pgd_init_tensor(
    init: PGDInitStrategy,
    shape: tuple[int, ...] | torch.Size,
    device: torch.device | str,
) -> Float[Tensor, "... shape"]:
    match init:
        case "random":
            return torch.rand(shape, device=device)
        case "ones":
            return torch.ones(shape, device=device)
        case "zeroes":
            return torch.zeros(shape, device=device)


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
) -> dict[str, ComponentsMaskInfo]:
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
        component_masks=interpolate_component_mask(ci, adv_sources_components),
        weight_deltas_and_masks=weight_deltas_and_masks,
        routing_masks=routing_masks,
    )


def _forward_with_adv_sources(
    model: ComponentModel,
    batch: Any,
    adv_sources: dict[str, Float[Tensor, "*batch_dim_or_ones mask_c"]],
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    routing_masks: RoutingMasks,
    target_out: Tensor,
    batch_dims: tuple[int, ...],
    reconstruction_loss: ReconstructionLoss,
) -> tuple[Float[Tensor, ""], int]:
    mask_infos = _construct_mask_infos_from_adv_sources(
        adv_sources=adv_sources,
        ci=ci,
        weight_deltas=weight_deltas,
        routing_masks=routing_masks,
        batch_dims=batch_dims,
    )
    out = model(batch, mask_infos=mask_infos)
    return reconstruction_loss(out, target_out)


def pgd_masked_recon_loss_update(
    model: ComponentModel,
    batch: Any,
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    target_out: Tensor,
    router: Router,
    pgd_config: PGDConfig,
    reconstruction_loss: ReconstructionLoss,
) -> tuple[Float[Tensor, ""], int]:
    """Per-step PGD masked recon.

    Inits fresh adversarial sources, runs `pgd_config.n_steps` of inner sign-PGD against
    the recon objective, returns `(sum_loss, n_examples)` evaluated at the final sources.
    """
    batch_dims = next(iter(ci.values())).shape[:-1]
    routing_masks = router.get_masks(module_names=model.target_module_paths, mask_shape=batch_dims)
    adv_sources = _init_adv_sources(model, batch_dims, target_out.device, weight_deltas, pgd_config)

    fwd_pass = partial(
        _forward_with_adv_sources,
        model=model,
        batch=batch,
        adv_sources=adv_sources,
        ci=ci,
        weight_deltas=weight_deltas,
        routing_masks=routing_masks,
        target_out=target_out,
        batch_dims=batch_dims,
        reconstruction_loss=reconstruction_loss,
    )
    return _run_pgd_loop(adv_sources, pgd_config, fwd_pass)
