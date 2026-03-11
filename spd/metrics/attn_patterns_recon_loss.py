import math
from fnmatch import fnmatch
from typing import Any, ClassVar, override

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.distributed import ReduceOp

from spd.configs import SamplingType
from spd.metrics.base import Metric
from spd.models.component_model import CIOutputs, ComponentModel
from spd.models.components import ComponentsMaskInfo, make_mask_infos
from spd.routing import AllLayersRouter
from spd.utils.component_utils import calc_stochastic_component_mask_info
from spd.utils.distributed_utils import all_reduce
from spd.utils.general_utils import get_obj_device


def _resolve_paths(pattern: str, model: ComponentModel) -> list[str]:
    """Resolve an fnmatch pattern against model target module paths."""
    matches = [p for p in model.target_module_paths if fnmatch(p, pattern)]
    assert matches, f"Pattern {pattern!r} matched no target module paths"
    return sorted(matches)


def _resolve_qk_paths(
    model: ComponentModel,
    q_proj_path: str | None,
    k_proj_path: str | None,
    c_attn_path: str | None,
) -> tuple[list[str], list[str], bool]:
    """Resolve Q/K projection paths, returning (q_paths, k_paths, is_combined).

    For separate Q/K projections: returns matched paths paired by sorted order.
    For combined c_attn: returns the same paths for both Q and K.
    """
    if c_attn_path is not None:
        paths = _resolve_paths(c_attn_path, model)
        return paths, paths, True
    assert q_proj_path is not None and k_proj_path is not None
    q_paths = _resolve_paths(q_proj_path, model)
    k_paths = _resolve_paths(k_proj_path, model)
    assert len(q_paths) == len(k_paths), f"Q/K path counts differ: {len(q_paths)} vs {len(k_paths)}"
    return q_paths, k_paths, False


def _resolve_attn_modules(
    model: ComponentModel,
    q_paths: list[str],
) -> list[nn.Module | None]:
    """Derive parent attention module from Q paths, returning it if it has RoPE support.

    For each Q path (e.g. "h.0.attn.q_proj"), strips the last segment to get the parent
    attention module (e.g. "h.0.attn"). Returns the module if it has `apply_rotary_pos_emb`,
    otherwise None.
    """
    result: list[nn.Module | None] = []
    for q_path in q_paths:
        parent_path = q_path.rsplit(".", 1)[0]
        attn_module = model.target_model.get_submodule(parent_path)
        if hasattr(attn_module, "apply_rotary_pos_emb"):
            result.append(attn_module)
        else:
            result.append(None)
    return result


def _compute_attn_patterns(
    q: Float[Tensor, "batch seq d"],
    k: Float[Tensor, "batch seq d"],
    n_heads: int,
    attn_module: nn.Module | None,
) -> Float[Tensor, "batch n_heads seq seq"]:
    """Compute causal attention patterns from Q and K projections.

    If attn_module is provided (has RoPE), applies rotary positional embeddings to Q and K
    before computing the dot-product attention.
    """
    B, S, D = q.shape
    head_dim = D // n_heads
    q = q.view(B, S, n_heads, head_dim).transpose(1, 2)
    k = k.view(B, S, n_heads, head_dim).transpose(1, 2)

    if attn_module is not None:
        position_ids = torch.arange(S, device=q.device).unsqueeze(0)
        cos = attn_module.rotary_cos[position_ids].to(q.dtype)  # pyright: ignore[reportIndexIssue]
        sin = attn_module.rotary_sin[position_ids].to(q.dtype)  # pyright: ignore[reportIndexIssue]
        q, k = attn_module.apply_rotary_pos_emb(q, k, cos, sin)  # pyright: ignore[reportCallIssue]

    attn = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
    causal_mask = torch.triu(torch.ones(S, S, device=q.device, dtype=torch.bool), diagonal=1)
    attn = attn.masked_fill(causal_mask, float("-inf"))
    return F.softmax(attn, dim=-1)


def _split_combined_qkv(
    output: Float[Tensor, "... d"],
) -> tuple[Float[Tensor, "..."], Float[Tensor, "..."]]:
    """Split combined QKV output into Q and K projections."""
    d = output.shape[-1] // 3
    return output[..., :d], output[..., d : 2 * d]


def _attn_patterns_recon_loss_update(
    model: ComponentModel,
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    pre_weight_acts: dict[str, Float[Tensor, "..."]],
    mask_infos_list: list[dict[str, ComponentsMaskInfo]],
    q_paths: list[str],
    k_paths: list[str],
    is_combined: bool,
    n_heads: int,
    attn_modules: list[nn.Module | None],
) -> tuple[Float[Tensor, ""], int]:
    """Shared update logic for both CI-masked and stochastic variants."""
    # 1. Compute target attention patterns from pre_weight_acts
    target_patterns: list[Float[Tensor, "batch n_heads seq seq"]] = []
    for i, (q_path, k_path) in enumerate(zip(q_paths, k_paths, strict=True)):
        if is_combined:
            assert q_path == k_path
            target_out = model.components[q_path](pre_weight_acts[q_path])
            target_q, target_k = _split_combined_qkv(target_out)
        else:
            target_q = model.components[q_path](pre_weight_acts[q_path])
            target_k = model.components[k_path](pre_weight_acts[k_path])
        target_patterns.append(
            _compute_attn_patterns(target_q, target_k, n_heads, attn_modules[i]).detach()
        )

    # 2. Compute masked attention patterns and KL divergence
    device = get_obj_device(pre_weight_acts)
    sum_kl = torch.tensor(0.0, device=device)
    n_distributions = 0

    for mask_infos in mask_infos_list:
        comp_cache = model(batch, mask_infos=mask_infos, cache_type="input").cache

        for i, (q_path, k_path) in enumerate(zip(q_paths, k_paths, strict=True)):
            if is_combined:
                assert q_path == k_path
                masked_out = model.components[q_path](
                    comp_cache[q_path],
                    mask=mask_infos[q_path].component_mask,
                    weight_delta_and_mask=mask_infos[q_path].weight_delta_and_mask,
                )
                masked_q, masked_k = _split_combined_qkv(masked_out)
            else:
                masked_q = model.components[q_path](
                    comp_cache[q_path],
                    mask=mask_infos[q_path].component_mask,
                    weight_delta_and_mask=mask_infos[q_path].weight_delta_and_mask,
                )
                masked_k = model.components[k_path](
                    comp_cache[k_path],
                    mask=mask_infos[k_path].component_mask,
                    weight_delta_and_mask=mask_infos[k_path].weight_delta_and_mask,
                )

            masked_patterns = _compute_attn_patterns(masked_q, masked_k, n_heads, attn_modules[i])
            # KL(target || masked): sum over attention distribution dimension
            kl = F.kl_div(
                masked_patterns.clamp(min=1e-12).log(),
                target_patterns[i],
                reduction="sum",
            )
            sum_kl = sum_kl + kl
            # Count: batch * n_heads * seq (one distribution per query position per head)
            n_distributions += target_patterns[i].shape[0] * n_heads * target_patterns[i].shape[2]

    return sum_kl, n_distributions


def _attn_patterns_recon_loss_compute(
    sum_kl: Float[Tensor, ""],
    n_distributions: Int[Tensor, ""] | int,
) -> Float[Tensor, ""]:
    return sum_kl / n_distributions


# --- CI-masked variant ---


class CIMaskedAttnPatternsReconLoss(Metric):
    """Attention pattern reconstruction loss using CI masks."""

    metric_section: ClassVar[str] = "loss"

    def __init__(
        self,
        model: ComponentModel,
        device: str,
        n_heads: int,
        q_proj_path: str | None,
        k_proj_path: str | None,
        c_attn_path: str | None,
    ) -> None:
        self.model = model
        self.n_heads = n_heads
        self.q_paths, self.k_paths, self.is_combined = _resolve_qk_paths(
            model, q_proj_path, k_proj_path, c_attn_path
        )
        self.attn_modules = _resolve_attn_modules(model, self.q_paths)
        self.sum_kl = torch.tensor(0.0, device=device)
        self.n_distributions = torch.tensor(0, device=device)

    @override
    def update(
        self,
        *,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        pre_weight_acts: dict[str, Float[Tensor, "..."]],
        ci: CIOutputs,
        **_: Any,
    ) -> None:
        mask_infos = make_mask_infos(ci.lower_leaky, weight_deltas_and_masks=None)
        sum_kl, n_distributions = _attn_patterns_recon_loss_update(
            model=self.model,
            batch=batch,
            pre_weight_acts=pre_weight_acts,
            mask_infos_list=[mask_infos],
            q_paths=self.q_paths,
            k_paths=self.k_paths,
            is_combined=self.is_combined,
            n_heads=self.n_heads,
            attn_modules=self.attn_modules,
        )
        self.sum_kl += sum_kl
        self.n_distributions += n_distributions

    @override
    def compute(self) -> Float[Tensor, ""]:
        sum_kl = all_reduce(self.sum_kl, op=ReduceOp.SUM)
        n_distributions = all_reduce(self.n_distributions, op=ReduceOp.SUM)
        return _attn_patterns_recon_loss_compute(sum_kl, n_distributions)


# --- Stochastic variant ---


class StochasticAttnPatternsReconLoss(Metric):
    """Attention pattern reconstruction loss with stochastic masks."""

    metric_section: ClassVar[str] = "loss"

    def __init__(
        self,
        model: ComponentModel,
        device: str,
        sampling: SamplingType,
        use_delta_component: bool,
        n_mask_samples: int,
        n_heads: int,
        q_proj_path: str | None,
        k_proj_path: str | None,
        c_attn_path: str | None,
    ) -> None:
        self.model = model
        self.sampling: SamplingType = sampling
        self.use_delta_component = use_delta_component
        self.n_mask_samples = n_mask_samples
        self.n_heads = n_heads
        self.q_paths, self.k_paths, self.is_combined = _resolve_qk_paths(
            model, q_proj_path, k_proj_path, c_attn_path
        )
        self.attn_modules = _resolve_attn_modules(model, self.q_paths)
        self.sum_kl = torch.tensor(0.0, device=device)
        self.n_distributions = torch.tensor(0, device=device)

    @override
    def update(
        self,
        *,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        pre_weight_acts: dict[str, Float[Tensor, "..."]],
        ci: CIOutputs,
        weight_deltas: dict[str, Float[Tensor, "d_out d_in"]],
        **_: Any,
    ) -> None:
        mask_infos_list = [
            calc_stochastic_component_mask_info(
                causal_importances=ci.lower_leaky,
                component_mask_sampling=self.sampling,
                weight_deltas=weight_deltas if self.use_delta_component else None,
                router=AllLayersRouter(),
            )
            for _ in range(self.n_mask_samples)
        ]
        sum_kl, n_distributions = _attn_patterns_recon_loss_update(
            model=self.model,
            batch=batch,
            pre_weight_acts=pre_weight_acts,
            mask_infos_list=mask_infos_list,
            q_paths=self.q_paths,
            k_paths=self.k_paths,
            is_combined=self.is_combined,
            n_heads=self.n_heads,
            attn_modules=self.attn_modules,
        )
        self.sum_kl += sum_kl
        self.n_distributions += n_distributions

    @override
    def compute(self) -> Float[Tensor, ""]:
        sum_kl = all_reduce(self.sum_kl, op=ReduceOp.SUM)
        n_distributions = all_reduce(self.n_distributions, op=ReduceOp.SUM)
        return _attn_patterns_recon_loss_compute(sum_kl, n_distributions)
