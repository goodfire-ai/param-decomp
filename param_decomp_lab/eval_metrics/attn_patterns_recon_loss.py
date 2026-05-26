import math
from abc import ABC
from fnmatch import fnmatch
from typing import Literal, Self, override

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from pydantic import model_validator
from torch import Tensor, nn
from torch.distributed import ReduceOp

from param_decomp.base_config import BaseConfig
from param_decomp.component_model import ComponentModel
from param_decomp.distributed import all_reduce
from param_decomp.masks import (
    AllLayersRouter,
    ComponentsMaskInfo,
    calc_stochastic_component_mask_info,
    make_mask_infos,
)
from param_decomp.metrics.base import Metric, MetricResult
from param_decomp.metrics.context import MetricContext


class _AttnPatternsBaseConfig(BaseConfig):
    """Shared config for attention-pattern recon metrics.

    Supports standard attention and RoPE (auto-detected from the parent attention
    module). ALiBi / QK-norm / sliding window are not supported.

    Either `(q_proj_path, k_proj_path)` or `c_attn_path` must be set (combined QKV with
    output split as `[Q | K | V]` along the last dim) — not both, not neither.
    """

    n_heads: int
    q_proj_path: str | None = None
    k_proj_path: str | None = None
    c_attn_path: str | None = None

    @model_validator(mode="after")
    def _validate_paths(self) -> Self:
        has_separate = self.q_proj_path is not None and self.k_proj_path is not None
        has_combined = self.c_attn_path is not None
        assert has_separate != has_combined, (
            "Specify either (q_proj_path, k_proj_path) or c_attn_path, not both/neither"
        )
        return self


class CIMaskedAttnPatternsReconLossConfig(_AttnPatternsBaseConfig):
    type: Literal["CIMaskedAttnPatternsReconLoss"] = "CIMaskedAttnPatternsReconLoss"


class StochasticAttnPatternsReconLossConfig(_AttnPatternsBaseConfig):
    type: Literal["StochasticAttnPatternsReconLoss"] = "StochasticAttnPatternsReconLoss"


def _resolve_paths(pattern: str, model: ComponentModel) -> list[str]:
    matches = [p for p in model.target_module_paths if fnmatch(p, pattern)]
    assert matches, f"Pattern {pattern!r} matched no target module paths"
    return sorted(matches)


def _resolve_qk_paths(
    model: ComponentModel,
    q_proj_path: str | None,
    k_proj_path: str | None,
    c_attn_path: str | None,
) -> tuple[list[str], list[str], bool]:
    if c_attn_path is not None:
        paths = _resolve_paths(c_attn_path, model)
        return paths, paths, True
    assert q_proj_path is not None and k_proj_path is not None
    q_paths = _resolve_paths(q_proj_path, model)
    k_paths = _resolve_paths(k_proj_path, model)
    assert len(q_paths) == len(k_paths), f"Q/K path counts differ: {len(q_paths)} vs {len(k_paths)}"
    return q_paths, k_paths, False


def _resolve_attn_modules(model: ComponentModel, q_paths: list[str]) -> list[nn.Module | None]:
    result: list[nn.Module | None] = []
    for q_path in q_paths:
        parent_path = q_path.rsplit(".", 1)[0]
        attn_module = model.target_model.get_submodule(parent_path)
        result.append(attn_module if hasattr(attn_module, "apply_rotary_pos_emb") else None)
    return result


def _compute_attn_patterns(
    q: Float[Tensor, "batch seq d"],
    k: Float[Tensor, "batch seq d"],
    n_heads: int,
    attn_module: nn.Module | None,
) -> Float[Tensor, "batch n_heads seq seq"]:
    B, S, D = q.shape
    head_dim = D // n_heads
    q = q.view(B, S, n_heads, head_dim).transpose(1, 2)
    n_kv_heads = k.shape[-1] // head_dim
    assert n_heads % n_kv_heads == 0, (
        f"n_heads ({n_heads}) must be a multiple of n_kv_heads ({n_kv_heads})"
    )
    k = k.view(B, S, n_kv_heads, head_dim).transpose(1, 2)
    if n_kv_heads != n_heads:
        k = k.repeat_interleave(n_heads // n_kv_heads, dim=1)
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

    device = next(iter(pre_weight_acts.values())).device
    sum_kl = torch.zeros((), device=device)
    n_distributions = 0
    for mask_infos in mask_infos_list:
        comp_cache = model(batch, mask_infos=mask_infos, cache_type="input").cache
        for i, (q_path, k_path) in enumerate(zip(q_paths, k_paths, strict=True)):
            if is_combined:
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
            kl = F.kl_div(
                masked_patterns.clamp(min=1e-12).log(),
                target_patterns[i],
                reduction="sum",
            )
            sum_kl = sum_kl + kl
            n_distributions += target_patterns[i].shape[0] * n_heads * target_patterns[i].shape[2]
    return sum_kl, n_distributions


class _AttnPatternsBase(Metric[_AttnPatternsBaseConfig], ABC):
    """Shared bind/reset/accumulate/compute for both attn-pattern metrics.

    Accumulates per-distribution KLs and a count of distributions across all matched
    attention layers; `compute` returns the mean.
    """

    @override
    def bind(self, *, model: ComponentModel, device: str) -> None:
        super().bind(model=model, device=device)
        self.n_heads = self.cfg.n_heads
        self.q_paths, self.k_paths, self.is_combined = _resolve_qk_paths(
            model, self.cfg.q_proj_path, self.cfg.k_proj_path, self.cfg.c_attn_path
        )
        self.attn_modules = _resolve_attn_modules(model, self.q_paths)

    @override
    def reset(self) -> None:
        self.sum_kl = torch.zeros((), device=self.device)
        self.n_distributions = torch.zeros((), device=self.device, dtype=torch.long)

    def _accumulate(self, sum_kl: Float[Tensor, ""], n: int) -> Float[Tensor, ""]:
        self.sum_kl += sum_kl.detach()
        self.n_distributions += n
        return sum_kl / n

    @override
    def compute(self) -> MetricResult:
        sum_kl = all_reduce(self.sum_kl, op=ReduceOp.SUM)
        n_distributions = all_reduce(self.n_distributions, op=ReduceOp.SUM)
        return sum_kl / n_distributions


class CIMaskedAttnPatternsReconLoss(_AttnPatternsBase):
    """Attention pattern KL between CI-masked components and full target components."""

    log_namespace = "loss"
    short_name = "CIAttnRecon"

    @override
    def update(self, ctx: MetricContext) -> Tensor:
        mask_infos = make_mask_infos(ctx.ci.lower_leaky, weight_deltas_and_masks=None)
        sum_kl, n = _attn_patterns_recon_loss_update(
            model=self.model,
            batch=ctx.batch,
            pre_weight_acts=ctx.pre_weight_acts,
            mask_infos_list=[mask_infos],
            q_paths=self.q_paths,
            k_paths=self.k_paths,
            is_combined=self.is_combined,
            n_heads=self.n_heads,
            attn_modules=self.attn_modules,
        )
        return self._accumulate(sum_kl, n)


class StochasticAttnPatternsReconLoss(_AttnPatternsBase):
    """Attention pattern KL between stochastically-masked components and full ones.

    Averages over `ctx.n_mask_samples` mask draws per batch.
    """

    log_namespace = "loss"
    short_name = "StochAttnRecon"

    @override
    def update(self, ctx: MetricContext) -> Tensor:
        wd = ctx.weight_deltas if ctx.use_delta_component else None
        mask_infos_list = [
            calc_stochastic_component_mask_info(
                causal_importances=ctx.ci.lower_leaky,
                component_mask_sampling=ctx.sampling,
                weight_deltas=wd,
                router=AllLayersRouter(),
            )
            for _ in range(ctx.n_mask_samples)
        ]
        sum_kl, n = _attn_patterns_recon_loss_update(
            model=self.model,
            batch=ctx.batch,
            pre_weight_acts=ctx.pre_weight_acts,
            mask_infos_list=mask_infos_list,
            q_paths=self.q_paths,
            k_paths=self.k_paths,
            is_combined=self.is_combined,
            n_heads=self.n_heads,
            attn_modules=self.attn_modules,
        )
        return self._accumulate(sum_kl, n)
