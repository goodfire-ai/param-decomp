"""Minimal single-file Parameter Decomposition implementation.

The method itself has zero dependencies on the `param_decomp` package.
Sibling entry points wire it up to specific target models:

  - `pile_4L.py`         — VPD paper's 4-layer LlamaSimpleMLP on the Pile
  - `simplestories_4L.py` — 2-layer LlamaSimpleMLP on SimpleStories

Launch via `torchrun -m` (the entry points use relative imports, so they
must be run as modules from the repo root, not as scripts):

    # 8-GPU single-node
    torchrun --standalone --nproc_per_node=8 -m nano_param_decomp.pile_4L
    torchrun --standalone --nproc_per_node=8 -m nano_param_decomp.simplestories_4L

    # Single-GPU smoke test
    python -m nano_param_decomp.pile_4L

The file is structured for paper readers — everything the method needs is here:

  A. Config (method hyperparameters)
  B. Leaky-hard sigmoids (straight-through for `lower_leaky`)
  C. ComponentLinear wrapper that replaces each target `nn.Linear`
  D. CI transformer (the `global_shared_transformer` causal-importance function)
  E. Losses + mask/routing sampling
  F. Persistent PGD (adversarial sources persisted across steps)
  G. LR schedule
  H. Distributed setup + SPDModule container
  I. Training loop (faithfulness warmup + main loop)
  J. Eval metrics (CI L0, CE/KL with various mask strategies, hidden-acts recon, PGD recon)

The `decompose` entry point takes a target model plus train and eval data iterators
(yielding `[B, S]` int64 tensors of token ids); the entry-point files own the
dataset/tokenization specifics.

What this file deliberately does NOT support:

  - checkpointing, W&B workspace views
  - alternative PPGD scopes, routing types, sigmoid types, CI fn types
  - loss classes other than the four used by the 4L run
  - YAML parsing / experiment registry / sweep machinery
  - models that don't take a sequence of tokens as input

LM-specific assumptions: the target takes integer `input_ids` and returns
logits over a vocab; reconstruction loss is KL on logits and the eval reports
next-token CE. The rest of the method (ComponentLinear, CI transformer, PPGD,
losses) only requires that each decomposed `nn.Linear` sees a `[B, S, d_in]`
input, so any sequential target would work in principle — but the eval metrics
are written for tokenized LM data.
"""

# nn.Module buffer attribute access is typed as `Tensor | Module` by basedpyright; suppress.
# pyright: reportIndexIssue=false, reportArgumentType=false, reportOperatorIssue=false, reportUnnecessaryComparison=false

import math
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal, cast, override

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel

# =============================================================================
# Section A: Config
# =============================================================================


@dataclass
class Config:
    """Configuration for the Parameter Decomposition method.

    Defaults are set to the values used by the pile-4L decomposition in the VPD paper.
    """

    # Maps each `nn.Linear` submodule path to its component count. Required; tied to the
    # specific target model.
    C_per_module: dict[str, int]

    # Training schedule
    n_steps: int = 400_000
    batch_size: int = 64  # global batch (divided across ranks)
    seq_len: int = 512
    seed: int = 0

    # Main optimizer (AdamW over components + CI transformer)
    main_lr: float = 5e-5
    main_lr_final_frac: float = 0.1

    # Faithfulness warmup (AdamW over components only; runs before main loop)
    faithfulness_warmup_steps: int = 400
    faithfulness_warmup_lr: float = 1e-3

    # Loss coefficients
    coeff_faith: float = 1e7
    coeff_imp: float = 2e-4
    coeff_stoch: float = 0.5
    coeff_ppgd: float = 0.5

    # Importance minimality (L_p with linear p-anneal, plus log description term)
    p_start: float = 2.0
    p_end: float = 0.4
    imp_eps: float = 1e-12
    imp_beta: float = 0.5

    # Leaky-hard sigmoid slope outside [0, 1]
    leaky_alpha: float = 0.01

    # CI transformer (global_shared_transformer)
    ci_d_model: int = 2048
    ci_n_blocks: int = 8
    ci_n_heads: int = 16
    ci_mlp_hidden: int = 8192
    ci_rope_base: float = 10000.0

    # Persistent PGD (per_batch_per_position scope, Adam)
    ppgd_lr: float = 0.01
    ppgd_lr_final_frac: float = 0.1
    ppgd_warmup_pct: float = 0.025
    ppgd_beta1: float = 0.5
    ppgd_beta2: float = 0.99
    ppgd_eps: float = 1e-8
    ppgd_inner_steps: int = 2

    # Gradient clipping
    grad_clip_components: float = 0.01

    # Evaluation
    eval_freq: int = 1000
    slow_eval_freq: int = 10000
    slow_eval_on_first_step: bool = True
    eval_batch_size: int = 128
    ci_alive_threshold: float = 0.0
    rounding_threshold: float = 0.0
    pgd_eval_step_size: float = 0.1
    pgd_eval_n_steps: int = 20

    # Logging
    log_every: int = 200
    use_wandb: bool = False
    wandb_project: str = "param-decomp"
    wandb_run_name: str | None = None


# =============================================================================
# Section B: Leaky-hard sigmoids
# =============================================================================


class _LowerLeakyHardSigmoid(torch.autograd.Function):
    """Forward: `clamp(x, 0, 1)`. Backward: pass-through on `(0, 1)`; in the `x <= 0` region,
    only return `alpha * grad_output` when `grad_output < 0` (i.e. gradient wants `y` to
    increase — this can 'resurrect' a dead component). When `grad_output >= 0` in that
    region, return 0 — no point pushing `x` further negative when `y` is already saturated.
    Above 1, gradient is blocked.
    """

    @staticmethod
    @override
    def forward(ctx: Any, x: Tensor, alpha: float) -> Tensor:  # type: ignore[override]
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return x.clamp(0.0, 1.0)

    @staticmethod
    @override
    def backward(ctx: Any, *grad_outputs: Tensor) -> tuple[Tensor, None]:  # type: ignore[override]
        grad_output = grad_outputs[0]
        (x,) = ctx.saved_tensors
        alpha: float = ctx.alpha
        zero = torch.zeros_like(grad_output)
        grad = torch.where(
            x <= 0,
            torch.where(grad_output < 0, alpha * grad_output, zero),
            torch.where(x <= 1, grad_output, zero),
        )
        return grad, None


def lower_leaky(x: Tensor, alpha: float) -> Tensor:
    return cast(Tensor, _LowerLeakyHardSigmoid.apply(x, alpha))


def upper_leaky(x: Tensor, alpha: float) -> Tensor:
    """For `x > 1` return `1 + alpha*(x-1)` (linear continuation, grad alpha); otherwise `clamp(x, 0, 1)`.
    Native autograd gives the correct profile (0 below 0, 1 in-range, alpha above 1), so no
    custom Function is needed.
    """
    return torch.where(x > 1, 1 + alpha * (x - 1), x.clamp(0.0, 1.0))


# =============================================================================
# Section C: ComponentLinear wrapper + install helper
# =============================================================================


class ComponentLinear(nn.Module):
    """Replaces one `nn.Linear` in the target model.

    Parameters:
        V: [d_in, C], the component basis in input space.
        U: [C, d_out], the component output transforms.
    The original weight and bias are kept as frozen buffers (`W_target`, `bias`).

    Two forward modes:
        "target":     returns F.linear(x, W_target, bias) and caches x for the CI function.
        "component":  returns the masked component output, optionally routed against the
                      target output per (batch, pos) by `routing_mask`.

    In "component" mode the caller sets `self.mask` ([B, S, C]), `self.delta_mask` ([B, S]),
    and optionally `self.routing_mask` ([B, S], bool). `None` routing_mask means "component
    everywhere".

    The weight delta is `W_target - (V @ U).T`; it is masked per-position by the scalar
    `delta_mask`. The delta acts as an extra "spillover" component that absorbs whatever
    `V @ U` doesn't yet explain.
    """

    def __init__(self, linear: nn.Linear, C: int) -> None:
        super().__init__()
        d_out, d_in = linear.weight.shape
        self.C = C
        self.register_buffer("W_target", linear.weight.detach().clone())
        # Stubs type `nn.Linear.bias` as `Parameter`, but at runtime it is None when `bias=False`.
        linear_bias = cast(Tensor | None, linear.bias)
        self.register_buffer(
            "bias", linear_bias.detach().clone() if linear_bias is not None else None
        )
        self.V = nn.Parameter(torch.empty(d_in, C).normal_(0.0, 1.0 / math.sqrt(d_in)))
        self.U = nn.Parameter(torch.empty(C, d_out).normal_(0.0, 1.0 / math.sqrt(C)))
        # Transient state set per forward from the training loop
        self.mode: Literal["target", "component"] = "target"
        self.mask: Tensor | None = None
        self.delta_mask: Tensor | None = None
        self.routing_mask: Tensor | None = None
        self.last_input: Tensor | None = None
        self.cache_output: bool = False
        self.last_output: Tensor | None = None

    def weight_delta(self) -> Tensor:
        return self.W_target - (self.V @ self.U).T

    @override
    def forward(self, x: Tensor) -> Tensor:
        if self.mode == "target":
            self.last_input = x.detach()
            out = F.linear(x, self.W_target, self.bias)
            if self.cache_output:
                self.last_output = out.detach()
            return out
        assert self.mask is not None and self.delta_mask is not None
        comp_acts = x @ self.V  # [B, S, C]
        comp_out = (comp_acts * self.mask) @ self.U
        if self.bias is not None:
            comp_out = comp_out + self.bias
        delta_out = F.linear(x, self.weight_delta())  # [B, S, d_out]
        comp_out = comp_out + self.delta_mask.unsqueeze(-1) * delta_out
        if self.routing_mask is not None:
            target_out = F.linear(x, self.W_target, self.bias)
            comp_out = torch.where(self.routing_mask.unsqueeze(-1), comp_out, target_out)
        if self.cache_output:
            self.last_output = comp_out.detach()
        return comp_out


def install_components(model: nn.Module, module_to_c: dict[str, int]) -> dict[str, ComponentLinear]:
    """Freeze every target parameter and replace each listed `nn.Linear` in place."""
    for p in model.parameters():
        p.requires_grad_(False)
    wrappers: dict[str, ComponentLinear] = {}
    for path, C in module_to_c.items():
        parent_path, _, attr = path.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        linear = model.get_submodule(path)
        assert isinstance(linear, nn.Linear), f"{path} is not nn.Linear: {type(linear)}"
        wrapper = ComponentLinear(linear, C)
        setattr(parent, attr, wrapper)
        wrappers[path] = wrapper
    return wrappers


# =============================================================================
# Section D: CI transformer (global_shared_transformer)
# =============================================================================


def precompute_rope(
    seq_len: int, head_dim: int, base: float, device: torch.device
) -> tuple[Tensor, Tensor]:
    assert head_dim % 2 == 0
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)  # [seq_len, half]
    return freqs.cos(), freqs.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Split-in-half RoPE. x: [B, H, S, head_dim]; cos/sin: [S, half]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos[None, None, :, :].to(x.dtype)
    sin = sin[None, None, :, :].to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class CIAttention(nn.Module):
    """Bidirectional multi-head self-attention with RoPE on Q, K."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    @override
    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).reshape(B, S, self.n_heads * self.head_dim)
        return self.o_proj(out)


class CIBlock(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.attn = CIAttention(cfg.ci_d_model, cfg.ci_n_heads)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.ci_d_model, cfg.ci_mlp_hidden),
            nn.GELU(),
            nn.Linear(cfg.ci_mlp_hidden, cfg.ci_d_model),
        )

    @override
    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.attn(F.rms_norm(x, (x.shape[-1],)), cos, sin)
        x = x + self.mlp(F.rms_norm(x, (x.shape[-1],)))
        return x


class CITransformer(nn.Module):
    """The causal-importance function: a shared transformer that sees all layers at once.

    Inputs: dict of pre-weight activations (one entry per target module, shape [B, S, d_in_m]).
    Each is RMS-normed (no learned scale) and concatenated along the feature dim; a linear
    projection maps the concatenation to d_model, `n_blocks` transformer blocks run over the
    sequence, and a second linear projection maps back to the total component dimension.
    The output is split per module and passed through the two leaky-hard sigmoids.

    Modules are concatenated in alphabetical-path order (see `module_order`), so the layout
    of `proj_in` columns / `proj_out` rows is independent of the input dict's iteration order.
    """

    def __init__(
        self, d_in_per_module: dict[str, int], c_per_module: dict[str, int], cfg: Config
    ) -> None:
        super().__init__()
        self.module_order = sorted(d_in_per_module.keys())
        self.cfg = cfg
        total_in = sum(d_in_per_module.values())
        total_C = sum(c_per_module[name] for name in self.module_order)
        self.proj_in = nn.Linear(total_in, cfg.ci_d_model)
        self.blocks = nn.ModuleList([CIBlock(cfg) for _ in range(cfg.ci_n_blocks)])
        self.proj_out = nn.Linear(cfg.ci_d_model, total_C)
        self.c_splits: list[int] = [c_per_module[n] for n in self.module_order]
        head_dim = cfg.ci_d_model // cfg.ci_n_heads
        cos, sin = precompute_rope(cfg.seq_len, head_dim, cfg.ci_rope_base, torch.device("cpu"))
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    @override
    def forward(
        self, acts: dict[str, Tensor]
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
        normed = [F.rms_norm(acts[n], (acts[n].shape[-1],)) for n in self.module_order]
        x = torch.cat(normed, dim=-1)
        x = self.proj_in(x)
        S = x.shape[1]
        cos, sin = self.rope_cos[:S], self.rope_sin[:S]
        for block in self.blocks:
            x = block(x, cos, sin)
        logits = self.proj_out(x)  # [B, S, total_C]
        per_module = dict(zip(self.module_order, logits.split(self.c_splits, dim=-1), strict=True))
        alpha = self.cfg.leaky_alpha
        ci_lower = {n: lower_leaky(v, alpha) for n, v in per_module.items()}
        ci_upper = {n: upper_leaky(v, alpha) for n, v in per_module.items()}
        return ci_lower, ci_upper, per_module


# =============================================================================
# Section E: Losses + mask/routing sampling
# =============================================================================


def faithfulness_loss(wrappers: dict[str, ComponentLinear]) -> Tensor:
    """Sum of squared weight-delta errors divided by total delta element count."""
    sum_sq = torch.zeros((), device=next(iter(wrappers.values())).V.device)
    numel = 0
    for w in wrappers.values():
        delta = w.weight_delta()
        sum_sq = sum_sq + delta.pow(2).sum()
        numel += delta.numel()
    return sum_sq / numel


def anneal_p(step: int, total_steps: int, p_start: float, p_end: float) -> float:
    t = min(max(step / total_steps, 0.0), 1.0)
    return p_start + (p_end - p_start) * t


def importance_minimality_loss(
    ci_upper: dict[str, Tensor], p: float, eps: float, beta: float, world_size: int
) -> Tensor:
    """Per-module: sum_c [ mean[c] + beta * mean[c] * log2(1 + sum[c] * world_size) ].

    `mean` / `sum` are over batch and sequence dims (local to this rank). The `* world_size`
    term rescales the local sum into an estimate of the global per-component total.
    """
    total = torch.zeros((), device=next(iter(ci_upper.values())).device)
    for v in ci_upper.values():
        vals = (v + eps).pow(p)  # [B, S, C]
        batch_seq_dims = tuple(range(vals.ndim - 1))
        sum_c = vals.sum(dim=batch_seq_dims)  # [C]
        n = math.prod(vals.shape[:-1])
        mean_c = sum_c / n
        total = total + (mean_c + beta * mean_c * torch.log2(1 + sum_c * world_size)).sum()
    return total


def kl_logits(pred: Tensor, target: Tensor) -> Tensor:
    """KL(softmax(target) || softmax(pred)) averaged over all non-vocab positions.

    `target` is treated as the reference distribution (equivalent to the original model's
    output); we don't backprop through it.
    """
    log_q = F.log_softmax(pred, dim=-1)
    p = F.softmax(target.detach(), dim=-1)
    kl_per_pos = F.kl_div(log_q, p, reduction="none").sum(dim=-1)
    return kl_per_pos.mean()


def sample_continuous_masks(
    ci_lower: dict[str, Tensor],
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """mask = ci + (1 - ci) * U(0, 1); delta_mask ~ U(0, 1) per (batch, pos)."""
    masks: dict[str, Tensor] = {}
    delta_masks: dict[str, Tensor] = {}
    for name, ci in ci_lower.items():
        u = torch.rand_like(ci)
        masks[name] = ci + (1 - ci) * u
        delta_masks[name] = torch.rand(*ci.shape[:-1], device=ci.device, dtype=ci.dtype)
    return masks, delta_masks


def sample_uniform_k_subset_routing(
    module_names: list[str], batch_dims: tuple[int, ...], device: torch.device
) -> dict[str, Tensor]:
    """For each (batch, pos), sample k ~ Uniform{1..M} and route to a random k-subset."""
    M = len(module_names)
    k = torch.randint(1, M + 1, batch_dims, device=device)  # [*batch_dims]
    noise = torch.rand(M, *batch_dims, device=device)
    ranks = noise.argsort(dim=0).argsort(dim=0)  # [M, *batch_dims]
    return {name: ranks[i] < k for i, name in enumerate(module_names)}


def set_wrapper_masks(
    wrappers: dict[str, ComponentLinear],
    masks: dict[str, Tensor],
    delta_masks: dict[str, Tensor],
    routing: dict[str, Tensor] | None,
) -> None:
    for name, w in wrappers.items():
        w.mode = "component"
        w.mask = masks[name]
        w.delta_mask = delta_masks[name]
        w.routing_mask = None if routing is None else routing[name]


def clear_wrapper_masks(wrappers: dict[str, ComponentLinear]) -> None:
    for w in wrappers.values():
        w.mode = "target"
        w.mask = None
        w.delta_mask = None
        w.routing_mask = None


def stochastic_recon_loss(
    target_model: nn.Module,
    wrappers: dict[str, ComponentLinear],
    input_ids: Tensor,
    target_logits: Tensor,
    ci_lower: dict[str, Tensor],
) -> Tensor:
    """One-sample stochastic-mask reconstruction with uniform-k-subset per-position routing."""
    B, S = input_ids.shape
    masks, delta_masks = sample_continuous_masks(ci_lower)
    routing = sample_uniform_k_subset_routing(list(wrappers), (B, S), input_ids.device)
    set_wrapper_masks(wrappers, masks, delta_masks, routing)
    try:
        pred = target_model(input_ids)
    finally:
        clear_wrapper_masks(wrappers)
    return kl_logits(pred, target_logits)


# =============================================================================
# Section F: Persistent PGD
# =============================================================================


class PersistentPGD:
    """Per-module adversarial sources that persist across training steps.

    Scope is `per_batch_per_position`: sources have shape `[local_B, S, C + 1]` on each
    rank (+1 for the delta component's scalar mask), with no cross-rank sync. We maintain
    Adam state (m, v) alongside each source and a global step counter t.

    Each training step:
      1) `warmup`: run `inner_steps` (=2) PGD updates on the current batch, all-layers routed.
      2) `recon_loss`: forward once with current sources (no mutation) to produce the loss
         term that enters the main total; the caller extracts source grads from this via
         `torch.autograd.grad(loss, sources, retain_graph=True)` before `total.backward()`.
      3) `external_step`: Adam-update sources with the extracted grads after `total.backward()`.
    """

    def __init__(
        self,
        wrappers: dict[str, ComponentLinear],
        local_B: int,
        seq_len: int,
        device: torch.device,
        cfg: Config,
    ) -> None:
        self.cfg = cfg
        self.sources: dict[str, Tensor] = {}
        self.m: dict[str, Tensor] = {}
        self.v: dict[str, Tensor] = {}
        for name, w in wrappers.items():
            shape = (local_B, seq_len, w.C + 1)
            src = torch.rand(shape, device=device).requires_grad_(True)
            self.sources[name] = src
            self.m[name] = torch.zeros(shape, device=device)
            self.v[name] = torch.zeros(shape, device=device)
        self.t = 0

    def _masks_from_sources(
        self, ci_lower: dict[str, Tensor]
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        masks: dict[str, Tensor] = {}
        delta_masks: dict[str, Tensor] = {}
        for name, ci in ci_lower.items():
            s = self.sources[name]
            masks[name] = ci + (1 - ci) * s[..., : ci.shape[-1]]
            delta_masks[name] = s[..., -1]
        return masks, delta_masks

    def _forward_loss(
        self,
        target_model: nn.Module,
        wrappers: dict[str, ComponentLinear],
        input_ids: Tensor,
        target_logits: Tensor,
        ci_lower: dict[str, Tensor],
    ) -> Tensor:
        masks, delta_masks = self._masks_from_sources(ci_lower)
        set_wrapper_masks(wrappers, masks, delta_masks, routing=None)  # all-layers
        try:
            pred = target_model(input_ids)
        finally:
            clear_wrapper_masks(wrappers)
        return kl_logits(pred, target_logits)

    def warmup(
        self,
        target_model: nn.Module,
        wrappers: dict[str, ComponentLinear],
        input_ids: Tensor,
        target_logits: Tensor,
        ci_lower: dict[str, Tensor],
        lr: float,
    ) -> None:
        for _ in range(self.cfg.ppgd_inner_steps):
            loss = self._forward_loss(target_model, wrappers, input_ids, target_logits, ci_lower)
            grads = torch.autograd.grad(loss, list(self.sources.values()), retain_graph=False)
            self._adam_step(dict(zip(self.sources, grads, strict=True)), lr)

    def recon_loss(
        self,
        target_model: nn.Module,
        wrappers: dict[str, ComponentLinear],
        input_ids: Tensor,
        target_logits: Tensor,
        ci_lower: dict[str, Tensor],
    ) -> Tensor:
        return self._forward_loss(target_model, wrappers, input_ids, target_logits, ci_lower)

    def external_step(self, grads: dict[str, Tensor], lr: float) -> None:
        self._adam_step(grads, lr)

    def _adam_step(self, grads: dict[str, Tensor], lr: float) -> None:
        self.t += 1
        bc1 = 1 - self.cfg.ppgd_beta1**self.t
        bc2 = 1 - self.cfg.ppgd_beta2**self.t
        with torch.no_grad():
            for name, src in self.sources.items():
                g = grads[name]
                m, v = self.m[name], self.v[name]
                m.mul_(self.cfg.ppgd_beta1).add_(g, alpha=1 - self.cfg.ppgd_beta1)
                v.mul_(self.cfg.ppgd_beta2).addcmul_(g, g, value=1 - self.cfg.ppgd_beta2)
                src.add_(lr * (m / bc1) / ((v / bc2).sqrt() + self.cfg.ppgd_eps))
                src.clamp_(0.0, 1.0)


# =============================================================================
# Section G: LR schedule
# =============================================================================


def cosine_lr(
    step: int, total: int, start: float, final_frac: float, warmup_pct: float = 0.0
) -> float:
    """Linear warmup from 0 to `start` over `warmup_pct * total` steps, then half-period cosine
    decay from `start` to `start * final_frac`. `progress` reaches 1 at `step == total - 1`."""
    warmup_steps = int(warmup_pct * total)
    decay_steps = total - warmup_steps
    if warmup_steps > 0 and step < warmup_steps:
        return start * (step / warmup_steps)
    if decay_steps <= 1:
        return start
    progress = (step - warmup_steps) / (decay_steps - 1)
    progress = min(max(progress, 0.0), 1.0)
    final = start * final_frac
    return final + 0.5 * (start - final) * (1 + math.cos(math.pi * progress))


# =============================================================================
# Section H: Distributed setup + SPDModule container
# =============================================================================


def init_dist() -> tuple[int, int, int, torch.device]:
    """Returns (rank, world_size, local_rank, device). Falls back to single-process."""
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank, torch.device(f"cuda:{local_rank}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return 0, 1, 0, device


class SPDModule(nn.Module):
    """Container so DDP tracks both target-model component params and CI transformer params.

    `forward(input_ids)` runs the target-only forward (caching pre-weight activations in
    each `ComponentLinear`) and then the CI transformer. The two additional masked forwards
    for `stochastic_recon_loss` and PPGD go through `self.target` directly — DDP's grad sync
    fires on the parameters themselves regardless of which forward visited them.
    """

    def __init__(
        self,
        target: nn.Module,
        ci_fn: CITransformer,
        wrappers: dict[str, ComponentLinear],
    ) -> None:
        super().__init__()
        self.target = target
        self.ci_fn = ci_fn
        self._wrappers = wrappers

    @override
    def forward(
        self, input_ids: Tensor
    ) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
        clear_wrapper_masks(self._wrappers)
        target_logits = self.target(input_ids)
        acts = {n: _require(w.last_input) for n, w in self._wrappers.items()}
        ci_lower, ci_upper, ci_pre_sigmoid = self.ci_fn(acts)
        return target_logits, ci_lower, ci_upper, ci_pre_sigmoid


def _require(x: Tensor | None) -> Tensor:
    assert x is not None
    return x


# =============================================================================
# Section I: Training loop
# =============================================================================


def decompose(
    target_model: nn.Module,
    cfg: Config,
    train_loader: Iterator[Tensor],
    eval_loader: Iterator[Tensor],
) -> None:
    """Decompose `target_model` using SPD. `cfg.C_per_module` names the `nn.Linear` submodules
    to decompose; `target_model.forward(input_ids)` must return logits.

    Currently assumes that `train_loader` / `eval_loader` yield `[local_B, seq_len]` int64 token-id
    tensors, where `local_B = (batch_size or eval_batch_size) // world_size`. They must be sharded
    across rank by the caller — see the entry-point files for the reference Pile / SimpleStories
    loaders."""
    rank, world_size, local_rank, device = init_dist()
    assert cfg.batch_size % world_size == 0, "global batch size must be divisible by world size"
    local_B = cfg.batch_size // world_size

    # Same seed on every rank so V/U/CI params match after init.
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    def _log(msg: str) -> None:
        if rank == 0:
            print(f"[rank0] {msg}", flush=True)

    target_model.eval()
    wrappers = install_components(target_model, cfg.C_per_module)
    _log(f"installed {len(wrappers)} components")
    d_in_per_module: dict[str, int] = {
        name: int(w.W_target.shape[1]) for name, w in wrappers.items()
    }
    ci_fn = CITransformer(d_in_per_module, cfg.C_per_module, cfg)
    _log(f"built CI transformer ({sum(p.numel() for p in ci_fn.parameters()):,} params)")
    target_model.to(device)
    ci_fn.to(device)
    _log(f"moved models to {device}")

    # Faithfulness warmup — pre-DDP. Gradients are a deterministic function of W_target,
    # V, and U, all identical across ranks, so no sync is needed.
    component_params = [p for w in wrappers.values() for p in (w.V, w.U)]
    warmup_opt = torch.optim.AdamW(
        component_params, lr=cfg.faithfulness_warmup_lr, weight_decay=0.0
    )
    _log(f"starting faithfulness warmup ({cfg.faithfulness_warmup_steps} steps)")
    for _ in range(cfg.faithfulness_warmup_steps):
        warmup_opt.zero_grad()
        faithfulness_loss(wrappers).backward()
        warmup_opt.step()
    _log("faithfulness warmup done")

    # Now divergence between ranks is OK — seed per-rank for data + PPGD + sampling streams.
    torch.manual_seed(cfg.seed + rank)
    torch.cuda.manual_seed_all(cfg.seed + rank)

    spd = SPDModule(target_model, ci_fn, wrappers).to(device)
    spd_wrapped: nn.Module
    if world_size > 1:
        spd_wrapped = DistributedDataParallel(
            spd, device_ids=[local_rank], output_device=local_rank
        )
    else:
        spd_wrapped = spd
    _log("DDP wrap complete")

    ppgd = PersistentPGD(wrappers, local_B, cfg.seq_len, device, cfg)
    ci_params = list(ci_fn.parameters())
    opt = torch.optim.AdamW(component_params + ci_params, lr=cfg.main_lr, weight_decay=0.0)
    _log("PPGD, optimizer, dataloader ready")

    if rank == 0 and cfg.use_wandb:
        import wandb  # type: ignore[import-untyped]

        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name)
        _log(f"wandb url: {wandb.run.url if wandb.run else '?'}")

    for step in range(cfg.n_steps):
        main_lr = cosine_lr(step, cfg.n_steps, cfg.main_lr, cfg.main_lr_final_frac)
        ppgd_lr = cosine_lr(
            step, cfg.n_steps, cfg.ppgd_lr, cfg.ppgd_lr_final_frac, cfg.ppgd_warmup_pct
        )
        for g in opt.param_groups:
            g["lr"] = main_lr

        input_ids = next(train_loader).to(device)

        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            target_logits, ci_lower, ci_upper, _ci_pre = spd_wrapped(input_ids)

            ppgd.warmup(target_model, wrappers, input_ids, target_logits, ci_lower, lr=ppgd_lr)

            loss_faith = faithfulness_loss(wrappers)
            loss_imp = importance_minimality_loss(
                ci_upper,
                anneal_p(step, cfg.n_steps, cfg.p_start, cfg.p_end),
                cfg.imp_eps,
                cfg.imp_beta,
                world_size,
            )
            loss_stoch = stochastic_recon_loss(
                target_model, wrappers, input_ids, target_logits, ci_lower
            )
            loss_ppgd = ppgd.recon_loss(target_model, wrappers, input_ids, target_logits, ci_lower)

        # Total-loss summation runs outside autocast so the coeff*loss sum stays in fp32.
        total = (
            cfg.coeff_faith * loss_faith
            + cfg.coeff_imp * loss_imp
            + cfg.coeff_stoch * loss_stoch
            + cfg.coeff_ppgd * loss_ppgd
        )

        # Extract PPGD source grads before the main backward. Per-rank, no all-reduce.
        ppgd_grads = torch.autograd.grad(loss_ppgd, list(ppgd.sources.values()), retain_graph=True)
        ppgd_grads_dict = dict(zip(ppgd.sources, ppgd_grads, strict=True))

        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(component_params, cfg.grad_clip_components)
        opt.step()
        ppgd.external_step(ppgd_grads_dict, ppgd_lr)

        is_regular_eval = step % cfg.eval_freq == 0
        is_slow_eval = step % cfg.slow_eval_freq == 0 or (step == 0 and cfg.slow_eval_on_first_step)
        if is_regular_eval or is_slow_eval:
            eval_batch = next(eval_loader).to(device)
            eval_metrics = run_eval(
                target_model,
                ci_fn,
                wrappers,
                cfg,
                world_size,
                eval_batch,
                is_slow=is_slow_eval,
            )
            if rank == 0 and cfg.use_wandb:
                import wandb  # type: ignore[import-untyped]

                to_log: dict[str, Any] = {}
                for k, v in eval_metrics.items():
                    to_log[k] = wandb.Histogram(v.tolist()) if isinstance(v, Tensor) else v
                wandb.log(to_log, step=step)

        if rank == 0 and step % cfg.log_every == 0:
            metrics = {
                "loss/faith": loss_faith.detach().item(),
                "loss/imp": loss_imp.detach().item(),
                "loss/stoch": loss_stoch.detach().item(),
                "loss/ppgd": loss_ppgd.detach().item(),
                "lr/main": main_lr,
                "lr/ppgd": ppgd_lr,
                "step": step,
            }
            if cfg.use_wandb:
                import wandb  # type: ignore[import-untyped]

                wandb.log(metrics, step=step)
            else:
                print(
                    " ".join(
                        f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in metrics.items()
                    ),
                    flush=True,
                )

    if world_size > 1:
        dist.destroy_process_group()


# =============================================================================
# Section J: Eval metrics
# =============================================================================
# Each function runs on a single eval batch and returns a `dict[str, float | Tensor]`.
# 1D tensors are converted to `wandb.Histogram` at log time; scalars are logged as floats.
# Cross-rank reduction is `dist.ReduceOp.AVG` for scalar metrics and `all_gather` for
# histograms.


def _all_reduce_mean(t: Tensor, world_size: int) -> Tensor:
    """Mean-reduce a tensor across ranks, in place. No-op for single-process runs."""
    if world_size > 1:
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return t


def _ce_next_token(logits: Tensor, input_ids: Tensor) -> float:
    """CE loss over shifted next-token targets. Masks the first position of each batch item with
    -100 so the boundary between batch items does not contribute a fake transition."""
    masked = input_ids.clone()
    masked[:, 0] = -100
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = masked.reshape(-1)
    return F.cross_entropy(flat_logits[:-1], flat_labels[1:], ignore_index=-100).item()


def eval_ci_l0(ci_lower: dict[str, Tensor], threshold: float, world_size: int) -> dict[str, float]:
    """Per-module mean count of components with `ci > threshold` per position."""
    out: dict[str, float] = {}
    for name, ci in ci_lower.items():
        l0 = (ci > threshold).float().sum(-1).mean()
        l0 = _all_reduce_mean(l0.clone(), world_size)
        out[f"eval/l0/{name}"] = l0.item()
    return out


def _forward_with_masks(
    target_model: nn.Module,
    wrappers: dict[str, ComponentLinear],
    input_ids: Tensor,
    masks: dict[str, Tensor],
    delta_masks: dict[str, Tensor],
) -> Tensor:
    """All-layers routed forward with explicit per-module masks. Returns logits."""
    set_wrapper_masks(wrappers, masks, delta_masks, routing=None)
    try:
        return target_model(input_ids)
    finally:
        clear_wrapper_masks(wrappers)


def eval_ce_kl_losses(
    target_model: nn.Module,
    wrappers: dict[str, ComponentLinear],
    input_ids: Tensor,
    target_logits: Tensor,
    ci_lower: dict[str, Tensor],
    rounding_threshold: float,
    world_size: int,
) -> dict[str, float]:
    """Six forward passes with different mask strategies; compute KL (vs target) and CE (vs
    next-token) for each. Only the `stoch_masked` strategy routes through the weight delta
    (with U(0,1) mask); the other five pass `delta_mask=0` to disable the delta component.
    """
    B, S = input_ids.shape
    device = input_ids.device
    zeros_bs = torch.zeros(B, S, device=device)

    def strat_ci():
        return {n: ci for n, ci in ci_lower.items()}, {n: zeros_bs for n in ci_lower}

    def strat_stoch():
        return sample_continuous_masks(ci_lower)

    def strat_unmasked():
        return {n: torch.ones_like(c) for n, c in ci_lower.items()}, {n: zeros_bs for n in ci_lower}

    def strat_random():
        return {n: torch.rand_like(c) for n, c in ci_lower.items()}, {n: zeros_bs for n in ci_lower}

    def strat_rounded():
        return (
            {n: (c > rounding_threshold).to(c.dtype) for n, c in ci_lower.items()},
            {n: zeros_bs for n in ci_lower},
        )

    def strat_zero():
        return {n: torch.zeros_like(c) for n, c in ci_lower.items()}, {
            n: zeros_bs for n in ci_lower
        }

    strategies = {
        "ci_masked": strat_ci,
        "unmasked": strat_unmasked,
        "stoch_masked": strat_stoch,
        "random_masked": strat_random,
        "rounded_masked": strat_rounded,
        "zero_masked": strat_zero,
    }
    kls: dict[str, float] = {}
    ces: dict[str, float] = {}
    for name, build in strategies.items():
        masks, delta_masks = build()
        logits = _forward_with_masks(target_model, wrappers, input_ids, masks, delta_masks)
        kls[name] = kl_logits(logits, target_logits).item()
        ces[name] = _ce_next_token(logits, input_ids)

    target_ce = _ce_next_token(target_logits, input_ids)
    zero_ce = ces["zero_masked"]

    out: dict[str, float] = {}
    for name in strategies:
        out[f"eval/ce_kl/kl_{name}"] = _all_reduce_mean(
            torch.tensor(kls[name], device=device), world_size
        ).item()
    for name in [k for k in strategies if k != "zero_masked"]:
        ce_diff = ces[name] - target_ce
        out[f"eval/ce_kl/ce_difference_{name}"] = _all_reduce_mean(
            torch.tensor(ce_diff, device=device), world_size
        ).item()
        denom = zero_ce - target_ce
        unrecov = ce_diff / denom if denom != 0 else float("nan")
        out[f"eval/ce_kl/ce_unrecovered_{name}"] = _all_reduce_mean(
            torch.tensor(unrecov, device=device), world_size
        ).item()
    return out


def eval_pgd_recon(
    target_model: nn.Module,
    wrappers: dict[str, ComponentLinear],
    input_ids: Tensor,
    target_logits: Tensor,
    ci_lower: dict[str, Tensor],
    step_size: float,
    n_steps: int,
    world_size: int,
) -> dict[str, float]:
    """Adversarially find high-loss masks via sign-SGD PGD, then report the resulting recon
    loss. One source of shape [1, 1, C+1] per module is shared across the batch — broadcast
    from rank 0 at init and gradient-averaged across ranks before each step.
    """
    B, S = input_ids.shape
    device = input_ids.device
    sources: dict[str, Tensor] = {}
    for name, w in wrappers.items():
        src = torch.rand((1, 1, w.C + 1), device=device)
        if world_size > 1:
            dist.broadcast(src, src=0)
        sources[name] = src.requires_grad_(True)

    def build_masks() -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        masks: dict[str, Tensor] = {}
        delta_masks: dict[str, Tensor] = {}
        for name, ci in ci_lower.items():
            s_expanded = sources[name].expand(B, S, -1)
            masks[name] = ci + (1 - ci) * s_expanded[..., :-1]
            delta_masks[name] = s_expanded[..., -1]
        return masks, delta_masks

    with torch.enable_grad():
        for _ in range(n_steps):
            masks, delta_masks = build_masks()
            set_wrapper_masks(wrappers, masks, delta_masks, routing=None)
            try:
                logits = target_model(input_ids)
            finally:
                clear_wrapper_masks(wrappers)
            loss = kl_logits(logits, target_logits)
            grads = torch.autograd.grad(loss, list(sources.values()))
            with torch.no_grad():
                for name, g in zip(sources, grads, strict=True):
                    if world_size > 1:
                        dist.all_reduce(g, op=dist.ReduceOp.AVG)
                    sources[name].add_(step_size * g.sign())
                    sources[name].clamp_(0.0, 1.0)

        masks, delta_masks = build_masks()
        set_wrapper_masks(wrappers, masks, delta_masks, routing=None)
        try:
            logits = target_model(input_ids)
        finally:
            clear_wrapper_masks(wrappers)
        final_loss = kl_logits(logits, target_logits)

    return {"eval/pgd_recon_loss": _all_reduce_mean(final_loss.detach().clone(), world_size).item()}


def eval_component_activation_density(
    ci_lower: dict[str, Tensor], threshold: float, world_size: int
) -> dict[str, Tensor]:
    """Per-component fraction of (batch, pos) where ci > threshold. Returned as 1D per-module tensors
    (one entry per component); logged as `wandb.Histogram` at the call site."""
    out: dict[str, Tensor] = {}
    for name, ci in ci_lower.items():
        B, S = ci.shape[:2]
        active = (ci > threshold).to(ci.dtype).sum(dim=(0, 1)) / (B * S)
        out[f"eval/density/{name}"] = _all_reduce_mean(active.clone(), world_size).cpu()
    return out


def eval_ci_mean_per_component(ci_lower: dict[str, Tensor], world_size: int) -> dict[str, Tensor]:
    """Per-module per-component mean CI value; 1D tensor per module."""
    out: dict[str, Tensor] = {}
    for name, ci in ci_lower.items():
        mean_c = ci.mean(dim=(0, 1))
        out[f"eval/ci_mean/{name}"] = _all_reduce_mean(mean_c.clone(), world_size).cpu()
    return out


def eval_ci_histograms(
    ci_lower: dict[str, Tensor],
    ci_pre_sigmoid: dict[str, Tensor],
    world_size: int,
) -> dict[str, Tensor]:
    """Flattened CI value distributions (both post- and pre-sigmoid) gathered across ranks."""
    out: dict[str, Tensor] = {}
    for name in ci_lower:
        for prefix, src in [("ci_hist", ci_lower), ("ci_hist_pre_sigmoid", ci_pre_sigmoid)]:
            t = src[name].detach().flatten()
            if world_size > 1:
                buf = torch.empty(world_size * t.numel(), device=t.device, dtype=t.dtype)
                dist.all_gather_into_tensor(buf, t)
                t = buf
            out[f"eval/{prefix}/{name}"] = t.cpu()
    return out


def eval_hidden_acts_recon(
    target_model: nn.Module,
    wrappers: dict[str, ComponentLinear],
    input_ids: Tensor,
    ci_lower: dict[str, Tensor],
    world_size: int,
    *,
    stochastic: bool,
) -> dict[str, float]:
    """Compare per-module outputs between target forward and masked forward. Returns per-module MSEs
    plus a `/total` average.

    Requires `cache_output=True` on wrappers (set and cleared by `run_eval`)."""
    for w in wrappers.values():
        w.mode = "target"
    _ = target_model(input_ids)
    target_acts = {n: _require(w.last_output) for n, w in wrappers.items()}

    if stochastic:
        masks, delta_masks = sample_continuous_masks(ci_lower)
    else:
        B, S = input_ids.shape
        ones_bs = torch.ones(B, S, device=input_ids.device)
        masks = {n: ci for n, ci in ci_lower.items()}
        delta_masks = {n: torch.zeros_like(ones_bs) for n in ci_lower}
    set_wrapper_masks(wrappers, masks, delta_masks, routing=None)
    try:
        _ = target_model(input_ids)
    finally:
        clear_wrapper_masks(wrappers)
    comp_acts = {n: _require(w.last_output) for n, w in wrappers.items()}

    tag = "stoch" if stochastic else "ci"
    per_module: dict[str, float] = {}
    total_sq = torch.zeros((), device=input_ids.device)
    total_n = 0
    for name in target_acts:
        mse = F.mse_loss(comp_acts[name], target_acts[name], reduction="mean")
        mse = _all_reduce_mean(mse.clone(), world_size)
        per_module[name] = mse.item()
        total_sq = total_sq + mse * target_acts[name].numel()
        total_n += target_acts[name].numel()
    out = {f"eval/hidden_recon_{tag}/{n}": v for n, v in per_module.items()}
    out[f"eval/hidden_recon_{tag}/total"] = (total_sq / total_n).item()
    return out


def run_eval(
    target_model: nn.Module,
    ci_fn: CITransformer,
    wrappers: dict[str, ComponentLinear],
    cfg: Config,
    world_size: int,
    eval_batch: Tensor,
    is_slow: bool,
) -> dict[str, float | Tensor]:
    """Compute all applicable eval metrics on one batch. Returns a flat dict of scalars and 1D
    tensors (tensors are converted to `wandb.Histogram` at log time)."""
    metrics: dict[str, float | Tensor] = {}
    for w in wrappers.values():
        w.cache_output = False
    try:
        # Target forward + CI forward (no grad). Uses `last_input` cache populated by target mode.
        with torch.no_grad():
            clear_wrapper_masks(wrappers)
            target_logits = target_model(eval_batch)
            acts = {n: _require(w.last_input) for n, w in wrappers.items()}
            ci_lower, _ci_upper, ci_pre_sigmoid = ci_fn(acts)

            metrics.update(eval_ci_l0(ci_lower, cfg.ci_alive_threshold, world_size))
            metrics.update(
                eval_ce_kl_losses(
                    target_model,
                    wrappers,
                    eval_batch,
                    target_logits,
                    ci_lower,
                    cfg.rounding_threshold,
                    world_size,
                )
            )

            if is_slow:
                metrics.update(
                    eval_component_activation_density(ci_lower, cfg.ci_alive_threshold, world_size)
                )
                metrics.update(eval_ci_mean_per_component(ci_lower, world_size))
                metrics.update(eval_ci_histograms(ci_lower, ci_pre_sigmoid, world_size))
                for w in wrappers.values():
                    w.cache_output = True
                for stoch in (True, False):
                    metrics.update(
                        eval_hidden_acts_recon(
                            target_model,
                            wrappers,
                            eval_batch,
                            ci_lower,
                            world_size,
                            stochastic=stoch,
                        )
                    )
                for w in wrappers.values():
                    w.cache_output = False

        # PGD needs grad through sources.
        metrics.update(
            eval_pgd_recon(
                target_model,
                wrappers,
                eval_batch,
                target_logits,
                ci_lower,
                cfg.pgd_eval_step_size,
                cfg.pgd_eval_n_steps,
                world_size,
            )
        )
    finally:
        for w in wrappers.values():
            w.cache_output = False
            w.last_output = None
        clear_wrapper_masks(wrappers)
    return metrics
