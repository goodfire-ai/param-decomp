"""Minimal single-file *from-scratch* Parameter Decomposition.

Unlike `run.py` (which decomposes a frozen pretrained target), this trains an
interpretable model *from scratch*: there is no target model. Every decomposed
`nn.Linear` is replaced by a pure UV decomposition `out = ((x @ V) * mask) @ U (+ bias)`,
and *all* parameters — components (V/U), the CI transformer, embeddings, and norms —
are trained jointly. The reconstruction objective is next-token cross-entropy, not
KL-vs-target.

Training losses:
  - importance minimality (with the log-description frequency term)
  - next-token CE under stochastic-subset masking (decomposability pressure)
  - next-token CE under adversarial (persistent PGD) masking

The unmasked-forward next-token CE is *eval-only* (reported as `UnmaskedReconLoss`), not a
training term: stochastic-subset routing already trains non-selected layers at mask=1.0, so
the ones-mask anchor is exercised partially without a dedicated term. We watch the eval to
decide whether to add it back.

What's deliberately gone vs `run.py`: the target model, the weight-delta / spillover
component, faithfulness loss + warmup, KL losses, and the target/component mode toggle.

Launch via `torchrun -m` from the repo root (entry points use relative imports):

    torchrun --standalone --nproc_per_node=8 -m nano_param_decomp.from_scratch_simplestories
    python -m nano_param_decomp.from_scratch_simplestories   # single-GPU smoke

File structure (mirrors `run.py` for paper readers):

  A. Config            B. Leaky-hard sigmoids   C. ComponentLinear + build
  D. CI transformer    E. Losses + mask sampling F. Persistent PGD
  G. LR schedule       H. Distributed + SPDModule
  I. Training loop     J. Eval metrics
"""

# nn.Module buffer attribute access is typed as `Tensor | Module` by basedpyright; suppress.
# pyright: reportIndexIssue=false, reportArgumentType=false, reportOperatorIssue=false, reportUnnecessaryComparison=false

import io
import math
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, cast, override

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import wandb  # type: ignore[import-untyped]
from PIL import Image
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel

# --- Section A: Config ---


@dataclass
class Config:
    """Configuration for from-scratch Parameter Decomposition."""

    # Maps each `nn.Linear` submodule path to its component count.
    C_per_module: dict[str, int]

    # Training schedule
    n_steps: int = 400_000
    batch_size: int = 64  # global batch (divided across ranks)
    seq_len: int = 512
    seed: int = 0

    # Main optimizer (AdamW over *all* trainable params). The low LR is inherited from the
    # decompose-a-frozen-target setting: components (V/U) and the CI fn must move slowly so the
    # persistent adversary can track them. Here it also governs the from-scratch scaffold
    # (embeddings, norms), which trains at the same low rate.
    main_lr: float = 5e-5
    main_lr_final_frac: float = 0.1
    main_warmup_pct: float = 0.01
    weight_decay: float = 0.01  # applied to dim>=2 params (V/U, embed, CI matrices) only
    grad_clip: float = 1.0

    # Loss coefficients (unmasked-forward CE is eval-only, not trained — see module docstring)
    coeff_imp: float = 1e-3
    coeff_stoch: float = 1.0
    coeff_ppgd: float = 1.0

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
    ppgd_lr_final_frac: float = 1.0
    ppgd_warmup_pct: float = 0.025
    ppgd_beta1: float = 0.5
    ppgd_beta2: float = 0.99
    ppgd_eps: float = 1e-8
    ppgd_inner_steps: int = 2

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


# --- Section B: Leaky-hard sigmoids ---


class _LowerLeakyHardSigmoid(torch.autograd.Function):
    """Forward: `clamp(x, 0, 1)`. Backward: pass-through on `(0, 1)`; in the `x <= 0` region,
    only return `alpha * grad_output` when `grad_output < 0` (gradient wants `y` to increase —
    can 'resurrect' a dead component). Above 1, gradient is blocked.
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
    """For `x > 1` return `1 + alpha*(x-1)` (linear continuation); otherwise `clamp(x, 0, 1)`."""
    return torch.where(x > 1, 1 + alpha * (x - 1), x.clamp(0.0, 1.0))


# --- Section C: ComponentLinear wrapper + build helper ---


class ComponentLinear(nn.Module):
    """A linear layer that *is* a UV decomposition. There is no target weight.

    Parameters:
        V: [d_in, C], the component basis in input space.
        U: [C, d_out], the component output transforms.
        bias: [d_out] or None.

    Forward is always `((x @ V) * mask) @ U (+ bias)`. `self.mask` ([B, S, C]) is set per
    forward by the training loop; `None` means an all-ones (unmasked) forward. The input `x`
    is cached (detached) on every forward so the CI function can read each layer's input
    activations after the unmasked forward.
    """

    def __init__(self, d_in: int, d_out: int, C: int, *, bias: bool) -> None:
        super().__init__()
        self.C = C
        self.V = nn.Parameter(torch.empty(d_in, C).normal_(0.0, 1.0 / math.sqrt(d_in)))
        self.U = nn.Parameter(torch.empty(C, d_out).normal_(0.0, 1.0 / math.sqrt(C)))
        self.bias = nn.Parameter(torch.zeros(d_out)) if bias else None
        self.mask: Tensor | None = None
        self.last_input: Tensor | None = None
        self.cache_output: bool = False
        self.last_output: Tensor | None = None

    @override
    def forward(self, x: Tensor) -> Tensor:
        self.last_input = x.detach()
        comp_acts = x @ self.V  # [B, S, C]
        if self.mask is not None:
            comp_acts = comp_acts * self.mask
        out = comp_acts @ self.U
        if self.bias is not None:
            out = out + self.bias
        if self.cache_output:
            self.last_output = out.detach()
        return out


def build_components(model: nn.Module, module_to_c: dict[str, int]) -> dict[str, ComponentLinear]:
    """Replace each listed `nn.Linear` in `model` with a `ComponentLinear`, in place. The
    original (random) weights are discarded — only the shape is read. Nothing is frozen."""
    wrappers: dict[str, ComponentLinear] = {}
    for path, C in module_to_c.items():
        parent_path, _, attr = path.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        linear = model.get_submodule(path)
        assert isinstance(linear, nn.Linear), f"{path} is not nn.Linear: {type(linear)}"
        d_out, d_in = linear.weight.shape
        wrapper = ComponentLinear(d_in, d_out, C, bias=linear.bias is not None)
        setattr(parent, attr, wrapper)
        wrappers[path] = wrapper
    return wrappers


# --- Section D: CI transformer (global_shared_transformer) ---


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
    """Causal-importance function: a shared transformer that sees all layers at once.

    Inputs: dict of per-module input activations ([B, S, d_in_m]). Each is RMS-normed and
    concatenated, projected to d_model, run through `n_blocks` blocks, projected back to the
    total component dimension, split per module, and passed through the two leaky-hard
    sigmoids. Modules are concatenated in alphabetical-path order (`module_order`).
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


# --- Section E: Losses + mask sampling ---


def ce_next_token(logits: Tensor, input_ids: Tensor) -> Tensor:
    """Next-token CE. Masks the first position of each batch item with -100 so the boundary
    between packed items does not contribute a fake transition."""
    masked = input_ids.clone()
    masked[:, 0] = -100
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = masked.reshape(-1)
    return F.cross_entropy(flat_logits[:-1], flat_labels[1:], ignore_index=-100)


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


def sample_uniform_k_subset_routing(
    module_names: list[str], batch_dims: tuple[int, ...], device: torch.device
) -> dict[str, Tensor]:
    """For each (batch, pos), sample k ~ Uniform{1..M} and route to a random k-subset."""
    M = len(module_names)
    k = torch.randint(1, M + 1, batch_dims, device=device)  # [*batch_dims]
    noise = torch.rand(M, *batch_dims, device=device)
    ranks = noise.argsort(dim=0).argsort(dim=0)  # [M, *batch_dims]
    return {name: ranks[i] < k for i, name in enumerate(module_names)}


def sample_stochastic_subset_masks(
    ci_lower: dict[str, Tensor], batch_dims: tuple[int, ...], device: torch.device
) -> dict[str, Tensor]:
    """Selected layers (uniform-k-subset per position) get `ci + (1-ci)*U(0,1)`; non-selected
    layers get mask 1.0 (full components). With no target, "not routed" means run unmasked."""
    routing = sample_uniform_k_subset_routing(list(ci_lower), batch_dims, device)
    masks: dict[str, Tensor] = {}
    for name, ci in ci_lower.items():
        u = torch.rand_like(ci)
        stoch = ci + (1 - ci) * u
        masks[name] = torch.where(routing[name].unsqueeze(-1), stoch, torch.ones_like(ci))
    return masks


def set_wrapper_masks(wrappers: dict[str, ComponentLinear], masks: dict[str, Tensor]) -> None:
    for name, w in wrappers.items():
        w.mask = masks[name]


def clear_wrapper_masks(wrappers: dict[str, ComponentLinear]) -> None:
    for w in wrappers.values():
        w.mask = None


def forward_with_masks(
    model: nn.Module,
    wrappers: dict[str, ComponentLinear],
    input_ids: Tensor,
    masks: dict[str, Tensor],
) -> Tensor:
    set_wrapper_masks(wrappers, masks)
    try:
        return model(input_ids)
    finally:
        clear_wrapper_masks(wrappers)


def stochastic_recon_loss(
    model: nn.Module,
    wrappers: dict[str, ComponentLinear],
    input_ids: Tensor,
    ci_lower: dict[str, Tensor],
) -> Tensor:
    """One-sample stochastic-subset reconstruction, scored by next-token CE."""
    B, S = input_ids.shape
    masks = sample_stochastic_subset_masks(ci_lower, (B, S), input_ids.device)
    pred = forward_with_masks(model, wrappers, input_ids, masks)
    return ce_next_token(pred, input_ids)


# --- Section F: Persistent PGD ---


class PersistentPGD:
    """Per-module adversarial sources that persist across training steps.

    Scope is `per_batch_per_position`: sources have shape `[local_B, S, C]` on each rank,
    no cross-rank sync. Adam state (m, v) is maintained alongside each source. All layers
    are masked (no routing): the adversary controls every component's mask.
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
            shape = (local_B, seq_len, w.C)
            self.sources[name] = torch.rand(shape, device=device).requires_grad_(True)
            self.m[name] = torch.zeros(shape, device=device)
            self.v[name] = torch.zeros(shape, device=device)
        self.t = 0

    def _masks_from_sources(self, ci_lower: dict[str, Tensor]) -> dict[str, Tensor]:
        return {name: ci + (1 - ci) * self.sources[name] for name, ci in ci_lower.items()}

    def recon_loss(
        self,
        model: nn.Module,
        wrappers: dict[str, ComponentLinear],
        input_ids: Tensor,
        ci_lower: dict[str, Tensor],
    ) -> Tensor:
        masks = self._masks_from_sources(ci_lower)
        pred = forward_with_masks(model, wrappers, input_ids, masks)
        return ce_next_token(pred, input_ids)

    def warmup(
        self,
        model: nn.Module,
        wrappers: dict[str, ComponentLinear],
        input_ids: Tensor,
        ci_lower: dict[str, Tensor],
        lr: float,
    ) -> None:
        for _ in range(self.cfg.ppgd_inner_steps):
            loss = self.recon_loss(model, wrappers, input_ids, ci_lower)
            grads = torch.autograd.grad(loss, list(self.sources.values()), retain_graph=False)
            self._adam_step(dict(zip(self.sources, grads, strict=True)), lr)

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


# --- Section G: LR schedule ---


def cosine_lr(
    step: int, total: int, start: float, final_frac: float, warmup_pct: float = 0.0
) -> float:
    """Linear warmup from 0 to `start` over `warmup_pct * total` steps, then half-period cosine
    decay from `start` to `start * final_frac`."""
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


# --- Section H: Distributed setup + SPDModule container ---


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
    """Container so DDP tracks both the (decomposed) model params and the CI transformer.

    `forward(input_ids)` runs the unmasked component forward — every `ComponentLinear` at
    mask=None — producing both the logits for the unmasked-CE anchor and the cached per-layer
    input activations the CI function consumes. The stochastic / PPGD masked forwards go
    through `self.model` directly; DDP's grad sync fires on the parameters themselves
    regardless of which forward visited them.
    """

    def __init__(
        self, model: nn.Module, ci_fn: CITransformer, wrappers: dict[str, ComponentLinear]
    ) -> None:
        super().__init__()
        self.model = model
        self.ci_fn = ci_fn
        self._wrappers = wrappers

    @override
    def forward(self, input_ids: Tensor) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
        clear_wrapper_masks(self._wrappers)
        logits = self.model(input_ids)
        acts = {n: _require(w.last_input) for n, w in self._wrappers.items()}
        ci_lower, ci_upper, _ci_pre = self.ci_fn(acts)
        return logits, ci_lower, ci_upper


def _require(x: Tensor | None) -> Tensor:
    assert x is not None
    return x


# --- Section I: Training loop ---


def decompose(
    model: nn.Module,
    cfg: Config,
    train_loader: Iterator[Tensor],
    eval_loader: Iterator[Tensor],
) -> None:
    """Train `model` from scratch as a decomposable LM. `cfg.C_per_module` names the
    `nn.Linear` submodules to turn into `ComponentLinear`s; `model.forward(input_ids)` must
    return logits. Loaders yield `[local_B, seq_len]` int64 token ids, sharded by rank."""
    rank, world_size, local_rank, device = init_dist()
    assert cfg.batch_size % world_size == 0, "global batch size must be divisible by world size"
    local_B = cfg.batch_size // world_size

    # Same seed on every rank so all params match after init.
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    def _log(msg: str) -> None:
        if rank == 0:
            print(f"[rank0] {msg}", flush=True)

    wrappers = build_components(model, cfg.C_per_module)
    _log(f"installed {len(wrappers)} components")
    d_in_per_module = {name: int(w.V.shape[0]) for name, w in wrappers.items()}
    ci_fn = CITransformer(d_in_per_module, cfg.C_per_module, cfg)
    _log(f"built CI transformer ({sum(p.numel() for p in ci_fn.parameters()):,} params)")

    spd = SPDModule(model, ci_fn, wrappers).to(device)
    _log(f"moved model + CI fn to {device}")

    # Per-rank seed for data + PPGD + sampling streams (params already matched above).
    torch.manual_seed(cfg.seed + rank)
    torch.cuda.manual_seed_all(cfg.seed + rank)

    spd_wrapped: nn.Module
    if world_size > 1:
        spd_wrapped = DistributedDataParallel(
            spd, device_ids=[local_rank], output_device=local_rank
        )
    else:
        spd_wrapped = spd
    _log("DDP wrap complete")

    ppgd = PersistentPGD(wrappers, local_B, cfg.seq_len, device, cfg)
    trainable = [p for p in spd.parameters() if p.requires_grad]
    decay = [p for p in trainable if p.dim() >= 2]
    no_decay = [p for p in trainable if p.dim() < 2]
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.main_lr,
    )
    _log(f"optimizer ready ({sum(p.numel() for p in trainable):,} trainable params)")

    if rank == 0 and cfg.use_wandb:
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name)
        _log(f"wandb url: {wandb.run.url if wandb.run else '?'}")

    for step in range(cfg.n_steps):
        main_lr = cosine_lr(
            step, cfg.n_steps, cfg.main_lr, cfg.main_lr_final_frac, cfg.main_warmup_pct
        )
        ppgd_lr = cosine_lr(
            step, cfg.n_steps, cfg.ppgd_lr, cfg.ppgd_lr_final_frac, cfg.ppgd_warmup_pct
        )
        for g in opt.param_groups:
            g["lr"] = main_lr

        input_ids = next(train_loader).to(device)

        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            logits, ci_lower, ci_upper = spd_wrapped(input_ids)

            ppgd.warmup(model, wrappers, input_ids, ci_lower, lr=ppgd_lr)

            loss_imp = importance_minimality_loss(
                ci_upper,
                anneal_p(step, cfg.n_steps, cfg.p_start, cfg.p_end),
                cfg.imp_eps,
                cfg.imp_beta,
                world_size,
            )
            loss_stoch = stochastic_recon_loss(model, wrappers, input_ids, ci_lower)
            loss_ppgd = ppgd.recon_loss(model, wrappers, input_ids, ci_lower)
            # Unmasked CE is not trained (eval-only); computed here only for the train log.
            loss_unmasked = ce_next_token(logits, input_ids)

        # Total-loss summation runs outside autocast so the coeff*loss sum stays in fp32.
        total = cfg.coeff_imp * loss_imp + cfg.coeff_stoch * loss_stoch + cfg.coeff_ppgd * loss_ppgd

        # Extract PPGD source grads before the main backward. Per-rank, no all-reduce.
        ppgd_grads = torch.autograd.grad(loss_ppgd, list(ppgd.sources.values()), retain_graph=True)
        ppgd_grads_dict = dict(zip(ppgd.sources, ppgd_grads, strict=True))

        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
        opt.step()
        ppgd.external_step(ppgd_grads_dict, ppgd_lr)

        is_regular_eval = step % cfg.eval_freq == 0
        is_slow_eval = step % cfg.slow_eval_freq == 0 or (step == 0 and cfg.slow_eval_on_first_step)
        if is_regular_eval or is_slow_eval:
            eval_batch = next(eval_loader).to(device)
            eval_metrics = run_eval(
                model,
                ci_fn,
                wrappers,
                cfg,
                world_size,
                eval_batch,
                is_slow=is_slow_eval,
                imp_p=anneal_p(step, cfg.n_steps, cfg.p_start, cfg.p_end),
            )
            if rank == 0 and cfg.use_wandb:
                wandb.log(eval_metrics, step=step)

        if rank == 0 and step % cfg.log_every == 0:
            metrics = {
                "loss/imp": loss_imp.detach().item(),
                "loss/unmasked": loss_unmasked.detach().item(),
                "loss/stoch": loss_stoch.detach().item(),
                "loss/ppgd": loss_ppgd.detach().item(),
                "lr/main": main_lr,
                "lr/ppgd": ppgd_lr,
                "step": step,
            }
            if cfg.use_wandb:
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


# --- Section J: Eval metrics ---
# Key naming mirrors `run.py` / the main LM experiment so nano and main runs overlay in W&B.


def _all_reduce_mean(t: Tensor, world_size: int) -> Tensor:
    if world_size > 1:
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return t


def eval_ci_l0(
    ci_lower: dict[str, Tensor], threshold: float, world_size: int, use_wandb: bool
) -> dict[str, Any]:
    per_module: dict[str, float] = {}
    for name, ci in ci_lower.items():
        l0 = (ci > threshold).float().sum(-1).mean()
        per_module[name] = _all_reduce_mean(l0.clone(), world_size).item()
    total = sum(per_module.values())
    out: dict[str, Any] = {f"eval/l0/{threshold}_{n}": v for n, v in per_module.items()}
    out[f"eval/l0/{threshold}_total"] = total
    if use_wandb:
        table_data = list(per_module.items()) + [("total", total)]
        out["eval/l0/bar_chart"] = wandb.plot.bar(
            table=wandb.Table(columns=["layer", "l0"], data=table_data),
            label="layer",
            value="l0",
            title=f"L0_{threshold}",
        )
    return out


def eval_ce_losses(
    model: nn.Module,
    wrappers: dict[str, ComponentLinear],
    input_ids: Tensor,
    ci_lower: dict[str, Tensor],
    rounding_threshold: float,
    world_size: int,
) -> dict[str, float]:
    """Next-token CE under several mask strategies. With no target, the unmasked forward is
    the reference and `zero_masked` is the floor: `ce_unrecovered = (ce - ce_unmasked) /
    (ce_zero - ce_unmasked)`."""
    strategies: dict[str, dict[str, Tensor]] = {
        "unmasked": {n: torch.ones_like(c) for n, c in ci_lower.items()},
        "ci_masked": {n: ci for n, ci in ci_lower.items()},
        "stoch_masked": sample_stochastic_subset_masks(ci_lower, input_ids.shape, input_ids.device),
        "random_masked": {n: torch.rand_like(c) for n, c in ci_lower.items()},
        "rounded_masked": {n: (c > rounding_threshold).to(c.dtype) for n, c in ci_lower.items()},
        "zero_masked": {n: torch.zeros_like(c) for n, c in ci_lower.items()},
    }
    ces: dict[str, float] = {}
    for name, masks in strategies.items():
        ces[name] = ce_next_token(
            forward_with_masks(model, wrappers, input_ids, masks), input_ids
        ).item()

    device = input_ids.device
    unmasked_ce = ces["unmasked"]
    zero_ce = ces["zero_masked"]
    out: dict[str, float] = {}
    for name in strategies:
        out[f"eval/ce/ce_{name}"] = _all_reduce_mean(
            torch.tensor(ces[name], device=device), world_size
        ).item()
    for name in [k for k in strategies if k not in ("zero_masked", "unmasked")]:
        denom = zero_ce - unmasked_ce
        unrecov = (ces[name] - unmasked_ce) / denom if denom != 0 else float("nan")
        out[f"eval/ce/ce_unrecovered_{name}"] = _all_reduce_mean(
            torch.tensor(unrecov, device=device), world_size
        ).item()
    return out


def eval_pgd_recon(
    model: nn.Module,
    wrappers: dict[str, ComponentLinear],
    input_ids: Tensor,
    ci_lower: dict[str, Tensor],
    step_size: float,
    n_steps: int,
    world_size: int,
) -> dict[str, float]:
    """Adversarially find high-CE masks via sign-SGD PGD, then report the resulting CE. One
    source of shape [1, 1, C] per module is shared across the batch — broadcast from rank 0
    at init and gradient-averaged across ranks before each step."""
    B, S = input_ids.shape
    device = input_ids.device
    sources: dict[str, Tensor] = {}
    for name, w in wrappers.items():
        src = torch.rand((1, 1, w.C), device=device)
        if world_size > 1:
            dist.broadcast(src, src=0)
        sources[name] = src.requires_grad_(True)

    def compute_loss() -> Tensor:
        masks = {
            name: ci + (1 - ci) * sources[name].expand(B, S, -1) for name, ci in ci_lower.items()
        }
        return ce_next_token(forward_with_masks(model, wrappers, input_ids, masks), input_ids)

    with torch.enable_grad():
        for _ in range(n_steps):
            loss = compute_loss()
            grads = torch.autograd.grad(loss, list(sources.values()))
            with torch.no_grad():
                for name, g in zip(sources, grads, strict=True):
                    if world_size > 1:
                        dist.all_reduce(g, op=dist.ReduceOp.AVG)
                    sources[name].add_(step_size * g.sign())
                    sources[name].clamp_(0.0, 1.0)
        final_loss = compute_loss()

    return {
        "eval/loss/PGDReconLoss": _all_reduce_mean(final_loss.detach().clone(), world_size).item()
    }


def _plot_mean_ci_per_component(mean_component_cis: dict[str, Tensor], log_y: bool) -> Image.Image:
    n_modules = len(mean_component_cis)
    max_rows = 6
    n_cols = (n_modules + max_rows - 1) // max_rows
    n_rows = min(n_modules, max_rows)
    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(8 * n_cols, 3 * n_rows), dpi=200, squeeze=False
    )
    axs = np.array(axs).reshape(n_rows, n_cols)
    for i in range(n_modules, n_rows * n_cols):
        axs[i % n_rows, i // n_rows].set_visible(False)
    for i, (module_name, mean_ci) in enumerate(mean_component_cis.items()):
        sorted_cis = torch.sort(mean_ci, descending=True)[0].detach().float().cpu().numpy()
        ax = axs[i % n_rows, i // n_rows]
        if log_y:
            ax.set_yscale("log")
        ax.scatter(range(len(sorted_cis)), sorted_cis, marker="x", s=10)
        if i % n_rows == n_rows - 1 or i == n_modules - 1:
            ax.set_xlabel("Component")
        ax.set_ylabel("mean CI")
        ax.set_title(module_name, fontsize=10)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def eval_ci_mean_per_component_figs(
    ci_lower: dict[str, Tensor], world_size: int, use_wandb: bool
) -> dict[str, Any]:
    if not use_wandb:
        return {}
    mean_per_module: dict[str, Tensor] = {}
    for name, ci in ci_lower.items():
        mean_c = ci.mean(dim=tuple(range(ci.ndim - 1)))
        mean_per_module[name] = _all_reduce_mean(mean_c.clone(), world_size).cpu()
    return {
        "eval/figures/ci_mean_per_component": wandb.Image(
            _plot_mean_ci_per_component(mean_per_module, log_y=False)
        ),
        "eval/figures/ci_mean_per_component_log": wandb.Image(
            _plot_mean_ci_per_component(mean_per_module, log_y=True)
        ),
    }


def eval_hidden_acts_recon(
    model: nn.Module,
    wrappers: dict[str, ComponentLinear],
    input_ids: Tensor,
    ci_lower: dict[str, Tensor],
    world_size: int,
    *,
    stochastic: bool,
) -> dict[str, float]:
    """Per-module MSE between the unmasked-forward layer outputs and the masked-forward layer
    outputs (CI-mask or stochastic-mask). Self-consistency, target-free. Requires
    `cache_output=True` (set/cleared by `run_eval`)."""
    clear_wrapper_masks(wrappers)
    _ = model(input_ids)
    unmasked_acts = {n: _require(w.last_output) for n, w in wrappers.items()}

    if stochastic:
        masks = sample_stochastic_subset_masks(ci_lower, input_ids.shape, input_ids.device)
    else:
        masks = {n: ci for n, ci in ci_lower.items()}
    _ = forward_with_masks(model, wrappers, input_ids, masks)
    comp_acts = {n: _require(w.last_output) for n, w in wrappers.items()}

    metric_name = "StochasticHiddenActsReconLoss" if stochastic else "CIHiddenActsReconLoss"
    per_module: dict[str, float] = {}
    total_sq = torch.zeros((), device=input_ids.device)
    total_n = 0
    for name in unmasked_acts:
        mse = F.mse_loss(comp_acts[name], unmasked_acts[name], reduction="mean")
        mse = _all_reduce_mean(mse.clone(), world_size)
        per_module[name] = mse.item()
        total_sq = total_sq + mse * unmasked_acts[name].numel()
        total_n += unmasked_acts[name].numel()
    out = {f"eval/loss/{metric_name}/{n}": v for n, v in per_module.items()}
    out[f"eval/loss/{metric_name}/total"] = (total_sq / total_n).item()
    return out


def eval_train_losses(
    model: nn.Module,
    wrappers: dict[str, ComponentLinear],
    eval_batch: Tensor,
    logits: Tensor,
    ci_lower: dict[str, Tensor],
    ci_upper: dict[str, Tensor],
    cfg: Config,
    world_size: int,
    imp_p: float,
) -> dict[str, float]:
    imp = importance_minimality_loss(ci_upper, imp_p, cfg.imp_eps, cfg.imp_beta, world_size)
    unmasked = ce_next_token(logits, eval_batch)
    stoch = stochastic_recon_loss(model, wrappers, eval_batch, ci_lower)
    return {
        "eval/loss/ImportanceMinimalityLoss": _all_reduce_mean(
            imp.detach().clone(), world_size
        ).item(),
        "eval/loss/UnmaskedReconLoss": _all_reduce_mean(
            unmasked.detach().clone(), world_size
        ).item(),
        "eval/loss/StochasticReconSubsetLoss": _all_reduce_mean(
            stoch.detach().clone(), world_size
        ).item(),
    }


def run_eval(
    model: nn.Module,
    ci_fn: CITransformer,
    wrappers: dict[str, ComponentLinear],
    cfg: Config,
    world_size: int,
    eval_batch: Tensor,
    is_slow: bool,
    imp_p: float,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for w in wrappers.values():
        w.cache_output = False
    try:
        with torch.no_grad():
            clear_wrapper_masks(wrappers)
            logits = model(eval_batch)
            acts = {n: _require(w.last_input) for n, w in wrappers.items()}
            ci_lower, ci_upper, _ci_pre = ci_fn(acts)

            metrics.update(eval_ci_l0(ci_lower, cfg.ci_alive_threshold, world_size, cfg.use_wandb))
            metrics.update(
                eval_ce_losses(
                    model, wrappers, eval_batch, ci_lower, cfg.rounding_threshold, world_size
                )
            )
            metrics.update(
                eval_train_losses(
                    model, wrappers, eval_batch, logits, ci_lower, ci_upper, cfg, world_size, imp_p
                )
            )

            if is_slow:
                metrics.update(eval_ci_mean_per_component_figs(ci_lower, world_size, cfg.use_wandb))
                for w in wrappers.values():
                    w.cache_output = True
                for stoch in (True, False):
                    metrics.update(
                        eval_hidden_acts_recon(
                            model, wrappers, eval_batch, ci_lower, world_size, stochastic=stoch
                        )
                    )
                for w in wrappers.values():
                    w.cache_output = False

        metrics.update(
            eval_pgd_recon(
                model,
                wrappers,
                eval_batch,
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
