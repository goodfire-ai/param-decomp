"""Faithful OthelloGPT: the 8-layer GPT from Li et al. 2022 (arXiv:2210.13382).

Mirrors the canonical Neel TransformerLens "synthetic" model so its published weights load
cleanly (see `param_decomp_lab/experiments/lm/convert_othello_gpt.py`). That checkpoint has
LayerNorm folded into the following linear weights, so the norms here are parameter-free
`LayerNormPre` (centre + RMS-normalise, no affine) and `lm_head` carries the unembed bias.
Unlike `GPT2Simple` the head is untied and the activation is the exact (erf) GELU.
"""

import inspect
from pathlib import Path
from typing import Literal, override

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn import functional as F

from param_decomp.base_config import BaseConfig
from param_decomp_lab.distributed import log0
from param_decomp_lab.experiments.lm.pretrain.run_info import PretrainRunInfo


class OthelloGPTConfig(BaseConfig):
    model_type: Literal["OthelloGPT"]
    block_size: int = 59
    vocab_size: int = 61
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    layer_norm_eps: float = 1e-5


class LayerNormPre(nn.Module):
    """Parameter-free LayerNorm (centre + RMS-normalise), matching folded TransformerLens weights."""

    def __init__(self, eps: float):
        super().__init__()
        self.eps = eps

    @override
    def forward(self, x: Float[Tensor, "batch pos d_model"]) -> Float[Tensor, "batch pos d_model"]:
        x = x - x.mean(-1, keepdim=True)
        scale = (x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        return x / scale


class CausalSelfAttention(nn.Module):
    def __init__(self, config: OthelloGPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.q_proj = nn.Linear(config.n_embd, config.n_embd)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd)

    @override
    def forward(self, x: Float[Tensor, "batch pos d_model"]) -> Float[Tensor, "batch pos d_model"]:
        B, T, C = x.size()
        head_dim = C // self.n_head
        q = self.q_proj(x).view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


class MLP(nn.Module):
    def __init__(self, config: OthelloGPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.down_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    @override
    def forward(self, x: Float[Tensor, "... dim"]) -> Float[Tensor, "... dim"]:
        return self.down_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config: OthelloGPTConfig):
        super().__init__()
        self.ln_1 = LayerNormPre(config.layer_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNormPre(config.layer_norm_eps)
        self.mlp = MLP(config)

    @override
    def forward(self, x: Float[Tensor, "batch pos d_model"]) -> Float[Tensor, "batch pos d_model"]:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class OthelloGPT(nn.Module):
    def __init__(self, config: OthelloGPTConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNormPre(config.layer_norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=True)

    @override
    def forward(
        self,
        idx: Int[Tensor, "batch pos"],
        targets: Int[Tensor, "batch pos"] | None = None,
        return_logits: bool = True,
    ) -> tuple[Float[Tensor, "batch pos vocab"] | None, Float[Tensor, ""] | None]:
        _b, t = idx.size()
        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        )
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.to(torch.long).view(-1),
                ignore_index=-1,
            )

        return (logits if return_logits else None), loss

    @classmethod
    def from_run_info(cls, run_info: PretrainRunInfo) -> "OthelloGPT":
        model = cls(OthelloGPTConfig(**run_info.model_config_dict))
        state_dict = torch.load(run_info.checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        return model

    @classmethod
    def from_pretrained(cls, model_path: str | Path) -> "OthelloGPT":
        return cls.from_run_info(PretrainRunInfo.from_path(model_path))

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
        zero_stage: int,
    ) -> torch.optim.Optimizer:
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if zero_stage == 1:
            log0("using ZeroRedundancyOptimizer")
            optimizer: torch.optim.Optimizer = ZeroRedundancyOptimizer(
                decay_params,
                optimizer_class=torch.optim.AdamW,
                lr=learning_rate,
                betas=betas,
                fused=use_fused,
                weight_decay=weight_decay,
            )
            optimizer.add_param_group({"params": nodecay_params, "weight_decay": 0.0})
        else:
            optimizer = torch.optim.AdamW(
                optim_groups, lr=learning_rate, betas=betas, fused=use_fused
            )
        return optimizer
