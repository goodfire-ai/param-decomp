"""Version of GPT-2 with separate projection layers for query, key, and value."""

from __future__ import annotations

import inspect
import math
from pathlib import Path
from typing import TYPE_CHECKING, Literal, override

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn import functional as F

from param_decomp.base_config import BaseConfig
from param_decomp.interfaces import LoadableModule
from param_decomp.utils.distributed_utils import log0

if TYPE_CHECKING:
    from param_decomp.pretrain.run_info import PretrainRunInfo

# Suppress issues with nn.Module buffer access and @torch.no_grad() decorator
# pyright: reportIndexIssue=false, reportUntypedFunctionDecorator=false


class GPT2SimpleConfig(BaseConfig):
    model_type: Literal["GPT2Simple"]
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    flash_attention: bool = True


class NewGELU(nn.Module):
    @override
    def forward(self, input: Float[Tensor, "... dim"]) -> Float[Tensor, "... dim"]:
        return (
            0.5
            * input
            * (
                1.0
                + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0)))
            )
        )


class LayerNorm(nn.Module):
    def __init__(self, n_embd: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.bias = nn.Parameter(torch.zeros(n_embd))
        # Use pre-stored stds instead of computing them on the fly
        self.std: float | None = None

    @override
    def forward(
        self, residual: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        residual_mean = residual.mean(dim=-1, keepdim=True)
        if self.std is None:
            residual_std = (residual.var(dim=-1, keepdim=True, unbiased=False) + self.eps).sqrt()
        else:
            residual_std = self.std

        residual = (residual - residual_mean) / residual_std
        return residual * self.weight + self.bias


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPT2SimpleConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.flash_attention = config.flash_attention
        # key, query, value projections for all heads, but in a batch
        self.q_proj = nn.Linear(config.n_embd, config.n_embd)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd)
        # output projection
        self.o_proj = nn.Linear(config.n_embd, config.n_embd)
        object.__setattr__(self.o_proj, "LLMC_RESIDUAL_SCALE_FLAG", True)
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
            persistent=False,
        )

    @override
    def forward(
        self,
        x: Float[Tensor, "batch pos d_model"],
    ) -> Float[Tensor, "batch pos d_model"]:
        B, T, C = x.size()
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if self.flash_attention:
            # use PyTorch SDPA
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
            )
        else:
            # manual implementation of attention
            # this materializes the large (T,T) matrix for all the queries and keys
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side
        y = self.o_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config: GPT2SimpleConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = NewGELU()
        self.down_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        object.__setattr__(self.down_proj, "LLMC_RESIDUAL_SCALE_FLAG", True)

    @override
    def forward(self, x: Float[Tensor, "... dim"]) -> Float[Tensor, "... dim"]:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.down_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPT2SimpleConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, eps=1e-5)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, eps=1e-5)
        self.mlp = MLP(config)

    @override
    def forward(
        self,
        x: Float[Tensor, "batch pos d_model"],
    ) -> Float[Tensor, "batch pos d_model"]:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2Simple(LoadableModule):
    def __init__(self, config: GPT2SimpleConfig):
        super().__init__()
        self.config = config

        self.wte: nn.Embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe: nn.Embedding = nn.Embedding(config.block_size, config.n_embd)
        self._h: list[Block] = [Block(config) for _ in range(config.n_layer)]
        self.h: nn.ModuleList = nn.ModuleList(self._h)
        self.ln_f: LayerNorm = LayerNorm(config.n_embd, eps=1e-5)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        object.__setattr__(self.lm_head, "LLMC_SKIP_INIT", True)
        self.wte.weight = self.lm_head.weight  # type: ignore[assignment]

        # init all weights, use a torch rng object to be very careful
        self.init_rng = torch.Generator()
        self.init_rng.manual_seed(42)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = (
                0.02
                if not hasattr(module, "LLMC_RESIDUAL_SCALE_FLAG")
                else 0.02 / math.sqrt(2 * self.config.n_layer)
            )
            if not hasattr(module, "LLMC_SKIP_INIT"):
                torch.nn.init.normal_(module.weight, mean=0.0, std=std, generator=self.init_rng)
            if getattr(module, "bias", None) is not None:
                torch.nn.init.zeros_(module.bias)  # type: ignore[arg-type]
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02, generator=self.init_rng)

    @override
    def forward(
        self,
        idx: Int[Tensor, "batch pos"],
        targets: Int[Tensor, "batch pos"] | None = None,
        return_logits: bool = True,
    ) -> tuple[
        Float[Tensor, "batch pos vocab"] | None,
        Float[Tensor, ""] | None,
    ]:
        device = idx.device
        _b, t = idx.size()
        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        )
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.wte(idx)  # (b, t, n_embd)
        pos_emb = self.wpe(pos)  # (t, n_embd)
        x = tok_emb + pos_emb

        for block in self._h:
            x = block(x)
        x = self.ln_f(x)

        logits: Tensor = self.lm_head(x)
        loss: Tensor | None
        if targets is not None:
            if targets.dtype != torch.long:
                targets = targets.to(torch.long)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            loss = None

        out_logits: Tensor | None = logits
        if not return_logits:
            out_logits = None

        return out_logits, loss

    @classmethod
    @override
    def from_run_info(cls, run_info: PretrainRunInfo) -> GPT2Simple:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Create a GPT-2 model from a PretrainRunInfo, loading weights from its checkpoint."""
        model = cls(GPT2SimpleConfig(**run_info.model_config_dict))
        state_dict = torch.load(run_info.checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        return model

    @classmethod
    @override
    def from_pretrained(cls, model_path: str | Path) -> GPT2Simple:
        """Create a GPT-2 model from a wandb string or a local path."""
        from param_decomp.pretrain.run_info import PretrainRunInfo

        run_info = PretrainRunInfo.from_path(model_path)
        return cls.from_run_info(run_info)

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
        zero_stage: int,
    ) -> torch.optim.Optimizer:
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        log0(
            f"num decayed parameter tensors: {len(decay_params)}, "
            f"with {num_decay_params:,} parameters"
        )
        log0(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, "
            f"with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        log0(f"using fused AdamW: {use_fused}")
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
            log0("using regular AdamW")
            optimizer = torch.optim.AdamW(
                optim_groups, lr=learning_rate, betas=betas, fused=use_fused
            )
        return optimizer

    @torch.no_grad()
    def generate(
        self,
        idx: Float[Tensor, "... pos"],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ) -> Float[Tensor, "... pos"]:
        # Keep track of whether input was 1D and ensure input has batch dimension
        is_1d = idx.dim() == 1
        if is_1d:
            idx = idx.unsqueeze(0)

        batch_size = idx.size(0)
        not_completed = torch.ones(batch_size, dtype=torch.bool, device=idx.device)

        for _ in range(max_new_tokens):
            if not not_completed.any():
                break

            idx_cond = (
                idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            )
            logits, _ = self(idx_cond)
            assert logits is not None
            logits = logits[:, -1, :]
            if temperature > 0:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(logits, dim=-1)
            else:
                probs = torch.zeros_like(logits)
                probs.scatter_(1, logits.argmax(dim=-1, keepdim=True), 1.0)
            idx_next = torch.multinomial(probs, num_samples=1)

            if eos_token_id is not None:
                not_completed = not_completed & (idx_next[:, -1] != eos_token_id)
                update_mask = not_completed.unsqueeze(-1)
                idx_next = torch.where(
                    update_mask, idx_next, torch.full_like(idx_next, eos_token_id)
                )

            idx = torch.cat((idx, idx_next), dim=1)

        if is_1d:
            idx = idx.squeeze(0)

        return idx
