"""Version of GPT-2 with separate projection layers for query, key, and value."""

import inspect
import math
from pathlib import Path
from typing import Any, Literal, cast, override

import torch
import torch.nn as nn
import torch.utils.checkpoint
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn import functional as F
from transformers import GPT2LMHeadModel

from param_decomp.base_config import BaseConfig
from param_decomp_lab.distributed import log0
from param_decomp_lab.experiments.lm.pretrain.run_info import PretrainRunInfo

HF_GPT2_VARIANTS: dict[str, dict[str, int]] = {
    "gpt2": {"n_layer": 12, "n_head": 12, "n_embd": 768},
    "gpt2-medium": {"n_layer": 24, "n_head": 16, "n_embd": 1024},
    "gpt2-large": {"n_layer": 36, "n_head": 20, "n_embd": 1280},
    "gpt2-xl": {"n_layer": 48, "n_head": 25, "n_embd": 1600},
}

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


class GPT2Simple(nn.Module):
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

        # Per-block gradient checkpointing toggle. Off by default; enabled at PD-training
        # time via `enable_activation_checkpointing()` to trade ~33% extra compute for
        # ~10-15x less stored activation memory through the target forward.
        self._use_activation_checkpointing: bool = False

        # init all weights, use a torch rng object to be very careful
        self.init_rng = torch.Generator()
        self.init_rng.manual_seed(42)
        self.apply(self._init_weights)

    def enable_activation_checkpointing(self) -> None:
        """Wrap each Block in `torch.utils.checkpoint.checkpoint` during forward.

        Only block inputs are stored; intermediates (q/k/v, mlp.c_fc out, gelu out) are
        recomputed during backward. ~10-15x less activation memory at ~33% extra compute.
        """
        self._use_activation_checkpointing = True

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

        if self._use_activation_checkpointing:
            for block in self._h:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
        else:
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
    def from_run_info(cls, run_info: PretrainRunInfo) -> "GPT2Simple":
        """Create a GPT-2 model from a PretrainRunInfo, loading weights from its checkpoint."""
        model = cls(GPT2SimpleConfig(**run_info.model_config_dict))
        state_dict = torch.load(run_info.checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        return model

    @classmethod
    def from_pretrained(cls, model_path: str | Path) -> "GPT2Simple":
        """Create a GPT-2 model from a wandb string or a local path."""
        from param_decomp_lab.experiments.lm.pretrain.run_info import PretrainRunInfo

        run_info = PretrainRunInfo.from_path(model_path)
        return cls.from_run_info(run_info)

    @classmethod
    def from_hf_pretrained(cls, model_type: str) -> "GPT2Simple":
        """Load HF GPT-2 pretrained weights, splitting fused c_attn into q/k/v.

        Args:
            model_type: ``gpt2`` / ``gpt2-medium`` / ``gpt2-large`` / ``gpt2-xl``,
                or any equivalent HF id (e.g. ``openai-community/gpt2-xl``).
        """
        short_name = model_type.split("/")[-1]
        assert short_name in HF_GPT2_VARIANTS, (
            f"Unknown GPT-2 variant {model_type!r}; expected one of {list(HF_GPT2_VARIANTS)}"
        )
        log0(f"loading HF weights into vendored unfused GPT2Simple: {model_type}")

        config_args = HF_GPT2_VARIANTS[short_name] | {"vocab_size": 50257, "block_size": 1024}
        config = GPT2SimpleConfig(model_type="GPT2Simple", **cast(dict[str, Any], config_args))
        model = cls(config)

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd = model.state_dict()
        n_embd = config.n_embd

        with torch.no_grad():
            for i in range(config.n_layer):
                # Split fused c_attn into q/k/v. HF Conv1D stores weight as
                # [in_features=n_embd, out_features=3*n_embd] (transpose of nn.Linear),
                # so we split along dim=1 (out) and transpose before assigning.
                c_attn_w = sd_hf[f"transformer.h.{i}.attn.c_attn.weight"]
                c_attn_b = sd_hf[f"transformer.h.{i}.attn.c_attn.bias"]
                assert c_attn_w.shape == (n_embd, 3 * n_embd)
                assert c_attn_b.shape == (3 * n_embd,)
                q_w, k_w, v_w = c_attn_w.split(n_embd, dim=1)
                q_b, k_b, v_b = c_attn_b.split(n_embd, dim=0)
                sd[f"h.{i}.attn.q_proj.weight"].copy_(q_w.t())
                sd[f"h.{i}.attn.q_proj.bias"].copy_(q_b)
                sd[f"h.{i}.attn.k_proj.weight"].copy_(k_w.t())
                sd[f"h.{i}.attn.k_proj.bias"].copy_(k_b)
                sd[f"h.{i}.attn.v_proj.weight"].copy_(v_w.t())
                sd[f"h.{i}.attn.v_proj.bias"].copy_(v_b)

                # attn.c_proj → o_proj. Conv1D weight is transpose of nn.Linear.
                sd[f"h.{i}.attn.o_proj.weight"].copy_(
                    sd_hf[f"transformer.h.{i}.attn.c_proj.weight"].t()
                )
                sd[f"h.{i}.attn.o_proj.bias"].copy_(sd_hf[f"transformer.h.{i}.attn.c_proj.bias"])

                # mlp.c_fc: transpose; same name.
                sd[f"h.{i}.mlp.c_fc.weight"].copy_(sd_hf[f"transformer.h.{i}.mlp.c_fc.weight"].t())
                sd[f"h.{i}.mlp.c_fc.bias"].copy_(sd_hf[f"transformer.h.{i}.mlp.c_fc.bias"])

                # mlp.c_proj → down_proj: transpose.
                sd[f"h.{i}.mlp.down_proj.weight"].copy_(
                    sd_hf[f"transformer.h.{i}.mlp.c_proj.weight"].t()
                )
                sd[f"h.{i}.mlp.down_proj.bias"].copy_(sd_hf[f"transformer.h.{i}.mlp.c_proj.bias"])

                # LayerNorms: direct copy.
                sd[f"h.{i}.ln_1.weight"].copy_(sd_hf[f"transformer.h.{i}.ln_1.weight"])
                sd[f"h.{i}.ln_1.bias"].copy_(sd_hf[f"transformer.h.{i}.ln_1.bias"])
                sd[f"h.{i}.ln_2.weight"].copy_(sd_hf[f"transformer.h.{i}.ln_2.weight"])
                sd[f"h.{i}.ln_2.bias"].copy_(sd_hf[f"transformer.h.{i}.ln_2.bias"])

            # Embeddings + final LN. wte ties to lm_head; copying wte updates both.
            sd["wte.weight"].copy_(sd_hf["transformer.wte.weight"])
            sd["wpe.weight"].copy_(sd_hf["transformer.wpe.weight"])
            sd["ln_f.weight"].copy_(sd_hf["transformer.ln_f.weight"])
            sd["ln_f.bias"].copy_(sd_hf["transformer.ln_f.bias"])

        del model_hf
        return model

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
