"""LlamaSimple variant using standard GELU MLP instead of SwiGLU.

This model is the same as LlamaSimple but replaces the SwiGLU MLP with a
standard GELU MLP (like GPT-2).
"""

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


class LlamaSimpleMLPConfig(BaseConfig):
    model_type: Literal["LlamaSimpleMLP"]
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    n_intermediate: int = 768 * 4  # Standard 4x expansion for GELU MLP
    mlp_bias: bool = False
    attn_bias: bool = False
    rotary_adjacent_pairs: bool = False
    rotary_dim: int = 768 // 12  # i.e. same as d_head
    rotary_base: int = 10000
    n_ctx: int = 1024
    n_key_value_heads: int = (
        12 // 4
    )  # Note that llama 3.1 n_key_value_heads does not scale with n_heads
    use_grouped_query_attention: bool = True
    flash_attention: bool = True
    rms_norm_eps: float = 1e-6


class CausalSelfAttention(nn.Module):
    def __init__(self, config: LlamaSimpleMLPConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.use_grouped_query_attention = config.use_grouped_query_attention
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head  # Head size
        self.n_key_value_heads = config.n_key_value_heads
        self.repeat_kv_heads = config.n_head // config.n_key_value_heads  # Will be 1 if not GQA
        self.rotary_dim = self.head_dim  # Align rotary_dim with head_dim for simplicity
        self.rotary_adjacent_pairs = config.rotary_adjacent_pairs
        self.rotary_base = config.rotary_base
        self.n_ctx = config.n_ctx  # Max context length for precomputation
        self.flash_attention = config.flash_attention
        if self.use_grouped_query_attention:
            self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.attn_bias)
            self.k_proj = nn.Linear(
                config.n_embd, self.n_key_value_heads * self.head_dim, bias=config.attn_bias
            )
            self.v_proj = nn.Linear(
                config.n_embd, self.n_key_value_heads * self.head_dim, bias=config.attn_bias
            )
        else:
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.attn_bias)

        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.attn_bias)
        object.__setattr__(self.o_proj, "LLMC_RESIDUAL_SCALE_FLAG", True)

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
            persistent=False,  # Set persistent=False if not part of model state dict
        )
        sin, cos = self.calculate_sin_cos_rotary(self.rotary_dim, self.n_ctx, base=self.rotary_base)
        self.register_buffer("rotary_sin", sin, persistent=False)
        self.register_buffer("rotary_cos", cos, persistent=False)

    def calculate_sin_cos_rotary(
        self,
        rotary_dim: int,
        n_ctx: int,
        base: int = 10000,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[Tensor, Tensor]:
        """Precomputes sin and cos for rotary embeddings"""
        high_precision = torch.float32 if dtype != torch.float64 else torch.float64
        pos = torch.arange(n_ctx, dtype=high_precision)  # Positions 0..n_ctx-1
        dim = torch.arange(rotary_dim // 2, dtype=high_precision)  # Dimensions 0..rotary_dim/2-1
        freq = base ** (dim / (rotary_dim / 2))  # Frequencies based on base and dimension

        if self.rotary_adjacent_pairs:
            freq = freq.unsqueeze(1).repeat(1, 2).flatten()
        else:
            freq = freq.repeat(2)

        # Calculate angles: (n_ctx, rotary_dim)
        angles = pos[:, None] / freq[None, :]

        # Calculate sin and cos, cast to desired dtype
        sin = torch.sin(angles).to(dtype)
        cos = torch.cos(angles).to(dtype)
        return sin, cos

    def get_offset_position_ids(
        self,
        past_kv_pos_offset: int,
        attention_mask: Int[Tensor, "batch offset_pos"],
    ) -> Int[Tensor, "batch pos"]:
        shifted_position_ids = attention_mask.cumsum(dim=1) - 1
        position_ids = shifted_position_ids.masked_fill(shifted_position_ids < 0, 0)
        return position_ids[:, past_kv_pos_offset:].long()  # Ensure long type for indexing

    def rotate_every_two(self, x: Tensor) -> Tensor:
        """Rotates pairs of elements in the last dimension (handles adjacent_pairs logic)"""
        x_rot = x.clone()
        if self.rotary_adjacent_pairs:
            x_rot[..., ::2] = -x[..., 1::2]
            x_rot[..., 1::2] = x[..., ::2]
        else:
            n = x.shape[-1] // 2
            x_rot[..., :n] = -x[..., n:]
            x_rot[..., n:] = x[..., :n]
        return x_rot

    def apply_rotary_pos_emb(
        self,
        q: Float[Tensor, "batch n_head seq_len head_dim"],
        k: Float[Tensor, "batch n_kv_head seq_len head_dim"],
        cos: Float[Tensor, "batch seq_len rotary_dim"],
        sin: Float[Tensor, "batch seq_len rotary_dim"],
    ) -> tuple[
        Float[Tensor, "batch n_head seq_len head_dim"],
        Float[Tensor, "batch n_kv_head seq_len head_dim"],
    ]:
        # Unsqueeze cos/sin for broadcasting: (batch, 1, seq_len, rotary_dim)
        # This aligns with the head dimension of q and k
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        # Select the part of q and k to be rotated
        q_rot = q[..., : self.rotary_dim]
        k_rot = k[..., : self.rotary_dim]

        # Apply rotation using the formula: x_rotated = x * cos + rotate_half(x) * sin
        # Using self.rotate_every_two to handle standard RoPE and adjacent pairs logic
        q_rotated = (q_rot * cos) + (self.rotate_every_two(q_rot) * sin)
        k_rotated = (k_rot * cos) + (self.rotate_every_two(k_rot) * sin)

        # If rotary_dim is less than head_dim, combine rotated part with the non-rotated part
        if self.rotary_dim < self.head_dim:
            q_pass = q[..., self.rotary_dim :]
            k_pass = k[..., self.rotary_dim :]
            q_embed = torch.cat((q_rotated, q_pass), dim=-1)
            k_embed = torch.cat((k_rotated, k_pass), dim=-1)
        else:
            q_embed = q_rotated
            k_embed = k_rotated

        return q_embed.to(q.dtype), k_embed.to(k.dtype)

    @override
    def forward(  # type: ignore[override]
        self,
        x: Float[Tensor, "batch pos d_model"],
        attention_mask: Int[Tensor, "batch offset_pos"] | None = None,
        position_ids: Int[Tensor, "batch pos"] | None = None,
        past_key_value: tuple[Tensor, Tensor] | None = None,
    ) -> Float[Tensor, "batch pos d_model"]:
        B, T, C = x.size()  # Batch size, Sequence length, Embedding dimension

        if self.use_grouped_query_attention:
            q = self.q_proj(x)  # (B, T, C)
            k = self.k_proj(x)  # (B, T, n_kv_heads * head_dim)
            v = self.v_proj(x)  # (B, T, n_kv_heads * head_dim)

            # Reshape for multi-head attention
            q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
            k = k.view(B, T, self.n_key_value_heads, self.head_dim).transpose(
                1, 2
            )  # (B, n_kv_heads, T, head_dim)
            v = v.view(B, T, self.n_key_value_heads, self.head_dim).transpose(
                1, 2
            )  # (B, n_kv_heads, T, head_dim)
        else:
            # Standard MHA: Compute Q, K, V from combined projection
            qkv = self.c_attn(x)  # (B, T, 3*C)
            q, k, v = qkv.split(self.n_embd, dim=2)  # (B, T, C) each
            # Reshape for multi-head attention
            q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
            k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
            v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)

        past_kv_pos_offset = 0  # TODO: Handle this properly if implementing KV caching
        if position_ids is None:
            if attention_mask is not None:
                # Derive position IDs from attention mask for the current sequence part
                position_ids = self.get_offset_position_ids(
                    past_kv_pos_offset, attention_mask
                )  # (B, T)
            else:
                # Assume sequential positions if no mask/ids provided
                position_ids = torch.arange(
                    past_kv_pos_offset, past_kv_pos_offset + T, dtype=torch.long, device=x.device
                ).unsqueeze(0)  # (1, T) -> broadcasts to (B, T)
        else:
            # Use provided position_ids, selecting the relevant part
            position_ids = position_ids[:, past_kv_pos_offset : past_kv_pos_offset + T]

        position_ids = position_ids.clamp(max=self.n_ctx - 1)

        cos = self.rotary_cos[position_ids].to(q.dtype)
        sin = self.rotary_sin[position_ids].to(q.dtype)

        q, k = self.apply_rotary_pos_emb(q, k, cos, sin)

        # Repeat K/V heads if using Grouped Query Attention
        if self.use_grouped_query_attention and self.repeat_kv_heads > 1:
            k = k.repeat_interleave(self.repeat_kv_heads, dim=1)
            v = v.repeat_interleave(self.repeat_kv_heads, dim=1)

        if self.flash_attention:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
            )
        else:
            # Manual attention calculation
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            y = att @ v  # (B, n_head, T, head_dim)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        y = self.o_proj(y)  # (B, T, C)

        return y


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


class MLP(nn.Module):
    def __init__(self, config: LlamaSimpleMLPConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_intermediate, bias=config.mlp_bias)
        self.gelu = NewGELU()
        self.down_proj = nn.Linear(config.n_intermediate, config.n_embd, bias=config.mlp_bias)
        object.__setattr__(self.down_proj, "LLMC_RESIDUAL_SCALE_FLAG", True)

    @override
    def forward(self, x: Float[Tensor, "... dim"]) -> Float[Tensor, "... dim"]:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.down_proj(x)
        return x


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        """LlamaRMSNorm is equivalent to T5LayerNorm"""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    @override
    def forward(self, hidden_states: Float[Tensor, "... dim"]) -> Float[Tensor, "... dim"]:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    @override
    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Block(nn.Module):
    def __init__(self, config: LlamaSimpleMLPConfig):
        super().__init__()
        self.rms_1 = LlamaRMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.rms_2 = LlamaRMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.mlp = MLP(config)

    @override
    def forward(self, x: Float[Tensor, "... pos d_model"]) -> Float[Tensor, "... pos d_model"]:
        x = x + self.attn(self.rms_1(x))
        x = x + self.mlp(self.rms_2(x))
        return x


class LlamaSimpleMLP(LoadableModule):
    def __init__(self, config: LlamaSimpleMLPConfig):
        super().__init__()
        self.config = config
        self.wte: nn.Embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self._h: list[Block] = [Block(config) for _ in range(config.n_layer)]
        self.h: nn.ModuleList = nn.ModuleList(self._h)
        self.ln_f: LlamaRMSNorm = LlamaRMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        object.__setattr__(self.lm_head, "LLMC_SKIP_INIT", True)

        # Tie embeddings and lm_head weights
        self.wte.weight = self.lm_head.weight  # type: ignore[assignment]
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
                bias = getattr(module, "bias", None)
                if bias is not None:
                    torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02, generator=self.init_rng)

    @override
    def forward(
        self,
        idx: Float[Tensor, "batch pos"],
        targets: Float[Tensor, "batch pos vocab"] | None = None,
        return_logits: bool = True,
    ) -> tuple[Float[Tensor, "batch pos vocab"] | None, Float[Tensor, ""] | None]:
        _b, t = idx.size()
        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        )
        tok_emb = self.wte(idx)
        x = tok_emb
        for block in self._h:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            targets = targets.long()
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        if not return_logits:
            logits = None
        return logits, loss

    @classmethod
    @override
    def from_run_info(cls, run_info: PretrainRunInfo) -> LlamaSimpleMLP:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Create a LlamaSimpleMLP model from a PretrainRunInfo, loading weights from its checkpoint."""
        model = cls(LlamaSimpleMLPConfig(**run_info.model_config_dict))
        state_dict = torch.load(run_info.checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        return model

    @classmethod
    @override
    def from_pretrained(cls, model_path: str | Path) -> LlamaSimpleMLP:
        """Create a LlamaSimpleMLP model from a wandb string or a local path."""
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
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for _n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        log0(f"num decayed tensors: {len(decay_params)}, {num_decay_params:,} params")
        log0(f"num non-decayed tensors: {len(nodecay_params)}, {num_nodecay_params:,} params")
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
        """Generate tokens."""
        # Keep track of whether input was 1D and ensure input has batch dimension
        is_1d = idx.dim() == 1
        if is_1d:
            idx = idx.unsqueeze(0)

        # Initialize not_completed mask for the batch
        batch_size = idx.size(0)
        not_completed = torch.ones(batch_size, dtype=torch.bool, device=idx.device)

        for _ in range(max_new_tokens):
            # If all sequences are completed, stop early
            if not not_completed.any():
                break

            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            )

            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :]
            if temperature > 0:
                logits = logits / temperature
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")

                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
            else:
                # Create one-hot vector with 1 at position of max logit
                probs = torch.zeros_like(logits)
                probs.scatter_(1, logits.argmax(dim=-1, keepdim=True), 1.0)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            if eos_token_id is not None:
                not_completed = not_completed & (idx_next[:, -1] != eos_token_id)
                update_mask = not_completed.unsqueeze(-1)
                idx_next = torch.where(
                    update_mask, idx_next, torch.full_like(idx_next, eos_token_id)
                )

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)

        # Remove batch dimension if input was 1D
        if is_1d:
            idx = idx.squeeze(0)

        return idx
