"""A tiny Llama-ish transformer used solely as a frozen target for the
two-pool wall-clock benchmark.

Kept minimal so that the benchmark exercises ComponentModel / the 2-pool layout
rather than any specific real-world target. The decomposable sites are the seven
linears per block (Q/K/V/O on attention, gate/up/down on SwiGLU MLP).
"""

# pyright: reportArgumentType=false

from typing import override

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TinyAttention(nn.Module):
    def __init__(self, d: int, n_heads: int) -> None:
        super().__init__()
        assert d % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

    @override
    def forward(self, x: Tensor) -> Tensor:
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, S, self.n_heads * self.head_dim)
        return self.o_proj(out)


class TinySwiGLU(nn.Module):
    def __init__(self, d: int, d_mlp: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d, d_mlp, bias=False)
        self.up_proj = nn.Linear(d, d_mlp, bias=False)
        self.down_proj = nn.Linear(d_mlp, d, bias=False)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TinyBlock(nn.Module):
    def __init__(self, d: int, n_heads: int, d_mlp: int) -> None:
        super().__init__()
        self.attn = TinyAttention(d, n_heads)
        self.mlp = TinySwiGLU(d, d_mlp)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(F.rms_norm(x, (x.shape[-1],)))
        x = x + self.mlp(F.rms_norm(x, (x.shape[-1],)))
        return x


class TinyTransformer(nn.Module):
    def __init__(
        self,
        vocab: int,
        d: int,
        n_blocks: int,
        n_heads: int,
        d_mlp: int,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([TinyBlock(d, n_heads, d_mlp) for _ in range(n_blocks)])
        self.unembed = nn.Linear(d, vocab, bias=False)

    @override
    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        x = F.rms_norm(x, (x.shape[-1],))
        return self.unembed(x)


def sites_for_block(block_idx: int) -> list[str]:
    return [
        f"blocks.{block_idx}.attn.q_proj",
        f"blocks.{block_idx}.attn.k_proj",
        f"blocks.{block_idx}.attn.v_proj",
        f"blocks.{block_idx}.attn.o_proj",
        f"blocks.{block_idx}.mlp.gate_proj",
        f"blocks.{block_idx}.mlp.up_proj",
        f"blocks.{block_idx}.mlp.down_proj",
    ]
