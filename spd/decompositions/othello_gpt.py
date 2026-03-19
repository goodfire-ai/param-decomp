"""OthelloGPT model for SPD decomposition.

8-layer, 8-head GPT-2-style transformer (d_model=512, d_mlp=2048, vocab=61)
trained to predict valid Othello moves from game history.

Weights from Baidicoot/Othello-GPT-Transformer-Lens on HuggingFace.
"""

import os
from typing import override

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from torch.nn import functional as F

N_LAYER = 8
N_HEAD = 8
N_EMBD = 512
VOCAB_SIZE = 61
BLOCK_SIZE = 59


class CausalSelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(N_EMBD, N_EMBD)
        self.k_proj = nn.Linear(N_EMBD, N_EMBD)
        self.v_proj = nn.Linear(N_EMBD, N_EMBD)
        self.o_proj = nn.Linear(N_EMBD, N_EMBD)

    @override
    def forward(self, x: Float[Tensor, "batch pos d_model"]) -> Float[Tensor, "batch pos d_model"]:
        B, T, C = x.size()
        d_head = C // N_HEAD
        q = self.q_proj(x).view(B, T, N_HEAD, d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, N_HEAD, d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, N_HEAD, d_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c_fc = nn.Linear(N_EMBD, 4 * N_EMBD)
        self.gelu = nn.GELU()
        self.down_proj = nn.Linear(4 * N_EMBD, N_EMBD)

    @override
    def forward(self, x: Float[Tensor, "... dim"]) -> Float[Tensor, "... dim"]:
        return self.down_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(N_EMBD)
        self.attn = CausalSelfAttention()
        self.ln_2 = nn.LayerNorm(N_EMBD)
        self.mlp = MLP()

    @override
    def forward(self, x: Float[Tensor, "batch pos d_model"]) -> Float[Tensor, "batch pos d_model"]:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class OthelloGPT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.wte = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.wpe = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self._h = [Block() for _ in range(N_LAYER)]
        self.h = nn.ModuleList(self._h)
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE, bias=False)

    @override
    def forward(
        self, idx: Int[Tensor, "batch pos"]
    ) -> tuple[Float[Tensor, "batch pos vocab"], None]:
        _b, t = idx.size()
        assert t <= BLOCK_SIZE, f"Sequence length {t} exceeds block size {BLOCK_SIZE}"
        pos = torch.arange(t, dtype=torch.long, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for block in self._h:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x), None

    @classmethod
    def from_pretrained(cls, _model_name: str = "synthetic") -> "OthelloGPT":
        model = cls()
        old_val = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER")
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(
                repo_id="Baidicoot/Othello-GPT-Transformer-Lens",
                filename="final.pth",
            )
        finally:
            if old_val is None:
                os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
            else:
                os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = old_val

        src = torch.load(path, map_location="cpu", weights_only=True)
        dst: dict[str, Tensor] = {}

        dst["wte.weight"] = src["tok_emb.weight"]
        dst["wpe.weight"] = src["pos_emb"].squeeze(0)
        dst["ln_f.weight"] = src["ln_f.weight"]
        dst["ln_f.bias"] = src["ln_f.bias"]
        dst["lm_head.weight"] = src["head.weight"]

        for i in range(N_LAYER):
            s, d = f"blocks.{i}", f"h.{i}"

            dst[f"{d}.ln_1.weight"] = src[f"{s}.ln1.weight"]
            dst[f"{d}.ln_1.bias"] = src[f"{s}.ln1.bias"]
            dst[f"{d}.ln_2.weight"] = src[f"{s}.ln2.weight"]
            dst[f"{d}.ln_2.bias"] = src[f"{s}.ln2.bias"]

            dst[f"{d}.attn.q_proj.weight"] = src[f"{s}.attn.query.weight"]
            dst[f"{d}.attn.q_proj.bias"] = src[f"{s}.attn.query.bias"]
            dst[f"{d}.attn.k_proj.weight"] = src[f"{s}.attn.key.weight"]
            dst[f"{d}.attn.k_proj.bias"] = src[f"{s}.attn.key.bias"]
            dst[f"{d}.attn.v_proj.weight"] = src[f"{s}.attn.value.weight"]
            dst[f"{d}.attn.v_proj.bias"] = src[f"{s}.attn.value.bias"]
            dst[f"{d}.attn.o_proj.weight"] = src[f"{s}.attn.proj.weight"]
            dst[f"{d}.attn.o_proj.bias"] = src[f"{s}.attn.proj.bias"]

            dst[f"{d}.mlp.c_fc.weight"] = src[f"{s}.mlp.0.weight"]
            dst[f"{d}.mlp.c_fc.bias"] = src[f"{s}.mlp.0.bias"]
            dst[f"{d}.mlp.down_proj.weight"] = src[f"{s}.mlp.2.weight"]
            dst[f"{d}.mlp.down_proj.bias"] = src[f"{s}.mlp.2.bias"]

        model.load_state_dict(dst, strict=True)
        return model
