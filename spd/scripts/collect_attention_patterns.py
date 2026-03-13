"""Shared utility for collecting attention weights from LlamaSimpleMLP models.

Used by the detect_* head characterization scripts.
"""

import math

import torch
from torch.nn import functional as F

from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLP


def collect_attention_patterns(
    model: LlamaSimpleMLP,
    input_ids: torch.Tensor,
) -> list[torch.Tensor]:
    """Run forward pass and return attention weights for each layer.

    Returns list of (batch, n_heads, seq_len, seq_len) tensors.
    """
    B, T = input_ids.shape
    x = model.wte(input_ids)
    patterns: list[torch.Tensor] = []

    for block in model._h:
        attn_input = block.rms_1(x)
        attn = block.attn

        q = attn.q_proj(attn_input).view(B, T, attn.n_head, attn.head_dim).transpose(1, 2)
        k = (
            attn.k_proj(attn_input)
            .view(B, T, attn.n_key_value_heads, attn.head_dim)
            .transpose(1, 2)
        )
        v = (
            attn.v_proj(attn_input)
            .view(B, T, attn.n_key_value_heads, attn.head_dim)
            .transpose(1, 2)
        )

        position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)
        cos = attn.rotary_cos[position_ids].to(q.dtype)  # pyright: ignore[reportIndexIssue]
        sin = attn.rotary_sin[position_ids].to(q.dtype)  # pyright: ignore[reportIndexIssue]
        q, k = attn.apply_rotary_pos_emb(q, k, cos, sin)

        if attn.repeat_kv_heads > 1:
            k = k.repeat_interleave(attn.repeat_kv_heads, dim=1)
            v = v.repeat_interleave(attn.repeat_kv_heads, dim=1)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(attn.head_dim))
        att = att.masked_fill(attn.bias[:, :, :T, :T] == 0, float("-inf"))  # pyright: ignore[reportIndexIssue]
        att = F.softmax(att, dim=-1)
        patterns.append(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, attn.n_embd)
        y = attn.o_proj(y)
        x = x + y
        x = x + block.mlp(block.rms_2(x))

    return patterns


def collect_attention_patterns_with_logits(
    model: LlamaSimpleMLP,
    input_ids: torch.Tensor,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Run forward pass and return (softmax_weights, pre_softmax_logits) per layer.

    Each element is a tuple of (batch, n_heads, seq_len, seq_len) tensors.
    Logits have -inf at causally masked positions.
    """
    B, T = input_ids.shape
    x = model.wte(input_ids)
    results: list[tuple[torch.Tensor, torch.Tensor]] = []

    for block in model._h:
        attn_input = block.rms_1(x)
        attn = block.attn

        q = attn.q_proj(attn_input).view(B, T, attn.n_head, attn.head_dim).transpose(1, 2)
        k = (
            attn.k_proj(attn_input)
            .view(B, T, attn.n_key_value_heads, attn.head_dim)
            .transpose(1, 2)
        )
        v = (
            attn.v_proj(attn_input)
            .view(B, T, attn.n_key_value_heads, attn.head_dim)
            .transpose(1, 2)
        )

        position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)
        cos = attn.rotary_cos[position_ids].to(q.dtype)  # pyright: ignore[reportIndexIssue]
        sin = attn.rotary_sin[position_ids].to(q.dtype)  # pyright: ignore[reportIndexIssue]
        q, k = attn.apply_rotary_pos_emb(q, k, cos, sin)

        if attn.repeat_kv_heads > 1:
            k = k.repeat_interleave(attn.repeat_kv_heads, dim=1)
            v = v.repeat_interleave(attn.repeat_kv_heads, dim=1)

        logits = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(attn.head_dim))
        logits = logits.masked_fill(attn.bias[:, :, :T, :T] == 0, float("-inf"))  # pyright: ignore[reportIndexIssue]
        att = F.softmax(logits, dim=-1)
        results.append((att, logits))

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, attn.n_embd)
        y = attn.o_proj(y)
        x = x + y
        x = x + block.mlp(block.rms_2(x))

    return results
