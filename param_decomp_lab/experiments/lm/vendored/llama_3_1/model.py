"""The plain Llama-3.1 architecture (RMSNorm, RoPE with the "llama3" frequency scaling,
grouped-query attention, SwiGLU MLP, untied `lm_head`), specialised for decomposition training:
no KV cache, full causal forward (RoPE cos/sin computed per-forward from `inv_freq`, HF-style —
no precomputed table, no `block_size` cap baked into the model). `componentize_llama`
(in `components.py`) turns a frozen `VendoredLlama` into a mask-threading `ComponentLlama`.

Module paths match HF with the `model.` prefix stripped (e.g. `layers.18.mlp.gate_proj`), so
`from_hf_pretrained` is a direct `load_state_dict` after a prefix strip.
"""

import math
from typing import Any, override

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor, nn

from param_decomp_lab.distributed import log0
from param_decomp_lab.experiments.lm.vendored.llama_3_1.config import (
    Llama3RopeScaling,
    VendoredLlamaConfig,
)

# pyright: reportUninitializedInstanceVariable=false

# -----------------------------------------------------------------------------------------------
# Numeric kernels copied VERBATIM from HuggingFace transformers v4.57.3 (do not reimplement):
#   transformers/models/llama/modeling_llama.py — LlamaRMSNorm, rotate_half, apply_rotary_pos_emb,
#                                                  repeat_kv, and the LlamaRotaryEmbedding.forward
#                                                  cos/sin computation
#   transformers/modeling_rope_utils.py         — _compute_default_rope_parameters /
#                                                  _compute_llama3_parameters (the inv_freq + llama3
#                                                  wavelength rescaling)
# Only the plumbing is adapted (config-object attribute reads → explicit scalars; the rotary
# `forward` is inlined into `_attend` without the KV-cache `position_ids` path). Every
# COMPUTATIONAL line is upstream's — re-copy from the same files on a transformers bump.
# -----------------------------------------------------------------------------------------------


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """LlamaRMSNorm is equivalent to T5LayerNorm"""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    @override
    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def _vendored_config_from_hf(hf_cfg: Any) -> VendoredLlamaConfig:
    """Translate a transformers `LlamaConfig` (dynamic attrs, hence `Any`) to the vendored config."""
    assert hf_cfg.model_type == "llama", f"expected a llama config, got {hf_cfg.model_type}"
    rs = hf_cfg.rope_scaling
    scaling = (
        Llama3RopeScaling(
            factor=rs["factor"],
            low_freq_factor=rs["low_freq_factor"],
            high_freq_factor=rs["high_freq_factor"],
            original_max_position_embeddings=rs["original_max_position_embeddings"],
        )
        if rs is not None
        else None
    )
    assert not hf_cfg.tie_word_embeddings, "vendored Llama assumes an untied lm_head"
    return VendoredLlamaConfig(
        model_type="VendoredLlama",
        max_position_embeddings=hf_cfg.max_position_embeddings,
        vocab_size=hf_cfg.vocab_size,
        n_layer=hf_cfg.num_hidden_layers,
        n_head=hf_cfg.num_attention_heads,
        n_key_value_heads=hf_cfg.num_key_value_heads,
        n_embd=hf_cfg.hidden_size,
        n_intermediate=hf_cfg.intermediate_size,
        rope_theta=hf_cfg.rope_theta,
        rope_scaling=scaling,
        rms_norm_eps=hf_cfg.rms_norm_eps,
    )


def _llama3_inv_freq(head_dim: int, base: float, scaling: Llama3RopeScaling | None) -> Tensor:
    """inv_freq for RoPE. Body verbatim from `_compute_default_rope_parameters` +
    `_compute_llama3_parameters` (config reads replaced by the passed scalars)."""
    dim = head_dim
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(dtype=torch.float) / dim)
    )
    if scaling is None:
        return inv_freq
    factor = scaling.factor  # `8` in the original implementation
    low_freq_factor = scaling.low_freq_factor  # `1` in the original implementation
    high_freq_factor = scaling.high_freq_factor  # `4` in the original implementation
    old_context_len = (
        scaling.original_max_position_embeddings
    )  # `8192` in the original implementation

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq
    # wavelen < high_freq_wavelen: do nothing
    # wavelen > low_freq_wavelen: divide by factor
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    # otherwise: interpolate between the two, using a smooth factor
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smoothed_inv_freq = (
        1 - smooth_factor
    ) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
    return inv_freq_llama


def rotate_half(x: Tensor) -> Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, unsqueeze_dim: int = 1
) -> tuple[Tensor, Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep): (batch, num_key_value_heads, seqlen,
    head_dim) -> (batch, num_attention_heads, seqlen, head_dim)."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttention(nn.Module):
    # q/k/v are separate projections (k/v narrower under GQA) — HF Llama is already unfused
    # (unlike GPT-2's fused c_attn), so each is an independent decomposition target with no
    # split step needed. Thin shell: no KV cache, sdpa only, RoPE inlined from the copied kernels.
    inv_freq: Tensor

    def __init__(self, config: VendoredLlamaConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_key_value_heads
        self.head_dim = config.n_embd // config.n_head
        self.n_rep = self.n_head // self.n_kv_head
        self.q_proj = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, config.n_embd, bias=False)
        inv_freq = _llama3_inv_freq(self.head_dim, config.rope_theta, config.rope_scaling)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rope_cos_sin(self, x: Tensor, seq_len: int) -> tuple[Tensor, Tensor]:
        """cos/sin for positions [0, seq_len). Body verbatim from `LlamaRotaryEmbedding.forward`
        with position_ids = arange(seq_len) and attention_scaling = 1.0 (llama3)."""
        position_ids = torch.arange(seq_len, device=x.device)[None, :]
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def _attend(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        B, T, _ = q.shape
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        cos, sin = self._rope_cos_sin(q, T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)


class LlamaMLP(nn.Module):
    def __init__(self, config: VendoredLlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embd, config.n_intermediate, bias=False)
        self.up_proj = nn.Linear(config.n_embd, config.n_intermediate, bias=False)
        self.down_proj = nn.Linear(config.n_intermediate, config.n_embd, bias=False)

    @override
    def forward(self, x: Float[Tensor, "... dim"]) -> Float[Tensor, "... dim"]:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaBlock(nn.Module):
    def __init__(self, config: VendoredLlamaConfig):
        super().__init__()
        self.input_layernorm = LlamaRMSNorm(config.n_embd, config.rms_norm_eps)
        self.self_attn = LlamaAttention(config)
        self.post_attention_layernorm = LlamaRMSNorm(config.n_embd, config.rms_norm_eps)
        self.mlp = LlamaMLP(config)

    @override
    def forward(self, x: Float[Tensor, "b t d"]) -> Float[Tensor, "b t d"]:
        h = self.input_layernorm(x)
        a = self.self_attn
        x = x + a.o_proj(a._attend(a.q_proj(h), a.k_proj(h), a.v_proj(h)))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class VendoredLlama(nn.Module):
    """Plain Llama-3.1 target (untied `lm_head`). `componentize_llama` turns it into a
    `ComponentLlama` with a mask-threading forward."""

    def __init__(self, config: VendoredLlamaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)
        self._layers: list[LlamaBlock] = [LlamaBlock(config) for _ in range(config.n_layer)]
        self.layers = nn.ModuleList(self._layers)
        self.norm = LlamaRMSNorm(config.n_embd, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self._use_activation_checkpointing: bool = False

    def enable_activation_checkpointing(self) -> None:
        self._use_activation_checkpointing = True

    @override
    def forward(self, idx: Int[Tensor, "b t"]) -> Float[Tensor, "b t vocab"]:
        _b, t = idx.size()
        assert t <= self.config.max_position_embeddings, (
            f"seq len {t} > max_position_embeddings {self.config.max_position_embeddings}"
        )
        x = self.embed_tokens(idx)
        for block in self._layers:
            x = block(x)
        return self.lm_head(self.norm(x))

    @classmethod
    def from_hf_pretrained(cls, model_name: str) -> "VendoredLlama":
        from transformers import LlamaForCausalLM

        log0(f"loading HF weights into vendored Llama: {model_name}")
        # local_files_only: the target is vendored and pre-cached in HF_HUB_CACHE, so never
        # touch HF Hub at load time. Without this, every rank issues etag/metadata requests —
        # an N-rank thunderherd that read-times-out against the HF CDN and stalls launch/resume.
        # (Pre-cache the model once if the cache is cold; we never download it at run time.)
        hf = LlamaForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, local_files_only=True
        )
        model = cls(_vendored_config_from_hf(hf.config))
        stripped = {k.removeprefix("model."): v for k, v in hf.state_dict().items()}
        missing, unexpected = model.load_state_dict(stripped, strict=False)
        # persistent=False rotary buffers are absent from both sides; nothing real may be missing.
        assert not missing, f"missing keys loading HF Llama: {missing}"
        assert not unexpected, f"unexpected keys loading HF Llama: {unexpected}"
        del hf
        return model

    @classmethod
    def from_hf_config_random(cls, model_name: str) -> "VendoredLlama":
        """Random-init at the HF model's config shapes — reads only the cached config json.

        Benchmarking scaffolding (`target.spec kind: random_weights_in_vendored`):
        FLOP-identical to `from_hf_pretrained` without the N-rank weight load.
        """
        from transformers import LlamaConfig

        log0(f"random-init vendored Llama at the config shapes of: {model_name}")
        hf_cfg = LlamaConfig.from_pretrained(model_name, local_files_only=True)
        return cls(_vendored_config_from_hf(hf_cfg))
