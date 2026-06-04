"""Config for the vendored Llama-3.1 decomposition target."""

from typing import Literal

from param_decomp.base_config import BaseConfig


class Llama3RopeScaling(BaseConfig):
    """The "llama3" RoPE frequency rescaling (Llama-3.1+). Reshapes inv_freq by wavelength:
    low-frequency components divided by `factor`, high-frequency untouched, smooth interpolation
    between. `original_max_position_embeddings` is the pre-scaling context the thresholds are
    defined against, NOT the actual sequence length."""

    factor: float = 8.0
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    original_max_position_embeddings: int = 8192


class VendoredLlamaConfig(BaseConfig):
    model_type: Literal["VendoredLlama"]
    max_position_embeddings: int = 131072  # Llama-3.1's native context; only a forward sanity cap
    vocab_size: int = 128256
    n_layer: int = 32
    n_head: int = 32
    n_key_value_heads: int = 8
    n_embd: int = 4096
    n_intermediate: int = 14336
    rope_theta: float = 500000.0
    rope_scaling: Llama3RopeScaling | None = Llama3RopeScaling()
    rms_norm_eps: float = 1e-5
