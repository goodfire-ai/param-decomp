"""Encoder configuration for transcoder architectures.

Vendored from https://github.com/bartbussmann/nn_decompositions (MIT license).
Only EncoderConfig is used; CLTConfig and SAEConfig are omitted.
"""

from dataclasses import dataclass, field
from typing import Literal

import torch


@dataclass
class EncoderConfig:
    """Base config for encoder architectures (SAE and Transcoder)."""

    # Architecture
    input_size: int
    output_size: int
    dict_size: int = 12288

    # Encoder type
    encoder_type: Literal["vanilla", "topk", "batchtopk", "jumprelu"] = "topk"

    # Training
    seed: int = 49
    batch_size: int = 4096
    lr: float = 3e-4
    num_tokens: int = int(1e9)
    l1_coeff: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.99
    max_grad_norm: float = 1.0

    # Device
    device: str = "cuda:0"
    dtype: torch.dtype = field(default=torch.float32)

    # Dead feature tracking
    n_batches_to_dead: int = 50

    # Optional features
    input_unit_norm: bool = False
    pre_enc_bias: bool = False

    # TopK specific
    top_k: int = 32
    top_k_aux: int = 512
    aux_penalty: float = 1 / 32

    # JumpReLU specific
    bandwidth: float = 0.001

    # Logging
    run_name: str | None = None
    wandb_project: str = "encoders"
    perf_log_freq: int = 1000
    checkpoint_freq: int | Literal["final"] = "final"
    n_eval_seqs: int = 8

    @property
    def name(self) -> str:
        if self.run_name is not None:
            return self.run_name
        base = f"{self.dict_size}_{self.encoder_type}"
        if self.encoder_type in ("topk", "batchtopk"):
            base += f"_k{self.top_k}"
        return f"{base}_{self.lr}"
