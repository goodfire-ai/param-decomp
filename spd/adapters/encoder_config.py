"""Encoder configuration for transcoder architectures.

Originally by Bart Bussmann, vendored from https://github.com/bartbussmann/nn_decompositions (MIT license).
Only EncoderConfig is used; CLTConfig and SAEConfig are omitted.
"""

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass
class EncoderConfig:
    """Base config for encoder architectures (SAE and Transcoder).

    All fields are required — values come from the config.json saved with each checkpoint.
    """

    input_size: int
    output_size: int
    dict_size: int
    encoder_type: Literal["vanilla", "topk", "batchtopk", "jumprelu"]
    seed: int
    batch_size: int
    lr: float
    num_tokens: int
    l1_coeff: float
    beta1: float
    beta2: float
    max_grad_norm: float
    device: str
    dtype: torch.dtype
    n_batches_to_dead: int
    input_unit_norm: bool
    pre_enc_bias: bool
    top_k: int
    top_k_aux: int
    aux_penalty: float
    bandwidth: float
    run_name: str | None
    wandb_project: str
    perf_log_freq: int
    checkpoint_freq: int | Literal["final"]
    n_eval_seqs: int

    @property
    def name(self) -> str:
        if self.run_name is not None:
            return self.run_name
        base = f"{self.dict_size}_{self.encoder_type}"
        if self.encoder_type in ("topk", "batchtopk"):
            base += f"_k{self.top_k}"
        return f"{base}_{self.lr}"
