"""BatchTopK Transcoder nn.Module and EncoderConfig.

Originally by Bart Bussmann, vendored from https://github.com/bartbussmann/nn_decompositions (MIT license).
"""

from dataclasses import dataclass
from typing import Any, Literal, override

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class EncoderConfig:
    """Config for BatchTopK transcoder checkpoints.

    All fields are required — values come from the config.json saved with each checkpoint.
    """

    input_size: int
    output_size: int
    dict_size: int
    encoder_type: Literal["batchtopk"]
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
        return f"{self.dict_size}_batchtopk_k{self.top_k}_{self.lr}"


class BatchTopKTranscoder(nn.Module):
    """BatchTopK sparse transcoder (encoder-decoder).

    Supports both SAE mode (input = target) and Transcoder mode (input != target).
    """

    def __init__(self, cfg: EncoderConfig):
        super().__init__()

        self.cfg = cfg
        torch.manual_seed(cfg.seed)

        self.input_size = cfg.input_size
        self.output_size = cfg.output_size
        self.dict_size = cfg.dict_size

        self.b_dec = nn.Parameter(torch.zeros(cfg.output_size))
        self.b_enc = nn.Parameter(torch.zeros(cfg.dict_size))
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(cfg.input_size, cfg.dict_size))
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(cfg.dict_size, cfg.output_size))
        )
        if cfg.input_size == cfg.output_size:
            self.W_dec.data[:] = self.W_enc.t().data
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        self.num_batches_not_active = torch.zeros((cfg.dict_size,)).to(cfg.device)

        self.to(cfg.dtype).to(cfg.device)

    def encode(self, x: Tensor) -> Tensor:
        use_pre_enc_bias = self.cfg.pre_enc_bias and self.input_size == self.output_size
        x_enc = x - self.b_dec if use_pre_enc_bias else x
        acts = F.relu(x_enc @ self.W_enc + self.b_enc)
        acts_topk = torch.topk(acts.flatten(), self.cfg.top_k * x.shape[0], dim=-1)
        return (
            torch.zeros_like(acts.flatten())
            .scatter(-1, acts_topk.indices, acts_topk.values)
            .reshape(acts.shape)
        )

    def decode(self, acts: Tensor) -> Tensor:
        return acts @ self.W_dec + self.b_dec

    @override
    def forward(self, x_in: Tensor, y_target: Tensor) -> dict[str, Any]:
        x_in_proc = x_in
        y_target_proc = y_target
        y_mean, y_std = None, None
        if self.cfg.input_unit_norm:
            x_mean = x_in.mean(dim=-1, keepdim=True)
            x_in_proc = (x_in - x_mean) / (x_in.std(dim=-1, keepdim=True) + 1e-5)
            y_mean = y_target.mean(dim=-1, keepdim=True)
            y_std = y_target.std(dim=-1, keepdim=True)
            y_target_proc = (y_target - y_mean) / (y_std + 1e-5)

        acts_dense = F.relu(x_in_proc @ self.W_enc + self.b_enc)
        acts_topk = torch.topk(acts_dense.flatten(), self.cfg.top_k * x_in.shape[0], dim=-1)
        acts = (
            torch.zeros_like(acts_dense.flatten())
            .scatter(-1, acts_topk.indices, acts_topk.values)
            .reshape(acts_dense.shape)
        )

        y_pred = acts @ self.W_dec + self.b_dec
        y_pred_out = y_pred
        if y_mean is not None:
            assert y_std is not None
            y_pred_out = y_pred * y_std + y_mean

        self.num_batches_not_active += (acts.sum(0) == 0).float()
        self.num_batches_not_active[acts.sum(0) > 0] = 0

        l2_loss = (y_pred.float() - y_target_proc.float()).pow(2).mean()
        l0_norm = (acts > 0).float().sum(-1).mean()
        l1_loss = self.cfg.l1_coeff * acts.float().abs().sum(-1).mean()

        dead_features = self.num_batches_not_active >= self.cfg.n_batches_to_dead
        aux_loss: Tensor
        if dead_features.sum() > 0:
            residual = y_target_proc.float() - y_pred.float()
            acts_topk_aux = torch.topk(
                acts_dense[:, dead_features],
                min(self.cfg.top_k_aux, int(dead_features.sum().item())),
                dim=-1,
            )
            acts_aux = torch.zeros_like(acts_dense[:, dead_features]).scatter(
                -1, acts_topk_aux.indices, acts_topk_aux.values
            )
            y_pred_aux = acts_aux @ self.W_dec[dead_features]
            aux_loss = self.cfg.aux_penalty * (y_pred_aux.float() - residual.float()).pow(2).mean()
        else:
            aux_loss = torch.tensor(0, dtype=y_target.dtype, device=y_target.device)

        return {
            "output": y_pred_out,
            "feature_acts": acts,
            "num_dead_features": (self.num_batches_not_active > self.cfg.n_batches_to_dead).sum(),
            "loss": l2_loss + l1_loss + aux_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": acts.float().abs().sum(-1).mean(),
            "l1_loss": l1_loss,
            "aux_loss": aux_loss,
        }
