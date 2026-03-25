"""Rank-1 LoRA baseline for comparison with SPD component editing."""

import random
from typing import Any

import torch
import torch.nn.functional as F
from jaxtyping import Int
from torch import Tensor


class LoRATrainer:
    """Rank-1 LoRA on a single linear layer. Reusable across runs without model reload."""

    def __init__(
        self,
        target_model: torch.nn.Module,
        layer_path: str,
        reg_seqs: list[Int[Tensor, " seq"]],
        lr: float = 1e-3,
    ):
        self.target_model = target_model
        self.target_model.eval()

        mod: Any = target_model
        for part in layer_path.split("."):
            mod = getattr(mod, part)
        assert isinstance(mod, torch.nn.Linear)
        self.linear = mod

        d_out, d_in = int(self.linear.weight.shape[0]), int(self.linear.weight.shape[1])
        self.A = torch.randn(1, d_in, device="cuda") * 0.01
        self.B = torch.zeros(d_out, 1, device="cuda")
        self.A.requires_grad = True
        self.B.requires_grad = True
        self.lr = lr

        self._hook = self.linear.register_forward_hook(self._fwd_hook)
        self.optimizer = torch.optim.AdamW([self.A, self.B], lr=lr)

        max_len = max(t.shape[0] for t in reg_seqs)
        self.reg_batch = torch.zeros(len(reg_seqs), max_len, dtype=torch.long, device="cuda")
        self.reg_lens = [t.shape[0] for t in reg_seqs]
        for i, t in enumerate(reg_seqs):
            self.reg_batch[i, : t.shape[0]] = t

        self.reg_base_batch = self._compute_reg_baselines()

    def _compute_reg_baselines(self) -> Tensor:
        base_list = []
        with torch.no_grad():
            for i, seq_len in enumerate(self.reg_lens):
                t = self.reg_batch[i, :seq_len]
                base_list.append(self.forward(t.unsqueeze(0))[0].softmax(-1))
        vocab = base_list[0].shape[-1]
        result = torch.zeros(len(self.reg_lens), self.reg_batch.shape[1], vocab, device="cuda")
        for i, (bl, seq_len) in enumerate(zip(base_list, self.reg_lens, strict=True)):
            result[i, :seq_len] = bl[:seq_len]
        return result

    def _fwd_hook(self, _mod: torch.nn.Module, _inp: tuple[Any, ...], out: Tensor) -> Tensor:
        return out + (_inp[0] @ self.A.T) @ self.B.T

    def forward(self, tokens: Tensor) -> Tensor:
        return self.target_model(tokens)[0]

    def train_step(
        self,
        train_seqs: list[tuple[Int[Tensor, " seq"], list[int]]],
        kl_weight: float = 0.0,
        batch_size: int = 8,
    ) -> float:
        """One training step with optional KL regularization. Returns total loss."""
        idxs = random.sample(range(len(train_seqs)), min(batch_size, len(train_seqs)))
        batch = [train_seqs[j] for j in idxs]
        max_len = max(t.shape[0] for t, _ in batch)
        batch_t = torch.zeros(len(batch), max_len, dtype=torch.long, device="cuda")
        batch_pos = []
        for i, (t, pos) in enumerate(batch):
            batch_t[i, : t.shape[0]] = t
            batch_pos.append(pos)

        logits = self.forward(batch_t)
        ce = torch.tensor(0.0, device="cuda")
        for i, pos in enumerate(batch_pos):
            ce = ce + F.cross_entropy(
                logits[i, pos], batch_t[i, torch.tensor(pos, device="cuda") + 1]
            )
        ce = ce / len(batch)

        kl_reg = torch.tensor(0.0, device="cuda")
        if kl_weight > 0:
            logits_reg = self.forward(self.reg_batch)
            probs_reg = logits_reg.softmax(-1)
            kl_all = (
                probs_reg * ((probs_reg + 1e-10).log() - (self.reg_base_batch + 1e-10).log())
            ).sum(-1)
            for i, seq_len in enumerate(self.reg_lens):
                kl_reg = kl_reg + kl_all[i, :seq_len].mean()
            kl_reg = kl_reg / len(self.reg_lens)

        total = ce + kl_weight * kl_reg
        self.optimizer.zero_grad()
        total.backward()
        self.optimizer.step()
        return total.item()

    def reset(self) -> None:
        """Reset LoRA params to init. No model reload needed."""
        d_in = int(self.linear.weight.shape[1])
        with torch.no_grad():
            self.A.copy_(torch.randn(1, d_in, device="cuda") * 0.01)
            self.B.zero_()
        self.optimizer = torch.optim.AdamW([self.A, self.B], lr=self.lr)
        self.reg_base_batch = self._compute_reg_baselines()

    def cleanup(self) -> None:
        """Remove forward hook."""
        self._hook.remove()
