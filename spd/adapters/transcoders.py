"""Transcoder nn.Module implementations (Vanilla, TopK, BatchTopK, JumpReLU).

Vendored from https://github.com/bartbussmann/nn_decompositions (MIT license).
"""

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from spd.adapters.encoder_config import EncoderConfig


class SharedTranscoder(nn.Module):
    """Base class for encoder-decoder models (SAE and Transcoder).

    Supports both SAE mode (input = target) and Transcoder mode (input != target).
    All subclasses use forward(x_in, y_target) signature.
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
        # Initialize W_dec from W_enc only if input_size == output_size (SAE case)
        if cfg.input_size == cfg.output_size:
            self.W_dec.data[:] = self.W_enc.t().data
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        self.num_batches_not_active = torch.zeros((cfg.dict_size,)).to(cfg.device)

        self.to(cfg.dtype).to(cfg.device)

    def encode(
        self, x: torch.Tensor, return_dense: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encode input to sparse activations. Subclasses must implement.

        If return_dense=True, also returns the dense (pre-sparsification) activations
        as a second element.
        """
        raise NotImplementedError

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.W_dec + self.b_dec

    def preprocess_input(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if self.cfg.input_unit_norm:
            x_mean = x.mean(dim=-1, keepdim=True)
            x = x - x_mean
            x_std = x.std(dim=-1, keepdim=True)
            x = x / (x_std + 1e-5)
            return x, x_mean, x_std
        return x, None, None

    def postprocess_output(
        self, out: torch.Tensor, mean: torch.Tensor | None, std: torch.Tensor | None
    ) -> torch.Tensor:
        if self.cfg.input_unit_norm and mean is not None:
            return out * std + mean
        return out

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed

    def update_inactive_features(self, acts: torch.Tensor):
        self.num_batches_not_active += (acts.sum(0) == 0).float()
        self.num_batches_not_active[acts.sum(0) > 0] = 0

    def _get_auxiliary_loss(
        self, y_target: torch.Tensor, y_pred: torch.Tensor, acts: torch.Tensor
    ) -> torch.Tensor:
        dead_features = self.num_batches_not_active >= self.cfg.n_batches_to_dead
        if dead_features.sum() > 0:
            residual = y_target.float() - y_pred.float()
            acts_topk_aux = torch.topk(
                acts[:, dead_features],
                min(self.cfg.top_k_aux, dead_features.sum()),
                dim=-1,
            )
            acts_aux = torch.zeros_like(acts[:, dead_features]).scatter(
                -1, acts_topk_aux.indices, acts_topk_aux.values
            )
            y_pred_aux = acts_aux @ self.W_dec[dead_features]
            return self.cfg.aux_penalty * (y_pred_aux.float() - residual.float()).pow(2).mean()
        return torch.tensor(0, dtype=y_target.dtype, device=y_target.device)

    def _build_loss_dict(
        self,
        y_target: torch.Tensor,
        y_pred: torch.Tensor,
        acts: torch.Tensor,
        y_pred_out: torch.Tensor,
        l0_norm: torch.Tensor,
        extra_losses: dict[str, torch.Tensor] | None = None,
    ) -> dict:
        l2_loss = (y_pred.float() - y_target.float()).pow(2).mean()
        l1_norm = acts.float().abs().sum(-1).mean()
        num_dead = (self.num_batches_not_active > self.cfg.n_batches_to_dead).sum()

        loss = l2_loss
        if extra_losses:
            loss = loss + sum(extra_losses.values())

        result = {
            "output": y_pred_out,
            "feature_acts": acts,
            "num_dead_features": num_dead,
            "loss": loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
        }
        if extra_losses:
            result.update(extra_losses)
        return result


class VanillaTranscoder(SharedTranscoder):
    def encode(self, x: torch.Tensor, return_dense: bool = False):
        use_pre_enc_bias = self.cfg.pre_enc_bias and self.input_size == self.output_size
        x_enc = x - self.b_dec if use_pre_enc_bias else x
        acts = F.relu(x_enc @ self.W_enc + self.b_enc)
        return (acts, acts) if return_dense else acts

    def forward(self, x_in: torch.Tensor, y_target: torch.Tensor) -> dict:
        x_in, _, _ = self.preprocess_input(x_in)
        y_target, y_mean, y_std = self.preprocess_input(y_target)

        acts = self.encode(x_in)
        y_pred = self.decode(acts)
        y_pred_out = self.postprocess_output(y_pred, y_mean, y_std)

        self.update_inactive_features(acts)

        l0_norm = (acts > 0).float().sum(-1).mean()
        l1_loss = self.cfg.l1_coeff * acts.float().abs().sum(-1).mean()
        return self._build_loss_dict(
            y_target,
            y_pred,
            acts,
            y_pred_out,
            l0_norm,
            extra_losses={"l1_loss": l1_loss},
        )


class TopKTranscoder(SharedTranscoder):
    def encode(self, x: torch.Tensor, return_dense: bool = False):
        use_pre_enc_bias = self.cfg.pre_enc_bias and self.input_size == self.output_size
        x_enc = x - self.b_dec if use_pre_enc_bias else x
        acts = F.relu(x_enc @ self.W_enc + self.b_enc)
        acts_topk = torch.topk(acts, self.cfg.top_k, dim=-1)
        acts_sparse = torch.zeros_like(acts).scatter(-1, acts_topk.indices, acts_topk.values)
        return (acts_sparse, acts) if return_dense else acts_sparse

    def forward(self, x_in: torch.Tensor, y_target: torch.Tensor) -> dict:
        x_in, _, _ = self.preprocess_input(x_in)
        y_target, y_mean, y_std = self.preprocess_input(y_target)

        acts, acts_dense = self.encode(x_in, return_dense=True)
        y_pred = self.decode(acts)
        y_pred_out = self.postprocess_output(y_pred, y_mean, y_std)

        self.update_inactive_features(acts)

        l0_norm = (acts > 0).float().sum(-1).mean()
        l1_loss = self.cfg.l1_coeff * acts.float().abs().sum(-1).mean()
        aux_loss = self._get_auxiliary_loss(y_target, y_pred, acts_dense)
        return self._build_loss_dict(
            y_target,
            y_pred,
            acts,
            y_pred_out,
            l0_norm,
            extra_losses={"l1_loss": l1_loss, "aux_loss": aux_loss},
        )


class BatchTopKTranscoder(SharedTranscoder):
    def encode(self, x: torch.Tensor, return_dense: bool = False):
        use_pre_enc_bias = self.cfg.pre_enc_bias and self.input_size == self.output_size
        x_enc = x - self.b_dec if use_pre_enc_bias else x
        acts = F.relu(x_enc @ self.W_enc + self.b_enc)
        acts_topk = torch.topk(acts.flatten(), self.cfg.top_k * x.shape[0], dim=-1)
        acts_sparse = (
            torch.zeros_like(acts.flatten())
            .scatter(-1, acts_topk.indices, acts_topk.values)
            .reshape(acts.shape)
        )
        return (acts_sparse, acts) if return_dense else acts_sparse

    def forward(self, x_in: torch.Tensor, y_target: torch.Tensor) -> dict:
        x_in, _, _ = self.preprocess_input(x_in)
        y_target, y_mean, y_std = self.preprocess_input(y_target)

        acts, acts_dense = self.encode(x_in, return_dense=True)
        y_pred = self.decode(acts)
        y_pred_out = self.postprocess_output(y_pred, y_mean, y_std)

        self.update_inactive_features(acts)

        l0_norm = (acts > 0).float().sum(-1).mean()
        l1_loss = self.cfg.l1_coeff * acts.float().abs().sum(-1).mean()
        aux_loss = self._get_auxiliary_loss(y_target, y_pred, acts_dense)
        return self._build_loss_dict(
            y_target,
            y_pred,
            acts,
            y_pred_out,
            l0_norm,
            extra_losses={"l1_loss": l1_loss, "aux_loss": aux_loss},
        )


class RectangleFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return ((x > -0.5) & (x < 0.5)).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(x <= -0.5) | (x >= 0.5)] = 0
        return grad_input


class JumpReLUFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, log_threshold, bandwidth):
        ctx.save_for_backward(x, log_threshold, torch.tensor(bandwidth))
        threshold = torch.exp(log_threshold)
        return x * (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, log_threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        threshold = torch.exp(log_threshold)
        x_grad = (x > threshold).float() * grad_output
        threshold_grad = (
            -(threshold / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None


class JumpReLUActivation(nn.Module):
    def __init__(self, feature_size: int, bandwidth: float, device: str = "cpu"):
        super().__init__()
        self.log_threshold = nn.Parameter(torch.zeros(feature_size, device=device))
        self.bandwidth = bandwidth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return JumpReLUFunction.apply(x, self.log_threshold, self.bandwidth)


class StepFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, log_threshold, bandwidth):
        ctx.save_for_backward(x, log_threshold, torch.tensor(bandwidth))
        threshold = torch.exp(log_threshold)
        return (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, log_threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        threshold = torch.exp(log_threshold)
        x_grad = torch.zeros_like(x)
        threshold_grad = (
            -(1.0 / bandwidth) * RectangleFunction.apply((x - threshold) / bandwidth) * grad_output
        )
        return x_grad, threshold_grad, None


class JumpReLUTranscoder(SharedTranscoder):
    def __init__(self, cfg: EncoderConfig):
        super().__init__(cfg)
        self.jumprelu = JumpReLUActivation(cfg.dict_size, cfg.bandwidth, cfg.device)

    def encode(self, x: torch.Tensor, return_dense: bool = False):
        use_pre_enc_bias = self.cfg.pre_enc_bias and self.input_size == self.output_size
        x_enc = x - self.b_dec if use_pre_enc_bias else x
        pre_acts = F.relu(x_enc @ self.W_enc + self.b_enc)
        acts = self.jumprelu(pre_acts)
        return (acts, pre_acts) if return_dense else acts

    def forward(self, x_in: torch.Tensor, y_target: torch.Tensor) -> dict:
        x_in, _, _ = self.preprocess_input(x_in)
        y_target, y_mean, y_std = self.preprocess_input(y_target)

        acts, pre_acts = self.encode(x_in, return_dense=True)
        y_pred = self.decode(acts)
        y_pred_out = self.postprocess_output(y_pred, y_mean, y_std)

        self.update_inactive_features(acts)

        l0_norm = (
            StepFunction.apply(pre_acts, self.jumprelu.log_threshold, self.cfg.bandwidth)
            .sum(dim=-1)
            .mean()
        )
        sparsity_loss = self.cfg.l1_coeff * l0_norm
        return self._build_loss_dict(
            y_target,
            y_pred,
            acts,
            y_pred_out,
            l0_norm,
            extra_losses={"sparsity_loss": sparsity_loss},
        )
