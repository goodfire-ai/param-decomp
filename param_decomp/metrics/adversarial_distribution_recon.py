"""Adversarial-distribution reconstruction loss, its config, and the adversary head.

A small head branches off the CI fn's transformer trunk (it consumes the trunk's
post-block features, detached, so only the head's own params train against this loss) and
emits the parameters of a per-component probability distribution. A reparameterized sample
from that distribution is the adversarial mask source: `mask = ci + (1 - ci) * source`.
Components and the CI fn *descend* the resulting recon loss; the adversary head *ascends*
it via its own AdamW, stepped from `after_backward` on the gradients the outer
`total_loss.backward()` leaves on the head params (negated for ascent).

Two distributions (`distribution`):

- `gaussian_sigmoid`: the head emits `(mu, raw_sigma)`; `sigma = softplus(raw_sigma) + eps`;
  `z = mu + sigma * eps_noise` (explicit reparam) and `source = sigmoid(z)`.
- `beta`: the head emits `(raw_a, raw_b)`; `alpha, beta = softplus(.) + eps`; `source` is a
  `Beta(alpha, beta).rsample()` — supported on `[0, 1]`, with implicit reparameterization
  gradients (via the Gamma path) so gradients reach the head.

Unlike `PersistentPGDReconLoss` there is no separate adversary network, no persistent
per-datapoint source state, and no inner warmup loop — the trunk is shared with the CI fn
and the source is a fresh sample each step.
"""

from typing import Any, ClassVar, Literal, override

import torch
import torch.nn.functional as F
from jaxtyping import Float
from pydantic import Field, NonNegativeFloat, PositiveFloat, PositiveInt
from torch import Tensor, nn
from torch.autograd import Function
from torch.distributed import ReduceOp
from torch.distributions import Beta
from torch.nn.utils import clip_grad_norm_

from param_decomp.base_config import BaseConfig, Probability
from param_decomp.batch_and_loss_fns import ReconstructionLoss
from param_decomp.ci_fns import GlobalCiFnWrapper
from param_decomp.ci_nn_blocks import Linear
from param_decomp.component_model import ComponentModel
from param_decomp.components import EmbeddingComponents
from param_decomp.distributed import all_reduce, broadcast_tensor
from param_decomp.metrics.base import LossMetricConfig, Metric, MetricResult
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.persistent_pgd_state import get_ppgd_mask_infos
from param_decomp.metrics.stochastic_hidden_acts_recon import (
    calc_hidden_acts_mse,
    compute_per_module_metrics,
)
from param_decomp.schedule import ScheduleConfig, get_scheduled_value

AdversaryDistribution = Literal["deterministic", "gaussian_sigmoid", "beta"]
"""How the head turns its outputs into a source in `[0, 1]`.

- `deterministic`: `source = sigmoid(logit)` (no sampling) — exactly `gaussian_sigmoid` with
  sigma -> 0, so it isolates "does the *distribution* help" from the trunk/head architecture.
  Uses a plain sigmoid (not a hard/leaky one) so the gradient vanishes at saturation and the
  adversary's logits self-limit; a non-vanishing rail leak here drives logits to +-inf under
  the ascending + reverse-trunk setup.
- `gaussian_sigmoid`: `source = sigmoid(mu + sigma * eps)`.
- `beta`: `source = Beta(alpha, beta).rsample()`.
"""

TrunkGrad = Literal["detach", "reverse"]
"""How the adversary head couples to the shared CI trunk in the backward pass.

- `detach`: head reads stop-grad trunk features; only the head trains against this loss
  (the CI trunk stays a pure causal-importance representation).
- `reverse`: a gradient-reversal layer sits between trunk and head, so the defender's
  descent on the trunk becomes *ascent* on the source path — the trunk is co-opted to make
  attacks stronger (gives the adversary the trunk's depth, but the CI representation is no
  longer shaped purely for causal importance).
"""

AdversarySources = dict[str, Float[Tensor, "*batch_dims source_c"]]
# Per-module distribution params for logging: {module: {param_name: tensor}}.
DistParams = dict[str, dict[str, Float[Tensor, "*batch_dims source_c"]]]

# Per-channel scalars the head must emit for each parameterization.
_PARAMS_PER_CHANNEL: dict[AdversaryDistribution, int] = {
    "deterministic": 1,
    "gaussian_sigmoid": 2,
    "beta": 2,
}
# Softplus floor keeping sigma strictly positive.
_POSITIVE_EPS = 1e-4
# Beta concentration is mapped into [BETA_MIN, BETA_MAX] via a sigmoid: BETA_MIN >= 1 keeps
# the density unimodal (no boundary mass spikes) so `rsample`'s implicit-reparam gradients
# stay finite; the upper bound stops it running away. Both ends keep gradient (smooth map).
_BETA_MIN = 1.0
_BETA_MAX = 100.0
# Saturation band for `source_frac_saturated`.
_SATURATION_EPS = 1e-3


class _GradientReversal(Function):
    """Identity forward; scales the gradient by `-lambda` on the backward pass."""

    @override
    @staticmethod
    def forward(ctx: Any, x: Tensor, lambda_: float) -> Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @override
    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> tuple[Tensor, None]:
        return -ctx.lambda_ * grad_outputs[0], None


def _build_head(d_model: int, hidden_dims: list[int], out_dim: int) -> nn.Module:
    """Adversary readout on the trunk: a single linear layer, or an MLP if hidden_dims given."""
    if not hidden_dims:
        return Linear(d_model, out_dim, nonlinearity="linear")
    layers: list[nn.Module] = []
    in_dim = d_model
    for h in hidden_dims:
        layers.append(Linear(in_dim, h, nonlinearity="relu"))
        layers.append(nn.GELU())
        in_dim = h
    layers.append(Linear(in_dim, out_dim, nonlinearity="linear"))
    return nn.Sequential(*layers)


def _bounded_concentration(raw: Tensor) -> Tensor:
    """Map a raw head output into `[_BETA_MIN, _BETA_MAX]` smoothly (gradient everywhere)."""
    return _BETA_MIN + (_BETA_MAX - _BETA_MIN) * torch.sigmoid(raw)


class AdversaryHeadOptimizerConfig(BaseConfig):
    """AdamW hyperparameters + LR schedule for the adversary head.

    Kept here rather than in `configs.py` to avoid an import cycle through the loss-metric
    union. The schedule is keyed on the global training step, like `ci_fn_optimizer`.
    """

    lr_schedule: ScheduleConfig
    weight_decay: NonNegativeFloat = 0.0
    betas: tuple[Probability, Probability] = (0.9, 0.999)
    grad_clip_norm: PositiveFloat | None = None


class AdversarialDistributionReconLossConfig(LossMetricConfig):
    """Config for `AdversarialDistributionReconLoss`.

    `update()` returns `None` before `start_frac` of training (the head is dormant and not
    stepped). `n_samples` independent reparameterized draws are averaged per step.
    """

    type: Literal["AdversarialDistributionReconLoss"] = "AdversarialDistributionReconLoss"
    optimizer: AdversaryHeadOptimizerConfig
    distribution: AdversaryDistribution = "gaussian_sigmoid"
    trunk_grad: TrunkGrad = Field(
        default="detach",
        description=(
            "How the head couples to the CI trunk: `detach` (head-only) or `reverse` "
            "(gradient-reversal so the shared trunk is co-opted adversarially)."
        ),
    )
    grl_lambda: PositiveFloat = Field(
        default=1.0, description="Gradient-reversal scale; only used when trunk_grad='reverse'."
    )
    head_hidden_dims: list[PositiveInt] = Field(
        default_factory=list,
        description=(
            "Hidden layer widths for the adversary head MLP on the trunk features. Empty = a "
            "single linear readout (fine when trunk_grad='reverse' supplies depth)."
        ),
    )
    start_frac: Probability = 0.0
    n_samples: PositiveInt = 1


class _RunningMoments:
    """Streaming mean/std/min/max of a quantity, aggregated across all elements and ranks."""

    def __init__(self, device: str) -> None:
        self._sum = torch.zeros((), device=device)
        self._sumsq = torch.zeros((), device=device)
        self._count = torch.zeros((), device=device, dtype=torch.long)
        self._min = torch.tensor(float("inf"), device=device)
        self._max = torch.tensor(float("-inf"), device=device)

    def update(self, t: Tensor) -> None:
        t = t.detach()
        self._sum += t.sum()
        self._sumsq += t.square().sum()
        self._count += t.numel()
        self._min = torch.minimum(self._min, t.min())
        self._max = torch.maximum(self._max, t.max())

    def reduced(self, prefix: str) -> dict[str, Float[Tensor, ""]]:
        total = all_reduce(self._sum.clone())
        total_sq = all_reduce(self._sumsq.clone())
        count = all_reduce(self._count.clone().float())
        mean = total / count
        std = (total_sq / count - mean.square()).clamp_min(0.0).sqrt()
        return {
            f"{prefix}/mean": mean,
            f"{prefix}/std": std,
            f"{prefix}/min": all_reduce(self._min.clone(), ReduceOp.MIN),
            f"{prefix}/max": all_reduce(self._max.clone(), ReduceOp.MAX),
        }


class AdversaryHeadState:
    """The adversary head (a single `Linear` on the CI trunk), its AdamW, and the sampler.

    The head is *not* a submodule of the CI fn, so its params never enter the trainer's
    `ci_fn_optimizer` group — it is optimized only here, by ascent. It lives outside the DDP
    wrapper, so its params are broadcast at init and its grads all-reduced before each step.
    """

    def __init__(
        self,
        *,
        model: ComponentModel,
        device: str,
        use_delta_component: bool,
        distribution: AdversaryDistribution,
        trunk_grad: TrunkGrad,
        grl_lambda: float,
        head_hidden_dims: list[int],
        optimizer_cfg: AdversaryHeadOptimizerConfig,
        n_samples: int,
        reconstruction_loss: ReconstructionLoss,
    ) -> None:
        self._device = device
        self._distribution: AdversaryDistribution = distribution
        self._trunk_grad: TrunkGrad = trunk_grad
        self._grl_lambda = grl_lambda
        self._n_params = _PARAMS_PER_CHANNEL[distribution]
        self._n_samples = n_samples
        self._reconstruction_loss = reconstruction_loss
        self._lr_schedule = optimizer_cfg.lr_schedule
        self._grad_clip_norm = optimizer_cfg.grad_clip_norm

        ci_fn = model.ci_fn
        assert isinstance(ci_fn, GlobalCiFnWrapper), (
            "AdversarialDistributionReconLoss requires a global_shared_transformer CI fn "
            f"(the head shares its trunk); got {type(ci_fn).__name__}"
        )
        for name, component in model.components.items():
            assert not isinstance(component, EmbeddingComponents), (
                f"AdversarialDistributionReconLoss does not support embedding decomposition "
                f"targets (found at {name!r})."
            )
        self._ci_fn = ci_fn
        d_model = ci_fn.transformer.d_model

        # One source channel per component, plus one for the weight-delta channel when present
        # (the last channel, matching the `[..., :-1]` / `[..., -1]` split in get_ppgd_mask_infos).
        self._layer_order = sorted(model.module_to_c)
        self._source_c = {
            name: model.module_to_c[name] + (1 if use_delta_component else 0)
            for name in self._layer_order
        }
        out_dim = self._n_params * sum(self._source_c.values())
        self.head = _build_head(d_model, head_hidden_dims, out_dim).to(device)
        for param in self.head.parameters():
            broadcast_tensor(param.data)

        self.optimizer = torch.optim.AdamW(
            self.head.parameters(),
            lr=optimizer_cfg.lr_schedule.start_val,
            betas=optimizer_cfg.betas,
            weight_decay=optimizer_cfg.weight_decay,
        )

    def _head_params(
        self, pre_weight_acts: dict[str, Float[Tensor, "... d_in"]]
    ) -> dict[str, Float[Tensor, "*batch_dims source_c n_params"]]:
        """Per-module raw distribution params from the trunk + head.

        With `trunk_grad='detach'` the trunk is stop-grad (head-only ascent); with `'reverse'`
        a gradient-reversal layer routes the source-path gradient into the trunk as ascent.
        """
        match self._trunk_grad:
            case "detach":
                with torch.no_grad():
                    trunk_x, added_seq_dim = self._ci_fn.trunk_features(pre_weight_acts)
            case "reverse":
                trunk_x, added_seq_dim = self._ci_fn.trunk_features(pre_weight_acts)
                trunk_x = _GradientReversal.apply(trunk_x, self._grl_lambda)
        raw = self.head(trunk_x)
        if added_seq_dim:
            raw = raw.squeeze(-2)
        chunks = torch.split(
            raw, [self._n_params * self._source_c[name] for name in self._layer_order], dim=-1
        )
        out: dict[str, Tensor] = {}
        for name, chunk in zip(self._layer_order, chunks, strict=True):
            out[name] = chunk.reshape(*chunk.shape[:-1], self._source_c[name], self._n_params)
        return out

    def _sample(
        self, params: Float[Tensor, "*batch_dims source_c n_params"]
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Source in `[0, 1]` plus the named params for logging. Reparameterized where stochastic."""
        match self._distribution:
            case "deterministic":
                logit = params[..., 0]
                return torch.sigmoid(logit), {"logit": logit}
            case "gaussian_sigmoid":
                mu = params[..., 0]
                sigma = F.softplus(params[..., 1]) + _POSITIVE_EPS
                source = torch.sigmoid(mu + sigma * torch.randn_like(mu))
                return source, {"mu": mu, "sigma": sigma}
            case "beta":
                alpha = _bounded_concentration(params[..., 0])
                beta = _bounded_concentration(params[..., 1])
                source = Beta(alpha, beta).rsample()
                return source, {"alpha": alpha, "beta": beta}

    def generate_sources(self, head_params: dict[str, Tensor]) -> AdversarySources:
        return {name: self._sample(head_params[name])[0] for name in head_params}

    def distribution_params_and_sources(
        self, pre_weight_acts: dict[str, Float[Tensor, "... d_in"]]
    ) -> tuple[DistParams, AdversarySources]:
        head_params = self._head_params(pre_weight_acts)
        params: DistParams = {}
        sources: AdversarySources = {}
        for name, raw in head_params.items():
            source, named = self._sample(raw)
            params[name] = named
            sources[name] = source
        return params, sources

    def compute_recon_sum_and_n(
        self,
        model: ComponentModel,
        batch: Tensor,
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
        pre_weight_acts: dict[str, Float[Tensor, "... d_in"]],
        weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    ) -> tuple[Float[Tensor, ""], int]:
        """Recon forward returning `(sum_loss, n_examples)` summed over all adversary draws."""
        batch_dims = next(iter(ci.values())).shape[:-1]
        device = next(iter(ci.values())).device
        head_params = self._head_params(pre_weight_acts)
        sum_loss = torch.zeros((), device=device)
        n_examples = 0
        for _ in range(self._n_samples):
            mask_infos = get_ppgd_mask_infos(
                ci=ci,
                weight_deltas=weight_deltas,
                ppgd_sources=self.generate_sources(head_params),
                routing_masks="all",
                batch_dims=batch_dims,
            )
            out = model(batch, mask_infos=mask_infos)
            loss, n = self._reconstruction_loss(pred=out, target=target_out)
            sum_loss = sum_loss + loss
            n_examples += n
        return sum_loss, n_examples

    def update_lr(self, step: int, total_steps: int) -> None:
        lr = get_scheduled_value(step, total_steps, self._lr_schedule)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def ascend(self) -> None:
        """One gradient-ascent step using the grads the outer backward left on the head params.

        Negating the descent grad turns the AdamW step into ascent. Grads are averaged across
        ranks first (the head lives outside DDP), then optionally clipped.
        """
        for param in self.head.parameters():
            if param.grad is None:
                continue
            all_reduce(param.grad, op=ReduceOp.AVG)
            param.grad.neg_()
        if self._grad_clip_norm is not None:
            clip_grad_norm_(self.head.parameters(), self._grad_clip_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def state_dict(self) -> dict[str, Any]:
        return {"head": self.head.state_dict(), "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.head.load_state_dict(state["head"])
        self.optimizer.load_state_dict(state["optimizer"])


class AdversarialDistributionReconLoss(Metric[AdversarialDistributionReconLossConfig]):
    """Recon loss under masks whose adversarial sources are sampled from a learned distribution.

    Drives components + CI fn to reconstruct the target output under masks the adversary head
    makes as hard as possible; the head ascends the same loss. Mirrors the eval-time breakdown
    of `PersistentPGDReconLoss` (live recon loss on train steps, plus output- and
    hidden-activation-MSE on eval batches) and additionally logs streaming stats of the
    distribution params and sampled sources to surface pathologies (sigma collapse, alpha/beta
    blow-up, source saturation).
    """

    log_namespace: ClassVar[str] = "loss"
    slow: ClassVar[bool] = True
    short_name = "AdvDistRecon"

    def __init__(self, cfg: AdversarialDistributionReconLossConfig) -> None:
        super().__init__(cfg)
        self.state: AdversaryHeadState | None = None
        self._pending_resume_state: dict[str, Any] | None = None
        self._should_ascend = False

    def _ensure_state(self, ctx: MetricContext) -> None:
        if self.state is not None:
            return
        self.state = AdversaryHeadState(
            model=self.model,
            device=self.device,
            use_delta_component=ctx.use_delta_component,
            distribution=self.cfg.distribution,
            trunk_grad=self.cfg.trunk_grad,
            grl_lambda=self.cfg.grl_lambda,
            head_hidden_dims=self.cfg.head_hidden_dims,
            optimizer_cfg=self.cfg.optimizer,
            n_samples=self.cfg.n_samples,
            reconstruction_loss=ctx.reconstruction_loss,
        )
        if self._pending_resume_state is not None:
            self.state.load_state_dict(self._pending_resume_state)
            self._pending_resume_state = None

    @override
    def reset(self) -> None:
        self._recon_sum_loss = torch.zeros((), device=self.device)
        self._recon_n_examples = torch.zeros((), device=self.device, dtype=torch.long)
        self._hidden_sum_mse: dict[str, Tensor] = {}
        self._hidden_n: dict[str, Tensor] = {}
        self._param_moments: dict[str, _RunningMoments] = {}
        self._source_sat_sum = torch.zeros((), device=self.device)
        self._source_sat_n = torch.zeros((), device=self.device, dtype=torch.long)

    @override
    def update(self, ctx: MetricContext) -> Tensor | None:
        if ctx.current_frac_of_training < self.cfg.start_frac:
            self._should_ascend = False
            return None
        self._ensure_state(ctx)
        assert self.state is not None
        if not ctx.is_eval:
            self.state.update_lr(step=ctx.step, total_steps=ctx.total_steps)

        weight_deltas = ctx.weight_deltas if ctx.use_delta_component else None

        sum_loss, n_examples = self.state.compute_recon_sum_and_n(
            model=self.model,
            batch=ctx.batch,
            target_out=ctx.target_out,
            ci=ctx.ci.lower_leaky,
            pre_weight_acts=ctx.pre_weight_acts,
            weight_deltas=weight_deltas,
        )

        # Skip the head step on eval and on the final step (the trainer also skips its own
        # optimizer steps on the last step — it is only for plotting/logging).
        self._should_ascend = not ctx.is_eval and ctx.step != ctx.total_steps

        if ctx.is_eval:
            self._recon_sum_loss += sum_loss.detach()
            self._recon_n_examples += n_examples
            self._accum_hidden_acts(ctx, weight_deltas)
            self._accum_adv_param_stats(ctx)

        return sum_loss / n_examples

    def _accum_hidden_acts(
        self,
        ctx: MetricContext,
        weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    ) -> None:
        assert self.state is not None
        target_acts = self.model(ctx.batch, cache_type="output").cache
        batch_dims = ctx.target_out.shape[:-1]
        _, sources = self.state.distribution_params_and_sources(ctx.pre_weight_acts)
        mask_infos = get_ppgd_mask_infos(
            ci=ctx.ci.lower_leaky,
            weight_deltas=weight_deltas,
            ppgd_sources=sources,
            routing_masks="all",
            batch_dims=batch_dims,
        )
        per_module, _ = calc_hidden_acts_mse(
            model=self.model, batch=ctx.batch, mask_infos=mask_infos, target_acts=target_acts
        )
        for key, (mse, n) in per_module.items():
            if key not in self._hidden_sum_mse:
                self._hidden_sum_mse[key] = torch.zeros((), device=self.device)
                self._hidden_n[key] = torch.zeros((), device=self.device, dtype=torch.long)
            self._hidden_sum_mse[key] += mse.detach()
            self._hidden_n[key] += n

    def _accum_adv_param_stats(self, ctx: MetricContext) -> None:
        assert self.state is not None
        with torch.no_grad():
            params, sources = self.state.distribution_params_and_sources(ctx.pre_weight_acts)
        for named in params.values():
            for param_name, value in named.items():
                if param_name not in self._param_moments:
                    self._param_moments[param_name] = _RunningMoments(self.device)
                self._param_moments[param_name].update(value)
        if "source" not in self._param_moments:
            self._param_moments["source"] = _RunningMoments(self.device)
        for source in sources.values():
            self._param_moments["source"].update(source)
            saturated = (source < _SATURATION_EPS) | (source > 1.0 - _SATURATION_EPS)
            self._source_sat_sum += saturated.sum()
            self._source_sat_n += source.numel()

    @override
    def compute(self) -> MetricResult:
        cls = type(self).__name__
        out: dict[str, Float[Tensor, ""]] = {}
        if self._hidden_sum_mse:
            out.update(
                compute_per_module_metrics(
                    class_name=f"{cls}/hidden_acts",
                    per_module_sum_mse=self._hidden_sum_mse,
                    per_module_n_examples=self._hidden_n,
                )
            )
        if self._recon_n_examples.item() > 0:
            sum_loss = all_reduce(self._recon_sum_loss.clone())
            n = all_reduce(self._recon_n_examples.clone())
            out[f"{cls}/output_recon"] = sum_loss / n
        for param_name, moments in self._param_moments.items():
            out.update(moments.reduced(f"{cls}/adv_params/{param_name}"))
        if self._source_sat_n.item() > 0:
            sat_sum = all_reduce(self._source_sat_sum.clone())
            sat_n = all_reduce(self._source_sat_n.clone().float())
            out[f"{cls}/adv_params/source_frac_saturated"] = sat_sum / sat_n
        return out

    @override
    def after_backward(self) -> None:
        if self._should_ascend:
            assert self.state is not None
            self.state.ascend()
            self._should_ascend = False

    @override
    def state_dict(self) -> dict[str, Any]:
        if self.state is None:
            return {}
        return self.state.state_dict()

    @override
    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not state:
            self._pending_resume_state = None
            return
        if self.state is None:
            self._pending_resume_state = state
        else:
            self.state.load_state_dict(state)
