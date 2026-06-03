"""Adversarial-network reconstruction loss, its config, and the adversary state machine.

An adversary network with the same architecture as the CI function (built from the same
`ci_config`) maps IID uniform-noise vectors to per-component adversarial sources. Those
sources form the masks for a reconstruction forward pass: `mask = ci + (1 - ci) * source`.
Components and the CI function *descend* the resulting recon loss as usual; the adversary
network *ascends* it via its own optimizer, stepped from `after_backward` on the gradients
the outer `total_loss.backward()` leaves on the adversary parameters.

Unlike `PersistentPGDReconLoss`:

- the adversarial sources are produced by a network from fresh noise every step — there is
  no persistent per-datapoint source state and no inner warmup loop;
- the network's outputs pass through a plain sigmoid (not the upper/lower-leaky sigmoids of
  the CI fn) so the adversary always receives gradient;
- the adversary's inputs are IID uniform noise on `[0, 1]`, not target-model activations,
  but they flow through the same architecture (and hence the same layer/RMS norm) the CI fn
  applies to its inputs.
"""

from typing import Annotated, Any, ClassVar, Literal, override

import torch
from jaxtyping import Float
from pydantic import Field, NonNegativeFloat, PositiveFloat, PositiveInt
from torch import Tensor
from torch.distributed import ReduceOp
from torch.nn.utils import clip_grad_norm_

from param_decomp.base_config import BaseConfig, Probability
from param_decomp.batch_and_loss_fns import ReconstructionLoss
from param_decomp.ci_fns import CiConfig, LayerwiseCiConfig, make_ci_fn_wrapper
from param_decomp.component_model import ComponentModel
from param_decomp.components import EmbeddingComponents, get_module_input_dim
from param_decomp.distributed import all_reduce, broadcast_tensor
from param_decomp.metrics.base import LossMetricConfig, Metric, MetricResult
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.persistent_pgd_state import get_ppgd_mask_infos
from param_decomp.metrics.stochastic_hidden_acts_recon import (
    calc_hidden_acts_mse,
    compute_per_module_metrics,
)
from param_decomp.schedule import ScheduleConfig, get_scheduled_value

AdversarySources = dict[str, Float[Tensor, "*batch_dims source_c"]]


class AdversaryOptimizerConfig(BaseConfig):
    """AdamW hyperparameters + LR schedule for the adversary network.

    Mirrors `OptimizerConfig`, kept here rather than in `configs.py` to avoid an import
    cycle through the loss-metric union. The schedule is keyed on the global training step,
    exactly like `components_optimizer` and `ci_fn_optimizer`.
    """

    lr_schedule: ScheduleConfig
    weight_decay: NonNegativeFloat = 0.0
    betas: tuple[Probability, Probability] = (0.9, 0.999)
    grad_clip_norm: PositiveFloat | None = None


class AdversarialNetworkReconLossConfig(LossMetricConfig):
    """Config for `AdversarialNetworkReconLoss`.

    `update()` returns `None` before `start_frac` of training (the adversary is dormant and
    not stepped). `n_samples` independent noise draws are averaged per step.
    """

    type: Literal["AdversarialNetworkReconLoss"] = "AdversarialNetworkReconLoss"
    optimizer: AdversaryOptimizerConfig
    architecture: Annotated[CiConfig, Field(discriminator="mode")] | None = Field(
        default=None,
        description=(
            "Adversary-network architecture, in the same shape as `pd.ci_config` (pick its "
            "`d_model`, `n_blocks`, `mlp_hidden_dim`, `attn_config`, etc. here to size the "
            "adversary independently of the CI fn). When omitted, the adversary mirrors the "
            "target's CI fn architecture."
        ),
    )
    start_frac: Probability = 0.0
    n_samples: PositiveInt = 1


class AdversaryNetworkState:
    """The adversary network, its AdamW optimizer, and the noise -> source -> mask machinery.

    The network has a CI-fn-shaped architecture (built by `make_ci_fn_wrapper` from
    `ci_config` — either the target's CI fn config or a per-loss override) but emits one extra
    channel per target when `use_delta_component` is set, so its per-module output matches the
    `[..., C (+1)]` source layout consumed by `get_ppgd_mask_infos`.
    """

    def __init__(
        self,
        *,
        model: ComponentModel,
        ci_config: CiConfig,
        device: str,
        use_delta_component: bool,
        optimizer_cfg: AdversaryOptimizerConfig,
        n_samples: int,
        reconstruction_loss: ReconstructionLoss,
    ) -> None:
        self._device = device
        self._n_samples = n_samples
        self._reconstruction_loss = reconstruction_loss
        self._lr_schedule = optimizer_cfg.lr_schedule
        self._grad_clip_norm = optimizer_cfg.grad_clip_norm

        assert not (isinstance(ci_config, LayerwiseCiConfig) and ci_config.fn_type == "mlp"), (
            "AdversarialNetworkReconLoss does not support the per-component-scalar 'mlp' CI fn "
            "type: the adversary consumes raw noise vectors, not component activations."
        )
        for name, component in model.components.items():
            assert not isinstance(component, EmbeddingComponents), (
                f"AdversarialNetworkReconLoss does not support embedding decomposition targets "
                f"(found at {name!r}); the adversary needs a scalar input dim per target."
            )

        self._input_dims = {
            name: get_module_input_dim(model.target_model.get_submodule(name))
            for name in model.module_to_c
        }
        # One source per component plus, when present, one for the weight-delta channel.
        source_c_per_module = {
            name: c + (1 if use_delta_component else 0) for name, c in model.module_to_c.items()
        }
        self.network = make_ci_fn_wrapper(
            target_model=model.target_model,
            module_to_c=source_c_per_module,
            components=model.components,
            ci_config=ci_config,
        ).to(device)
        # Keep adversary replicas identical across ranks (it lives outside the DDP wrapper).
        for param in self.network.parameters():
            broadcast_tensor(param.data)

        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=optimizer_cfg.lr_schedule.start_val,
            betas=optimizer_cfg.betas,
            weight_decay=optimizer_cfg.weight_decay,
        )

    def generate_sources(self, batch_dims: tuple[int, ...]) -> AdversarySources:
        """Sample IID uniform-`[0, 1]` noise per target and map it through the adversary.

        Returns per-module sources in `(0, 1)` via the final sigmoid, shaped
        `[*batch_dims, source_c]`.
        """
        noise = {
            name: torch.rand(*batch_dims, input_dim, device=self._device)
            for name, input_dim in self._input_dims.items()
        }
        logits = self.network(noise)
        return {name: torch.sigmoid(value) for name, value in logits.items()}

    def compute_recon_sum_and_n(
        self,
        model: ComponentModel,
        batch: Tensor,
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
        weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    ) -> tuple[Float[Tensor, ""], int]:
        """Recon forward returning `(sum_loss, n_examples)` summed over all noise draws."""
        batch_dims = next(iter(ci.values())).shape[:-1]
        device = next(iter(ci.values())).device
        sum_loss = torch.zeros((), device=device)
        n_examples = 0
        for _ in range(self._n_samples):
            sources = self.generate_sources(batch_dims)
            mask_infos = get_ppgd_mask_infos(
                ci=ci,
                weight_deltas=weight_deltas,
                ppgd_sources=sources,
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
        """One gradient-ascent step on the adversary using the grads left by the outer backward.

        The outer `total_loss.backward()` populates `param.grad` with the descent gradient
        of the recon loss; negating it turns the AdamW step into ascent. Grads are averaged
        across ranks first (the network lives outside DDP), then optionally clipped.
        """
        for param in self.network.parameters():
            if param.grad is None:
                continue
            all_reduce(param.grad, op=ReduceOp.AVG)
            param.grad.neg_()
        if self._grad_clip_norm is not None:
            clip_grad_norm_(self.network.parameters(), self._grad_clip_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def state_dict(self) -> dict[str, Any]:
        return {"network": self.network.state_dict(), "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.network.load_state_dict(state["network"])
        self.optimizer.load_state_dict(state["optimizer"])


class AdversarialNetworkReconLoss(Metric[AdversarialNetworkReconLossConfig]):
    """Recon loss under masks whose adversarial sources are emitted by a learned network.

    Drives components + CI fn to reconstruct the target output under masks the adversary
    network makes as hard as possible; the network is trained to maximise the same loss.
    Mirrors the eval-time breakdown of `PersistentPGDReconLoss`: the live recon loss on
    training steps, plus output- and hidden-activation-MSE accumulators on eval batches.
    """

    log_namespace: ClassVar[str] = "loss"
    slow: ClassVar[bool] = True
    short_name = "AdvNetRecon"

    def __init__(self, cfg: AdversarialNetworkReconLossConfig) -> None:
        super().__init__(cfg)
        self.state: AdversaryNetworkState | None = None
        self._pending_resume_state: dict[str, Any] | None = None
        self._should_ascend = False

    def _ensure_state(self, ctx: MetricContext) -> None:
        if self.state is not None:
            return
        ci_config = (
            self.cfg.architecture if self.cfg.architecture is not None else self.model.ci_config
        )
        self.state = AdversaryNetworkState(
            model=self.model,
            ci_config=ci_config,
            device=self.device,
            use_delta_component=ctx.use_delta_component,
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
            weight_deltas=weight_deltas,
        )

        # Skip the adversary step on eval and on the final step (the trainer also skips its
        # own optimizer steps there — the last step is only for plotting/logging).
        self._should_ascend = not ctx.is_eval and ctx.step != ctx.total_steps

        if ctx.is_eval:
            self._recon_sum_loss += sum_loss.detach()
            self._recon_n_examples += n_examples
            self._accum_hidden_acts(ctx, weight_deltas)

        return sum_loss / n_examples

    def _accum_hidden_acts(
        self,
        ctx: MetricContext,
        weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    ) -> None:
        assert self.state is not None
        target_acts = self.model(ctx.batch, cache_type="output").cache
        batch_dims = ctx.target_out.shape[:-1]
        mask_infos = get_ppgd_mask_infos(
            ci=ctx.ci.lower_leaky,
            weight_deltas=weight_deltas,
            ppgd_sources=self.state.generate_sources(batch_dims),
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

    @override
    def compute(self) -> MetricResult:
        out: dict[str, Float[Tensor, ""]] = {}
        if self._hidden_sum_mse:
            class_name = f"{type(self).__name__}/hidden_acts"
            out.update(
                compute_per_module_metrics(
                    class_name=class_name,
                    per_module_sum_mse=self._hidden_sum_mse,
                    per_module_n_examples=self._hidden_n,
                )
            )
        if self._recon_n_examples.item() > 0:
            sum_loss = all_reduce(self._recon_sum_loss)
            n = all_reduce(self._recon_n_examples)
            out[f"{type(self).__name__}/output_recon"] = sum_loss / n
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
