"""Head-initialized PGD reconstruction loss: a learned amortized initializer for PGD.

The eval success metric (`PGDReconLoss`) attacks the frozen decomposition with a 20-step
sign-PGD search over the mask box. A single-shot learned adversary can't match that, so the
defender trained against it isn't PGD-robust. Here instead:

1. a (detached, deterministic) head predicts an initial source `s0 = sigmoid(head(trunk))`;
2. sign-PGD refines `s0` for a **random number of steps** (`pgd_steps_min..max`) into `s_k`
   — optionally also from a fresh **random restart** init, keeping the higher-loss endpoint;
3. the **defender** (components + CI fn) descends recon under `s_k.detach()` — a PGD-strength
   attack, same family as eval, so robustness transfers;
4. the **head** is trained by *distillation*: `MSE(s0, s_k.detach())` — it learns to predict
   where PGD ends up, so over training PGD needs fewer steps from the head's init.

The two randomnesses are deliberately distinct: the **step count** (how many refinement
steps) and the **restart** (whether a second PGD runs from a random init). They never share
a name. The head reads `trunk.detach()` and is optimized only by its own MSE — no gradient
from this loss enters the CI trunk, so the CI fn trains purely on its own objective.
"""

from typing import Any, ClassVar, Literal, override

import torch
import torch.nn.functional as F
from jaxtyping import Float
from pydantic import Field, PositiveInt, model_validator
from torch import Tensor, nn
from torch.distributed import ReduceOp
from torch.nn.utils import clip_grad_norm_

from param_decomp.base_config import Probability
from param_decomp.batch_and_loss_fns import ReconstructionLoss
from param_decomp.ci_fns import GlobalCiFnWrapper
from param_decomp.ci_nn_blocks import Linear
from param_decomp.component_model import ComponentModel
from param_decomp.components import EmbeddingComponents
from param_decomp.distributed import all_reduce, broadcast_tensor
from param_decomp.metrics.adversarial_distribution_recon import AdversaryHeadOptimizerConfig
from param_decomp.metrics.base import LossMetricConfig, Metric, MetricResult
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.persistent_pgd_state import get_ppgd_mask_infos
from param_decomp.metrics.stochastic_hidden_acts_recon import (
    calc_hidden_acts_mse,
    compute_per_module_metrics,
)
from param_decomp.schedule import get_scheduled_value

AdversarySources = dict[str, Float[Tensor, "*batch_dims source_c"]]

_SATURATION_EPS = 1e-3


def _build_head(d_model: int, hidden_dims: list[int], out_dim: int) -> nn.Module:
    """Single linear readout on the trunk, or an MLP if hidden_dims given."""
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


class HeadInitPGDReconLossConfig(LossMetricConfig):
    """Config for `HeadInitPGDReconLoss`.

    The number of PGD refinement steps is drawn uniformly from `[pgd_steps_min, pgd_steps_max]`
    on every call (the "step count" randomness). `random_restart` is the separate, orthogonal
    randomness: when true, a second PGD run starts from a fresh uniform-random source and the
    higher-loss endpoint of {head-init, random-init} is used.
    """

    type: Literal["HeadInitPGDReconLoss"] = "HeadInitPGDReconLoss"
    optimizer: AdversaryHeadOptimizerConfig
    pgd_step_size: float = 0.1
    pgd_steps_min: PositiveInt = 1
    pgd_steps_max: PositiveInt = 8
    random_restart: bool = True
    defender_target: Literal["winner_take_all", "head_and_random"] = Field(
        default="winner_take_all",
        description=(
            "Which PGD endpoint(s) the defender trains against. `winner_take_all`: only the "
            "higher-loss of {head-init, random-init} — but a good head always wins, so the "
            "random restart is discarded and the defender overfits to the head's narrow attack "
            "mode (brittle/oscillatory on the broad eval PGD). `head_and_random`: pool both "
            "endpoints so the defender also hardens against fresh random-init (eval-like) "
            "attacks. Requires `random_restart`."
        ),
    )
    head_hidden_dims: list[PositiveInt] = Field(default_factory=lambda: [2048, 2048])
    start_frac: Probability = 0.0

    @model_validator(mode="after")
    def _validate(self) -> "HeadInitPGDReconLossConfig":
        assert self.pgd_steps_min <= self.pgd_steps_max, "pgd_steps_min must be <= pgd_steps_max"
        if self.defender_target == "head_and_random":
            assert self.random_restart, "defender_target='head_and_random' requires random_restart"
        return self


def _refine_pgd(
    *,
    model: ComponentModel,
    batch: Tensor,
    ci_detached: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    target_out: Float[Tensor, "... vocab"],
    init_sources: AdversarySources,
    step_size: float,
    n_steps: int,
    reconstruction_loss: ReconstructionLoss,
) -> tuple[AdversarySources, Float[Tensor, ""]]:
    """Sign-PGD ascent on the recon loss over `sources`, from `init_sources`, with CI fixed.

    `ci_detached` is treated as constant (the attack optimizes only the sources). Returns the
    detached final sources and the (mean) recon loss at them. Mirrors the eval `PGDReconLoss`
    attack but starts from a provided init rather than a random one.
    """
    batch_dims = next(iter(ci_detached.values())).shape[:-1]
    sources = {k: v.detach().clone().requires_grad_(True) for k, v in init_sources.items()}

    def loss_at(srcs: AdversarySources) -> Float[Tensor, ""]:
        mask_infos = get_ppgd_mask_infos(
            ci=ci_detached,
            weight_deltas=weight_deltas,
            ppgd_sources=srcs,
            routing_masks="all",
            batch_dims=batch_dims,
        )
        out = model(batch, mask_infos=mask_infos)
        loss, n = reconstruction_loss(pred=out, target=target_out)
        return loss / n

    for _ in range(n_steps):
        with torch.enable_grad():
            loss = loss_at(sources)
        grads = torch.autograd.grad(loss, list(sources.values()))
        with torch.no_grad():
            for k, g in zip(sources, grads, strict=True):
                sources[k].add_(step_size * g.sign())
                sources[k].clamp_(0.0, 1.0)

    detached = {k: v.detach() for k, v in sources.items()}
    with torch.no_grad():
        final_loss = loss_at(detached)
    return detached, final_loss


class HeadInitPGDState:
    """The amortized PGD-initializer head, its optimizer, and the attack/distill machinery.

    The head is not a submodule of the CI fn (its params never enter `ci_fn_optimizer`); it is
    optimized only here, by distillation toward the PGD endpoint. It lives outside the DDP
    wrapper, so params are broadcast at init and grads all-reduced before each step.
    """

    def __init__(
        self,
        *,
        model: ComponentModel,
        device: str,
        use_delta_component: bool,
        cfg: HeadInitPGDReconLossConfig,
        reconstruction_loss: ReconstructionLoss,
    ) -> None:
        self._device = device
        self._cfg = cfg
        self._reconstruction_loss = reconstruction_loss
        self._lr_schedule = cfg.optimizer.lr_schedule
        self._grad_clip_norm = cfg.optimizer.grad_clip_norm

        ci_fn = model.ci_fn
        assert isinstance(ci_fn, GlobalCiFnWrapper), (
            "HeadInitPGDReconLoss requires a global_shared_transformer CI fn (head shares its "
            f"trunk); got {type(ci_fn).__name__}"
        )
        for name, component in model.components.items():
            assert not isinstance(component, EmbeddingComponents), (
                f"HeadInitPGDReconLoss does not support embedding decomposition targets ({name!r})."
            )
        self._ci_fn = ci_fn
        d_model = ci_fn.transformer.d_model

        self._layer_order = sorted(model.module_to_c)
        self._source_c = {
            name: model.module_to_c[name] + (1 if use_delta_component else 0)
            for name in self._layer_order
        }
        self.head = _build_head(d_model, cfg.head_hidden_dims, sum(self._source_c.values())).to(
            device
        )
        for param in self.head.parameters():
            broadcast_tensor(param.data)

        self.optimizer = torch.optim.AdamW(
            self.head.parameters(),
            lr=cfg.optimizer.lr_schedule.start_val,
            betas=cfg.optimizer.betas,
            weight_decay=cfg.optimizer.weight_decay,
        )

    def predict_sources(
        self, pre_weight_acts: dict[str, Float[Tensor, "... d_in"]]
    ) -> AdversarySources:
        """`s0 = sigmoid(head(trunk.detach()))`, split per module (grad flows to the head only)."""
        with torch.no_grad():
            trunk_x, added_seq_dim = self._ci_fn.trunk_features(pre_weight_acts)
        raw = self.head(trunk_x)
        if added_seq_dim:
            raw = raw.squeeze(-2)
        chunks = torch.split(raw, [self._source_c[n] for n in self._layer_order], dim=-1)
        return {n: torch.sigmoid(c) for n, c in zip(self._layer_order, chunks, strict=True)}

    def _sample_n_steps(self) -> int:
        """Draw the PGD *step count* (depth) for this call (the step-count randomness)."""
        lo, hi = self._cfg.pgd_steps_min, self._cfg.pgd_steps_max
        return int(torch.randint(lo, hi + 1, ()).item())

    def run_attack(
        self,
        *,
        model: ComponentModel,
        batch: Tensor,
        ci: dict[str, Float[Tensor, "... C"]],
        weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
        target_out: Float[Tensor, "... vocab"],
        head_sources: AdversarySources,
    ) -> tuple[AdversarySources, AdversarySources | None, int, bool]:
        """Refine the head init with sign-PGD (random step count); optional random restart.

        Returns `(head_endpoint, random_endpoint, n_steps_used, restart_won)`, both detached.
        `random_endpoint` is None when `random_restart` is off. CI is detached inside PGD so
        the attack optimizes only the sources. `restart_won` is whether the random-restart
        endpoint reached a higher loss than the head-initialized one (diagnostic for head init
        quality; also picks the head's distillation target). The caller decides which
        endpoint(s) the defender trains against (`defender_target`).
        """
        ci_detached = {k: v.detach() for k, v in ci.items()}
        n_steps = self._sample_n_steps()
        refine_kwargs = dict(
            model=model,
            batch=batch,
            ci_detached=ci_detached,
            weight_deltas=weight_deltas,
            target_out=target_out,
            step_size=self._cfg.pgd_step_size,
            n_steps=n_steps,
            reconstruction_loss=self._reconstruction_loss,
        )
        head_endpoint, loss_head = _refine_pgd(init_sources=head_sources, **refine_kwargs)  # pyright: ignore[reportArgumentType]
        random_endpoint: AdversarySources | None = None
        restart_won = False
        if self._cfg.random_restart:
            random_init = {name: torch.rand_like(src) for name, src in head_sources.items()}
            random_endpoint, loss_rand = _refine_pgd(init_sources=random_init, **refine_kwargs)  # pyright: ignore[reportArgumentType]
            restart_won = loss_rand.item() > loss_head.item()
        return head_endpoint, random_endpoint, n_steps, restart_won

    def distill_loss(
        self, head_sources: AdversarySources, target: AdversarySources
    ) -> Float[Tensor, ""]:
        """`MSE(s0, s_k.detach())` averaged over modules — trains the head to predict the PGD endpoint."""
        terms = [F.mse_loss(head_sources[n], target[n]) for n in self._layer_order]
        return torch.stack(terms).mean()

    def update_lr(self, step: int, total_steps: int) -> None:
        lr = get_scheduled_value(step, total_steps, self._lr_schedule)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def step_head(self, distill_loss: Float[Tensor, ""]) -> None:
        """One AdamW step descending the distillation MSE (head grads all-reduced across ranks)."""
        distill_loss.backward()
        for param in self.head.parameters():
            if param.grad is not None:
                all_reduce(param.grad, op=ReduceOp.AVG)
        if self._grad_clip_norm is not None:
            clip_grad_norm_(self.head.parameters(), self._grad_clip_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def state_dict(self) -> dict[str, Any]:
        return {"head": self.head.state_dict(), "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.head.load_state_dict(state["head"])
        self.optimizer.load_state_dict(state["optimizer"])


class HeadInitPGDReconLoss(Metric[HeadInitPGDReconLossConfig]):
    """Recon loss under a PGD-refined, head-initialized adversarial mask.

    The defender descends recon under the (detached) PGD endpoint; the head is trained by
    distillation toward that endpoint. Eval logs the live recon, per-module hidden-act MSE, the
    head distillation MSE, the mean PGD step count, and how often the random restart wins.
    """

    log_namespace: ClassVar[str] = "loss"
    slow: ClassVar[bool] = True
    short_name = "HeadInitPGDRecon"

    def __init__(self, cfg: HeadInitPGDReconLossConfig) -> None:
        super().__init__(cfg)
        self.state: HeadInitPGDState | None = None
        self._pending_resume_state: dict[str, Any] | None = None
        self._pending_distill: Float[Tensor, ""] | None = None

    def _ensure_state(self, ctx: MetricContext) -> None:
        if self.state is not None:
            return
        self.state = HeadInitPGDState(
            model=self.model,
            device=self.device,
            use_delta_component=ctx.use_delta_component,
            cfg=self.cfg,
            reconstruction_loss=ctx.reconstruction_loss,
        )
        if self._pending_resume_state is not None:
            self.state.load_state_dict(self._pending_resume_state)
            self._pending_resume_state = None

    @override
    def reset(self) -> None:
        self._recon_sum = torch.zeros((), device=self.device)
        self._recon_n = torch.zeros((), device=self.device, dtype=torch.long)
        self._hidden_sum_mse: dict[str, Tensor] = {}
        self._hidden_n: dict[str, Tensor] = {}
        self._distill_sum = torch.zeros((), device=self.device)
        self._nsteps_sum = torch.zeros((), device=self.device)
        self._restart_win_sum = torch.zeros((), device=self.device)
        self._source_sat_sum = torch.zeros((), device=self.device)
        self._source_n = torch.zeros((), device=self.device, dtype=torch.long)
        self._eval_batches = torch.zeros((), device=self.device, dtype=torch.long)

    @override
    def update(self, ctx: MetricContext) -> Tensor | None:
        if ctx.current_frac_of_training < self.cfg.start_frac:
            self._pending_distill = None
            return None
        self._ensure_state(ctx)
        assert self.state is not None
        if not ctx.is_eval:
            self.state.update_lr(step=ctx.step, total_steps=ctx.total_steps)

        weight_deltas = ctx.weight_deltas if ctx.use_delta_component else None
        head_sources = self.state.predict_sources(ctx.pre_weight_acts)
        head_ep, random_ep, n_steps, restart_won = self.state.run_attack(
            model=self.model,
            batch=ctx.batch,
            ci=ctx.ci.lower_leaky,
            weight_deltas=weight_deltas,
            target_out=ctx.target_out,
            head_sources=head_sources,
        )

        # The head distills toward the higher-loss endpoint (keeps predicting the strongest mode).
        distill_target = random_ep if (restart_won and random_ep is not None) else head_ep
        distill = self.state.distill_loss(head_sources, distill_target)
        self._pending_distill = distill if not ctx.is_eval else None

        # Defender recon. `head_and_random` pools both endpoints so the defender hardens against
        # fresh random-init (eval-like) attacks, not just the head's narrow mode; otherwise it
        # trains on the winner only.
        if self.cfg.defender_target == "head_and_random" and random_ep is not None:
            endpoints = [head_ep, random_ep]
        else:
            endpoints = [distill_target]
        sum_loss = torch.zeros((), device=self.device)
        n = 0
        for endpoint in endpoints:
            ep_loss, ep_n = self._recon_at(ctx, endpoint, weight_deltas)
            sum_loss = sum_loss + ep_loss
            n += ep_n

        if ctx.is_eval:
            self._recon_sum += sum_loss.detach()
            self._recon_n += n
            self._accum_eval(
                ctx, distill_target, distill.detach(), n_steps, restart_won, weight_deltas
            )

        return sum_loss / n

    def _recon_at(
        self,
        ctx: MetricContext,
        source: AdversarySources,
        weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    ) -> tuple[Float[Tensor, ""], int]:
        """Recon (with grad to components + CI fn) under masks built from a detached source."""
        mask_infos = get_ppgd_mask_infos(
            ci=ctx.ci.lower_leaky,
            weight_deltas=weight_deltas,
            ppgd_sources=source,
            routing_masks="all",
            batch_dims=ctx.target_out.shape[:-1],
        )
        out = self.model(ctx.batch, mask_infos=mask_infos)
        return ctx.reconstruction_loss(pred=out, target=ctx.target_out)

    def _accum_eval(
        self,
        ctx: MetricContext,
        s_k: AdversarySources,
        distill: Tensor,
        n_steps: int,
        restart_won: bool,
        weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    ) -> None:
        assert self.state is not None
        target_acts = self.model(ctx.batch, cache_type="output").cache
        batch_dims = ctx.target_out.shape[:-1]
        mask_infos = get_ppgd_mask_infos(
            ci=ctx.ci.lower_leaky,
            weight_deltas=weight_deltas,
            ppgd_sources=s_k,
            routing_masks="all",
            batch_dims=batch_dims,
        )
        per_module, _ = calc_hidden_acts_mse(
            model=self.model, batch=ctx.batch, mask_infos=mask_infos, target_acts=target_acts
        )
        for key, (mse, k) in per_module.items():
            if key not in self._hidden_sum_mse:
                self._hidden_sum_mse[key] = torch.zeros((), device=self.device)
                self._hidden_n[key] = torch.zeros((), device=self.device, dtype=torch.long)
            self._hidden_sum_mse[key] += mse.detach()
            self._hidden_n[key] += k
        self._distill_sum += distill
        self._nsteps_sum += n_steps
        self._restart_win_sum += 1.0 if restart_won else 0.0
        self._eval_batches += 1
        for src in s_k.values():
            self._source_sat_sum += ((src < _SATURATION_EPS) | (src > 1 - _SATURATION_EPS)).sum()
            self._source_n += src.numel()

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
        if self._recon_n.item() > 0:
            out[f"{cls}/output_recon"] = all_reduce(self._recon_sum.clone()) / all_reduce(
                self._recon_n.clone()
            )
        if self._eval_batches.item() > 0:
            nb = all_reduce(self._eval_batches.clone().float())
            out[f"{cls}/head_distill_mse"] = all_reduce(self._distill_sum.clone()) / nb
            out[f"{cls}/pgd_n_steps_mean"] = all_reduce(self._nsteps_sum.clone()) / nb
            out[f"{cls}/random_restart_win_frac"] = all_reduce(self._restart_win_sum.clone()) / nb
        if self._source_n.item() > 0:
            out[f"{cls}/source_frac_saturated"] = all_reduce(
                self._source_sat_sum.clone()
            ) / all_reduce(self._source_n.clone().float())
        return out

    @override
    def after_backward(self) -> None:
        if self._pending_distill is not None:
            assert self.state is not None
            self.state.step_head(self._pending_distill)
            self._pending_distill = None

    @override
    def state_dict(self) -> dict[str, Any]:
        return {} if self.state is None else self.state.state_dict()

    @override
    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not state:
            self._pending_resume_state = None
        elif self.state is None:
            self._pending_resume_state = state
        else:
            self.state.load_state_dict(state)
