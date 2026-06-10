"""Multi-strategy recon eval for targeted runs, on the target / nontarget distributions."""

from typing import ClassVar, Literal, override

import torch
from jaxtyping import Float
from torch import Tensor
from torch.distributed import ReduceOp

from param_decomp.base_config import BaseConfig
from param_decomp.distributed import all_reduce
from param_decomp.masks import (
    AllLayersRouter,
    ComponentsMaskInfo,
    WeightDeltaAndMask,
    calc_stochastic_component_mask_info,
    make_mask_infos,
)
from param_decomp.metrics.base import Metric, MetricResult
from param_decomp.metrics.context import MetricContext
from param_decomp.targeted import delta_override


class _ReconLossConfigBase(BaseConfig):
    rounding_threshold: float
    ci_alive_threshold: float = 0.1


class TargetReconLossConfig(_ReconLossConfigBase):
    type: Literal["TargetReconLoss"] = "TargetReconLoss"


class NontargetReconLossConfig(_ReconLossConfigBase):
    type: Literal["NontargetReconLoss"] = "NontargetReconLoss"


_STRATEGIES = ("stochastic", "ci_masked", "rounded", "delta_only")


class _TargetedReconLossBase[TConfig: TargetReconLossConfig | NontargetReconLossConfig](
    Metric[TConfig]
):
    """Recon error under four masking strategies plus summed CI L0.

    `delta_value` is the per-distribution delta mask: 0.0 on target data (components
    must do the work), 1.0 on nontarget data (the residual carries the output).
    `delta_only` always pins the delta to 1.0 — it reads ≈ 0 only on nontarget data
    at convergence (directional diagnostic, not pass/fail).
    """

    delta_value: ClassVar[float]

    @override
    def reset(self) -> None:
        self.sum_losses: dict[str, Tensor] = {
            s: torch.zeros((), device=self.device) for s in _STRATEGIES
        }
        self.n_examples: dict[str, Tensor] = {
            s: torch.zeros((), device=self.device, dtype=torch.long) for s in _STRATEGIES
        }
        self.l0_sum = torch.zeros((), device=self.device)
        self.l0_examples = torch.zeros((), device=self.device, dtype=torch.long)

    @override
    def update(self, ctx: MetricContext) -> None:
        assert ctx.use_delta_component, "targeted recon evals require use_delta_component"
        ci = ctx.ci.lower_leaky
        ci_sample = next(iter(ci.values()))
        leading_dims = ci_sample.shape[:-1]

        def pinned_deltas(value: float) -> dict[str, WeightDeltaAndMask]:
            return {
                layer: (
                    ctx.weight_deltas[layer],
                    torch.full(leading_dims, value, device=ci_sample.device, dtype=ci_sample.dtype),
                )
                for layer in ci
            }

        def accumulate(strategy: str, mask_infos: dict[str, ComponentsMaskInfo]) -> None:
            out = self.model(ctx.batch, mask_infos=mask_infos)
            sum_loss, n = ctx.reconstruction_loss(out, ctx.target_out)
            self.sum_losses[strategy] += sum_loss.detach()
            self.n_examples[strategy] += n

        with delta_override(self.delta_value):
            for _ in range(ctx.n_mask_samples):
                accumulate(
                    "stochastic",
                    calc_stochastic_component_mask_info(
                        causal_importances=ci,
                        component_mask_sampling=ctx.sampling,
                        weight_deltas=ctx.weight_deltas,
                        router=AllLayersRouter(),
                    ),
                )
        accumulate(
            "ci_masked",
            make_mask_infos(ci, weight_deltas_and_masks=pinned_deltas(self.delta_value)),
        )
        accumulate(
            "rounded",
            make_mask_infos(
                {k: (v > self.cfg.rounding_threshold).float() for k, v in ci.items()},
                weight_deltas_and_masks=pinned_deltas(self.delta_value),
            ),
        )
        accumulate(
            "delta_only",
            make_mask_infos(
                {k: torch.zeros_like(v) for k, v in ci.items()},
                weight_deltas_and_masks=pinned_deltas(1.0),
            ),
        )

        for layer_ci in ci.values():
            self.l0_sum += (layer_ci > self.cfg.ci_alive_threshold).float().sum(-1).sum().detach()
        self.l0_examples += leading_dims.numel()
        return None

    @override
    def compute(self) -> MetricResult:
        out: dict[str, Float[Tensor, ""]] = {}
        for strategy in _STRATEGIES:
            sum_loss = all_reduce(self.sum_losses[strategy], op=ReduceOp.SUM)
            n = all_reduce(self.n_examples[strategy], op=ReduceOp.SUM)
            out[strategy] = sum_loss / n
        l0_sum = all_reduce(self.l0_sum, op=ReduceOp.SUM)
        l0_examples = all_reduce(self.l0_examples, op=ReduceOp.SUM)
        out["total_l0"] = l0_sum / l0_examples
        return out


class TargetReconLoss(_TargetedReconLossBase[TargetReconLossConfig]):
    """Recon eval on the target distribution (delta off; components must do the work)."""

    log_namespace = "target_recon"
    short_name = "TgtRecon"
    delta_value: ClassVar[float] = 0.0


class NontargetReconLoss(_TargetedReconLossBase[NontargetReconLossConfig]):
    """Recon eval on the nontarget distribution (delta on; fed by the nontarget eval loop)."""

    log_namespace = "nontarget_recon"
    short_name = "NtgtRecon"
    eval_distribution = "nontarget"
    delta_value: ClassVar[float] = 1.0
