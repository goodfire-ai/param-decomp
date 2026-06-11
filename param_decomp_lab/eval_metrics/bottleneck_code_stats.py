"""Eval metrics for the CI fn's sparse bottleneck code.

`BottleneckCodeStats` emits cheap scalars every eval pass; `BottleneckCodeHistograms`
renders figures on the slow cadence. Both require a CI fn with a bottleneck configured.
"""

from typing import Any, Literal, override

import torch
from jaxtyping import Float
from matplotlib import pyplot as plt
from torch import Tensor
from torch.distributed import ReduceOp

from param_decomp.base_config import BaseConfig
from param_decomp.ci_fns import get_bottleneck
from param_decomp.ci_nn_blocks import SparseBottleneck
from param_decomp.distributed import all_reduce, gather_all_tensors
from param_decomp.metrics.base import Metric, MetricResult
from param_decomp.metrics.context import MetricContext
from param_decomp_lab.eval_metrics.plotting import _render_figure


def _require_bottleneck(metric: Metric[Any]) -> SparseBottleneck:
    assert metric.model.ci_fn is not None
    bottleneck = get_bottleneck(metric.model.ci_fn)
    assert bottleneck is not None, (
        f"{type(metric).__name__} requires a CI fn with a bottleneck "
        "(set ci_config.simple_transformer_ci_cfg.bottleneck)"
    )
    return bottleneck


class BottleneckCodeStatsConfig(BaseConfig):
    type: Literal["BottleneckCodeStats"] = "BottleneckCodeStats"


class BottleneckCodeStats(Metric[BottleneckCodeStatsConfig]):
    """Scalar stats of the bottleneck code.

    Emits exact code L0, dead-dim fraction, mean magnitude of active code values, the
    theta distribution extremes, and the fraction of active elements within the gate's
    rectangular-kernel gradient window (a proxy for whether theta can learn).
    """

    log_namespace = "bottleneck"
    short_name = "BneckStats"

    @override
    def reset(self) -> None:
        self._bottleneck = _require_bottleneck(self)
        d = self._bottleneck.bottleneck_dim
        self.dim_fired_counts = torch.zeros(d, device=self.device, dtype=torch.long)
        self.l0_sum = torch.zeros((), device=self.device)
        self.active_mag_sum = torch.zeros((), device=self.device)
        self.active_count = torch.zeros((), device=self.device, dtype=torch.long)
        self.in_window_count = torch.zeros((), device=self.device, dtype=torch.long)
        self.n_examples = torch.zeros((), device=self.device, dtype=torch.long)

    @override
    def update(self, ctx: MetricContext) -> None:
        codes = ctx.ci.bottleneck_codes
        assert codes is not None, "BottleneckCodeStats requires bottleneck codes in CIOutputs"
        codes = codes.detach()
        flat: Float[Tensor, "n D"] = codes.reshape(-1, codes.shape[-1])

        fired = flat != 0
        self.dim_fired_counts += fired.sum(dim=0)
        self.l0_sum += fired.sum(dim=-1).float().sum()
        self.active_mag_sum += flat.abs()[fired].sum()
        self.active_count += fired.sum()

        # Active elements sit above theta; count those within the upper half of the
        # rectangular kernel window (|z| - theta < bandwidth / 2), where theta still
        # receives gradient.
        theta = self._bottleneck.gate.theta.detach()
        in_window = fired & (flat.abs() - theta < self._bottleneck.gate.bandwidth / 2)
        self.in_window_count += in_window.sum()

        self.n_examples += flat.shape[0]
        return None

    @override
    def compute(self) -> MetricResult:
        dim_fired_counts = all_reduce(self.dim_fired_counts, op=ReduceOp.SUM)
        l0_sum = all_reduce(self.l0_sum, op=ReduceOp.SUM)
        active_mag_sum = all_reduce(self.active_mag_sum, op=ReduceOp.SUM)
        active_count = all_reduce(self.active_count, op=ReduceOp.SUM)
        in_window_count = all_reduce(self.in_window_count, op=ReduceOp.SUM)
        n_examples = all_reduce(self.n_examples, op=ReduceOp.SUM)
        assert n_examples.item() > 0, "compute() called before any update()"

        theta = self._bottleneck.gate.theta.detach()
        active_count_f = active_count.float().clamp(min=1)
        return {
            "code_l0": l0_sum / n_examples,
            "frac_dims_dead": (dim_fired_counts == 0).float().mean(),
            "mean_active_magnitude": active_mag_sum / active_count_f,
            "frac_active_in_theta_grad_window": in_window_count.float() / active_count_f,
            "theta_mean": theta.mean(),
            "theta_min": theta.min(),
            "theta_max": theta.max(),
        }


class BottleneckCodeHistogramsConfig(BaseConfig):
    """`n_batches_accum=None` accumulates every batch in the eval pass."""

    type: Literal["BottleneckCodeHistograms"] = "BottleneckCodeHistograms"
    n_batches_accum: int | None


class BottleneckCodeHistograms(Metric[BottleneckCodeHistogramsConfig]):
    """Figures for the bottleneck code: per-dim firing rates, active values, theta."""

    log_namespace = "figures"
    slow = True
    short_name = "BneckHist"

    @override
    def reset(self) -> None:
        self._bottleneck = _require_bottleneck(self)
        self.batches_seen = 0
        self.codes_list: list[Float[Tensor, "n D"]] = []

    @override
    def update(self, ctx: MetricContext) -> None:
        if self.cfg.n_batches_accum is not None and self.batches_seen >= self.cfg.n_batches_accum:
            return None
        self.batches_seen += 1
        codes = ctx.ci.bottleneck_codes
        assert codes is not None, "BottleneckCodeHistograms requires bottleneck codes in CIOutputs"
        self.codes_list.append(codes.detach().reshape(-1, codes.shape[-1]))
        return None

    @override
    def compute(self) -> MetricResult:
        assert self.batches_seen > 0, "compute() called before any update()"
        codes = torch.cat(gather_all_tensors(torch.cat(self.codes_list, dim=0)), dim=0)
        theta = self._bottleneck.gate.theta.detach().cpu()

        firing_rates = (codes != 0).float().mean(dim=0).cpu()
        active_vals = codes[codes != 0].cpu()

        fig, axs = plt.subplots(1, 3, figsize=(15, 4))

        axs[0].hist(firing_rates.numpy(), bins=50)
        n_dead = int((firing_rates == 0).sum())
        axs[0].set_xlabel("per-dim firing rate")
        axs[0].set_ylabel("# dims")
        axs[0].set_title(f"firing rates ({n_dead}/{firing_rates.numel()} dead)")

        if active_vals.numel() > 0:
            axs[1].hist(active_vals.numpy(), bins=100)
        axs[1].axvline(theta.mean().item(), color="red", linestyle="--", label="mean theta")
        axs[1].axvline(-theta.mean().item(), color="red", linestyle="--")
        axs[1].set_xlabel("active code value")
        axs[1].set_title("active (nonzero) code values")
        axs[1].legend()

        axs[2].scatter(theta.numpy(), firing_rates.numpy(), s=12)
        axs[2].set_xlabel("theta")
        axs[2].set_ylabel("firing rate")
        axs[2].set_title("per-dim theta vs firing rate")

        fig.tight_layout()
        img = _render_figure(fig)
        plt.close(fig)
        return {"bottleneck_code": img}
