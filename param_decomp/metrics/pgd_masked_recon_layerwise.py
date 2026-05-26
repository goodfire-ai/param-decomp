from typing import Literal, override

import torch
from torch import Tensor
from torch.distributed import ReduceOp

from param_decomp.distributed import all_reduce
from param_decomp.masks import LayerRouter
from param_decomp.metrics.base import Metric, MetricResult
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.pgd_utils import PGDConfig, pgd_masked_recon_loss_update


class PGDReconLayerwiseLossConfig(PGDConfig):
    type: Literal["PGDReconLayerwiseLoss"] = "PGDReconLayerwiseLoss"


class PGDReconLayerwiseLoss(Metric[PGDReconLayerwiseLossConfig]):
    """Per-layer PGD recon loss summed across layers.

    For each target layer, runs `cfg.n_steps` of per-step PGD on fresh adversarial
    sources routed to only that layer; sums the per-layer recon losses.
    """

    log_namespace = "loss"
    short_name = "PGDReconLayer"

    @override
    def reset(self) -> None:
        self.sum_loss = torch.zeros((), device=self.device)
        self.n_examples = torch.zeros((), device=self.device, dtype=torch.long)

    @override
    def update(self, ctx: MetricContext) -> Tensor:
        wd = ctx.weight_deltas if ctx.use_delta_component else None
        device = ctx.target_out.device
        sum_loss = torch.zeros((), device=device)
        n_examples = 0
        for layer in self.model.target_module_paths:
            sum_loss_layer, n_layer = pgd_masked_recon_loss_update(
                model=self.model,
                batch=ctx.batch,
                ci=ctx.ci.lower_leaky,
                weight_deltas=wd,
                target_out=ctx.target_out,
                router=LayerRouter(device=device, layer_name=layer),
                pgd_config=self.cfg,
                reconstruction_loss=ctx.reconstruction_loss,
            )
            sum_loss = sum_loss + sum_loss_layer
            n_examples += n_layer
        self.sum_loss += sum_loss.detach()
        self.n_examples += n_examples
        return sum_loss / n_examples

    @override
    def compute(self) -> MetricResult:
        sum_loss = all_reduce(self.sum_loss, op=ReduceOp.SUM)
        n_examples = all_reduce(self.n_examples, op=ReduceOp.SUM)
        return sum_loss / n_examples
