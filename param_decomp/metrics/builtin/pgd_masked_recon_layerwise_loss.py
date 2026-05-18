import torch
from jaxtyping import Float
from torch import Tensor
from torch.distributed import ReduceOp

from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.pgd_utils import PGDConfig, pgd_masked_recon_loss_update
from param_decomp.metrics.registry import register_metric
from param_decomp.models.component_model import ComponentModel
from param_decomp.routing import LayerRouter
from param_decomp.utils.distributed_utils import all_reduce


class PGDReconLayerwiseLossConfig(PGDConfig):
    pass


@register_metric
class PGDReconLayerwiseLoss:
    """Recon loss when masking with adversarially-optimized values and routing to one layer at a
    time."""

    section = "loss"
    config_type = PGDReconLayerwiseLossConfig
    short_name = "PGDReconLayer"

    def __init__(
        self, cfg: PGDReconLayerwiseLossConfig, *, model: ComponentModel, device: str
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.sum_loss = torch.zeros((), device=self.device)
        self.n_examples = torch.zeros((), device=self.device, dtype=torch.long)

    def update(self, ctx: MetricContext) -> Tensor:
        wd = ctx.weight_deltas if ctx.config.use_delta_component else None
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

    def compute(self) -> Float[Tensor, ""]:
        sum_loss = all_reduce(self.sum_loss, op=ReduceOp.SUM)
        n_examples = all_reduce(self.n_examples, op=ReduceOp.SUM)
        return sum_loss / n_examples
