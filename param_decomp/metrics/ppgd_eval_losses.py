from typing import Any, ClassVar, override

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import ReduceOp

from param_decomp.metrics.base import Metric
from param_decomp.metrics.hidden_acts_recon_loss import (
    calc_hidden_acts_mse,
    compute_per_module_metrics,
)
from param_decomp.models.batch_and_loss_fns import ReconstructionLoss
from param_decomp.models.component_model import CIOutputs, ComponentModel
from param_decomp.persistent_pgd import PersistentPGDState, get_ppgd_mask_infos
from param_decomp.utils.distributed_utils import all_reduce


class PPGDReconEval(Metric):
    """Eval losses using persistent PGD masks: hidden activation MSE and output reconstruction.

    Handles a single persistent PGD state, keyed by metric_name.
    """

    slow: ClassVar[bool] = True
    metric_section: ClassVar[str] = "loss"

    def __init__(
        self,
        model: ComponentModel,
        device: str,
        ppgd_state: PersistentPGDState,
        use_delta_component: bool,
        reconstruction_loss: ReconstructionLoss,
        metric_name: str,
    ) -> None:
        self.model = model
        self.use_delta_component = use_delta_component
        self.reconstruction_loss = reconstruction_loss
        self.device = device
        self._ppgd_state = ppgd_state
        self._metric_name = metric_name

        self._module_sum_mse: dict[str, Tensor] = {}
        self._module_n: dict[str, Tensor] = {}

        self._output_sum_loss = torch.tensor(0.0, device=device)
        self._output_n = torch.tensor(0, device=device)

    @override
    def update(
        self,
        *,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        ci: CIOutputs,
        weight_deltas: dict[str, Float[Tensor, "d_out d_in"]],
        target_out: Float[Tensor, "..."],
        **_: Any,
    ) -> None:
        target_acts = self.model(batch, cache_type="output").cache
        batch_dims = next(iter(ci.lower_leaky.values())).shape[:-1]

        mask_infos = get_ppgd_mask_infos(
            ci=ci.lower_leaky,
            weight_deltas=weight_deltas if self.use_delta_component else None,
            ppgd_sources=self._ppgd_state.get_effective_sources(),
            routing_masks="all",
            batch_dims=batch_dims,
        )
        per_module, comp_output = calc_hidden_acts_mse(
            model=self.model,
            batch=batch,
            mask_infos=mask_infos,
            target_acts=target_acts,
        )
        for key, (mse, n) in per_module.items():
            if key not in self._module_sum_mse:
                self._module_sum_mse[key] = torch.tensor(0.0, device=self.device)
                self._module_n[key] = torch.tensor(0, device=self.device)
            self._module_sum_mse[key] += mse.detach()
            self._module_n[key] += n

        output_loss, n_examples = self.reconstruction_loss(comp_output, target_out)
        self._output_sum_loss += output_loss.detach()
        self._output_n += n_examples

    @override
    def compute(self) -> dict[str, Float[Tensor, ""]]:
        out: dict[str, Float[Tensor, ""]] = {}
        per_module = compute_per_module_metrics(
            class_name=f"{self._metric_name}/hidden_acts",
            per_module_sum_mse=self._module_sum_mse,
            per_module_n_examples=self._module_n,
        )
        out.update(per_module)
        sum_loss = all_reduce(self._output_sum_loss, op=ReduceOp.SUM)
        n_examples = all_reduce(self._output_n, op=ReduceOp.SUM)
        out[f"{self._metric_name}/output_recon"] = sum_loss / n_examples
        return out
