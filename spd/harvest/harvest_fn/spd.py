from typing import override

import torch

from spd.adapters.spd import SPDAdapter
from spd.harvest.config import SPDHarvestConfig
from spd.harvest.harvest_fn.base import HarvestFn
from spd.harvest.schemas import HarvestBatch
from spd.utils.general_utils import extract_batch_data


class SPDHarvestFn(HarvestFn):
    def __init__(self, config: SPDHarvestConfig, adapter: SPDAdapter, device: torch.device):
        self._adapter = adapter
        self._activation_threshold = config.activation_threshold
        self._device = device

        self._adapter.component_model.to(device).eval()
        self._u_norms = {
            layer_name: component.U.norm(dim=1).to(device)
            for layer_name, component in self._adapter.component_model.components.items()
        }

    @override
    def __call__(self, batch_item: torch.Tensor) -> HarvestBatch:
        model = self._adapter.component_model

        batch = extract_batch_data(batch_item).to(self._device)

        out = model(batch, cache_type="input")
        probs = torch.softmax(out.output, dim=-1)

        ci_dict = model.calc_causal_importances(
            pre_weight_acts=out.cache,
            detach_inputs=True,
            sampling=self._adapter.spd_run_info.config.sampling,
        ).lower_leaky

        per_layer_acts = model.get_all_component_acts(out.cache)

        firings = {layer: ci > self._activation_threshold for layer, ci in ci_dict.items()}

        activations = {
            layer: {
                "causal_importance": ci_dict[layer],
                "component_activation": per_layer_acts[layer] * self._u_norms[layer],
            }
            for layer in model.target_module_paths
        }

        return HarvestBatch(
            tokens=batch,
            firings=firings,
            activations=activations,
            output_probs=probs,
        )
