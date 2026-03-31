"""CLT harvest function: computes sparse activations from a BatchTopK Cross-Layer Transcoder."""

from typing import override

import torch
from torch import Tensor

from spd.adapters.clt import CLTAdapter
from spd.harvest.harvest_fn.base import HarvestFn
from spd.harvest.schemas import HarvestBatch
from spd.utils.general_utils import extract_batch_data


class CLTHarvestFn(HarvestFn):
    def __init__(self, adapter: CLTAdapter, device: torch.device):
        self._adapter = adapter
        self._device = device

        adapter.base_model.to(device).eval()
        adapter.clt.to(device).eval()

    @override
    def __call__(self, batch_item: torch.Tensor) -> HarvestBatch:
        model = self._adapter.base_model
        clt = self._adapter.clt

        batch = extract_batch_data(batch_item).to(self._device)

        mlp_inputs: dict[int, Tensor] = {}
        hooks: list[torch.utils.hooks.RemovableHandle] = []
        for layer_idx in clt.layers:
            module = model.get_submodule(f"h.{layer_idx}.mlp")

            def _hook(
                _mod: torch.nn.Module,
                inp: tuple[Tensor, ...],
                _out: Tensor,
                idx: int = layer_idx,
            ) -> None:
                mlp_inputs[idx] = inp[0].detach()

            hooks.append(module.register_forward_hook(_hook))

        logits, _ = model(batch)
        for h in hooks:
            h.remove()

        assert logits is not None
        probs = torch.softmax(logits, dim=-1)

        firings: dict[str, Tensor] = {}
        activations: dict[str, dict[str, Tensor]] = {}
        for layer_idx in clt.layers:
            mlp_in = mlp_inputs[layer_idx]
            B, S, _ = mlp_in.shape
            flat = mlp_in.reshape(-1, clt.input_size)
            acts = clt.encode_layer(layer_idx, flat).reshape(B, S, -1)
            module_path = f"h.{layer_idx}.mlp"
            firings[module_path] = acts > 0
            activations[module_path] = {"activation": acts}

        return HarvestBatch(
            tokens=batch,
            firings=firings,
            activations=activations,
            output_probs=probs,
        )
