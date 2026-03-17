"""Transcoder harvest function: computes sparse activations from BatchTopK transcoders."""

from typing import override

import torch
from torch import Tensor

from spd.adapters.transcoder import TranscoderAdapter
from spd.harvest.harvest_fn.base import HarvestFn
from spd.harvest.schemas import HarvestBatch
from spd.utils.general_utils import extract_batch_data


class TranscoderHarvestFn(HarvestFn):
    def __init__(self, adapter: TranscoderAdapter, device: torch.device):
        self._adapter = adapter
        self._device = device

        adapter.base_model.to(device).eval()
        for tc in adapter.transcoders.values():
            tc.to(device).eval()

    @override
    def __call__(self, batch_item: torch.Tensor) -> HarvestBatch:
        model = self._adapter.base_model

        batch = extract_batch_data(batch_item).to(self._device)

        mlp_inputs: dict[str, Tensor] = {}
        hooks: list[torch.utils.hooks.RemovableHandle] = []
        for module_path in self._adapter.transcoders:
            module = model.get_submodule(module_path)

            def _hook(
                _mod: torch.nn.Module,
                inp: tuple[Tensor, ...],
                _out: Tensor,
                path: str = module_path,
            ) -> None:
                mlp_inputs[path] = inp[0].detach()

            hooks.append(module.register_forward_hook(_hook))

        logits, _ = model(batch)
        for h in hooks:
            h.remove()

        assert logits is not None
        probs = torch.softmax(logits, dim=-1)

        firings: dict[str, Tensor] = {}
        activations: dict[str, dict[str, Tensor]] = {}
        for module_path, tc in self._adapter.transcoders.items():
            mlp_in = mlp_inputs[module_path]
            B, S, _ = mlp_in.shape
            flat = mlp_in.reshape(-1, tc.input_size)
            acts = tc.encode(flat).reshape(B, S, -1)
            firings[module_path] = acts > 0
            activations[module_path] = {"activation": acts}

        return HarvestBatch(
            tokens=batch,
            firings=firings,
            activations=activations,
            output_probs=probs,
        )
