"""Cross-Layer Transcoder (CLT) encoder for harvest.

CLTs have per-layer encoders (W_enc.{i}, b_enc.{i}) and cross-layer decoders
(W_dec.{i} with shape [n_target_layers, dict_size, output_size]). For harvest,
only the encoder side is needed.

Vendored from https://github.com/bartbussmann/nn_decompositions (MIT license).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class CLTEncoderConfig:
    layers: list[int]
    input_size: int
    dict_size: int
    encoder_type: Literal["vanilla", "topk", "batchtopk", "jumprelu"]
    top_k: int
    bandwidth: float

    @staticmethod
    def from_checkpoint_json(cfg_raw: dict[str, Any]) -> "CLTEncoderConfig":
        layers_raw = cfg_raw["layers"]
        layers = json.loads(layers_raw) if isinstance(layers_raw, str) else layers_raw
        assert isinstance(layers, list), f"Expected list for layers, got {type(layers)}"
        return CLTEncoderConfig(
            layers=layers,
            input_size=cfg_raw["input_size"],
            dict_size=cfg_raw["dict_size"],
            encoder_type=cfg_raw["encoder_type"],
            top_k=cfg_raw.get("top_k", 0),
            bandwidth=cfg_raw.get("bandwidth", 0.001),
        )


class CrossLayerTranscoder(nn.Module):
    """Cross-Layer Transcoder encoder. Supports per-layer sparse encoding."""

    def __init__(self, config: CLTEncoderConfig, state_dict: dict[str, Tensor]):
        super().__init__()
        self.config = config
        self.layers = config.layers
        self.dict_size = config.dict_size
        self.input_size = config.input_size

        for i in self.layers:
            assert f"W_enc.{i}" in state_dict, f"Missing W_enc.{i} in checkpoint"
            self.register_buffer(f"W_enc_{i}", state_dict[f"W_enc.{i}"])
            self.register_buffer(f"b_enc_{i}", state_dict[f"b_enc.{i}"])

    def encode_layer(self, layer_idx: int, x: Tensor) -> Tensor:
        """Encode MLP input at a specific layer. x: [N, input_size] (flattened batch*seq)."""
        W_enc: Tensor = getattr(self, f"W_enc_{layer_idx}")
        b_enc: Tensor = getattr(self, f"b_enc_{layer_idx}")

        pre_acts = F.relu(x @ W_enc + b_enc)

        match self.config.encoder_type:
            case "vanilla":
                return pre_acts
            case "topk":
                topk = torch.topk(pre_acts, self.config.top_k, dim=-1)
                return torch.zeros_like(pre_acts).scatter(-1, topk.indices, topk.values)
            case "batchtopk":
                topk = torch.topk(pre_acts.flatten(), self.config.top_k * x.shape[0], dim=-1)
                return (
                    torch.zeros_like(pre_acts.flatten())
                    .scatter(-1, topk.indices, topk.values)
                    .reshape(pre_acts.shape)
                )
            case "jumprelu":
                raise NotImplementedError("JumpReLU CLT encoding not yet implemented")

    @staticmethod
    def from_checkpoint(checkpoint_dir: Path, device: str = "cpu") -> "CrossLayerTranscoder":
        with open(checkpoint_dir / "config.json") as f:
            cfg_raw = json.load(f)

        config = CLTEncoderConfig.from_checkpoint_json(cfg_raw)
        state_dict = torch.load(checkpoint_dir / "encoder.pt", map_location=device)
        model = CrossLayerTranscoder(config, state_dict)
        model.eval()
        return model.to(device)
