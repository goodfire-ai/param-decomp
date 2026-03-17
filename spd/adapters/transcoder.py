"""Transcoder adapter: loads trained transcoders from wandb artifacts."""

import json
from functools import cached_property
from pathlib import Path
from typing import Any, override

import torch
import wandb
from torch.utils.data import DataLoader

from spd.adapters.base import DecompositionAdapter, pretrain_dataloader
from spd.adapters.transcoder_model import BatchTopKTranscoder, EncoderConfig
from spd.autointerp.schemas import ModelMetadata
from spd.harvest.config import TranscoderHarvestConfig
from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLP
from spd.pretrain.run_info import PretrainRunInfo
from spd.topology import TransformerTopology

# E2e-trained transcoders save extra fields in config.json ("e2e", "e2e_cascading")
# that aren't part of EncoderConfig. Strip them so the dataclass constructor doesn't choke.
_ENCODER_CONFIG_FIELDS = frozenset(f.name for f in __import__("dataclasses").fields(EncoderConfig))


def _load_transcoder(checkpoint_dir: Path, device: str) -> BatchTopKTranscoder:
    with open(checkpoint_dir / "config.json") as f:
        cfg_dict: dict[str, Any] = json.load(f)
    cfg_dict["dtype"] = getattr(torch, cfg_dict.get("dtype", "torch.float32").replace("torch.", ""))
    cfg_dict["device"] = device
    filtered = {k: v for k, v in cfg_dict.items() if k in _ENCODER_CONFIG_FIELDS}
    cfg = EncoderConfig(**filtered)
    assert cfg.encoder_type == "batchtopk", f"Only batchtopk supported, got {cfg.encoder_type}"
    encoder = BatchTopKTranscoder(cfg)
    encoder.load_state_dict(torch.load(checkpoint_dir / "encoder.pt", map_location=device))
    encoder.eval()
    return encoder


def _download_artifact(artifact_path: str) -> Path:
    from spd.settings import SPD_OUT_DIR

    safe_name = artifact_path.replace("/", "_").replace(":", "_")
    dest = SPD_OUT_DIR / "checkpoints" / safe_name
    if dest.exists() and (dest / "encoder.pt").exists():
        return dest
    api = wandb.Api()
    artifact = api.artifact(artifact_path)
    artifact.download(root=str(dest))
    return dest


class TranscoderAdapter(DecompositionAdapter):
    def __init__(self, config: TranscoderHarvestConfig):
        self._config = config

    @cached_property
    def _run_info(self) -> PretrainRunInfo:
        return PretrainRunInfo.from_path(self._config.base_model_path)

    @cached_property
    def base_model(self) -> LlamaSimpleMLP:
        return LlamaSimpleMLP.from_run_info(self._run_info)

    @cached_property
    def _topology(self) -> TransformerTopology:
        return TransformerTopology(self.base_model)

    @cached_property
    def transcoders(self) -> dict[str, BatchTopKTranscoder]:
        result: dict[str, BatchTopKTranscoder] = {}
        for module_path, artifact_path in self._config.artifact_paths.items():
            checkpoint_dir = _download_artifact(artifact_path)
            result[module_path] = _load_transcoder(checkpoint_dir, "cpu")
        return result

    @property
    @override
    def decomposition_id(self) -> str:
        return self._config.id

    @property
    @override
    def vocab_size(self) -> int:
        return self.base_model.config.vocab_size

    @property
    @override
    def layer_activation_sizes(self) -> list[tuple[str, int]]:
        return [(path, tc.dict_size) for path, tc in self.transcoders.items()]

    @property
    @override
    def tokenizer_name(self) -> str:
        tok = self._run_info.hf_tokenizer_path
        assert tok is not None, "base model run missing hf_tokenizer_path"
        return tok

    @property
    @override
    def model_metadata(self) -> ModelMetadata:
        ds_cfg = self._run_info.config_dict.get("train_dataset_config", {})
        model_cls = type(self.base_model)
        return ModelMetadata(
            n_blocks=self._topology.n_blocks,
            model_class=f"{model_cls.__module__}.{model_cls.__qualname__}",
            dataset_name=ds_cfg.get("name", "unknown"),
            layer_descriptions={path: path.removeprefix("h.") for path in self.transcoders},
            seq_len=self.base_model.config.block_size,
            decomposition_method="transcoder",
        )

    @override
    def dataloader(self, batch_size: int) -> DataLoader[torch.Tensor]:
        return pretrain_dataloader(self._run_info, batch_size)
