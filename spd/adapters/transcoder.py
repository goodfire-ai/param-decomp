"""Transcoder adapter: loads trained transcoders from wandb artifacts."""

import json
from functools import cached_property
from pathlib import Path
from typing import Any, override

import torch
import wandb
from nn_decompositions.config import EncoderConfig
from nn_decompositions.transcoder import (
    BatchTopKTranscoder,
    JumpReLUTranscoder,
    SharedTranscoder,
    TopKTranscoder,
    VanillaTranscoder,
)
from torch.utils.data import DataLoader

from spd.adapters.base import DecompositionAdapter
from spd.autointerp.schemas import ModelMetadata
from spd.data import DatasetConfig, create_data_loader
from spd.harvest.config import TranscoderHarvestConfig
from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLP
from spd.pretrain.run_info import PretrainRunInfo
from spd.topology import TransformerTopology

_ENCODER_CLASSES: dict[str, type[SharedTranscoder]] = {
    "vanilla": VanillaTranscoder,
    "topk": TopKTranscoder,
    "batchtopk": BatchTopKTranscoder,
    "jumprelu": JumpReLUTranscoder,
}


def _load_transcoder(checkpoint_dir: Path, device: str) -> SharedTranscoder:
    with open(checkpoint_dir / "config.json") as f:
        cfg_dict: dict[str, Any] = json.load(f)
    cfg_dict["dtype"] = getattr(torch, cfg_dict.get("dtype", "torch.float32").replace("torch.", ""))
    cfg_dict["device"] = device
    cfg = EncoderConfig(**cfg_dict)
    encoder = _ENCODER_CLASSES[cfg.encoder_type](cfg)
    encoder.load_state_dict(torch.load(checkpoint_dir / "encoder.pt", map_location=device))
    encoder.eval()
    return encoder


def _download_artifact(artifact_path: str, dest: Path) -> Path:
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
    def _train_dataset_config(self) -> dict[str, Any]:
        cfg = self._run_info.config_dict.get("train_dataset_config")
        assert isinstance(cfg, dict), "base model run missing train_dataset_config"
        return cfg

    @cached_property
    def transcoders(self) -> dict[str, SharedTranscoder]:
        result: dict[str, SharedTranscoder] = {}
        for module_path, artifact_path in self._config.artifact_paths.items():
            safe_name = artifact_path.replace("/", "_").replace(":", "_")
            dest = Path(f"checkpoints/tc_{safe_name}")
            checkpoint_dir = _download_artifact(artifact_path, dest)
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
        return ModelMetadata(
            n_blocks=self._topology.n_blocks,
            model_class="spd.pretrain.models.llama_simple_mlp.LlamaSimpleMLP",
            dataset_name=self._train_dataset_config["name"],
            layer_descriptions={
                path: self._topology.target_to_canon(path) for path in self.transcoders
            },
        )

    @override
    def dataloader(self, batch_size: int) -> DataLoader[torch.Tensor]:
        ds_cfg = self._train_dataset_config
        dataset_config = DatasetConfig(
            name=ds_cfg["name"],
            is_tokenized=ds_cfg.get("is_tokenized", True),
            hf_tokenizer_path=self.tokenizer_name,
            streaming=True,
            split="train",
            n_ctx=self.base_model.config.block_size,
            column_name=ds_cfg.get("column_name", "input_ids"),
        )
        loader, _ = create_data_loader(
            dataset_config=dataset_config,
            batch_size=batch_size,
            buffer_size=1000,
        )
        return loader
