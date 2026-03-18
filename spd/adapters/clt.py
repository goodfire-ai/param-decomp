"""CLT adapter: loads a trained Cross-Layer Transcoder from a wandb artifact."""

from functools import cached_property
from typing import override

import torch
from torch.utils.data import DataLoader

from spd.adapters.base import DecompositionAdapter, pretrain_dataloader
from spd.adapters.clt_model import CrossLayerTranscoder
from spd.adapters.transcoder import _download_artifact
from spd.autointerp.schemas import ModelMetadata
from spd.harvest.config import CLTHarvestConfig
from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLP
from spd.pretrain.run_info import PretrainRunInfo
from spd.topology import TransformerTopology


class CLTAdapter(DecompositionAdapter):
    def __init__(self, config: CLTHarvestConfig):
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
    def clt(self) -> CrossLayerTranscoder:
        checkpoint_dir = _download_artifact(self._config.artifact_path)
        return CrossLayerTranscoder.from_checkpoint(checkpoint_dir, "cpu")

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
        return [(f"h.{i}.mlp", self.clt.dict_size) for i in self.clt.layers]

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
            layer_descriptions={f"h.{i}.mlp": f"{i}.mlp" for i in self.clt.layers},
            seq_len=self.base_model.config.block_size,
            decomposition_method="clt",
        )

    @override
    def dataloader(self, batch_size: int) -> DataLoader[torch.Tensor]:
        return pretrain_dataloader(self._run_info, batch_size)
