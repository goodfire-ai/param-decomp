from functools import cached_property
from typing import override

from torch import Tensor
from torch.utils.data import DataLoader

from param_decomp.adapters.base import DecompositionAdapter
from param_decomp.autointerp.schemas import ModelMetadata
from param_decomp.configs import LMTaskConfig
from param_decomp.data import train_loader_and_tokenizer
from param_decomp.models.component_model import ComponentModel, ParamDecompRunInfo
from param_decomp.topology import TransformerTopology
from param_decomp.utils.general_utils import runtime_cast
from param_decomp.utils.wandb_utils import parse_wandb_run_path


class ParamDecompAdapter(DecompositionAdapter):
    def __init__(self, wandb_path: str):
        self._wandb_path = wandb_path
        _, _, self._run_id = parse_wandb_run_path(wandb_path)

    @cached_property
    def pd_run_info(self):
        return ParamDecompRunInfo.from_path(self._wandb_path)

    @cached_property
    def component_model(self):
        return ComponentModel.from_run_info(self.pd_run_info)

    @cached_property
    def _topology(self) -> TransformerTopology:
        return TransformerTopology(self.component_model.target_model)

    @property
    @override
    def decomposition_id(self) -> str:
        return self._run_id

    @property
    @override
    def vocab_size(self) -> int:
        return self._topology.embedding_module.num_embeddings

    @property
    @override
    def layer_activation_sizes(self) -> list[tuple[str, int]]:
        cm = self.component_model
        return list(cm.module_to_c.items())

    @override
    def dataloader(self, batch_size: int) -> DataLoader[Tensor]:
        return train_loader_and_tokenizer(self.pd_run_info.config, batch_size)[0]

    @property
    @override
    def tokenizer_name(self) -> str:
        cfg = self.pd_run_info.config
        assert cfg.tokenizer_name is not None
        return cfg.tokenizer_name

    @property
    @override
    def model_metadata(self) -> ModelMetadata:
        cfg = self.pd_run_info.config
        task_cfg = runtime_cast(LMTaskConfig, cfg.task_config)
        return ModelMetadata(
            n_blocks=self._topology.n_blocks,
            model_class=cfg.pretrained_model_class,
            dataset_name=task_cfg.dataset_name,
            layer_descriptions={
                path: self._topology.target_to_canon(path)
                for path in self.component_model.target_module_paths
            },
            seq_len=task_cfg.max_seq_len,
            decomposition_method="pd",
        )
