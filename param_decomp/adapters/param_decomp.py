from functools import cached_property
from typing import override

from torch import Tensor
from torch.utils.data import DataLoader

from param_decomp.adapters.base import DecompositionAdapter
from param_decomp.autointerp.schemas import ModelMetadata
from param_decomp.experiments.lm.experiment import LMRunConfig
from param_decomp.models.component_model import ComponentModel
from param_decomp.run import RunConfig
from param_decomp.saved_run import PDRun
from param_decomp.topology import TransformerTopology
from param_decomp.utils.wandb_utils import parse_wandb_run_path


class PDAdapter(DecompositionAdapter):
    def __init__(self, wandb_path: str):
        self._wandb_path = wandb_path
        _, _, self._run_id = parse_wandb_run_path(wandb_path)

    @cached_property
    def pd_run(self) -> PDRun:
        return PDRun.from_path(self._wandb_path)

    @cached_property
    def run(self) -> RunConfig:
        return self.pd_run.run_cfg

    @cached_property
    def lm_run(self) -> LMRunConfig:
        run = self.run
        assert isinstance(run, LMRunConfig), (
            f"This method requires an LM run, got {type(run).__name__}"
        )
        return run

    @cached_property
    def component_model(self) -> ComponentModel:
        return self.pd_run.load_model()

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
        train_loader, _ = self.pd_run.load_dataloaders(
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
        )
        return train_loader

    @property
    @override
    def tokenizer_name(self) -> str:
        return self.lm_run.data.tokenizer_name

    @property
    @override
    def model_metadata(self) -> ModelMetadata:
        run = self.lm_run
        return ModelMetadata(
            n_blocks=self._topology.n_blocks,
            model_class=run.target.model_class,
            dataset_name=run.data.dataset_name,
            layer_descriptions={
                path: self._topology.target_to_canon(path)
                for path in self.component_model.target_module_paths
            },
            seq_len=run.data.max_seq_len,
            decomposition_method="pd",
        )
