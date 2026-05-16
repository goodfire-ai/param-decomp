from functools import cached_property
from typing import override

from torch import Tensor
from torch.utils.data import DataLoader

from param_decomp.adapters.base import DecompositionAdapter
from param_decomp.autointerp.schemas import ModelMetadata
from param_decomp.experiments.driver import ExperimentConfig
from param_decomp.experiments.lm.experiment import LMExperimentConfig
from param_decomp.models.component_model import ComponentModel
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
    def experiment_config(self) -> ExperimentConfig:
        exp = self.pd_run.experiment_config
        assert exp is not None, "PD run has no driver; cannot reconstruct experiment config"
        return exp

    @cached_property
    def lm_experiment_config(self) -> LMExperimentConfig:
        exp = self.experiment_config
        assert isinstance(exp, LMExperimentConfig), (
            f"This method requires an LM run, got {type(exp).__name__}"
        )
        return exp

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
        return self.lm_experiment_config.data.tokenizer_name

    @property
    @override
    def model_metadata(self) -> ModelMetadata:
        exp = self.lm_experiment_config
        return ModelMetadata(
            n_blocks=self._topology.n_blocks,
            model_class=exp.target.model_class,
            dataset_name=exp.data.dataset_name,
            layer_descriptions={
                path: self._topology.target_to_canon(path)
                for path in self.component_model.target_module_paths
            },
            seq_len=exp.data.max_seq_len,
            decomposition_method="pd",
        )
