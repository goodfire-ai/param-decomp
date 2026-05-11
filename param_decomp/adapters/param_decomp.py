from functools import cached_property
from typing import override

from torch import Tensor
from torch.utils.data import DataLoader

from param_decomp.adapters.base import DecompositionAdapter
from param_decomp.autointerp.schemas import ModelMetadata
from param_decomp.experiments.driver import ExperimentConfig
from param_decomp.experiments.lm.experiment import LMExperimentConfig
from param_decomp.load import load_pd
from param_decomp.models.component_model import ComponentModel, PDRunInfo
from param_decomp.topology import TransformerTopology
from param_decomp.utils.wandb_utils import parse_wandb_run_path


class PDAdapter(DecompositionAdapter):
    def __init__(self, wandb_path: str):
        self._wandb_path = wandb_path
        _, _, self._run_id = parse_wandb_run_path(wandb_path)

    @cached_property
    def pd_run_info(self) -> PDRunInfo:
        return PDRunInfo.from_path(self._wandb_path)

    @cached_property
    def experiment_config(self) -> ExperimentConfig:
        return self.pd_run_info.experiment_config

    @cached_property
    def component_model(self) -> ComponentModel:
        target = self.pd_run_info.load_target()
        return load_pd(self._wandb_path, target=target)

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
        train_loader, _ = self.pd_run_info.build_dataloaders(
            seed=None,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
        )
        return train_loader

    @property
    @override
    def tokenizer_name(self) -> str:
        exp = self.experiment_config
        assert isinstance(exp, LMExperimentConfig), f"No tokenizer for kind={exp.kind!r}"
        return exp.data.tokenizer_name

    @property
    @override
    def model_metadata(self) -> ModelMetadata:
        exp = self.experiment_config
        assert isinstance(exp, LMExperimentConfig), (
            f"`model_metadata` is not implemented for kind={exp.kind!r}"
        )
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


# Back-compat alias. Removed in step 9.
ParamDecompAdapter = PDAdapter
