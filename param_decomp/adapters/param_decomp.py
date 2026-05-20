from functools import cached_property
from typing import override

from torch import Tensor
from torch.utils.data import DataLoader

from param_decomp.adapters.base import DecompositionAdapter
from param_decomp.autointerp.schemas import ModelMetadata
from param_decomp.experiments.lm.experiment import lm_data, lm_target
from param_decomp.models.component_model import ComponentModel
from param_decomp.run import RunConfig
from param_decomp.saved_run import SavedRun
from param_decomp.topology import TransformerTopology
from param_decomp.utils.wandb_utils import parse_wandb_run_path


class PDAdapter(DecompositionAdapter):
    def __init__(self, wandb_path: str):
        self._wandb_path = wandb_path
        _, _, self._run_id = parse_wandb_run_path(wandb_path)

    @cached_property
    def pd_run(self) -> SavedRun:
        return SavedRun.from_path(self._wandb_path)

    @cached_property
    def run(self) -> RunConfig:
        assert self.pd_run.run_cfg is not None  # always set on from_path handles
        return self.pd_run.run_cfg

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
        # PDAdapter is LM-only; the LM driver ignores `device` because batches
        # are moved per-step.
        return self.pd_run.build_train_loader(device="cpu", batch_size_override=batch_size)

    @property
    @override
    def tokenizer_name(self) -> str:
        return lm_data(self.run).tokenizer_name

    @property
    @override
    def model_metadata(self) -> ModelMetadata:
        target = lm_target(self.run)
        data = lm_data(self.run)
        return ModelMetadata(
            n_blocks=self._topology.n_blocks,
            model_class=target.model_class,
            dataset_name=data.dataset_name,
            layer_descriptions={
                path: self._topology.target_to_canon(path)
                for path in self.component_model.target_module_paths
            },
            seq_len=data.max_seq_len,
            decomposition_method="pd",
        )
