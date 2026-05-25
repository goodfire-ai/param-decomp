from functools import cached_property
from typing import override

from torch import Tensor
from torch.utils.data import DataLoader

from param_decomp.component_model import ComponentModel
from param_decomp_lab.adapters.base import DecompositionAdapter
from param_decomp_lab.autointerp.schemas import ModelMetadata
from param_decomp_lab.experiments.lm.run import SavedLMRun, build_lm_loader
from param_decomp_lab.infra.wandb import parse_wandb_run_path
from param_decomp_lab.topology import TransformerTopology


class PDAdapter(DecompositionAdapter):
    def __init__(self, wandb_path: str):
        self._wandb_path = wandb_path
        _, _, self._run_id = parse_wandb_run_path(wandb_path)

    @cached_property
    def pd_run(self) -> SavedLMRun:
        return SavedLMRun.from_path(self._wandb_path)

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
        # PDAdapter is LM-only; build_lm_loader ignores `device` because batches are
        # moved per-step.
        return build_lm_loader(
            self.pd_run.cfg.target,
            self.pd_run.cfg.data,
            split="train",
            device="cpu",
            batch_size=batch_size,
        )

    @property
    @override
    def tokenizer_name(self) -> str:
        return self.pd_run.cfg.data.tokenizer_name

    @property
    @override
    def model_metadata(self) -> ModelMetadata:
        cfg = self.pd_run.cfg
        return ModelMetadata(
            n_blocks=self._topology.n_blocks,
            model_class=cfg.target.spec.model_class,
            dataset_name=cfg.data.dataset_name,
            layer_descriptions={
                path: self._topology.target_to_canon(path)
                for path in self.component_model.target_module_paths
            },
            seq_len=cfg.data.max_seq_len,
            decomposition_method="pd",
        )
