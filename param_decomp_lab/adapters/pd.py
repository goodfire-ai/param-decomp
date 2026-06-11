from functools import cached_property
from typing import override

import yaml
from torch import Tensor
from torch.utils.data import DataLoader

from param_decomp_lab.adapters.base import DecompositionAdapter
from param_decomp_lab.autointerp.schemas import ModelMetadata
from param_decomp_lab.component_model_io import HarvestableComponentModel
from param_decomp_lab.experiments.lm.run import SavedLMRun, build_lm_loader
from param_decomp_lab.experiments.lm.three_pool_run import SavedThreePoolLMRun
from param_decomp_lab.experiments.utils import EXPERIMENT_CONFIG_FILENAME
from param_decomp_lab.infra.run_files import resolve_config_path
from param_decomp_lab.infra.wandb import parse_wandb_run_path
from param_decomp_lab.topology import TransformerTopology


def load_saved_lm_run(path: str) -> SavedLMRun | SavedThreePoolLMRun:
    """Reload a single-pool or 3-pool LM run, dispatching on the saved config shape.

    3-pool configs carry `runtime.topology`; single-pool `RuntimeConfig` has no such field.
    Both saved-run types expose the same inference surface (`cfg.pd` / `cfg.target` /
    `cfg.data` / `load_model`), so callers consume the union uniformly.
    """
    config_path = resolve_config_path(path, config_filename=EXPERIMENT_CONFIG_FILENAME)
    raw = yaml.safe_load(config_path.read_text())
    is_three_pool = "topology" in raw["runtime"]
    return SavedThreePoolLMRun.from_path(path) if is_three_pool else SavedLMRun.from_path(path)


class PDAdapter(DecompositionAdapter):
    def __init__(self, wandb_path: str, target_module_paths: list[str] | None = None):
        self._wandb_path = wandb_path
        self._target_module_paths = target_module_paths
        _, _, self._run_id = parse_wandb_run_path(wandb_path)

    @cached_property
    def pd_run(self) -> SavedLMRun | SavedThreePoolLMRun:
        return load_saved_lm_run(self._wandb_path)

    @cached_property
    def component_model(self) -> HarvestableComponentModel:
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
        if self._target_module_paths is None:
            return list(cm.module_to_c.items())
        missing = set(self._target_module_paths) - set(cm.module_to_c)
        assert not missing, f"target_module_paths not in decomposition: {sorted(missing)}"
        return [(path, cm.module_to_c[path]) for path in self._target_module_paths]

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
            dataset_name=cfg.data.dataset_name,
            layer_descriptions={
                path: self._topology.target_to_canon(path)
                for path in self.component_model.target_module_paths
            },
            seq_len=cfg.data.max_seq_len,
            decomposition_method="pd",
        )
