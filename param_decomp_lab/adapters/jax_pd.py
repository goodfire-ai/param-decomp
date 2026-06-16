from functools import cached_property
from pathlib import Path
from typing import Any, override

import yaml
from torch.utils.data import DataLoader

from param_decomp.decomposition_targets import resolve_decomposition_targets
from param_decomp_config.lm import LMExperimentConfig
from param_decomp_lab.adapters.base import DecompositionAdapter
from param_decomp_lab.autointerp.schemas import ModelMetadata
from param_decomp_lab.experiments.lm.run import build_target
from param_decomp_lab.experiments.utils import EXPERIMENT_CONFIG_FILENAME
from param_decomp_lab.harvest.schemas import get_harvest_dir
from param_decomp_lab.topology import TransformerTopology


def is_jax_run(decomposition_id: str) -> bool:
    """A JAX single-pool run dir pins the `pd-jax-lm` wrapper as `config.yaml` (it carries
    a `torch_config:` key) and checkpoints with orbax under `ckpts/`; a torch run instead
    has `model_*.pth`. The wrapper key is the explicit marker."""
    wrapper = get_harvest_dir(decomposition_id).parent / "config.yaml"
    if not wrapper.exists():
        return False
    raw = yaml.safe_load(wrapper.read_text())
    return isinstance(raw, dict) and "torch_config" in raw


class JaxPDAdapter(DecompositionAdapter):
    """Autointerp/clustering adapter for a JAX single-pool run, read from its pinned
    config. Autointerp consumes harvest output plus run metadata only — no trained
    components — so this builds the target *architecture* from config (no orbax restore,
    no PD checkpoint) purely to derive `n_blocks` and canonical layer descriptions."""

    def __init__(self, decomposition_id: str):
        self._run_id = decomposition_id

    @cached_property
    def cfg(self) -> LMExperimentConfig:
        config_path = get_harvest_dir(self._run_id).parent / EXPERIMENT_CONFIG_FILENAME
        assert config_path.exists(), f"config not found: {config_path}"
        return LMExperimentConfig.from_file(config_path)

    @cached_property
    def _topology(self) -> TransformerTopology:
        return TransformerTopology(build_target(self.cfg.target))

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
        targets = resolve_decomposition_targets(
            self._topology.target_model, self.cfg.pd.decomposition_targets
        )
        return [(t.module_path, t.C) for t in targets]

    @override
    def dataloader(self, batch_size: int) -> DataLoader[Any]:
        raise NotImplementedError(
            "JaxPDAdapter does not build a torch dataloader; the JAX harvest worker reads "
            "pre-tokenized parquet via the trainer's ShardServer."
        )

    @property
    @override
    def tokenizer_name(self) -> str:
        return self.cfg.data.tokenizer_name

    @property
    @override
    def model_metadata(self) -> ModelMetadata:
        cfg = self.cfg
        return ModelMetadata(
            n_blocks=self._topology.n_blocks,
            dataset_name=self._semantic_dataset_name(),
            layer_descriptions={
                path: self._topology.target_to_canon(path)
                for path, _ in self.layer_activation_sizes
            },
            seq_len=cfg.data.max_seq_len,
            decomposition_method="pd",
        )

    def _semantic_dataset_name(self) -> str:
        """The corpus identity for `DATASET_DESCRIPTIONS`. The JAX trainer reads
        pre-tokenized parquet, so its `dataset_name` is the loader name `"parquet"`, not
        the corpus — recover the corpus from the shard directory (e.g.
        `.../pile_neox_tok_512/*.parquet` -> `pile_neox_tok_512`)."""
        data = self.cfg.data
        if data.dataset_name != "parquet":
            return data.dataset_name
        assert data.data_files is not None, "parquet data config without data_files"
        return Path(data.data_files).parent.name
