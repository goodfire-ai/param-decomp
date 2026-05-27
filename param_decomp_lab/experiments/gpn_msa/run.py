"""GPN-MSA PD experiment: YAML -> `optimize()` glue, plus the `SavedGPNMSARun` reload class.

GPN-MSA's `forward` takes structured input — `input_ids` *and* `aux_features` (encoding
the multi-species alignment column at each position) — so we cannot use
`param_decomp_lab.batch_and_loss_fns.make_run_batch` (which only does `model(batch)` or
`model(batch[k])`). Instead the batch is a `dict[str, Tensor]` and `run_batch_gpn_msa`
unpacks it via `model(**batch).logits`. Core's `move_batch_to_device` handles dicts
already.

Real MSA data: see the GPN repo's data prep
(`https://github.com/songlab-cal/gpn/blob/main/analysis/gpn-msa_sapiens/`); we'd need
genome FASTAs + cross-species alignments. For scaffolding the experiment, we generate
synthetic random tensors of the right shape.

Run via `pd-gpn-msa path/to/config.yaml`.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, override

import fire
import torch
import torch.nn as nn
from pydantic import Field, PositiveInt
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset

from param_decomp.base_config import BaseConfig, runtime_cast
from param_decomp.batch_and_loss_fns import RunBatch
from param_decomp.component_model import ComponentModel
from param_decomp.distributed import DistributedState
from param_decomp.log import logger
from param_decomp.optimize import EvalLoop, optimize
from param_decomp_lab.batch_and_loss_fns import recon_loss_kl
from param_decomp_lab.component_model_io import load_component_model
from param_decomp_lab.distributed import get_device
from param_decomp_lab.eval_metrics import EVAL_METRIC_CLASSES
from param_decomp_lab.experiments.utils import (
    RUN_META_FILENAME,
    ExperimentConfig,
    init_pd_run,
)
from param_decomp_lab.infra.paths import ModelPath
from param_decomp_lab.infra.run_files import resolve_run_files
from param_decomp_lab.seed import set_seed


class GPNMSATargetConfig(BaseConfig):
    """Target HF model id for GPN-MSA (only one published checkpoint right now)."""

    model_name: str = Field(
        default="songlab/gpn-msa-sapiens",
        description="HF id for the GPN-MSA checkpoint",
    )


class GPNMSADataConfig(BaseConfig):
    """Synthetic-MSA dataset settings.

    `n_species` is the number of aligned species per position; the model one-hots each
    position to `n_species * aux_features_vocab_size` features internally. The published
    `songlab/gpn-msa-sapiens` has `n_aux_features=445` post-one-hot and
    `aux_features_vocab_size=5`, so `n_species=89` is the natural default.
    """

    seq_len: PositiveInt = Field(default=128)
    n_species: PositiveInt = Field(default=89)
    vocab_size: PositiveInt = Field(default=6)
    aux_features_vocab_size: PositiveInt = Field(default=5)
    # n_steps_per_epoch only matters for how the synthetic loader yields; the trainer
    # loops it indefinitely via `loop_dataloader`.
    n_synthetic_samples: PositiveInt = Field(default=1024)


class GPNMSAExperimentConfig(ExperimentConfig[GPNMSATargetConfig, GPNMSADataConfig]):
    pass


def build_target(target_cfg: GPNMSATargetConfig) -> nn.Module:
    """Load the GPN-MSA target via HF auto class.

    `import gpn.model` registers `GPNRoFormer` as an auto-class entry, so
    `AutoModelForMaskedLM.from_pretrained(...)` works without `trust_remote_code` from
    the hub. We still pass `trust_remote_code=True` for forward compatibility with any
    future hub-side reorganisations.
    """
    import gpn.model as _gpn_model  # registers GPNRoFormer with HF auto classes
    from transformers.models.auto.modeling_auto import AutoModelForMaskedLM

    assert _gpn_model is not None  # keep side-effect import alive after linting

    target_model = AutoModelForMaskedLM.from_pretrained(
        target_cfg.model_name, trust_remote_code=True
    )
    assert isinstance(target_model, nn.Module)
    target_model.eval()
    return target_model


class _SyntheticMSADataset(IterableDataset[dict[str, Tensor]]):
    """Yield random `(input_ids, aux_features)` pairs of the right shape.

    Used purely for scaffolding / smoke tests. Real MSA data would come from
    cross-species alignments (the GPN repo provides scripts to extract these from a
    reference + per-species FASTAs).
    """

    def __init__(
        self,
        *,
        seq_len: int,
        n_species: int,
        vocab_size: int,
        aux_features_vocab_size: int,
        n_samples: int,
        seed: int,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.n_species = n_species
        self.vocab_size = vocab_size
        self.aux_features_vocab_size = aux_features_vocab_size
        self.n_samples = n_samples
        self.seed = seed

    @override
    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.n_samples):
            input_ids = torch.randint(0, self.vocab_size, (self.seq_len,), generator=g)
            aux_features = torch.randint(
                0,
                self.aux_features_vocab_size,
                (self.seq_len, self.n_species),
                generator=g,
            )
            yield {"input_ids": input_ids, "aux_features": aux_features}


def build_gpn_msa_loader(
    target_cfg: GPNMSATargetConfig,
    data_cfg: GPNMSADataConfig,
    *,
    split: Literal["train", "eval"],
    device: str,
    batch_size: int,
    dist_state: DistributedState | None = None,
    seed: int | None = None,
) -> DataLoader[Any]:
    """Synthetic GPN-MSA `DataLoader`.

    `dist_state` is accepted for protocol-compat with other experiments but ignored —
    every rank yields its own RNG-disjoint synthetic stream. The eval seed is offset by
    1 to decouple from train.
    """
    del target_cfg, device, dist_state
    effective_seed = (seed or 0) + (1 if split == "eval" else 0)
    dataset = _SyntheticMSADataset(
        seq_len=data_cfg.seq_len,
        n_species=data_cfg.n_species,
        vocab_size=data_cfg.vocab_size,
        aux_features_vocab_size=data_cfg.aux_features_vocab_size,
        n_samples=data_cfg.n_synthetic_samples,
        seed=effective_seed,
    )
    return DataLoader(dataset, batch_size=batch_size, drop_last=True)


def run_batch_gpn_msa(model: nn.Module, batch: dict[str, Tensor]) -> Tensor:
    """Run GPN-MSA on a `{"input_ids", "aux_features"}` batch and return logits."""
    out = model(**batch)
    return runtime_cast(Tensor, out.logits)


def make_run_batch(target_cfg: GPNMSATargetConfig) -> RunBatch:
    del target_cfg
    return run_batch_gpn_msa


@dataclass(frozen=True)
class SavedGPNMSARun:
    """Handle to a completed GPN-MSA PD run on disk or in W&B."""

    cfg: GPNMSAExperimentConfig
    checkpoint_path: Path

    @classmethod
    def from_path(cls, path: ModelPath) -> "SavedGPNMSARun":
        files = resolve_run_files(
            path, config_filename=RUN_META_FILENAME, checkpoint_prefix="model"
        )
        return cls(
            cfg=GPNMSAExperimentConfig.from_file(files.config_path),
            checkpoint_path=files.checkpoint_path,
        )

    def load_model(self) -> ComponentModel:
        return load_component_model(
            pd_config=self.cfg.pd,
            checkpoint_path=self.checkpoint_path,
            target_model=build_target(self.cfg.target),
            run_batch=make_run_batch(self.cfg.target),
        )


def main(
    config_path: str | Path,
    *,
    group: str | None = None,
    tags: str | None = None,
) -> None:
    """Run a GPN-MSA PD experiment end-to-end from a YAML config."""
    cfg = GPNMSAExperimentConfig.from_file(config_path)

    set_seed(cfg.pd.seed)
    device = get_device()
    logger.info(f"Using device: {device}")

    target_model = build_target(cfg.target).to(device)
    cfg = cfg.model_copy(update={"runtime": cfg.runtime.model_copy(update={"device": device})})

    train_loader = build_gpn_msa_loader(
        cfg.target,
        cfg.data,
        split="train",
        device=device,
        batch_size=cfg.pd.batch_size,
        seed=cfg.pd.seed,
    )
    eval_loop = _build_eval_loop(cfg, device)

    sink = init_pd_run(cfg, group=group, tags=tags)

    try:
        optimize(
            target_model=target_model,
            train_loader=train_loader,
            run_batch=make_run_batch(cfg.target),
            reconstruction_loss=recon_loss_kl,
            pd_config=cfg.pd,
            runtime_config=cfg.runtime,
            sink=sink,
            cadence=cfg.cadence,
            eval_loop=eval_loop,
        )
    finally:
        sink.finish()


def _build_eval_loop(cfg: GPNMSAExperimentConfig, device: str) -> EvalLoop | None:
    if cfg.eval is None:
        return None
    eval_loader = build_gpn_msa_loader(
        cfg.target,
        cfg.data,
        split="eval",
        device=device,
        batch_size=cfg.eval.batch_size,
        seed=cfg.pd.seed,
    )
    return EvalLoop(
        loader=eval_loader,
        metrics=[EVAL_METRIC_CLASSES[m.type](m) for m in cfg.eval.metrics],
        n_steps=cfg.eval.n_steps,
        every=cfg.eval.every,
        slow_every=cfg.eval.slow_every,
        slow_on_first_step=cfg.eval.slow_on_first_step,
    )


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
