"""ESM2 PD experiment: YAML -> bundle/runner glue, plus the `SavedESM2Run` reload class.

ESM2 is a protein masked-language model (HF `EsmForMaskedLM`). The decomposition
target is the frozen MLM; recon loss is KL between component-replaced logits and the
target's logits at every position — the standard "match target model" recon loss used
by the LM experiment.

A masked-positions-only MLM cross-entropy (true training-style MLM loss) is *not* the
right recon loss for PD: PD recon compares component-replaced output against the
*target model's* output, not ground-truth labels. The masked-CE variant is a
ground-truth supervision signal — different objective.

Data is synthetic random amino-acid token ids by default — small, fast, no HF dataset
download required. The `data.kind: hf` branch is left as a typed stub
(`NotImplementedError`) and surfaces an abstraction gap (see report).

Run via `pd-esm2 path/to/config.yaml`.
"""

import importlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, override

import fire
import torch
import torch.nn as nn
from pydantic import Discriminator, Field, PositiveInt
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset

from param_decomp.base_config import BaseConfig
from param_decomp.batch_and_loss_fns import RunBatch
from param_decomp.component_model import ComponentModel
from param_decomp.distributed import DistributedState
from param_decomp.optimize import EvalLoop
from param_decomp_lab.batch_and_loss_fns import make_run_batch as _make_run_batch
from param_decomp_lab.batch_and_loss_fns import recon_loss_kl
from param_decomp_lab.component_model_io import load_component_model
from param_decomp_lab.eval_metrics import EVAL_METRIC_CLASSES
from param_decomp_lab.experiments.runner import ExperimentBundle, run_fresh
from param_decomp_lab.experiments.utils import RUN_META_FILENAME, ExperimentConfig
from param_decomp_lab.infra.paths import ModelPath
from param_decomp_lab.infra.run_files import resolve_run_files


def _resolve_class(fqn: str) -> type:
    module_path, _, class_name = fqn.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class ESM2TargetConfig(BaseConfig):
    """ESM2 HuggingFace masked-LM target.

    `output_extract` (passed to `make_run_batch`) pulls the logits tensor out of the HF
    forward output object; for `EsmForMaskedLM` this is `"logits"` of shape
    `(batch, seq, vocab=33)`.
    """

    model_class: str = "transformers.EsmForMaskedLM"
    model_name: str = "facebook/esm2_t30_150M_UR50D"
    output_extract: int | str | None = "logits"


class ESM2SyntheticData(BaseConfig):
    """Synthetic random amino-acid token-id loader.

    Generates batches of shape `(batch, max_seq_len)` with token ids uniformly in
    `[vocab_lo, vocab_hi)`. ESM2's amino-acid token ids are 4..23 (special tokens are
    0..3, X/B/U/Z/O are 24..30, `<mask>`=32). Default range avoids the special tokens.
    """

    kind: Literal["synthetic"] = "synthetic"
    max_seq_len: PositiveInt = Field(default=32)
    vocab_lo: int = Field(default=4)
    vocab_hi: int = Field(default=24)


class ESM2UniRef50Data(BaseConfig):
    """UniRef50 protein-sequences HF dataset config.

    Not implemented in the scaffolding; left as a typed stub so the YAML schema is honest
    about the eventual data source. See report for the abstraction gap (the existing
    `create_lm_data_loader` packs concatenated tokens cross-document, which is wrong for
    protein sequences — each sequence is one example).
    """

    kind: Literal["uniref50"] = "uniref50"
    dataset_name: str = "agemagician/uniref50"
    tokenizer_name: str = "facebook/esm2_t30_150M_UR50D"
    max_seq_len: PositiveInt = Field(default=128)
    train_split: str = "train"
    eval_split: str = "valid"


ESM2DataSpec = Annotated[ESM2SyntheticData | ESM2UniRef50Data, Discriminator("kind")]


class ESM2DataConfig(BaseConfig):
    spec: ESM2DataSpec = Field(..., description="Discriminated data source.")


class ESM2ExperimentConfig(ExperimentConfig[ESM2TargetConfig, ESM2DataConfig]):
    pass


def build_target(target_cfg: ESM2TargetConfig) -> nn.Module:
    """Load the ESM2 target model in eval mode."""
    cls = _resolve_class(target_cfg.model_class)
    target_model = cls.from_pretrained(target_cfg.model_name)
    target_model.eval()
    return target_model


class _SyntheticAATokens(IterableDataset[Tensor]):
    """Infinite stream of `(seq_len,)` random amino-acid token-id tensors."""

    def __init__(self, *, seq_len: int, vocab_lo: int, vocab_hi: int, seed: int):
        super().__init__()
        self.seq_len = seq_len
        self.vocab_lo = vocab_lo
        self.vocab_hi = vocab_hi
        self.seed = seed

    @override
    def __iter__(self) -> Iterator[Tensor]:
        gen = torch.Generator()
        gen.manual_seed(self.seed)
        while True:
            yield torch.randint(
                self.vocab_lo, self.vocab_hi, (self.seq_len,), generator=gen, dtype=torch.long
            )


def _collate_input_ids(batch: list[Tensor]) -> Tensor:
    return torch.stack(batch)


def build_esm2_loader(
    target_cfg: ESM2TargetConfig,
    data_cfg: ESM2DataConfig,
    *,
    split: Literal["train", "eval"],
    device: str,
    batch_size: int,
    dist_state: DistributedState | None = None,
    seed: int | None = None,
) -> DataLoader[Any]:
    """ESM2 `DataLoader` for the requested split.

    Synthetic loader is infinite and shared across DDP ranks (per-rank seed offset).
    Eval seed is offset by 1 to differ from train. `device` is unused — tensors are
    moved by `optimize()` via `move_batch_to_device`.
    """
    del target_cfg, device
    effective_seed = (seed or 0) + (1 if split == "eval" else 0)
    if dist_state is not None:
        effective_seed = effective_seed * 1000 + dist_state.rank

    spec = data_cfg.spec
    match spec:
        case ESM2SyntheticData():
            assert spec.vocab_lo < spec.vocab_hi, (
                f"vocab_lo ({spec.vocab_lo}) must be < vocab_hi ({spec.vocab_hi})"
            )
            dataset = _SyntheticAATokens(
                seq_len=spec.max_seq_len,
                vocab_lo=spec.vocab_lo,
                vocab_hi=spec.vocab_hi,
                seed=effective_seed,
            )
            return DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=_collate_input_ids,
                drop_last=True,
            )
        case ESM2UniRef50Data():
            # TODO: surfaced as abstraction issue — see report.
            raise NotImplementedError(
                "UniRef50 data path not implemented in scaffolding. The existing "
                "`create_lm_data_loader` packs concatenated tokens across documents, "
                "which is wrong for protein sequences (each sequence is one example, "
                "no cross-sequence concatenation). A protein-native loader would need "
                "per-sequence tokenisation, padding, and the model's attention_mask."
            )


def make_run_batch(target_cfg: ESM2TargetConfig) -> RunBatch:
    """`RunBatch` extracting `.logits` from the HF forward output.

    The synthetic loader yields `Tensor[batch, seq]` of token ids; `EsmForMaskedLM`
    accepts that as the first positional arg and returns a `MaskedLMOutput` with a
    `.logits` attribute. `_make_run_batch("logits")` does `getattr(model(batch), "logits")`.

    Note: for real UniRef50 data the batch would need to be `(input_ids, attention_mask)`
    and `RunBatch` would need to unpack and forward both — see report for the
    abstraction-stress angle.
    """
    return _make_run_batch(target_cfg.output_extract)


@dataclass(frozen=True)
class SavedESM2Run:
    """Handle to a completed ESM2 PD run on disk or in W&B."""

    cfg: ESM2ExperimentConfig
    checkpoint_path: Path

    @classmethod
    def from_path(cls, path: ModelPath) -> "SavedESM2Run":
        files = resolve_run_files(
            path, config_filename=RUN_META_FILENAME, checkpoint_prefix="model"
        )
        return cls(
            cfg=ESM2ExperimentConfig.from_file(files.config_path),
            checkpoint_path=files.checkpoint_path,
        )

    def load_model(self) -> ComponentModel:
        return load_component_model(
            pd_config=self.cfg.pd,
            checkpoint_path=self.checkpoint_path,
            target_model=build_target(self.cfg.target),
            run_batch=make_run_batch(self.cfg.target),
        )


def _build_eval_loop(
    cfg: ESM2ExperimentConfig, device: str, dist_state: DistributedState | None
) -> EvalLoop | None:
    if cfg.eval is None:
        return None
    eval_loader = build_esm2_loader(
        cfg.target,
        cfg.data,
        split="eval",
        device=device,
        batch_size=cfg.eval.batch_size,
        dist_state=dist_state,
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


_ESM2_BUNDLE = ExperimentBundle[ESM2ExperimentConfig](
    config_cls=ESM2ExperimentConfig,
    build_target=lambda cfg: build_target(cfg.target),
    build_train_loader=lambda cfg, device, dist_state: build_esm2_loader(
        cfg.target,
        cfg.data,
        split="train",
        device=device,
        batch_size=cfg.pd.batch_size,
        dist_state=dist_state,
        seed=cfg.pd.seed,
    ),
    build_eval_loop=_build_eval_loop,
    make_run_batch=lambda cfg: make_run_batch(cfg.target),
    reconstruction_loss=recon_loss_kl,
)


def main(
    config_path: str | Path,
    *,
    group: str | None = None,
    tags: str | None = None,
) -> None:
    """Run an ESM2 PD experiment end-to-end from a YAML config."""
    run_fresh(_ESM2_BUNDLE, Path(config_path), group=group, tags=tags)


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
