"""Carbon PD experiment: decompose `HuggingFaceBio/Carbon-500M` (Llama-style DNA model).

Mirrors `param_decomp_lab.experiments.lm.run` but specialises the target loader for the
HF Carbon family (custom DNA tokenizer requires `trust_remote_code=True`) and adds a
synthetic random-token data path so the smoke test can run without downloading the full
model weights or a genomic dataset.

The reconstruction loss is vanilla KL-on-logits today. A real Carbon training run would
use Factorised Nucleotide Supervision (FNS) — see the comment block at the bottom of
this module for the analysis of whether the `ReconstructionLoss` Protocol can express it.
"""

import importlib
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

from param_decomp.base_config import BaseConfig
from param_decomp.batch_and_loss_fns import RunBatch
from param_decomp.component_model import ComponentModel
from param_decomp.distributed import DistributedState
from param_decomp.optimize import EvalLoop
from param_decomp_lab.batch_and_loss_fns import make_run_batch as _make_run_batch
from param_decomp_lab.batch_and_loss_fns import recon_loss_kl
from param_decomp_lab.component_model_io import load_component_model
from param_decomp_lab.distributed import ensure_cached_and_call, with_distributed_cleanup
from param_decomp_lab.eval_metrics import EVAL_METRIC_CLASSES
from param_decomp_lab.experiments.lm.data import rank_batch_size
from param_decomp_lab.experiments.runner import ExperimentBundle, run_fresh
from param_decomp_lab.experiments.utils import RUN_META_FILENAME, ExperimentConfig
from param_decomp_lab.infra.paths import ModelPath
from param_decomp_lab.infra.run_files import resolve_run_files


def _resolve_class(fqn: str) -> type:
    """Load a class from a fully-qualified name, e.g. `transformers.LlamaForCausalLM`."""
    module_path, _, class_name = fqn.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class CarbonTargetConfig(BaseConfig):
    """Target spec for the Carbon experiment.

    `trust_remote_code` is true by default — Carbon ships a custom `HybridDNATokenizer`
    that requires it. `dtype` controls the loaded model dtype: bf16 keeps the 500M model
    well under typical smoke-test budgets.
    """

    model_class: str = "transformers.LlamaForCausalLM"
    model_name: str = "HuggingFaceBio/Carbon-500M"
    trust_remote_code: bool = True
    dtype: Literal["float32", "bfloat16", "float16"] = "bfloat16"
    output_extract: int | str | None = "logits"


class CarbonDataConfig(BaseConfig):
    """Data spec for the Carbon experiment.

    `kind: "synthetic"` skips HF dataset loading entirely and yields random token ids
    drawn from `[0, vocab_size)`. Useful for scaffolding / smoke tests where no real
    DNA corpus needs to flow through the loop. The HF-dataset path is intentionally
    not implemented yet — surface it when the experiment graduates beyond a smoke test.
    """

    kind: Literal["synthetic"] = "synthetic"
    vocab_size: PositiveInt = Field(default=155776, description="Carbon-500M tokenizer vocab size")
    seq_len: PositiveInt = 128
    n_train: PositiveInt = 1024
    n_eval: PositiveInt = 64


class CarbonExperimentConfig(ExperimentConfig[CarbonTargetConfig, CarbonDataConfig]):
    pass


def _resolve_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def build_target(target_cfg: CarbonTargetConfig) -> nn.Module:
    """Load the Carbon target model in eval mode.

    Goes through `ensure_cached_and_call` so DDP ranks share a single download. The
    `trust_remote_code` flag is required for the Carbon tokenizer/repo; the model itself
    is plain `LlamaForCausalLM`, but `from_pretrained` honours the same flag.
    """
    cls = _resolve_class(target_cfg.model_class)
    target_model = ensure_cached_and_call(
        cls.from_pretrained,
        target_cfg.model_name,
        trust_remote_code=target_cfg.trust_remote_code,
        torch_dtype=_resolve_dtype(target_cfg.dtype),
    )
    target_model.eval()
    return target_model


class _RandomTokenDataset(IterableDataset[Tensor]):
    """Infinite stream of random token-id sequences. Smoke-test stand-in for a DNA corpus."""

    def __init__(self, vocab_size: int, seq_len: int, n_samples: int, seed: int):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_samples = n_samples
        self.seed = seed

    @override
    def __iter__(self) -> Iterator[Tensor]:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed)
        for _ in range(self.n_samples):
            yield torch.randint(0, self.vocab_size, (self.seq_len,), generator=gen)


def build_carbon_loader(
    target_cfg: CarbonTargetConfig,
    data_cfg: CarbonDataConfig,
    *,
    split: Literal["train", "eval"],
    device: str,
    batch_size: int,
    dist_state: DistributedState | None = None,
    seed: int | None = None,
) -> DataLoader[Any]:
    """Carbon `DataLoader` for the requested split.

    Synthetic path: yields `(batch, seq_len)` int64 tensors of random token ids. The
    eval seed is offset by 1 so eval shuffles differently from train when both come
    from the same `pd_config.seed`.
    """
    del target_cfg, device
    effective_seed = (seed or 0) + (1 if split == "eval" else 0)
    n_samples = data_cfg.n_eval if split == "eval" else data_cfg.n_train
    per_rank_bs = rank_batch_size(batch_size, dist_state, label=f"{split}_batch_size")
    assert data_cfg.kind == "synthetic", (
        f"Only synthetic data is implemented for the Carbon experiment, got {data_cfg.kind!r}"
    )
    dataset = _RandomTokenDataset(
        vocab_size=data_cfg.vocab_size,
        seq_len=data_cfg.seq_len,
        n_samples=n_samples,
        seed=effective_seed + (dist_state.rank if dist_state is not None else 0),
    )

    def collate(items: list[Tensor]) -> Tensor:
        return torch.stack(items)

    return DataLoader(
        dataset,
        batch_size=per_rank_bs,
        collate_fn=collate,
        drop_last=True,
    )


def make_run_batch(target_cfg: CarbonTargetConfig) -> RunBatch:
    return _make_run_batch(target_cfg.output_extract)


@dataclass(frozen=True)
class SavedCarbonRun:
    """Handle to a completed Carbon PD run on disk or in W&B."""

    cfg: CarbonExperimentConfig
    checkpoint_path: Path

    @classmethod
    def from_path(cls, path: ModelPath) -> "SavedCarbonRun":
        files = resolve_run_files(
            path, config_filename=RUN_META_FILENAME, checkpoint_prefix="model"
        )
        return cls(
            cfg=CarbonExperimentConfig.from_file(files.config_path),
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
    cfg: CarbonExperimentConfig,
    device: str,
    dist_state: DistributedState | None,
) -> EvalLoop | None:
    if cfg.eval is None:
        return None
    eval_loader = build_carbon_loader(
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


_CARBON_BUNDLE = ExperimentBundle[CarbonExperimentConfig](
    config_cls=CarbonExperimentConfig,
    build_target=lambda cfg: build_target(cfg.target),
    build_train_loader=lambda cfg, device, dist_state: build_carbon_loader(
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
    # TODO: surfaced as abstraction issue — see report. Carbon's FNS recon loss
    # would need a richer Protocol than `(pred, target) -> (sum, n)`.
    reconstruction_loss=recon_loss_kl,
)


@with_distributed_cleanup
def main(
    config_path: str | Path,
    *,
    group: str | None = None,
    tags: str | None = None,
    run_id: str | None = None,
) -> None:
    """Run a Carbon PD experiment end-to-end from a YAML config.

    No SLURM submission path yet — invoke directly (use torchrun for DDP).
    `group` / `tags` are wandb-only (no-ops without `wandb:`).
    """
    run_fresh(_CARBON_BUNDLE, Path(config_path), group=group, tags=tags, run_id=run_id)


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()


# =============================================================================
# FNS RECONSTRUCTION-LOSS ANALYSIS
# =============================================================================
#
# Carbon was originally trained with a two-stage loss: vanilla CE, then "Factorised
# Nucleotide Supervision" (FNS) which decomposes each k-mer token loss into base-pair-
# level CE on the constituent nucleotides. The pieces FNS needs that the current
# `ReconstructionLoss` Protocol does NOT carry:
#
#   1. A vocab-side mapping `kmer_id -> (n1, n2, ..., nk)` indexing the 4 nucleotides.
#      This is a static, model-specific lookup table — not a tensor that flows out of
#      the forward pass.
#   2. A `k` (typically 6) factor that re-shapes per-token loss into per-base loss:
#      `(batch, seq, vocab)` would become `(batch, seq, k, 4)` after applying the
#      mapping and re-aggregating logits to per-base distributions.
#   3. The "loss" is no longer KL between two logit tensors — it is a structured sum
#      of `k` CEs against per-base targets derived from the *target model's* logits
#      (since this is a faithfulness/recon term, not a token-supervision term).
#
# The current Protocol is:
#
#     def __call__(self, pred: Tensor, target: Tensor) -> tuple[Tensor, int]: ...
#
# Where `pred` and `target` are both `(batch, seq, vocab)` and the contract is element-
# wise comparison reduced to `(sum, n_elements)`. There are two FNS-shaped issues:
#
# (A) Extra static config (the kmer→nucleotide map) needs to be bound somewhere. With
#     the current Protocol you'd close over it via `functools.partial` or a closure,
#     which works but is ugly — the kmer→nucleotide map effectively becomes part of
#     the recon-loss identity, but there's no place to store / version / log it.
#
# (B) The mapping changes `n_elements` semantics. With vanilla KL/MSE `n_elements` is
#     `pred.numel() // pred.shape[-1]` (number of positions). With FNS each position
#     contributes `k` base-pair predictions, so `n_elements = positions * k`. The
#     Protocol can express this fine — the second return is just an int — but the
#     reduction semantics are no longer "per-position mean", they're "per-base mean".
#     Callers that interpret the ratio as "loss per token" silently get the wrong unit.
#
# Net verdict: the `(pred, target) -> (sum, n_elements)` shape is technically rich
# enough — you can implement FNS as a closure-bound callable that internally re-shapes
# `pred` / `target` to `(..., k, 4)` and sums CE. **But** the Protocol gives no first-
# class home for the vocab-side static structure (the kmer map), and the int-counter
# return loses the unit. A cleaner extension would be:
#
#     class ReconstructionLoss(Protocol):
#         def __call__(
#             self, pred: Tensor, target: Tensor,
#         ) -> tuple[Tensor, int]: ...
#         # plus optional: aux_state attached to the callable (model-specific lookup
#         # tables, k, etc.) and a name/unit tag so the caller can label log keys.
#
# For the smoke test we use vanilla KL — see `recon_loss_kl` above.
