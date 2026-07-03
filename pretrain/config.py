"""The single self-contained pretrain run config (`pretrain.train` reads it directly).

Mirrors the torch `Config` recipe fields (next-token CE, AdamW, cosine LR + warmup, grad
clip) plus the run-instance fields the lab launcher stamps (`run_id`, `out_dir`). Data is
the offline pre-tokenized parquet artifact served by `param_decomp.data.ShardServer` —
NEVER streamed from HF at run time.
"""

from pathlib import Path
from typing import Annotated, Literal

from annotated_types import Ge, Gt, Le
from pydantic import Field, PositiveInt

from param_decomp.base_config import BaseConfig
from pretrain.models import (
    GPT2SimpleConfig,
    LlamaSimpleConfig,
    LlamaSimpleMLPConfig,
    ModelConfig,
)


class PretrainDataConfig(BaseConfig):
    dir: Path
    """Directory of `shard_*.parquet` int32 token shards (the prestage tool's output)."""
    tokenizer_name: str
    """HF tokenizer id, recorded in the cache for downstream display (not used at train time)."""


class PretrainWandbConfig(BaseConfig):
    project: str
    entity: str | None = None
    group: str | None = None
    tags: tuple[str, ...] = ()


class PretrainConfig(BaseConfig):
    seed: int = 45
    model: Annotated[ModelConfig, Field(discriminator="model_type")]
    data: PretrainDataConfig

    dp: PositiveInt | None = Field(
        default=None,
        description=(
            "Distributed world size — the number of data-parallel workers (nodes × 8). "
            "None means a single device (CPU / 1-GPU smoke, no jax.distributed). The "
            "single source of truth for distributedness; never inferred from SLURM env."
        ),
    )
    global_batch: Annotated[int, Gt(0)]
    num_iterations: Annotated[int, Gt(0)]
    learning_rate: Annotated[float, Gt(0)]
    warmup_iters: Annotated[int, Ge(0)]
    learning_rate_decay_frac: Annotated[float, Ge(0), Le(1)]
    """Fraction of the peak LR to decay to (0 → 0, 1 → no decay)."""
    weight_decay: Annotated[float, Ge(0)]
    grad_clip: Annotated[float, Gt(0)] | None
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    dtype: Literal["float32", "bfloat16"]
    """Compute dtype for the forward/backward. Masters are always fp32."""

    log_every: Annotated[int, Gt(0)] = 100
    val_every: Annotated[int, Gt(0)] = 1000
    val_steps: Annotated[int, Gt(0)] = 20
    save_every: Annotated[int, Gt(0)] = 1000
    keep_last: Annotated[int, Gt(0)] = 2

    run_id: str | None = None
    run_name: str
    out_dir: Path | None = None
    wandb: PretrainWandbConfig | None = None

    @property
    def block_size(self) -> int:
        return self.model.block_size

    @property
    def run_dir(self) -> Path:
        assert self.out_dir is not None and self.run_id is not None, (
            "out_dir / run_id are minted by the launcher; absent in a hand-authored config"
        )
        return self.out_dir / self.run_id


def load_pretrain_config(path: Path) -> PretrainConfig:
    return PretrainConfig.from_file(path)


__all__ = [
    "GPT2SimpleConfig",
    "LlamaSimpleConfig",
    "LlamaSimpleMLPConfig",
    "PretrainConfig",
    "PretrainDataConfig",
    "PretrainWandbConfig",
    "load_pretrain_config",
]
