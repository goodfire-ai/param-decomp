"""The single self-contained pretrain run config (`pretrain.train` reads it directly).

Mirrors the torch `Config` recipe fields (next-token CE, AdamW, cosine LR + warmup, grad
clip) plus the run-instance fields the lab launcher stamps (`run_id`, `data_root`). Data is
the offline pre-tokenized parquet artifact served by `param_decomp.pretrain.batch_data.ShardServer` —
NEVER streamed from HF at run time.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from annotated_types import Ge, Gt, Le
from pydantic import Field, PositiveInt

from param_decomp.core.base_config import BaseConfig
from param_decomp.infra.dataset_store import DatasetRef
from param_decomp.pretrain.models import (
    GPT2SimpleConfig,
    LlamaSimpleConfig,
    LlamaSimpleMLPConfig,
    ModelConfig,
)


class PretrainWandbConfig(BaseConfig):
    project: str
    entity: str | None = None
    group: str | None = None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class PretrainRunPaths:
    """The launcher's stamp, narrowed once: every path the run reads or writes hangs off
    `data_root`, so no downstream site re-asserts that the config was stamped."""

    data_root: Path
    run_id: str

    @property
    def run_dir(self) -> Path:
        return self.data_root / "runs" / self.run_id

    @property
    def compilation_cache_dir(self) -> Path:
        """A SIBLING of `runs/`, never per-run: every run and every rank shares it."""
        return self.data_root / "xla_compilation_cache"


class PretrainConfig(BaseConfig):
    seed: int = 45
    model: Annotated[ModelConfig, Field(discriminator="model_type")]
    data: DatasetRef
    """The shards to train on: a store name, or a tagged ad-hoc dir. Resolved against
    `data_root`; the dataset's own facts ride with its shards as `meta.json`."""

    gpus_per_node: PositiveInt = Field(
        default=8,
        description="GPUs per node — the launcher's node math and the trainer's topology assert.",
    )
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
    data_root: Path | None = None
    """The one root of the run's local world — the dataset store, the runs dir, the
    pretrain cache and the compilation cache all hang under it."""
    wandb: PretrainWandbConfig | None = None

    @property
    def block_size(self) -> int:
        return self.model.block_size

    @property
    def paths(self) -> PretrainRunPaths:
        assert self.data_root is not None and self.run_id is not None, (
            "run identity incomplete: data_root is authored or launcher-stamped,"
            " run_id is minted at entry"
        )
        return PretrainRunPaths(data_root=self.data_root, run_id=self.run_id)


def load_pretrain_config(path: Path) -> PretrainConfig:
    return PretrainConfig.from_file(path)


__all__ = [
    "GPT2SimpleConfig",
    "LlamaSimpleConfig",
    "LlamaSimpleMLPConfig",
    "PretrainConfig",
    "PretrainRunPaths",
    "PretrainWandbConfig",
    "load_pretrain_config",
]
