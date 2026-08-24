"""Runtime objects resolved from an authored LM config."""

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

import jax.numpy as jnp
from jax.typing import DTypeLike

from param_decomp.core.built_run import BuiltRun
from param_decomp.core.components import SiteC
from param_decomp.core.configs import PDConfig, TargetedPDConfig
from param_decomp.vendored_jax.llama import AttentionImplementation

WeightsDtype = Literal["float32", "bfloat16"]


def weights_jnp_dtype(dtype: WeightsDtype) -> DTypeLike:
    """The authored frozen-target dtype as the array dtype the target loaders cast to."""
    match dtype:
        case "float32":
            return jnp.float32
        case "bfloat16":
            return jnp.bfloat16


@dataclass(frozen=True)
class ResolvedLMData:
    """Pre-tokenized parquet shard directories: `dir` trains; `eval_dir` is the held-out
    split the eval pass reads."""

    dir: Path
    eval_dir: Path


@dataclass(frozen=True)
class TargetConfig:
    """An HF GLU-transformer target (`model_name` must be in `HF_MODEL_VARIANTS` —
    Llama-3.1-8B or a registered Qwen3 checkpoint)."""

    model_name: str
    sites: tuple[SiteC, ...]
    """Decomposed sites with per-site C, in canonical order (`canonical_site_cs`)."""
    weights_dtype: WeightsDtype
    """The authored `target.weights_dtype`, carried to the composition root's target load."""
    attention_implementation: AttentionImplementation

    supported_weights_dtypes: ClassVar[frozenset[WeightsDtype]] = frozenset({"bfloat16", "float32"})
    """Frozen-target weight dtypes the loader supports. `HFWeights` casts every tensor on
    read, so the family loaders honour whichever of the two the config names. A config
    requesting a dtype outside this set is refused at convert time — no silent downgrade
    (issue #727)."""


@dataclass(frozen=True)
class LlamaSimpleMLPTargetConfig:
    """The `LlamaSimpleMLP` lab-pretrained target (`param_decomp.targets.llama_simple_mlp`);
    weights from the store entry `pretrain_run_path` resolves to
    (`infra.pretrain_cache.resolved_cache_dir`)."""

    pretrain_run_path: str
    sites: tuple[SiteC, ...]
    """Decomposed sites with per-site C, in canonical order
    (`llama_simple_mlp.canonical_site_cs`)."""
    weights_dtype: WeightsDtype
    """The authored `target.weights_dtype`, carried to the composition root's target load."""
    attention_implementation: AttentionImplementation

    supported_weights_dtypes: ClassVar[frozenset[WeightsDtype]] = frozenset({"bfloat16", "float32"})
    """Frozen-target weight dtypes the loader supports — `_checkpoint_weight_getter` casts
    every safetensor on read. See `TargetConfig.supported_weights_dtypes`."""


AnyLMTargetConfig = TargetConfig | LlamaSimpleMLPTargetConfig
"""The closed set of LM target configs — what every LM `BuiltRun` carries and every LM
consumer (`build_target`, `run_metadata`, the targeted tokenizer route) dispatches on.
Non-LM targets (the toys) satisfy only the core `TargetSites` protocol and never enter
the LM aliases below."""


LMRun = BuiltRun[ResolvedLMData, AnyLMTargetConfig, PDConfig]
LMTargetedRun = BuiltRun[ResolvedLMData, AnyLMTargetConfig, TargetedPDConfig]
LMAnyRun = LMRun | LMTargetedRun
"""The stored-run consumers' view: the closed union of run shapes — consumers read only
the sections the shapes share."""
