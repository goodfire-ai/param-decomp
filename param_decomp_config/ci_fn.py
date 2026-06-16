"""Causal-importance function configs."""

from typing import Annotated, Literal, Self

from pydantic import BeforeValidator, Field, PositiveInt, model_validator

from param_decomp_config.base import BaseConfig

LayerwiseCiFnType = Literal["mlp", "vector_mlp", "shared_mlp"]


class LayerwiseCiConfig(BaseConfig):
    """Layerwise CI fns — one independent CI fn per decomposition target."""

    mode: Literal["layerwise"] = "layerwise"
    fn_type: LayerwiseCiFnType = Field(
        ..., description="Type of layerwise CI function: mlp, vector_mlp, or shared_mlp"
    )
    hidden_dims: list[PositiveInt] = Field(
        ..., description="Hidden dimensions for the CI function MLP"
    )

    @model_validator(mode="after")
    def validate_hidden_dims(self) -> Self:
        if self.fn_type in ("mlp", "vector_mlp") and not self.hidden_dims:
            raise ValueError(f"hidden_dims must be non-empty for fn_type={self.fn_type!r}")
        return self


class AttnConfig(BaseConfig):
    """Self-attention config for the transformer CI fn. Uses RoPE for length generalization."""

    n_heads: PositiveInt = Field(
        ...,
        description="Number of attention heads. Must divide the input dimension.",
    )
    max_len: PositiveInt = Field(
        default=2048,
        description="Maximum sequence length for RoPE embeddings.",
    )
    rope_base: float = Field(
        default=10000.0,
        description="Base for RoPE frequency computation.",
    )


class GlobalSharedTransformerCiConfig(BaseConfig):
    """Config for the global transformer CI fn.

    `d_model` must be divisible by `attn_config.n_heads` and the resulting per-head dim
    must be even (RoPE). `mlp_hidden_dim` defaults to `[4 * d_model]`.
    """

    d_model: PositiveInt
    n_blocks: PositiveInt
    mlp_hidden_dim: list[PositiveInt] | None = Field(
        default=None,
        description="Hidden dimension for transformer MLP blocks. "
        "If None, defaults to [4 * d_model].",
    )
    attn_config: AttnConfig

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        assert self.d_model % self.attn_config.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by "
            f"attn_config.n_heads ({self.attn_config.n_heads})"
        )
        d_head = self.d_model // self.attn_config.n_heads
        assert d_head % 2 == 0, (
            f"d_head ({d_head}) must be even for RoPE. "
            f"d_model={self.d_model}, "
            f"n_heads={self.attn_config.n_heads}"
        )
        return self


class GlobalSharedMlpCiConfig(BaseConfig):
    """A single global MLP CI fn that maps all layers jointly."""

    mode: Literal["global"] = "global"
    fn_type: Literal["global_shared_mlp"] = "global_shared_mlp"
    hidden_dims: list[PositiveInt] = Field(
        ..., description="Hidden dimensions for the global_shared_mlp CI function."
    )


class GlobalSharedTransformerCiFnConfig(BaseConfig):
    """A single global transformer CI fn that maps all layers jointly."""

    mode: Literal["global"] = "global"
    fn_type: Literal["global_shared_transformer"] = "global_shared_transformer"
    simple_transformer_ci_cfg: GlobalSharedTransformerCiConfig


# Stored global CI configs predate the split into per-fn_type classes: they wrote a
# single class with both `hidden_dims` and `simple_transformer_ci_cfg`, the inactive one
# left as `None`. Drop the inactive null so the now-`extra=forbid` per-fn_type classes
# accept old files. Delete once stored runs are migrated.
_NULLABLE_LEGACY_GLOBAL_CI_KEYS = frozenset({"hidden_dims", "simple_transformer_ci_cfg"})


def _drop_null_inactive_global_ci_field(value: object) -> object:
    if isinstance(value, dict):
        return {
            k: v
            for k, v in value.items()
            if not (k in _NULLABLE_LEGACY_GLOBAL_CI_KEYS and v is None)
        }
    return value


# Discriminated (by `fn_type`) union of the global CI-fn configs. Both share
# `mode="global"`; the `fn_type` literal selects MLP vs transformer.
GlobalCiConfig = Annotated[
    GlobalSharedMlpCiConfig | GlobalSharedTransformerCiFnConfig,
    Field(discriminator="fn_type"),
    BeforeValidator(_drop_null_inactive_global_ci_field),
]


# Nested discriminated union: the outer `mode` literal picks layerwise vs global, and
# the global branch's inner `fn_type` literal selects MLP vs transformer.
CiConfig = Annotated[
    LayerwiseCiConfig | GlobalCiConfig,
    Field(discriminator="mode"),
]
