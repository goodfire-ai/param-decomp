"""Eval-metric configs and the `AnyEvalMetricConfig` discriminated union.

Metric impls live in `param_decomp_lab/eval_metrics/`; YAML `eval.metrics` entries are
validated against `AnyEvalMetricConfig` and dispatched to the matching `Metric`
subclass via `EVAL_METRIC_CLASSES` (lab-side).
"""

from typing import Annotated, Literal, Self

from pydantic import Discriminator, Field, model_validator

from param_decomp_config.autointerp import LLMConfig, StrategyConfig
from param_decomp_config.base import BaseConfig
from param_decomp_config.losses import (
    ImportanceMinimalityLossConfig,
    PGDReconLossConfig,
    SmoothL0ImportanceMinimalityLossConfig,
    StochasticHiddenActsReconLossConfig,
)


class AutointerpLabelsConfig(BaseConfig):
    type: Literal["AutointerpLabels"] = "AutointerpLabels"
    k: int
    """Number of components to sample uniformly over the concatenated component space."""
    seed: int
    activation_threshold: float
    max_examples: int
    """Reservoir capacity (activation examples kept per sampled component)."""
    context_tokens_per_side: int
    llm: LLMConfig
    template_strategy: Annotated[StrategyConfig, Field(discriminator="type")]
    # Run/data facts the prompt needs that a bare ComponentModel doesn't carry. They
    # mirror `data.*` / are the eval data the metric renders — kept here so the metric
    # is self-contained (plain config dispatch). `n_blocks` / `layer_descriptions` are
    # derived from the model at `bind`.
    dataset_name: str
    seq_len: int
    tokenizer_name: str


class CEandKLLossesConfig(BaseConfig):
    """`rounding_threshold` binarises CI for the `*_rounded_masked` variant (`ci > threshold`)."""

    type: Literal["CEandKLLosses"] = "CEandKLLosses"
    rounding_threshold: float


class CIHiddenActsReconLossConfig(BaseConfig):
    type: Literal["CIHiddenActsReconLoss"] = "CIHiddenActsReconLoss"


class CIHistogramsConfig(BaseConfig):
    """`n_batches_accum=None` accumulates every batch in the eval pass."""

    type: Literal["CIHistograms"] = "CIHistograms"
    n_batches_accum: int | None


class CI_L0Config(BaseConfig):
    """`groups` maps `{group_name: [fnmatch-style layer pattern, ...]}`.

    Matching layers' L0s are summed into the group and logged under the group's name.
    """

    type: Literal["CI_L0"] = "CI_L0"
    groups: dict[str, list[str]] | None
    ci_alive_threshold: float = 0.0


class _AttnPatternsBaseConfig(BaseConfig):
    """Shared config for attention-pattern recon metrics.

    Supports standard attention and RoPE (auto-detected from the parent attention
    module). ALiBi / QK-norm / sliding window are not supported.

    Either `(q_proj_path, k_proj_path)` or `c_attn_path` must be set (combined QKV with
    output split as `[Q | K | V]` along the last dim) — not both, not neither.
    """

    n_heads: int
    q_proj_path: str | None = None
    k_proj_path: str | None = None
    c_attn_path: str | None = None

    @model_validator(mode="after")
    def _validate_paths(self) -> Self:
        has_separate = self.q_proj_path is not None and self.k_proj_path is not None
        has_combined = self.c_attn_path is not None
        assert has_separate != has_combined, (
            "Specify either (q_proj_path, k_proj_path) or c_attn_path, not both/neither"
        )
        return self


class CIMaskedAttnPatternsReconLossConfig(_AttnPatternsBaseConfig):
    type: Literal["CIMaskedAttnPatternsReconLoss"] = "CIMaskedAttnPatternsReconLoss"


class StochasticAttnPatternsReconLossConfig(_AttnPatternsBaseConfig):
    type: Literal["StochasticAttnPatternsReconLoss"] = "StochasticAttnPatternsReconLoss"


class CIMeanPerComponentConfig(BaseConfig):
    type: Literal["CIMeanPerComponent"] = "CIMeanPerComponent"


class ComponentActivationDensityConfig(BaseConfig):
    type: Literal["ComponentActivationDensity"] = "ComponentActivationDensity"
    ci_alive_threshold: float = 0.0


class IdentityCIErrorConfig(BaseConfig):
    """`identity_ci` / `dense_ci` list layers expected to produce Identity / Dense patterns."""

    type: Literal["IdentityCIError"] = "IdentityCIError"
    identity_ci: list[dict[str, str | int]] | None
    dense_ci: list[dict[str, str | int]] | None


class PermutedCIPlotsConfig(BaseConfig):
    """fnmatch patterns for layers permuted to align with the corresponding target solution.

    `identity_patterns` and `dense_patterns` are matched separately against the model.
    """

    type: Literal["PermutedCIPlots"] = "PermutedCIPlots"
    identity_patterns: list[str] | None
    dense_patterns: list[str] | None


class UVPlotsConfig(BaseConfig):
    """fnmatch patterns for layers permuted to align with the corresponding target solution.

    `identity_patterns` and `dense_patterns` are matched separately against the model.
    """

    type: Literal["UVPlots"] = "UVPlots"
    identity_patterns: list[str] | None
    dense_patterns: list[str] | None


AnyEvalMetricConfig = Annotated[
    AutointerpLabelsConfig
    | CEandKLLossesConfig
    | CIHiddenActsReconLossConfig
    | CIHistogramsConfig
    | CI_L0Config
    | CIMaskedAttnPatternsReconLossConfig
    | CIMeanPerComponentConfig
    | ComponentActivationDensityConfig
    | IdentityCIErrorConfig
    | ImportanceMinimalityLossConfig
    | PermutedCIPlotsConfig
    | PGDReconLossConfig
    | SmoothL0ImportanceMinimalityLossConfig
    | StochasticAttnPatternsReconLossConfig
    | StochasticHiddenActsReconLossConfig
    | UVPlotsConfig,
    Discriminator("type"),
]
