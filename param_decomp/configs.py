"""Config classes of various types."""

import importlib
from typing import Annotated, Any, Literal, Self

from pydantic import (
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    SerializeAsAny,
    field_validator,
    model_validator,
)

from param_decomp.base_config import BaseConfig
from param_decomp.metrics.base import LossMetricConfig, MetricConfig
from param_decomp.metrics.registry import METRIC_REGISTRY
from param_decomp.types import (
    GlobalCiFnType,
    LayerwiseCiFnType,
    Probability,
)


class LayerwiseCiConfig(BaseConfig):
    """Configuration for layerwise CI functions (one per layer)."""

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
    """Configuration for self-attention.

    Uses RoPE (Rotary Position Embeddings) for sequence length generalization.
    """

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


class GlobalCiConfig(BaseConfig):
    """Configuration for global CI function (single function for all layers)."""

    mode: Literal["global"] = "global"
    fn_type: GlobalCiFnType = Field(
        ...,
        description="Type of global CI function: global_shared_mlp or global_shared_transformer",
    )
    hidden_dims: list[PositiveInt] | None = Field(
        default=None,
        description="Hidden dimensions for global_shared_mlp CI function.",
    )
    simple_transformer_ci_cfg: GlobalSharedTransformerCiConfig | None = None

    @model_validator(mode="after")
    def validate_ci_config(self) -> Self:
        if self.fn_type == "global_shared_mlp":
            assert self.hidden_dims is not None, (
                "hidden_dims must be specified when fn_type='global_shared_mlp'"
            )
        elif self.fn_type == "global_shared_transformer":
            assert self.simple_transformer_ci_cfg is not None, (
                "simple_transformer_ci_cfg must be specified when fn_type='global_shared_transformer'"
            )
            assert self.hidden_dims is None, (
                "hidden_dims is only used for fn_type='global_shared_mlp'"
            )
        return self


CiConfig = LayerwiseCiConfig | GlobalCiConfig


class ScheduleConfig(BaseConfig):
    """Configuration for a schedule with warmup and decay."""

    start_val: PositiveFloat = Field(..., description="Starting/peak value (after warmup)")
    warmup_pct: Probability = Field(
        default=0.0, description="Fraction of total steps for linear warmup"
    )
    final_val_frac: NonNegativeFloat = Field(
        default=1.0,
        description="End value as fraction of start_val.",
    )
    fn_type: Literal["constant", "cosine", "linear"] = Field(
        default="constant", description="Decay function type after warmup"
    )

    @model_validator(mode="after")
    def validate_constant_schedule(self) -> Self:
        if self.fn_type == "constant" and self.final_val_frac != 1.0:
            raise ValueError("constant schedule requires final_val_frac == 1.0")
        return self


class OptimizerConfig(BaseConfig):
    """Configuration for one AdamW optimizer."""

    lr_schedule: ScheduleConfig = Field(..., description="Learning rate schedule")
    weight_decay: NonNegativeFloat = Field(default=0.0, description="AdamW weight decay")
    betas: tuple[Probability, Probability] = Field(
        default=(0.9, 0.999), description="AdamW (beta1, beta2)"
    )
    grad_clip_norm: PositiveFloat | None = Field(
        default=None,
        description="If set, clip the grad norm of this group's parameters to this value",
    )


class ModulePatternInfoConfig(BaseConfig):
    """Configuration for a module pattern with its number of components."""

    module_pattern: str = Field(..., description="fnmatch-style pattern to match module names")
    C: PositiveInt = Field(
        ..., description="Number of components for modules matching this pattern"
    )


# --- Subset routing (used by several metric configs) ---


class UniformKSubsetRoutingConfig(BaseConfig):
    type: Literal["uniform_k_subset"] = "uniform_k_subset"


class StaticProbabilityRoutingConfig(BaseConfig):
    type: Literal["static_probability"] = "static_probability"
    p: Probability


SubsetRoutingType = UniformKSubsetRoutingConfig | StaticProbabilityRoutingConfig


# --- Persistent PGD shared types (also imported by `metrics.persistent_pgd_recon`) ---


class SignPGDConfig(BaseConfig):
    type: Literal["sign"] = "sign"
    lr_schedule: ScheduleConfig


class AdamPGDConfig(BaseConfig):
    type: Literal["adam"] = "adam"
    beta1: Probability = Field(default=0.9, description="Adam beta1 for masks")
    beta2: Probability = Field(default=0.999, description="Adam beta2 for masks")
    eps: NonNegativeFloat = Field(default=1e-8, description="Adam epsilon for masks")
    lr_schedule: ScheduleConfig


PGDOptimizerConfig = SignPGDConfig | AdamPGDConfig


class SingleSourceScope(BaseConfig):
    type: Literal["single_source"] = "single_source"


class BroadcastAcrossBatchScope(BaseConfig):
    type: Literal["broadcast_across_batch"] = "broadcast_across_batch"


class RepeatAcrossBatchScope(BaseConfig):
    type: Literal["repeat_across_batch"] = "repeat_across_batch"
    n_sources: PositiveInt


class PerBatchPerPositionScope(BaseConfig):
    type: Literal["per_batch_per_position"] = "per_batch_per_position"


PersistentPGDSourceScope = Annotated[
    SingleSourceScope
    | BroadcastAcrossBatchScope
    | RepeatAcrossBatchScope
    | PerBatchPerPositionScope,
    Field(discriminator="type"),
]


class _PersistentPGDBaseConfig(LossMetricConfig):
    """Shared fields for persistent PGD configs."""

    optimizer: Annotated[PGDOptimizerConfig, Field(discriminator="type")]
    scope: PersistentPGDSourceScope
    use_sigmoid_parameterization: bool = False
    n_warmup_steps: NonNegativeInt = Field(
        default=0,
        description=(
            "Extra inner PGD source-optimization steps on each train batch before the final loss"
            " computation."
        ),
    )
    start_frac: Probability = 0.0
    n_samples: PositiveInt = 1


class PersistentPGDReconLossConfig(_PersistentPGDBaseConfig):
    pass


class PersistentPGDReconSubsetLossConfig(_PersistentPGDBaseConfig):
    routing: Annotated[
        SubsetRoutingType, Field(discriminator="type", default=UniformKSubsetRoutingConfig())
    ]


SamplingType = Literal["continuous", "binomial"]


# --- Metric resolution -------------------------------------------------------------


def _parse_metric_cfg(metric_name: str, raw: Any, *, train_loss: bool) -> MetricConfig:
    """Look up the metric by class name in METRIC_REGISTRY and validate its config."""
    assert metric_name in METRIC_REGISTRY, (
        f"unknown metric {metric_name!r} (registered: {sorted(METRIC_REGISTRY)})"
    )
    metric_cls = METRIC_REGISTRY[metric_name]
    cfg = raw if isinstance(raw, MetricConfig) else metric_cls.config_type.model_validate(raw or {})

    if train_loss:
        assert isinstance(cfg, LossMetricConfig), (
            f"{metric_name!r} is eval-only; move it under eval_metrics"
        )
        assert cfg.coeff is not None, f"loss_metrics.{metric_name!r} must set `coeff`"
    return cfg


class RuntimeConfig(BaseConfig):
    """Compute substrate the algorithm runs on.

    The three configs form a determinism ladder:

    1. Same ``PDConfig`` + same ``RuntimeConfig`` → bit-identical trained weights.
    2. Same ``PDConfig``, different ``RuntimeConfig`` → same algorithm, weights differ
       only via numerical effects (precision, device).
    3. Same ``PDConfig`` + same ``RuntimeConfig``, different ``LoggingConfig`` →
       bit-identical weights; only what was observed differs.

    ``RuntimeConfig`` is class 2: device placement, precision, parallelism degree —
    things that perturb numerics without changing the algorithm. Future home for
    NCCL flags, gradient accumulation steps, fp8 variants, etc.
    """

    autocast_bf16: bool = Field(
        default=True,
        description="Use torch.autocast with bfloat16 mixed precision in training and eval.",
    )
    device: Literal["cuda", "cpu"] = Field(
        default="cuda",
        description="Device to run on. Overridable ad-hoc with ``pd-run --device cpu``.",
    )
    dp: PositiveInt | None = Field(
        default=None,
        description="Number of GPUs for data parallelism. None = single GPU/CPU. Bounded by "
        "the cluster's GPUs-per-node for single-node DDP; multiples of that for multi-node. "
        "Declares the experiment's compute requirement; overridable ad-hoc by ``pd-run --dp N``.",
    )

    @model_validator(mode="after")
    def validate_device_dp(self) -> Self:
        from param_decomp.settings import GPUS_PER_NODE

        if self.dp is not None:
            assert self.device == "cuda", "dp requires device='cuda'"
            assert self.dp >= 2, "if set, dp must be at least 2 (pass None for single device)."
            assert self.dp <= GPUS_PER_NODE or self.dp % GPUS_PER_NODE == 0, (
                f"dp must be <= {GPUS_PER_NODE} (single node) or divisible by {GPUS_PER_NODE} "
                f"(multi-node), got {self.dp}"
            )
        return self


class LoggingConfig(BaseConfig):
    """Observation-only settings: cadence + eval-only metrics + display thresholds.

    Determinism class 3 in the PDConfig/RuntimeConfig/LoggingConfig ladder: fields
    here never touch the optimizer. Two runs with identical ``PDConfig`` +
    ``RuntimeConfig`` and different ``LoggingConfig`` produce bit-identical weights —
    only what you observed about the run differs.
    """

    train_log_freq: PositiveInt = Field(
        ...,
        description="Interval (in steps) at which to log training metrics",
    )
    eval_freq: PositiveInt = Field(
        ...,
        description="Interval (in steps) at which to log evaluation metrics",
    )
    eval_batch_size: PositiveInt = Field(
        ...,
        description="Batch size used for evaluation.",
    )
    slow_eval_freq: PositiveInt = Field(
        ...,
        description="Interval (in steps) at which to run slow evaluation metrics. "
        "Must be a multiple of `eval_freq`.",
    )
    n_eval_steps: PositiveInt = Field(
        ...,
        description="Number of steps to run evaluation for",
    )
    slow_eval_on_first_step: bool = Field(
        default=True,
        description="Whether to run slow evaluation on the first step",
    )
    save_freq: PositiveInt | None = Field(
        default=None,
        description="Interval (in steps) at which to save model checkpoints (None disables saving "
        "until the end of training).",
    )
    eval_metrics: dict[str, SerializeAsAny[MetricConfig]] = Field(
        default_factory=dict,
        description=(
            "Eval-only metrics keyed by metric class name. Metrics already set in"
            " `pd.loss_metrics` are evaluated automatically and should not be repeated here."
        ),
    )
    # wandb_run_name: str | None = Field(
    #     default=None,
    #     description="W&B run display name. None lets W&B auto-name.",
    # )

    @model_validator(mode="before")
    @classmethod
    def _discover_builtin_metrics(cls, data: Any) -> Any:
        """Ensure built-in `@register_metric` decorators have fired before `eval_metrics`
        looks names up in `METRIC_REGISTRY`. External metric modules are imported by
        `PDConfig._import_metric_modules`; rely on field ordering on the parent
        parent `Run` (pd validated before logging) for those to be visible here.
        """
        from param_decomp.metrics import discover_metrics

        discover_metrics()
        return data

    @field_validator("eval_metrics", mode="before")
    @classmethod
    def _parse_eval_metrics(cls, v: Any) -> dict[str, MetricConfig]:
        if v is None:
            return {}
        return {
            metric_name: _parse_metric_cfg(metric_name, raw, train_loss=False)
            for metric_name, raw in v.items()
        }

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        assert self.slow_eval_freq % self.eval_freq == 0, (
            "slow_eval_freq must be a multiple of eval_freq"
        )
        assert self.slow_eval_freq // self.eval_freq >= 1, (
            "slow_eval_freq must be at least eval_freq"
        )
        return self


class PDConfig(BaseConfig):
    """Algorithm specification.

    Determinism class 1 in the PDConfig/RuntimeConfig/LoggingConfig ladder: these are
    the fields that determine the trained weights given a fixed substrate. Two runs
    with identical ``PDConfig`` and identical ``RuntimeConfig`` produce bit-identical
    weights; flipping any field here changes what algorithm runs.
    """

    # --- General ---
    seed: int = Field(
        default=0,
        description="Random seed for reproducibility, including LM dataset shuffling.",
    )
    n_mask_samples: PositiveInt = Field(
        ...,
        description="Number of stochastic masks to sample when using stochastic recon losses",
    )
    ci_config: CiConfig = Field(
        ...,
        discriminator="mode",
        description="Configuration for the causal importance function.",
    )
    sampling: SamplingType = Field(
        default="continuous",
        description="Sampling mode for stochastic elements: 'continuous' (default) or 'binomial'",
    )
    sigmoid_type: Literal["normal", "hard", "leaky_hard", "upper_leaky_hard", "swish_hard"] = Field(
        default="leaky_hard",
        description="Type of sigmoid to use for causal importance calculation",
    )
    module_info: list[ModulePatternInfoConfig] = Field(
        ...,
        description="List of module patterns with C values specifying which modules to decompose.",
    )
    identity_module_info: list[ModulePatternInfoConfig] | None = Field(
        default=None,
        description="List of identity module patterns with C values.",
    )

    @property
    def all_module_info(self) -> list[ModulePatternInfoConfig]:
        result = list(self.module_info)
        if self.identity_module_info is not None:
            for info in self.identity_module_info:
                result.append(
                    ModulePatternInfoConfig(
                        module_pattern=f"{info.module_pattern}.pre_identity", C=info.C
                    )
                )
        return result

    use_delta_component: bool = Field(
        default=True,
        description="If True, use an extra component containing the difference between the target "
        "model and component weights.",
    )

    metric_modules: list[str] = Field(
        default_factory=list,
        description=(
            "Extra Python modules to import before validating `loss_metrics` / `eval_metrics`."
            " Each entry is a dotted module name (`my_pkg.my_metrics`) importable from the"
            " current environment. Imported side-effects (`@register_metric` decorators) expand"
            " `METRIC_REGISTRY` so user-defined metrics can be referenced by class name."
        ),
    )

    loss_metrics: dict[str, SerializeAsAny[LossMetricConfig]] = Field(
        default_factory=dict,
        description=(
            "Training-loss metrics keyed by metric class name. Each value's `coeff` weights the"
            " metric in the total training loss. Active loss metrics are automatically also"
            " evaluated."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _import_metric_modules(cls, data: Any) -> Any:
        """Import metric modules so their `@register_metric` decorators fire
        before the `loss_metrics` field validator looks names up in `METRIC_REGISTRY`.
        Idempotent: re-validation in the same process is a no-op. External-metric
        visibility on the sibling `LoggingConfig.eval_metrics` relies on field ordering
        in the parent `Run` (pd validated before logging).
        """
        from param_decomp.metrics import discover_metrics

        discover_metrics()
        if isinstance(data, dict):
            for spec in data.get("metric_modules", []) or []:
                importlib.import_module(spec)
        return data

    @field_validator("loss_metrics", mode="before")
    @classmethod
    def _parse_loss_metrics(cls, v: Any) -> dict[str, MetricConfig]:
        if v is None:
            return {}
        return {
            metric_name: _parse_metric_cfg(metric_name, raw, train_loss=True)
            for metric_name, raw in v.items()
        }

    # --- Training ---
    components_optimizer: OptimizerConfig = Field(
        ..., description="Optimizer config for the component (LinearComponent etc.) parameters"
    )
    ci_fn_optimizer: OptimizerConfig = Field(
        ..., description="Optimizer config for the CI function parameters"
    )
    steps: NonNegativeInt = Field(..., description="Total number of optimisation steps")
    batch_size: PositiveInt = Field(
        ...,
        description="Total batch size (may be divided across multiple devices).",
    )

    # --- Faithfulness Warmup ---
    faithfulness_warmup_steps: NonNegativeInt = Field(
        default=0,
        description="Number of warmup steps to optimize faithfulness loss before main training",
    )
    faithfulness_warmup_lr: PositiveFloat = Field(
        default=0.001,
        description="Learning rate for warmup phase (optimizing faithfulness loss only)",
    )
    faithfulness_warmup_weight_decay: NonNegativeFloat = Field(
        default=0.0,
        description="Weight decay for warmup phase optimizer",
    )

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        for metric_name, cfg in self.loss_metrics.items():
            assert cfg.coeff is not None, f"loss_metrics.{metric_name!r} must have a coeff"
        return self
