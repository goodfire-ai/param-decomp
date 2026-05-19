"""Config classes of various types"""

from typing import Annotated, Any, Literal, Self

from pydantic import (
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from param_decomp.base_config import BaseConfig
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
    """Configuration for global CI function (single function for all layers).

    For fn_type='global_shared_mlp': Concatenates all activations, processes through MLP.
    For fn_type='global_shared_transformer': Concatenates activations, projects to shared d_model,
    and applies transformer blocks over the sequence dimension.
    """

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
    """Configuration for a schedule with warmup and decay. Can be used for LR or other values."""

    start_val: PositiveFloat = Field(..., description="Starting/peak value (after warmup)")
    warmup_pct: Probability = Field(
        default=0.0, description="Fraction of total steps for linear warmup"
    )
    final_val_frac: NonNegativeFloat = Field(
        default=1.0,
        description="End value as fraction of start_val. Can be <1 (decay), =1 (no decay), or >1 (increase)",
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
    """Configuration for a module pattern with its number of components.

    Used in config files to specify which modules to decompose and how many
    components (C) to use for each module matching the pattern.
    """

    module_pattern: str = Field(..., description="fnmatch-style pattern to match module names")
    C: PositiveInt = Field(
        ..., description="Number of components for modules matching this pattern"
    )


#### Metrics that can be used as losses in training or eval ####
class LossMetricConfig(BaseConfig):
    coeff: float | None = Field(
        default=None,
        description="Loss coefficient. Required when set under `loss_metrics`; ignored under `eval_metrics`.",
    )


class FaithfulnessLossConfig(LossMetricConfig):
    classname: Literal["FaithfulnessLoss"] = "FaithfulnessLoss"


class ImportanceMinimalityLossConfig(LossMetricConfig):
    classname: Literal["ImportanceMinimalityLoss"] = "ImportanceMinimalityLoss"
    pnorm: NonNegativeFloat
    beta: NonNegativeFloat
    p_anneal_start_frac: Probability = 1.0
    p_anneal_final_p: NonNegativeFloat | None = None
    p_anneal_end_frac: Probability = 1.0
    eps: NonNegativeFloat = 1e-12


class UniformKSubsetRoutingConfig(BaseConfig):
    type: Literal["uniform_k_subset"] = "uniform_k_subset"


class StaticProbabilityRoutingConfig(BaseConfig):
    type: Literal["static_probability"] = "static_probability"
    p: Probability


SubsetRoutingType = UniformKSubsetRoutingConfig | StaticProbabilityRoutingConfig


class CIMaskedReconSubsetLossConfig(LossMetricConfig):
    classname: Literal["CIMaskedReconSubsetLoss"] = "CIMaskedReconSubsetLoss"
    routing: Annotated[
        SubsetRoutingType, Field(discriminator="type", default=UniformKSubsetRoutingConfig())
    ]


class CIMaskedReconLayerwiseLossConfig(LossMetricConfig):
    classname: Literal["CIMaskedReconLayerwiseLoss"] = "CIMaskedReconLayerwiseLoss"


class CIMaskedReconLossConfig(LossMetricConfig):
    classname: Literal["CIMaskedReconLoss"] = "CIMaskedReconLoss"


class StochasticReconLossConfig(LossMetricConfig):
    classname: Literal["StochasticReconLoss"] = "StochasticReconLoss"


class StochasticReconSubsetLossConfig(LossMetricConfig):
    classname: Literal["StochasticReconSubsetLoss"] = "StochasticReconSubsetLoss"
    routing: Annotated[
        SubsetRoutingType, Field(discriminator="type", default=UniformKSubsetRoutingConfig())
    ]


class StochasticReconLayerwiseLossConfig(LossMetricConfig):
    classname: Literal["StochasticReconLayerwiseLoss"] = "StochasticReconLayerwiseLoss"


class UnmaskedReconLossConfig(LossMetricConfig):
    classname: Literal["UnmaskedReconLoss"] = "UnmaskedReconLoss"


PGDInitStrategy = Literal["random", "ones", "zeroes"]

MaskScope = Literal["unique_per_datapoint", "shared_across_batch"]


class PGDConfig(LossMetricConfig):
    init: PGDInitStrategy
    step_size: float
    n_steps: int
    mask_scope: MaskScope


class PGDReconLossConfig(PGDConfig):
    classname: Literal["PGDReconLoss"] = "PGDReconLoss"


class PGDReconSubsetLossConfig(PGDConfig):
    classname: Literal["PGDReconSubsetLoss"] = "PGDReconSubsetLoss"
    routing: Annotated[
        SubsetRoutingType, Field(discriminator="type", default=UniformKSubsetRoutingConfig())
    ]


class PGDReconLayerwiseLossConfig(PGDConfig):
    classname: Literal["PGDReconLayerwiseLoss"] = "PGDReconLayerwiseLoss"


class PGDMultiBatchConfig(LossMetricConfig):
    init: PGDInitStrategy
    step_size: float
    n_steps: int
    gradient_accumulation_steps: int


class PGDMultiBatchReconLossConfig(PGDMultiBatchConfig):
    classname: Literal["PGDMultiBatchReconLoss"] = "PGDMultiBatchReconLoss"


class PGDMultiBatchReconSubsetLossConfig(PGDMultiBatchConfig):
    classname: Literal["PGDMultiBatchReconSubsetLoss"] = "PGDMultiBatchReconSubsetLoss"
    routing: Annotated[
        SubsetRoutingType, Field(discriminator="type", default=UniformKSubsetRoutingConfig())
    ]


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
    """Sources of shape (N, S, C) where N divides both batch_size and eval_batch_size.

    Repeated along batch dim at forward time: (N, S, C) -> (B, S, C).
    """

    type: Literal["repeat_across_batch"] = "repeat_across_batch"
    n_sources: PositiveInt


class PerBatchPerPositionScope(BaseConfig):
    """Sources of shape (B, S, C) — one source per batch element per position, separate across
    ranks.

    Unlike other scopes, gradients are NOT all-reduced across ranks, so each rank
    maintains fully independent sources for its own batch elements.
    """

    type: Literal["per_batch_per_position"] = "per_batch_per_position"


PersistentPGDSourceScope = Annotated[
    SingleSourceScope
    | BroadcastAcrossBatchScope
    | RepeatAcrossBatchScope
    | PerBatchPerPositionScope,
    Field(discriminator="type"),
]


class _PersistentPGDBaseConfig(LossMetricConfig):
    """Shared fields for persistent PGD configs.

    Persistent PGD maintains persistent masks that receive one gradient update per training step,
    amortizing PGD optimization across training.
    """

    optimizer: Annotated[PGDOptimizerConfig, Field(discriminator="type")]
    scope: PersistentPGDSourceScope
    use_sigmoid_parameterization: bool = False
    n_warmup_steps: Annotated[
        NonNegativeInt,
        Field(
            description="Number of additional inner PGD source-optimization steps to run on each "
            "batch before the final loss computation. Each training step always performs one PPGD "
            "source update (grad + step) as part of the outer loop; these warmup steps add extra "
            "source refinement iterations on the same batch in an inner loop beforehand."
        ),
    ] = 0
    start_frac: Probability = 0.0
    n_samples: PositiveInt = 1


class PersistentPGDReconLossConfig(_PersistentPGDBaseConfig):
    classname: Literal["PersistentPGDReconLoss"] = "PersistentPGDReconLoss"


class PersistentPGDReconSubsetLossConfig(_PersistentPGDBaseConfig):
    classname: Literal["PersistentPGDReconSubsetLoss"] = "PersistentPGDReconSubsetLoss"
    routing: Annotated[
        SubsetRoutingType, Field(discriminator="type", default=UniformKSubsetRoutingConfig())
    ]


class StochasticHiddenActsReconLossConfig(LossMetricConfig):
    classname: Literal["StochasticHiddenActsReconLoss"] = "StochasticHiddenActsReconLoss"


class CIHiddenActsReconLossConfig(BaseConfig):
    classname: Literal["CIHiddenActsReconLoss"] = "CIHiddenActsReconLoss"


class PersistentPGDReconEvalConfig(BaseConfig):
    classname: Literal["PersistentPGDReconEval"] = "PersistentPGDReconEval"


class PersistentPGDReconSubsetEvalConfig(BaseConfig):
    classname: Literal["PersistentPGDReconSubsetEval"] = "PersistentPGDReconSubsetEval"


class _AttnPatternsReconLossBaseConfig(BaseConfig):
    """Attention pattern reconstruction loss config.

    Supports standard attention and RoPE attention (auto-detected from the parent attention
    module). Models using ALiBi, QK-norm, sliding window, etc. are not supported.
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


class CIMaskedAttnPatternsReconLossConfig(_AttnPatternsReconLossBaseConfig):
    classname: Literal["CIMaskedAttnPatternsReconLoss"] = "CIMaskedAttnPatternsReconLoss"


class StochasticAttnPatternsReconLossConfig(_AttnPatternsReconLossBaseConfig):
    classname: Literal["StochasticAttnPatternsReconLoss"] = "StochasticAttnPatternsReconLoss"


#### Metrics that can only be used in eval ####
class CEandKLLossesConfig(BaseConfig):
    classname: Literal["CEandKLLosses"] = "CEandKLLosses"
    rounding_threshold: float


class CIHistogramsConfig(BaseConfig):
    classname: Literal["CIHistograms"] = "CIHistograms"
    n_batches_accum: int | None


class CI_L0Config(BaseConfig):
    classname: Literal["CI_L0"] = "CI_L0"
    groups: dict[str, list[str]] | None


class CIMeanPerComponentConfig(BaseConfig):
    classname: Literal["CIMeanPerComponent"] = "CIMeanPerComponent"


class ComponentActivationDensityConfig(BaseConfig):
    classname: Literal["ComponentActivationDensity"] = "ComponentActivationDensity"


class IdentityCIErrorConfig(BaseConfig):
    classname: Literal["IdentityCIError"] = "IdentityCIError"
    identity_ci: list[dict[str, str | int]] | None
    dense_ci: list[dict[str, str | int]] | None


class PermutedCIPlotsConfig(BaseConfig):
    classname: Literal["PermutedCIPlots"] = "PermutedCIPlots"
    identity_patterns: list[str] | None
    dense_patterns: list[str] | None


class StochasticReconSubsetCEAndKLConfig(BaseConfig):
    classname: Literal["StochasticReconSubsetCEAndKL"] = "StochasticReconSubsetCEAndKL"
    include_patterns: dict[str, list[str]] | None
    exclude_patterns: dict[str, list[str]] | None


class UVPlotsConfig(BaseConfig):
    classname: Literal["UVPlots"] = "UVPlots"
    identity_patterns: list[str] | None
    dense_patterns: list[str] | None


ReconLossConfigType = (
    UnmaskedReconLossConfig
    | CIMaskedReconLossConfig
    | CIMaskedReconSubsetLossConfig
    | CIMaskedReconLayerwiseLossConfig
    | StochasticReconLossConfig
    | StochasticReconSubsetLossConfig
    | StochasticReconLayerwiseLossConfig
    | PGDReconLossConfig
    | PGDReconSubsetLossConfig
    | PGDReconLayerwiseLossConfig
    | StochasticHiddenActsReconLossConfig
    | PersistentPGDReconLossConfig
    | PersistentPGDReconSubsetLossConfig
)

LossMetricConfigType = FaithfulnessLossConfig | ImportanceMinimalityLossConfig | ReconLossConfigType

EvalOnlyMetricConfigType = (
    CEandKLLossesConfig
    | CIHiddenActsReconLossConfig
    | CIHistogramsConfig
    | CI_L0Config
    | CIMeanPerComponentConfig
    | ComponentActivationDensityConfig
    | IdentityCIErrorConfig
    | PersistentPGDReconEvalConfig
    | PersistentPGDReconSubsetEvalConfig
    | PermutedCIPlotsConfig
    | UVPlotsConfig
    | StochasticReconSubsetCEAndKLConfig
    | PGDMultiBatchReconLossConfig
    | PGDMultiBatchReconSubsetLossConfig
    | CIMaskedAttnPatternsReconLossConfig
    | StochasticAttnPatternsReconLossConfig
)
MetricConfigType = LossMetricConfigType | EvalOnlyMetricConfigType


class _LossCapableMetricsConfig(BaseConfig):
    """Shared loss-capable metric fields used by both `LossMetricsConfig` and `EvalMetricsConfig`."""

    faithfulness: FaithfulnessLossConfig | None = None
    importance_minimality: ImportanceMinimalityLossConfig | None = None
    unmasked_recon: UnmaskedReconLossConfig | None = None
    ci_masked_recon: CIMaskedReconLossConfig | None = None
    ci_masked_recon_subset: CIMaskedReconSubsetLossConfig | None = None
    ci_masked_recon_layerwise: CIMaskedReconLayerwiseLossConfig | None = None
    stochastic_recon: StochasticReconLossConfig | None = None
    stochastic_recon_subset: StochasticReconSubsetLossConfig | None = None
    stochastic_recon_layerwise: StochasticReconLayerwiseLossConfig | None = None
    stochastic_hidden_acts_recon: StochasticHiddenActsReconLossConfig | None = None
    pgd_recon: PGDReconLossConfig | None = None
    pgd_recon_subset: PGDReconSubsetLossConfig | None = None
    pgd_recon_layerwise: PGDReconLayerwiseLossConfig | None = None
    persistent_pgd_recon: PersistentPGDReconLossConfig | None = None
    persistent_pgd_recon_subset: PersistentPGDReconSubsetLossConfig | None = None


class LossMetricsConfig(_LossCapableMetricsConfig):
    """Container of training-loss metric configs.

    Each field is a named, nullable metric config. Setting a field selects that metric for both
    training (weighted by `coeff`) and evaluation. Fields left as None are omitted.
    """

    def active(self) -> list[LossMetricConfigType]:
        return [v for _, v in self if v is not None]


class EvalMetricsConfig(_LossCapableMetricsConfig):
    """Container of eval metric configs.

    Includes all loss-capable metrics (set them here for eval-only computation; `coeff` is ignored)
    and additional eval-only metric fields.
    """

    ce_and_kl: CEandKLLossesConfig | None = None
    ci_hidden_acts_recon: CIHiddenActsReconLossConfig | None = None
    ci_histograms: CIHistogramsConfig | None = None
    ci_l0: CI_L0Config | None = None
    ci_mean_per_component: CIMeanPerComponentConfig | None = None
    component_activation_density: ComponentActivationDensityConfig | None = None
    identity_ci_error: IdentityCIErrorConfig | None = None
    persistent_pgd_recon_eval: PersistentPGDReconEvalConfig | None = None
    persistent_pgd_recon_subset_eval: PersistentPGDReconSubsetEvalConfig | None = None
    permuted_ci_plots: PermutedCIPlotsConfig | None = None
    uv_plots: UVPlotsConfig | None = None
    stochastic_recon_subset_ce_and_kl: StochasticReconSubsetCEAndKLConfig | None = None
    pgd_multibatch_recon: PGDMultiBatchReconLossConfig | None = None
    pgd_multibatch_recon_subset: PGDMultiBatchReconSubsetLossConfig | None = None
    ci_masked_attn_patterns_recon: CIMaskedAttnPatternsReconLossConfig | None = None
    stochastic_attn_patterns_recon: StochasticAttnPatternsReconLossConfig | None = None

    def active(self) -> list[MetricConfigType]:
        return [v for _, v in self if v is not None]


SamplingType = Literal["continuous", "binomial"]


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
        description="Device to run on.",
    )
    dp: PositiveInt | None = Field(
        default=None,
        description="Number of GPUs for data parallelism. None = single GPU/CPU. Bounded by "
        "the cluster's GPUs-per-node for single-node DDP; multiples of that for multi-node. ",
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
        description="Interval (in steps) at which to run slow evaluation metrics. Must be a multiple of `eval_freq`.",
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
    ci_alive_threshold: Probability = Field(
        default=0.0,
        description="Causal importance threshold above which a component is considered 'firing'. "
        "Used by L0 and component-activation-density metrics; doesn't affect training.",
    )
    eval_metrics: EvalMetricsConfig = Field(
        default_factory=EvalMetricsConfig,
        description=(
            "Eval-only metrics. Metrics already set in `pd.loss_metrics` are evaluated "
            "automatically and should not be repeated here."
        ),
    )
    wandb_run_name: str | None = Field(
        default=None,
        description="W&B run display name. None lets W&B auto-name.",
    )
    view_meta: dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form labels for downstream grouping/coloring/reports (e.g. "
        "`{'lr_ratio': 0.1, 'size': 'medium'}`). Populated by sweep generators; surfaced "
        "to W&B under a `view_meta/` prefix.",
    )

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
        description="Configuration for the causal importance function. "
        "Use LayerwiseCiConfig for per-layer CI functions or GlobalCiConfig for a single global CI function.",
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
        description="List of module patterns with C values specifying which modules to decompose. "
        "Example: [{module_pattern: 'h.*.mlp.c_fc', C: 10}, {module_pattern: 'h.*.attn.*', C: 20}]",
    )
    identity_module_info: list[ModulePatternInfoConfig] | None = Field(
        default=None,
        description="List of identity module patterns with C values. "
        "Identity operations will be inserted at these modules.",
    )

    @property
    def all_module_info(self) -> list[ModulePatternInfoConfig]:
        """Combine target and identity patterns with their C values.

        Returns list of ModulePatternInfoConfig with .pre_identity suffix added to identity patterns.
        """
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
        "model and component weights. This allows for removing the faithfulness loss.",
    )

    loss_metrics: LossMetricsConfig = Field(
        default_factory=LossMetricsConfig,
        description=(
            "Training-loss metrics. Each non-None field selects a loss; its `coeff` weights the "
            "training loss. Active loss metrics are automatically also evaluated."
        ),
    )
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
        for cfg in self.loss_metrics.active():
            assert cfg.coeff is not None, f"loss_metrics.{type(cfg).__name__} must have a coeff"
        return self
