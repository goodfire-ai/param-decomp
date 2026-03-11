"""Config classes of various types"""

from typing import Annotated, Any, ClassVar, Literal, Self

from pydantic import (
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from spd.base_config import BaseConfig
from spd.log import logger
from spd.spd_types import GlobalCiFnType, LayerwiseCiFnType, ModelPath, Probability


class LayerwiseCiConfig(BaseConfig):
    """Configuration for layerwise CI functions (one per layer)."""

    mode: Literal["layerwise"] = "layerwise"
    fn_type: LayerwiseCiFnType = Field(
        ..., description="Type of layerwise CI function: mlp, vector_mlp, or shared_mlp"
    )
    hidden_dims: list[NonNegativeInt] = Field(
        ..., description="Hidden dimensions for the CI function MLP"
    )


class BlockGroupConfig(BaseConfig):
    """Defines a group of modules processed together in global reverse residual CI.

    Modules within a block have their activations concatenated, projected to the residual
    stream dimension, and processed together by a single reader network.
    """

    name: str = Field(..., description="Block identifier (e.g. 'unembed', 'layer_2_mlp')")
    patterns: list[str] = Field(
        ...,
        description="Module patterns for this block (fnmatch-style, e.g. ['layers.2.mlp_*'])",
    )


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
    mlp_hidden_dim: list[NonNegativeInt] = Field(
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
    For fn_type='global_reverse_residual': Processes blocks in reverse order with residual stream.
    For fn_type='global_shared_transformer': Concatenates activations, projects to shared d_model,
    and applies transformer blocks over the sequence dimension.
    """

    mode: Literal["global"] = "global"
    fn_type: GlobalCiFnType = Field(
        ...,
        description="Type of global CI function: global_shared_mlp, "
        "global_reverse_residual, or global_shared_transformer",
    )
    hidden_dims: list[NonNegativeInt] | None = Field(
        default=None,
        description="Hidden dimensions for global_shared_mlp CI function. "
        "Use reader_hidden_dims for global_reverse_residual.",
    )
    reader_hidden_dims: list[NonNegativeInt] | None = Field(
        default=None,
        description="Hidden dimensions for reader MLPs in global_reverse_residual. "
        "Required when fn_type='global_reverse_residual', ignored otherwise.",
    )
    d_resid_ci_fn: PositiveInt | None = Field(
        default=None,
        description="Residual stream dimension for global_reverse_residual. "
        "Required when fn_type='global_reverse_residual', ignored otherwise.",
    )
    block_groups: list[BlockGroupConfig] | None = Field(
        default=None,
        description="Ordered list of block groups for global_reverse_residual. "
        "Order determines processing sequence (first = processed first, typically unembed). "
        "Required when fn_type='global_reverse_residual', ignored otherwise.",
    )
    transition_attn_config: AttnConfig | None = Field(
        default=None,
        description="Self-attention config for transitions in global_reverse_residual. "
        "If None, uses MLP-only transitions (original behavior). "
        "Only applies when fn_type='global_reverse_residual'.",
    )
    transition_hidden_dim: PositiveInt | None = Field(
        default=None,
        description="Hidden dimension for transition MLP in global_reverse_residual. "
        "MLP structure: d_resid_ci_fn -> transition_hidden_dim -> d_resid_ci_fn with GeLU. "
        "Required when fn_type='global_reverse_residual', ignored otherwise.",
    )
    simple_transformer_ci_cfg: GlobalSharedTransformerCiConfig | None = None

    @model_validator(mode="after")
    def validate_ci_config(self) -> Self:
        if self.fn_type == "global_reverse_residual":
            assert self.d_resid_ci_fn is not None, (
                "d_resid_ci_fn must be specified when fn_type='global_reverse_residual'"
            )
            assert self.block_groups is not None and len(self.block_groups) > 0, (
                "block_groups must be specified with at least one block when "
                "fn_type='global_reverse_residual'"
            )
            assert self.reader_hidden_dims is not None, (
                "reader_hidden_dims must be specified when fn_type='global_reverse_residual'"
            )
            if self.transition_attn_config is not None:
                assert self.d_resid_ci_fn % self.transition_attn_config.n_heads == 0, (
                    f"d_resid_ci_fn ({self.d_resid_ci_fn}) must be divisible by "
                    f"transition_attn_config.n_heads ({self.transition_attn_config.n_heads})"
                )
        elif self.fn_type == "global_shared_mlp":
            assert self.hidden_dims is not None, (
                "hidden_dims must be specified when fn_type='global_shared_mlp'"
            )
            assert self.transition_attn_config is None, (
                "transition_attn_config is only valid for global_reverse_residual"
            )
        elif self.fn_type == "global_shared_transformer":
            assert self.simple_transformer_ci_cfg is not None, (
                "simple_transformer_ci_cfg must be specified when fn_type='global_shared_transformer'"
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


def migrate_to_lr_schedule_config(config_dict: dict[str, Any]) -> None:
    """Migrate old LR config format (lr + lr_schedule + lr_warmup_pct) to ScheduleConfig.

    Modifies config_dict in place.
    """
    if "lr" not in config_dict:
        return

    logger.info("Migrating old LR config format to ScheduleConfig")

    old_lr = config_dict.pop("lr")
    old_fn_type = config_dict.pop("lr_schedule", "constant")
    old_warmup_pct = config_dict.pop("lr_warmup_pct", 0.0)

    # Old cosine decayed to 0, old constant stayed at 1
    final_val_frac = 0.0 if old_fn_type == "cosine" else 1.0

    config_dict["lr_schedule"] = {
        "start_val": old_lr,
        "fn_type": old_fn_type,
        "warmup_pct": old_warmup_pct,
        "final_val_frac": final_val_frac,
    }


# Task configs - these define task-specific parameters for SPD decomposition
class TMSTaskConfig(BaseConfig):
    task_name: Literal["tms"] = Field(
        default="tms",
        description="Task identifier for TMS",
    )
    feature_probability: Probability = Field(
        ...,
        description="Probability that a given feature is active in generated data",
    )
    data_generation_type: Literal["exactly_one_active", "at_least_zero_active"] = Field(
        default="at_least_zero_active",
        description="Strategy for generating synthetic data for TMS training",
    )


class ResidMLPTaskConfig(BaseConfig):
    task_name: Literal["resid_mlp"] = Field(
        default="resid_mlp",
        description="Identifier for the residual-MLP decomposition task",
    )
    feature_probability: Probability = Field(
        ...,
        description="Probability that a given feature is active in generated data",
    )
    data_generation_type: Literal[
        "exactly_one_active", "exactly_two_active", "at_least_zero_active"
    ] = Field(
        default="at_least_zero_active",
        description="Strategy for generating synthetic data for residual-MLP training",
    )


class IHTaskConfig(BaseConfig):
    task_name: Literal["ih"]
    prefix_window: PositiveInt | None = Field(
        default=None,
        description="Number of tokens to use as a prefix window for the induction head. If none, uses the full sequence length.",
    )


class LMTaskConfig(BaseConfig):
    task_name: Literal["lm"] = Field(
        default="lm",
        description="Identifier for the language-model decomposition task",
    )
    max_seq_len: PositiveInt = Field(
        default=512,
        description="Maximum sequence length to truncate or pad inputs to",
    )
    buffer_size: PositiveInt = Field(
        default=1000,
        description="Buffered sample count for streaming dataset shuffling",
    )
    dataset_name: str = Field(
        default="lennart-finke/SimpleStories",
        description="HuggingFace dataset identifier to use for the LM task",
    )
    column_name: str = Field(
        default="story",
        description="Dataset column that contains the text to train on",
    )
    train_data_split: str = Field(
        default="train",
        description="Name of the dataset split used for training",
    )
    eval_data_split: str = Field(
        default="test",
        description="Name of the dataset split used for evaluation",
    )
    shuffle_each_epoch: bool = Field(
        default=True,
        description="Whether to reshuffle data at each epoch. Set False in tests to keep fixed "
        "order across dp modes.",
    )
    is_tokenized: bool = Field(
        default=False,
        description="Whether the dataset is already tokenized",
    )
    streaming: bool = Field(
        default=False,
        description="Whether to use a streaming dataset",
    )
    dataset_seed: int | None = Field(
        default=None,
        description="Seed for dataset shuffling/sampling. When None, uses the global `seed`.",
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
        description="Loss coefficient. Used when metric is in loss_metric_configs.",
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

    @model_validator(mode="before")
    @classmethod
    def migrate_old_fields(cls, data: dict[str, Any]) -> dict[str, Any]:
        # Migrate pnorm_1 to pnorm (intermediate format)
        if "pnorm_1" in data and "pnorm" not in data:
            data["pnorm"] = data.pop("pnorm_1")
        elif "pnorm_1" in data:
            data.pop("pnorm_1")
        # Remove deprecated pnorm_2
        data.pop("pnorm_2", None)
        # Default beta if missing
        if "beta" not in data:
            logger.warning("beta not in ImportanceMinimalityLossConfig, defaulting to 0.0")
            data["beta"] = 0.0
        return data


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

    @model_validator(mode="before")
    @classmethod
    def migrate_step_size(cls, data: Any) -> Any:
        if isinstance(data, dict) and "step_size" in data and "lr_schedule" not in data:
            data["lr_schedule"] = {
                "start_val": data.pop("step_size"),
                "warmup_pct": 0.0,
                "final_val_frac": 1.0,
                "fn_type": "constant",
            }
        return data


class AdamPGDConfig(BaseConfig):
    type: Literal["adam"] = "adam"
    beta1: Probability = Field(default=0.9, description="Adam beta1 for masks")
    beta2: Probability = Field(default=0.999, description="Adam beta2 for masks")
    eps: NonNegativeFloat = Field(default=1e-8, description="Adam epsilon for masks")
    lr_schedule: ScheduleConfig

    @model_validator(mode="before")
    @classmethod
    def migrate_lr(cls, data: Any) -> Any:
        if isinstance(data, dict) and "lr" in data and "lr_schedule" not in data:
            data["lr_schedule"] = {
                "start_val": data.pop("lr"),
                "warmup_pct": 0.0,
                "final_val_frac": 1.0,
                "fn_type": "constant",
            }
        return data


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


def _coerce_ppgd_scope(config_dict: dict[str, Any]) -> None:
    """Backwards compat: migrate old scope format/names to current names."""
    scope = config_dict.get("scope")
    if isinstance(scope, str):
        scope = {"type": scope}
        config_dict["scope"] = scope
    if not isinstance(scope, dict):
        return
    match scope.get("type"):
        case "single_mask":
            scope["type"] = "single_source"
        case "batch_invariant":
            scope["type"] = "repeat_across_batch"
            if "n_masks" in scope:
                scope["n_sources"] = scope.pop("n_masks")
        case "per_batch" | "unique_per_batch_per_token":
            scope["type"] = "per_batch_per_position"
        case _:
            pass


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

    @model_validator(mode="before")
    @classmethod
    def _compat_scope(cls, data: Any) -> Any:
        if isinstance(data, dict):
            _coerce_ppgd_scope(data)
        return data


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

    @model_validator(mode="before")
    def handle_deprecated_config_keys(cls, config_dict: dict[str, Any]) -> dict[str, Any]:
        """Remove deprecated config keys and change names of any keys that have been renamed."""
        config_dict.pop("sigmoid_type", None)
        return config_dict


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

TaskConfig = TMSTaskConfig | ResidMLPTaskConfig | LMTaskConfig | IHTaskConfig

SamplingType = Literal["continuous", "binomial"]


class Config(BaseConfig):
    # --- WandB
    wandb_project: str | None = Field(
        default=None,
        description="Weights & Biases project name (set to None to disable WandB logging)",
    )
    wandb_run_name: str | None = Field(
        default=None,
        description="Explicit name for the WandB run (None generates an automatic name)",
    )
    wandb_run_name_prefix: str = Field(
        default="",
        description="Prefix prepended to an auto-generated WandB run name",
    )

    # --- General ---
    seed: int = Field(
        default=0,
        description="Random seed for reproducibility. Does not affect dataset shuffling if dataset_seed is set in TaskConfig.",
    )
    autocast_bf16: bool = Field(
        default=True,
        description="Whether to use torch.autocast with bfloat16 mixed precision",
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

    init_spd_checkpoint: str | None = Field(
        default=None,
        description="Path to a .pth checkpoint from a prior SPD run for component/CI initialization",
    )

    use_delta_component: bool = Field(
        default=True,
        description="If True, use an extra component containing the difference between the target "
        "model and component weights. This allows for removing the faithfulness loss.",
    )

    loss_metric_configs: list[Annotated[LossMetricConfigType, Field(discriminator="classname")]] = (
        Field(
            default=[],
            description=(
                "List of configs for loss metrics to compute (used for both training logs and eval); "
                "coefficients provided here are also used for weighting the training loss and eval loss/total."
            ),
        )
    )
    output_loss_type: Literal["mse", "kl"] = Field(
        ...,
        description="Metric used to measure recon error between model outputs and targets",
    )

    # --- Training ---
    lr_schedule: ScheduleConfig = Field(..., description="Learning rate schedule configuration")
    steps: NonNegativeInt = Field(..., description="Total number of optimisation steps")
    batch_size: PositiveInt = Field(
        ...,
        description="Total batch size (may be divided across multiple devices).",
    )
    grad_clip_norm_components: PositiveFloat | None = Field(
        default=None,
        description="If set, apply grad norm clipping to the parameters of the components",
    )
    grad_clip_norm_ci_fns: PositiveFloat | None = Field(
        default=None,
        description="If set, apply grad norm clipping to the parameters of the CI functions",
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

    # --- Logging & Saving ---
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
        description="Batch size used for evaluation. If None, uses the same as `batch_size`.",
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
    eval_metric_configs: list[Annotated[MetricConfigType, Field(discriminator="classname")]] = (
        Field(
            default=[],
            description="List of configs for metrics to use for evaluation",
        )
    )

    # --- Component Tracking ---
    ci_alive_threshold: Probability = Field(
        default=0.0,
        description="Causal importance threshold above which a component is considered 'firing'",
    )

    # --- Pretrained model info ---
    pretrained_model_class: str = Field(
        ...,
        description="Fully-qualified class name of the pretrained model to load. Can be defined "
        "locally or an in external package (e.g. 'transformers.LlamaForCausalLM' or "
        "'spd.experiments.resid_mlp.models.ResidMLP').",
    )
    pretrained_model_path: ModelPath | None = Field(
        default=None,
        description="Model identifier. Local path or wandb reference "
        "(e.g. 'wandb:goodfire/spd/runs/otxwx80v' or 'mnt/my_model/checkpoint.pth')",
    )
    pretrained_model_name: str | None = Field(
        default=None,
        description="hf model identifier. E.g. 'SimpleStories/SimpleStories-1.25M'",
    )
    pretrained_model_output_attr: str | None = Field(
        default=None,
        description="Name of the attribute on the forward output that contains logits or activations",
    )
    tokenizer_name: str | None = Field(
        default=None,
        description="Name or path of the tokenizer to use when loading an LM",
    )

    # --- Task Specific ---
    task_config: TaskConfig = Field(
        ...,
        discriminator="task_name",
        description="Nested task-specific configuration selected by the `task_name` discriminator",
    )

    DEPRECATED_CONFIG_KEYS: ClassVar[list[str]] = [
        "image_on_first_step",
        "image_freq",
        "metrics_fns",
        "figures_fns",
        "schatten_coeff",
        "embedding_recon_coeff",
        "is_embed_unembed_recon",
        "out_recon_coeff",
        "faithfulness_coeff",
        "stochastic_recon_coeff",
        "stochastic_recon_layerwise_coeff",
        "recon_coeff",
        "recon_layerwise_coeff",
        "ci_recon_coeff",
        "ci_recon_layerwise_coeff",
        "pnorm",
        "p_anneal_start_frac",
        "p_anneal_final_p",
        "p_anneal_end_frac",
        "importance_minimality_coeff",
        "dist_backend",
        "lr_exponential_halflife",
        "out_dir",
        "n_examples_until_dead",
        "gradient_accumulation_steps",
    ]
    RENAMED_CONFIG_KEYS: ClassVar[dict[str, str]] = {
        "grad_clip_norm": "grad_clip_norm_components",
        "print_freq": "eval_freq",
        "pretrained_model_name_hf": "pretrained_model_name",
        "recon_coeff": "ci_recon_coeff",
        "recon_layerwise_coeff": "ci_recon_layerwise_coeff",
    }

    @model_validator(mode="before")
    def handle_deprecated_config_keys(cls, config_dict: dict[str, Any]) -> dict[str, Any]:
        """Remove deprecated config keys and change names of any keys that have been renamed."""

        # We don't bother mapping the old ``eval_metrics`` to the new ``eval_metric_configs``.
        config_dict.pop("eval_metrics", None)

        cls._migrate_to_module_info(config_dict)
        cls._migrate_to_ci_config(config_dict)
        migrate_to_lr_schedule_config(config_dict)

        for key in list(config_dict.keys()):
            val = config_dict[key]
            if key in cls.DEPRECATED_CONFIG_KEYS:
                logger.warning(f"{key} is deprecated, but has value: {val}. Removing from config.")
                del config_dict[key]

            elif key in cls.RENAMED_CONFIG_KEYS:
                logger.info(f"Renaming {key} to {cls.RENAMED_CONFIG_KEYS[key]}")
                config_dict[cls.RENAMED_CONFIG_KEYS[key]] = val
                del config_dict[key]

            elif key in ("loss_metric_configs", "eval_metric_configs"):
                # We used to have an extra_init_kwargs field. This is hard to map. Just remove all
                # configs with it
                new_vals = [cfg for cfg in val if "extra_init_kwargs" not in cfg]
                config_dict[key] = new_vals

        # Remap simple_stories_train → spd.pretrain (models moved in-tree)
        pmc = config_dict.get("pretrained_model_class", "")
        if pmc.startswith("simple_stories_train.models."):
            config_dict["pretrained_model_class"] = pmc.replace(
                "simple_stories_train.models.", "spd.pretrain.models.", 1
            )

        if "eval_batch_size" not in config_dict:
            config_dict["eval_batch_size"] = config_dict["batch_size"]
        if "train_log_freq" not in config_dict:
            config_dict["train_log_freq"] = 50
        if "slow_eval_freq" not in config_dict:
            config_dict["slow_eval_freq"] = config_dict["eval_freq"]
        return config_dict

    @classmethod
    def _migrate_to_module_info(cls, config_dict: dict[str, Any]) -> None:
        """Migrate old config format (C + target_module_patterns) to new module_info format."""
        cond = "C" in config_dict or "target_module_patterns" in config_dict
        if not cond:
            return

        logger.warning(
            "Found old config keys for C definition, mapping old structure to new module_info structure"
        )
        global_c = config_dict["C"]
        config_dict["module_info"] = [
            {"module_pattern": p, "C": global_c} for p in config_dict["target_module_patterns"]
        ]
        del config_dict["C"]
        del config_dict["target_module_patterns"]

        identity_patterns = config_dict.pop("identity_module_patterns", None)
        if identity_patterns is not None:
            config_dict["identity_module_info"] = [
                {"module_pattern": p, "C": global_c} for p in identity_patterns
            ]

    @classmethod
    def _migrate_to_ci_config(cls, config_dict: dict[str, Any]) -> None:
        """Migrate old ci_fn_type/ci_fn_hidden_dims/use_global_ci to new ci_config structure."""
        has_old_fields = (
            "ci_fn_type" in config_dict
            or "ci_fn_hidden_dims" in config_dict
            or "use_global_ci" in config_dict
        )
        if not has_old_fields:
            return

        logger.info(
            "Migrating old ci_fn_type/ci_fn_hidden_dims/use_global_ci to ci_config structure"
        )

        ci_fn_type = config_dict.pop("ci_fn_type", "vector_mlp")
        ci_fn_hidden_dims = config_dict.pop("ci_fn_hidden_dims", [8])
        use_global_ci = config_dict.pop("use_global_ci", False)

        # Determine if this is a global CI function
        is_global = use_global_ci or ci_fn_type.startswith("global_")

        if is_global:
            # Map layerwise type to global type if use_global_ci was set
            if not ci_fn_type.startswith("global_"):
                ci_fn_type = "global_shared_mlp"
            config_dict["ci_config"] = {
                "mode": "global",
                "fn_type": ci_fn_type,
                "hidden_dims": ci_fn_hidden_dims,
            }
        else:
            config_dict["ci_config"] = {
                "mode": "layerwise",
                "fn_type": ci_fn_type,
                "hidden_dims": ci_fn_hidden_dims,
            }

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        assert self.slow_eval_freq % self.eval_freq == 0, (
            "slow_eval_freq must be a multiple of eval_freq"
        )
        assert self.slow_eval_freq // self.eval_freq >= 1, (
            "slow_eval_freq must be at least eval_freq"
        )

        for cfg in self.loss_metric_configs:
            assert cfg.coeff is not None, "All loss_metric_configs must have a coeff"

        if any(
            isinstance(cfg, PersistentPGDReconLossConfig | PersistentPGDReconSubsetLossConfig)
            for cfg in self.loss_metric_configs
        ):
            assert isinstance(self.task_config, LMTaskConfig), (
                "Persistent PGD losses are only supported with LM tasks"
            )

        return self
