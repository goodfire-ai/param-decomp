"""Top-level PD configs: `PDConfig`, `RuntimeConfig`, `Cadence`.

`PDConfig` is the algorithm spec; `RuntimeConfig` is the compute substrate; `Cadence`
governs when the loop emits train logs and checkpoints.
"""

from functools import cached_property
from typing import Annotated, Literal, Self

from pydantic import (
    Discriminator,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from param_decomp.base_config import BaseConfig, Probability
from param_decomp.ci_fns import CiConfig
from param_decomp.decomposition_targets import DecompositionTargetConfig
from param_decomp.masks import SamplingType
from param_decomp.metrics.ci_masked_recon import CIMaskedReconLossConfig
from param_decomp.metrics.ci_masked_recon_layerwise import CIMaskedReconLayerwiseLossConfig
from param_decomp.metrics.ci_masked_recon_subset import CIMaskedReconSubsetLossConfig
from param_decomp.metrics.faithfulness import FaithfulnessLossConfig
from param_decomp.metrics.importance_minimality import ImportanceMinimalityLossConfig
from param_decomp.metrics.persistent_pgd_recon import (
    PersistentPGDReconLossConfig,
    PersistentPGDReconSubsetLossConfig,
)
from param_decomp.metrics.pgd_masked_recon import PGDReconLossConfig
from param_decomp.metrics.pgd_masked_recon_layerwise import PGDReconLayerwiseLossConfig
from param_decomp.metrics.pgd_masked_recon_subset import PGDReconSubsetLossConfig
from param_decomp.metrics.stochastic_hidden_acts_recon import StochasticHiddenActsReconLossConfig
from param_decomp.metrics.stochastic_recon import StochasticReconLossConfig
from param_decomp.metrics.stochastic_recon_layerwise import StochasticReconLayerwiseLossConfig
from param_decomp.metrics.stochastic_recon_subset import StochasticReconSubsetLossConfig
from param_decomp.metrics.unmasked_recon import UnmaskedReconLossConfig
from param_decomp.schedule import ScheduleConfig


class OptimizerConfig(BaseConfig):
    lr_schedule: ScheduleConfig = Field(..., description="Learning rate schedule")
    weight_decay: NonNegativeFloat = Field(default=0.0, description="AdamW weight decay")
    betas: tuple[Probability, Probability] = Field(
        default=(0.9, 0.999), description="AdamW (beta1, beta2)"
    )
    grad_clip_norm: PositiveFloat | None = Field(
        default=None,
        description="If set, clip the grad norm of this group's parameters to this value",
    )


AnyLossMetricConfig = Annotated[
    CIMaskedReconLayerwiseLossConfig
    | CIMaskedReconLossConfig
    | CIMaskedReconSubsetLossConfig
    | FaithfulnessLossConfig
    | ImportanceMinimalityLossConfig
    | PersistentPGDReconLossConfig
    | PersistentPGDReconSubsetLossConfig
    | PGDReconLayerwiseLossConfig
    | PGDReconLossConfig
    | PGDReconSubsetLossConfig
    | StochasticHiddenActsReconLossConfig
    | StochasticReconLayerwiseLossConfig
    | StochasticReconLossConfig
    | StochasticReconSubsetLossConfig
    | UnmaskedReconLossConfig,
    Discriminator("type"),
]


class RuntimeConfig(BaseConfig):
    """Compute substrate: device, precision, data-parallelism degree.

    Perturbs numerics but doesn't change the algorithm. Future home for NCCL flags,
    gradient accumulation steps, fp8 variants, etc.
    """

    autocast_bf16: bool = Field(
        default=True,
        description="Use torch.autocast with bfloat16 mixed precision in training and eval.",
    )
    device: str = Field(
        default="cuda",
        description="Device to run on, e.g. 'cuda', 'cuda:0', or 'cpu'.",
    )
    dp: PositiveInt | None = Field(
        default=None,
        description="DDP world size, or None for single device.",
    )

    @model_validator(mode="after")
    def validate_device_dp(self) -> Self:
        assert self.device == "cpu" or self.device == "cuda" or self.device.startswith("cuda:"), (
            f"device must be 'cpu', 'cuda', or 'cuda:<index>', got {self.device!r}"
        )
        if self.dp is not None:
            assert self.device.startswith("cuda"), "dp requires a cuda device"
            assert self.dp >= 2, "if set, dp must be at least 2 (pass None for single device)."
        return self


class PDConfig(BaseConfig):
    """Algorithm specification: seed, CI function, losses, optimizers, target modules.

    Flipping any field here changes what algorithm runs. Pair with `RuntimeConfig`
    (substrate), `Cadence` (when to emit) and `RunSink` (where output goes) when
    calling `optimize`.
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
    decomposition_targets: list[DecompositionTargetConfig] = Field(
        ...,
        description="List of module patterns with C values specifying which modules to decompose.",
    )
    identity_decomposition_targets: list[DecompositionTargetConfig] | None = Field(
        default=None,
        description="List of identity module patterns with C values.",
    )

    @cached_property
    def all_decomposition_target_configs(self) -> list[DecompositionTargetConfig]:
        result = list(self.decomposition_targets)
        if self.identity_decomposition_targets is not None:
            for target in self.identity_decomposition_targets:
                result.append(
                    DecompositionTargetConfig(
                        module_pattern=f"{target.module_pattern}.pre_identity", C=target.C
                    )
                )
        return result

    use_delta_component: bool = Field(
        default=True,
        description="If True, use an extra component containing the difference between the target "
        "model and component weights.",
    )

    tied_weights: list[tuple[str, str]] | None = Field(
        default=None,
        description="Pairs (src, tgt) of component module names whose weights should be tied. "
        "After init, tgt's U/V are set to src's V.T / U.T. Ties make training nondeterministic.",
    )

    loss_metrics: list[AnyLossMetricConfig] = Field(
        default_factory=list,
        description=(
            "Training-loss metrics. Each entry's `type` field selects the concrete metric; "
            "`coeff` weights it in the total training loss. Active loss metrics are automatically"
            " also evaluated."
        ),
    )

    # --- Training ---
    components_optimizer: OptimizerConfig = Field(
        ..., description="Optimizer config for the component (LinearComponent etc.) parameters"
    )
    ci_fn_optimizer: OptimizerConfig = Field(
        ..., description="Optimizer config for the CI function parameters"
    )
    steps: PositiveInt = Field(..., description="Total number of optimisation steps")
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
    def validate_loss_metrics_have_coeff(self) -> Self:
        assert self.loss_metrics, "loss_metrics must contain at least one training loss"
        for cfg in self.loss_metrics:
            assert cfg.coeff is not None, f"loss_metrics.{cfg.type!r} must set `coeff`"
        return self


class Cadence(BaseConfig):
    """Rhythm of non-eval loop emissions: train-log and checkpoint periods.

    Held separately from `RunSink` so the sink only owns *where* output goes; `Cadence`
    owns *when* train logs and checkpoints fire. Eval timing lives on `EvalLoop`,
    alongside the runtime objects it depends on. `optimize()` always checkpoints at the
    final step regardless of `save_every`.
    """

    train_log_every: PositiveInt
    save_every: PositiveInt | None = None

    def should_log_train(self, step: int) -> bool:
        return step % self.train_log_every == 0

    def should_save(self, step: int) -> bool:
        if self.save_every is None or step == 0:
            return False
        return step % self.save_every == 0
