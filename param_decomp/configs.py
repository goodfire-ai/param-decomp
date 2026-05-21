"""Top-level PD configs: `PDConfig` (algorithm) and `RuntimeConfig` (substrate)."""

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

from param_decomp.base_config import BaseConfig
from param_decomp.ci_config import CiConfig
from param_decomp.metrics.ci_masked_recon_layerwise_loss import CIMaskedReconLayerwiseLossConfig
from param_decomp.metrics.ci_masked_recon_loss import CIMaskedReconLossConfig
from param_decomp.metrics.ci_masked_recon_subset_loss import CIMaskedReconSubsetLossConfig
from param_decomp.metrics.faithfulness_loss import FaithfulnessLossConfig
from param_decomp.metrics.hidden_acts_recon_loss import StochasticHiddenActsReconLossConfig
from param_decomp.metrics.importance_minimality_loss import ImportanceMinimalityLossConfig
from param_decomp.metrics.persistent_pgd import (
    PersistentPGDReconLossConfig,
    PersistentPGDReconSubsetLossConfig,
)
from param_decomp.metrics.pgd_masked_recon_layerwise_loss import PGDReconLayerwiseLossConfig
from param_decomp.metrics.pgd_masked_recon_loss import PGDReconLossConfig
from param_decomp.metrics.pgd_masked_recon_subset_loss import PGDReconSubsetLossConfig
from param_decomp.metrics.stochastic_recon_layerwise_loss import StochasticReconLayerwiseLossConfig
from param_decomp.metrics.stochastic_recon_loss import StochasticReconLossConfig
from param_decomp.metrics.stochastic_recon_subset_ce_and_kl import (
    StochasticReconSubsetCEAndKLConfig,
)
from param_decomp.metrics.stochastic_recon_subset_loss import StochasticReconSubsetLossConfig
from param_decomp.metrics.unmasked_recon_loss import UnmaskedReconLossConfig
from param_decomp.module_info import ModulePatternInfoConfig
from param_decomp.optimizer import OptimizerConfig
from param_decomp.routing import SamplingType

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
    | StochasticReconSubsetCEAndKLConfig
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


class PDConfig(BaseConfig):
    """Algorithm specification: seed, CI function, losses, optimizers, module info.

    Flipping any field here changes what algorithm runs. Pair with `RuntimeConfig`
    (substrate) and `RunSink` (cadence + outputs) when calling `optimize`.
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

    @cached_property
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
    def validate_loss_metrics_have_coeff(self) -> Self:
        for cfg in self.loss_metrics:
            assert cfg.coeff is not None, f"loss_metrics.{cfg.type!r} must set `coeff`"
        return self
