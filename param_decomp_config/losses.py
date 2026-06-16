"""Loss-metric configs.

One config per loss `Metric` in `param_decomp/metrics/` (plus the lab-side chunkwise
subset recon). Each carries a unique `type: Literal["<ClassName>"]` discriminator;
`AnyLossMetricConfig` in `param_decomp_config.pd` unions them for YAML validation.
"""

from typing import Annotated, Any, Literal

from pydantic import BeforeValidator, Field, NonNegativeFloat, NonNegativeInt, PositiveInt

from param_decomp_config.base import BaseConfig, Probability
from param_decomp_config.routing import SubsetRoutingType, UniformKSubsetRoutingConfig
from param_decomp_config.schedule import ScheduleConfig


class LossMetricConfig(BaseConfig):
    """Pydantic config for a metric that can also be used as a training loss.

    `coeff` is required when this metric is listed under `loss_metrics` (asserted by
    `PDConfig`'s field validator); ignored for eval-only instances.

    `name` overrides the class name as this instance's identity (`Metric.instance_key`),
    letting the same metric class appear under both `loss_metrics` and `eval.metrics`
    with different settings — e.g. a 1-step PGD training loss alongside a 20-step PGD
    eval probe. Leave `None` (the default) and the class name is used.
    """

    coeff: float | None = None
    name: str | None = None


class FaithfulnessLossConfig(LossMetricConfig):
    type: Literal["FaithfulnessLoss"] = "FaithfulnessLoss"


class ImportanceMinimalityLossConfig(LossMetricConfig):
    """Config for the `L_p`-style importance-minimality penalty on upper-leaky CI values.

    `pnorm` is the initial `p`; `beta` weights the entropy-like `mean * log2(1 + sum)`
    term added on top of the `L_p` term. `pnorm` is linearly annealed toward
    `p_anneal_final_p` between `p_anneal_start_frac` and `p_anneal_end_frac` of training
    (no-op when `p_anneal_final_p is None` or `p_anneal_start_frac == 1.0`).
    """

    type: Literal["ImportanceMinimalityLoss"] = "ImportanceMinimalityLoss"
    pnorm: NonNegativeFloat
    beta: NonNegativeFloat
    p_anneal_start_frac: Probability = 1.0
    p_anneal_final_p: NonNegativeFloat | None = None
    p_anneal_end_frac: Probability = 1.0
    eps: NonNegativeFloat = 1e-12


class CIMaskedReconLossConfig(LossMetricConfig):
    type: Literal["CIMaskedReconLoss"] = "CIMaskedReconLoss"


class CIMaskedReconLayerwiseLossConfig(LossMetricConfig):
    type: Literal["CIMaskedReconLayerwiseLoss"] = "CIMaskedReconLayerwiseLoss"


class CIMaskedReconSubsetLossConfig(LossMetricConfig):
    type: Literal["CIMaskedReconSubsetLoss"] = "CIMaskedReconSubsetLoss"
    routing: Annotated[
        SubsetRoutingType, Field(discriminator="type", default=UniformKSubsetRoutingConfig())
    ]


class StochasticReconLossConfig(LossMetricConfig):
    type: Literal["StochasticReconLoss"] = "StochasticReconLoss"


class StochasticReconLayerwiseLossConfig(LossMetricConfig):
    type: Literal["StochasticReconLayerwiseLoss"] = "StochasticReconLayerwiseLoss"


class StochasticReconSubsetLossConfig(LossMetricConfig):
    type: Literal["StochasticReconSubsetLoss"] = "StochasticReconSubsetLoss"
    routing: Annotated[
        SubsetRoutingType, Field(discriminator="type", default=UniformKSubsetRoutingConfig())
    ]


class StochasticHiddenActsReconLossConfig(LossMetricConfig):
    type: Literal["StochasticHiddenActsReconLoss"] = "StochasticHiddenActsReconLoss"


class UnmaskedReconLossConfig(LossMetricConfig):
    type: Literal["UnmaskedReconLoss"] = "UnmaskedReconLoss"


class ChunkwiseSubsetReconLossConfig(LossMetricConfig):
    """Reconstruction loss that mirrors the 3-pool / 2-pool chunkwise subset recon.

    The decomposed sites (`model.target_module_paths`, in order) are grouped into
    chunks of `sites_per_chunk`; each chunk runs `SubsetReconPlan(routing, n_samples)`
    — one masked suffix forward per generated routing, all the chunk's sites swapped in
    with a per-position routing draw — and the recon is the fused-linear-KL against the
    clean logits (when `use_fused_kl`). The total is the mean over all chunk forwards of
    `recon_loss / n_positions`, matching the 2-pool's per-step recon.

    The JAX single-pool trainer implements this natively: `recon.build_recon_terms`
    maps this `type` onto `recon.subset_chunk_plan` (a parameterization of the one
    `chunkwise_plan` builder), and the jitted step runs the chunk forwards directly —
    no vendored `LMComponentModel` or lab recon-plan machinery is involved.
    """

    type: Literal["ChunkwiseSubsetReconLoss"] = "ChunkwiseSubsetReconLoss"
    sites_per_chunk: PositiveInt
    routing: Annotated[
        SubsetRoutingType, Field(discriminator="type", default=UniformKSubsetRoutingConfig())
    ]
    n_samples: PositiveInt = 1
    use_fused_kl: bool = True


PGDInitStrategy = Literal["random", "ones", "zeroes"]
# Stored run configs predate the shape-literal scope names; alias exactly the literals
# that exist in stored data (`unique_per_datapoint` occurs only in LM runs, hence `bsc`).
# Delete once stored runs are migrated.
_LEGACY_MASK_SCOPE_ALIASES = {
    "shared_across_batch": "c",
    "unique_per_datapoint": "bsc",
}


def _alias_legacy_mask_scope(value: Any) -> Any:
    if isinstance(value, str):
        return _LEGACY_MASK_SCOPE_ALIASES.get(value, value)
    return value


# Scope literals spell the adversarial-source shape in tensor order (batch, seq, C).
# `c` is one shared vector, rank-polymorphic and DP-synced; `bc` (no seq axis) and
# `bsc` (LM) are independent per batch element, and must match the batch rank.
MaskScope = Annotated[Literal["c", "bc", "bsc"], BeforeValidator(_alias_legacy_mask_scope)]


class PGDConfig(LossMetricConfig):
    """Shared base for per-step PGD loss configs."""

    init: PGDInitStrategy
    step_size: float
    n_steps: int
    mask_scope: MaskScope


class PGDReconLossConfig(PGDConfig):
    type: Literal["PGDReconLoss"] = "PGDReconLoss"


class PGDReconLayerwiseLossConfig(PGDConfig):
    type: Literal["PGDReconLayerwiseLoss"] = "PGDReconLayerwiseLoss"


class PGDReconSubsetLossConfig(PGDConfig):
    type: Literal["PGDReconSubsetLoss"] = "PGDReconSubsetLoss"
    routing: Annotated[
        SubsetRoutingType, Field(discriminator="type", default=UniformKSubsetRoutingConfig())
    ]


class SignPGDConfig(BaseConfig):
    """Sign-PGD optimizer config (adds `lr * sign(grad)` to sources)."""

    type: Literal["sign"] = "sign"
    lr_schedule: ScheduleConfig


class AdamPGDConfig(BaseConfig):
    """Adam-style PGD optimizer config."""

    type: Literal["adam"] = "adam"
    beta1: Probability = Field(default=0.9, description="Adam beta1 for masks")
    beta2: Probability = Field(default=0.999, description="Adam beta2 for masks")
    eps: NonNegativeFloat = Field(default=1e-8, description="Adam epsilon for masks")
    lr_schedule: ScheduleConfig


PGDOptimizerConfig = SignPGDConfig | AdamPGDConfig


class CScope(BaseConfig):
    """PPGD source scope: one `[C]` source vector shared across all batch dims."""

    type: Literal["c"] = "c"


class SCScope(BaseConfig):
    """PPGD source scope: `[seq, C]` sources shared across batch elements, free per position."""

    type: Literal["sc"] = "sc"


class NSCScope(BaseConfig):
    """PPGD source scope: `n_sources` source vectors tiled along the batch dim.

    `n_sources` must divide the per-rank batch size.
    """

    type: Literal["nsc"] = "nsc"
    n_sources: PositiveInt


class BSCScope(BaseConfig):
    """PPGD source scope: an independent source per batch element and position.

    Skips cross-rank synchronization of source state.
    """

    type: Literal["bsc"] = "bsc"


# Stored run configs (`runs/*/experiment_config.yaml`) predate the shape-literal scope
# names; alias exactly the literals that exist in stored data so old runs keep loading.
# Delete once stored runs are migrated.
_LEGACY_SCOPE_TYPE_ALIASES = {
    "broadcast_across_batch": "sc",
    "per_batch_per_position": "bsc",
}


def _alias_legacy_scope_type(value: Any) -> Any:
    if isinstance(value, dict) and value.get("type") in _LEGACY_SCOPE_TYPE_ALIASES:
        return {**value, "type": _LEGACY_SCOPE_TYPE_ALIASES[value["type"]]}
    return value


# Scope literals spell the stored source shape, read left-to-right in tensor order
# (batch, seq, C). `c` is rank-polymorphic (all leading dims singleton); the
# seq-bearing scopes require a sequence axis and are illegal off-LM.
PersistentPGDSourceScope = Annotated[
    CScope | SCScope | NSCScope | BSCScope,
    Field(discriminator="type"),
    BeforeValidator(_alias_legacy_scope_type),
]


class PersistentPGDReconLossConfig(LossMetricConfig):
    """Persistent-PGD recon loss: adversarial mask sources persist across train steps,
    routed to all layers every forward.

    `update()` returns `None` before `start_frac` of training. Under
    `use_sigmoid_parameterization=True` sources are unconstrained and read via sigmoid;
    otherwise sources are clamped to `[0, 1]` after each step.
    """

    type: Literal["PersistentPGDReconLoss"] = "PersistentPGDReconLoss"
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
