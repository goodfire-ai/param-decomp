"""The torch-free pydantic config schema for the algorithm core.

Every algorithm-level config class lives here (or in the sibling `base_config` /
`schedule` modules): routing, the explicit (toy) site spec, loss-metric configs,
eval-metric configs, the top-level `PDConfig` / `Cadence`, the placement table the
engine resolves, and the `wandb.config` shaping helpers. Depends only on pydantic /
numpy / pyyaml / annotated-types (via `base_config`), so non-trainer consumers validate
the same YAML run configs without pulling jax/wandb.

Experiment-level schema (the `ExperimentConfig` base and its LM / TMS / ResidMLP
subclasses, each binding concrete `target`/`decomposition`/`data` sections) lives
lab-side under `param_decomp/experiments/` — including the authored
`decomposition.ci` configs, the tiled LM site specs, and the LM's compute substrate
(`runtime:`), which speak each domain's vocabulary. Core carries only the RESOLVED CI-fn
arches (`ci_fn.py`) and the resolved flat sites.
"""

import copy
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, Self

from pydantic import (
    BeforeValidator,
    Discriminator,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from param_decomp.core.base_config import BaseConfig, Probability
from param_decomp.core.schedule import ScheduleConfig

# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


class UniformKSubsetRoutingConfig(BaseConfig):
    """Route each position to a uniformly-sized random subset."""

    type: Literal["uniform_k_subset"] = "uniform_k_subset"


class StaticProbabilityRoutingConfig(BaseConfig):
    """Each position independently routes to each module with probability `p`."""

    type: Literal["static_probability"] = "static_probability"
    p: Probability


class AllRoutingConfig(BaseConfig):
    """Route every position to every module (the `"all"` fast path)."""

    type: Literal["all"] = "all"


# Discriminated union over the subset-routing configs (keyed by ``type``).
SubsetRoutingType = UniformKSubsetRoutingConfig | StaticProbabilityRoutingConfig | AllRoutingConfig


# ---------------------------------------------------------------------------
# Decomposition site (C) specs
# ---------------------------------------------------------------------------


class ExplicitSite(BaseConfig):
    name: str
    C: PositiveInt


class ExplicitCSpec(BaseConfig):
    """Arbitrary named sites with per-site C — the positionless toys (no arch grid, and the
    MLP CI fns have no chunk-homogeneity requirement)."""

    kind: Literal["explicit"] = "explicit"
    sites: list[ExplicitSite] = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# Loss-metric configs
# ---------------------------------------------------------------------------


class LossMetricConfig(BaseConfig):
    """Pydantic config for a metric that can also be used as a training loss.

    `coeff` is required when this metric is listed under `loss_metrics` and must be null
    when listed under `eval.metrics` — both directions are asserted
    (`PDConfig.validate_loss_metrics_have_coeff`;
    `param_decomp.experiments.eval_config.validate_eval_metrics`).

    `name` overrides the class name as this instance's identity (`Metric.instance_key`),
    letting the same metric class appear under both `loss_metrics` and `eval.metrics`
    with different settings — e.g. a 1-step PGD training loss alongside a 20-step PGD
    eval probe. Leave `None` (the default) and the class name is used.
    """

    coeff: float | None = None
    name: str | None = None


class FaithfulnessLossConfig(LossMetricConfig):
    type: Literal["FaithfulnessLoss"] = "FaithfulnessLoss"


class FrequencyMinimalityConfig(BaseConfig):
    """The frequency-minimality penalty riding on an imp-min term: a component's per-token
    firing frequency `f_c` (over the whole global batch) penalized by
    `f_c * log2(1 + reference_token_count * f_c)`, summed over components and scaled by
    `coeff`.

    `reference_token_count` (`a'`) is the token count the penalty is normalized against, so
    the curvature is invariant to batch size at a fixed firing rate. Setting it to the run's
    global `batch_size * seq_len` reproduces the implicit `B*T` the old rolled `beta` term
    baked inside its `log2`; coefficients then transfer as `coeff = old imp.coeff * old
    beta`. The `f=0 -> 0` cutoff is inherent to the form.
    """

    coeff: NonNegativeFloat
    reference_token_count: PositiveInt


class ImportanceMinimalityLossConfig(LossMetricConfig):
    """Config for the `L_p`-style importance-minimality penalty on upper-leaky CI values.

    `pnorm` is the exponent's full schedule (SPEC S9; canonical is the linear anneal
    `2.0 → 0.4`: `max_val=2.0` over knots `frac 1.0 → 0.2`). Its knots must keep
    `frac > 0` (asserted where the term is built) — a `p` touching 0 is never intended.
    `frequency` (when present) adds the batch-invariant frequency-minimality penalty over
    the same `(c + eps)^p` per-component sums.
    """

    type: Literal["ImportanceMinimalityLoss"] = "ImportanceMinimalityLoss"
    pnorm: ScheduleConfig
    frequency: FrequencyMinimalityConfig | None = None
    eps: NonNegativeFloat = 1e-12


class SmoothL0ImportanceMinimalityLossConfig(LossMetricConfig):
    """Geman–McClure smooth-L0 importance-minimality penalty on upper-leaky CI values.

    Per-value penalty `phi_gamma(c) = c^2 / (c^2 + gamma^2)` — a smooth approximation to
    the active-component count `1[c>0]`, exact only as `gamma -> 0` — fed through the same
    per-site `lp` mean (plus the optional `frequency` term) as `ImportanceMinimalityLoss`.
    Differs from the `L_p` penalty only in the per-value shape: `phi'(0) = 0` and
    `|phi'| <= 0.65/gamma` everywhere, so there is no singularity at the origin (no `eps`
    floor, no aggressive grad clip) — the gradient is localized on the threshold band
    `c ~ gamma/sqrt(3)` and redescends for clearly-on components.

    `gamma` is the width's full schedule (SPEC S9′); annealing it down (knots with
    decreasing `frac`) sharpens the count. Its knots must keep `frac > 0` (asserted
    where the term is built) — a `gamma` touching 0 is never intended.
    """

    type: Literal["SmoothL0ImportanceMinimalityLoss"] = "SmoothL0ImportanceMinimalityLoss"
    gamma: ScheduleConfig
    frequency: FrequencyMinimalityConfig | None = None


# The two imp-min penalties share the `coeff` + optional `frequency` surface and the
# `lp` mean aggregation; they differ only in the per-value penalty shape and its annealed
# parameter (`p` vs `gamma`). The trainer's single imp-min slot accepts either.
AnyImportanceMinimalityLossConfig = (
    ImportanceMinimalityLossConfig | SmoothL0ImportanceMinimalityLossConfig
)


class CIMaskedReconLossConfig(LossMetricConfig):
    type: Literal["CIMaskedReconLoss"] = "CIMaskedReconLoss"


class CIMaskedReconLayerwiseLossConfig(LossMetricConfig):
    type: Literal["CIMaskedReconLayerwiseLoss"] = "CIMaskedReconLayerwiseLoss"


class CIMaskedReconSubsetLossConfig(LossMetricConfig):
    type: Literal["CIMaskedReconSubsetLoss"] = "CIMaskedReconSubsetLoss"
    routing: Annotated[SubsetRoutingType, Field(discriminator="type")] = (
        UniformKSubsetRoutingConfig()
    )


class StochasticReconLossConfig(LossMetricConfig):
    type: Literal["StochasticReconLoss"] = "StochasticReconLoss"
    n_mask_samples: PositiveInt = 1


class StochasticReconLayerwiseLossConfig(LossMetricConfig):
    type: Literal["StochasticReconLayerwiseLoss"] = "StochasticReconLayerwiseLoss"
    n_mask_samples: PositiveInt = 1


class StochasticReconSubsetLossConfig(LossMetricConfig):
    type: Literal["StochasticReconSubsetLoss"] = "StochasticReconSubsetLoss"
    routing: Annotated[SubsetRoutingType, Field(discriminator="type")] = (
        UniformKSubsetRoutingConfig()
    )
    n_mask_samples: PositiveInt = 1


class StochasticHiddenActsReconLossConfig(LossMetricConfig):
    slow: ClassVar[bool] = True
    type: Literal["StochasticHiddenActsReconLoss"] = "StochasticHiddenActsReconLoss"
    n_mask_samples: PositiveInt = 1


class UnmaskedReconLossConfig(LossMetricConfig):
    type: Literal["UnmaskedReconLoss"] = "UnmaskedReconLoss"


class ChunkwiseSubsetReconLossConfig(LossMetricConfig):
    """Reconstruction loss that mirrors the 3-pool / 2-pool chunkwise subset recon.

    The decomposed sites (`model.target_module_paths`, in order) are grouped into
    chunks of `sites_per_chunk`; each chunk runs `SubsetReconPlan(routing, n_samples)`
    — one masked forward per generated routing, all the chunk's sites swapped in
    with a per-position routing draw — and the recon is KL against the clean logits. The
    total is the mean over all chunk forwards of `recon_loss / n_positions`, matching the
    2-pool's per-step recon.

    The JAX single-pool trainer implements this natively: `objective.build_objective`
    maps this `type` onto `recon.make_plan(live_groups(sites, sites_per_chunk), routing,
    StochasticSources(), n_samples)`, and the jitted step runs the chunk forwards
    directly — no vendored `LMComponentModel` or lab recon-plan machinery is involved.
    `routing` is honoured as authored; `recon.subset_chunk_plan` is the uniform-k
    parameterization the parity fixtures pin, not the path this config takes.
    """

    type: Literal["ChunkwiseSubsetReconLoss"] = "ChunkwiseSubsetReconLoss"
    sites_per_chunk: PositiveInt
    routing: Annotated[SubsetRoutingType, Field(discriminator="type")] = (
        UniformKSubsetRoutingConfig()
    )
    n_samples: PositiveInt = 1


PGDInitStrategy = Literal["random", "ones", "zeroes"]

SourceShape = Literal["c", "bc", "sc", "bsc"]
"""The stored adversarial-source shape, spelled over the waist axes in tensor order —
each letter names an axis the source keeps FULL; a missing letter is a size-1 broadcast
axis (the rank always matches the waist, so the elementwise combine broadcasts):

  positionless target:  `c (1, C+1)` · `bc (B, C+1)`   (`sc`/`bsc` are invalid — they
                        name a position axis the target lacks)
  positioned target:    `c (1, 1, C+1)` · `bc (B, 1, C+1)` — one source per batch
                        element shared over positions — · `sc (1, P, C+1)` ·
                        `bsc (B, P, C+1)`

One vocabulary for BOTH adversaries: persistent PGD implements all four; per-step
(fresh) PGD implements `c`/`bc`/`bsc` and rejects `sc` at validation."""


class PGDConfig(LossMetricConfig):
    """Shared base for per-step PGD loss configs."""

    init: PGDInitStrategy
    step_size: PositiveFloat
    n_steps: NonNegativeInt
    source_shape: SourceShape

    @model_validator(mode="after")
    def validate_source_shape_implemented(self) -> Self:
        if self.source_shape == "sc":
            raise ValueError("per-step PGD does not implement `sc` (persistent PGD does)")
        return self


class PGDReconLossConfig(PGDConfig):
    slow: ClassVar[bool] = False
    type: Literal["PGDReconLoss"] = "PGDReconLoss"


class PGDReconLayerwiseLossConfig(PGDConfig):
    type: Literal["PGDReconLayerwiseLoss"] = "PGDReconLayerwiseLoss"


class PGDReconSubsetLossConfig(PGDConfig):
    type: Literal["PGDReconSubsetLoss"] = "PGDReconSubsetLoss"
    routing: Annotated[SubsetRoutingType, Field(discriminator="type")] = (
        UniformKSubsetRoutingConfig()
    )


class AdamPGDConfig(BaseConfig):
    """Adam-style PGD optimizer config — the only implemented persistent-PGD optimizer."""

    type: Literal["adam"] = "adam"
    beta1: Probability = Field(default=0.9, description="Adam beta1 for masks")
    beta2: Probability = Field(default=0.999, description="Adam beta2 for masks")
    eps: NonNegativeFloat = Field(default=1e-8, description="Adam epsilon for masks")
    lr_schedule: ScheduleConfig


class PersistentPGDLossConfig(LossMetricConfig):
    """Shared adversary fields for the persistent-PGD loss terms (SPEC §4.4–4.5): the
    Adam-ascended source bundle's optimizer, stored shape, dtype, and warmup. Sources are
    clamped to `[0, 1]` after each step — the only implemented parameterization."""

    optimizer: AdamPGDConfig
    source_shape: SourceShape
    source_dtype: Literal["float32", "bfloat16"] = "float32"
    """Storage dtype for the persistent PPGD source tensors AND their Adam moments
    (`m`/`v`). `float32` (default) is SPEC N1 (fp32 SRC_STEP moments) and the only
    oracle-parity path. `bfloat16` halves the resident source+moment footprint (it scales
    with total source elements — dominant at large site counts) at some numerical risk: the
    second-moment `v` accumulates squared grads, which can underflow in bf16 for small
    grads — opt in only as an experiment."""
    n_warmup_steps: NonNegativeInt = Field(
        default=0,
        description=(
            "Extra inner PGD source-optimization steps on each train batch before the final loss"
            " computation."
        ),
    )


class PersistentPGDReconLossConfig(PersistentPGDLossConfig):
    """Persistent-PGD recon loss: the adversary's masks routed to all sites every
    forward."""

    type: Literal["PersistentPGDReconLoss"] = "PersistentPGDReconLoss"


class MergedStochasticSubsetPPGDReconLossConfig(PersistentPGDLossConfig):
    """ONE masked forward serving both recon pressures (SPEC S10' variation): each batch
    element is assigned adversarial with probability `adv_fraction` (mask sources = the
    persistent-PGD adversary's, every site routed) or stochastic otherwise (fresh
    `U[0,1]` sources, routed per `routing`) — the whole sequence takes one family, so no
    sample's loss is scored against a mixed-family attention context. `adv_fraction` is a
    `ScheduleConfig`, evaluated per step like the imp-min pnorm — a constant is the plain
    merge; a ramp anneals the adversarial share over training. `coeff` is the TOTAL:
    coeff 1.0 + constant adv_fraction 0.5 replaces the canonical 0.5 stochastic + 0.5
    persistent-PGD pair in expectation. Carries the persistent-adversary fields; one
    source bundle feeds this one term (SPEC S23) and the S14' final ascent rides its
    backward — no extra forward."""

    type: Literal["MergedStochasticSubsetPPGDReconLoss"] = "MergedStochasticSubsetPPGDReconLoss"
    adv_fraction: ScheduleConfig
    routing: SubsetRoutingType = Field(default_factory=UniformKSubsetRoutingConfig)

    @model_validator(mode="after")
    def validate_adv_fraction_is_probability(self) -> Self:
        # `frac` knots live in [0, 1] with the peak frac=1.0 attained, so the curve's
        # range is exactly [min_frac, 1.0] · max_val — the peak check is sufficient.
        if self.adv_fraction.max_val > 1.0:
            raise ValueError(f"adv_fraction must stay within [0, 1]: {self.adv_fraction}")
        return self


# ---------------------------------------------------------------------------
# Eval-metric configs
#
# Every eval metric declares `slow: ClassVar[bool]` beside its own definition. An eval
# costs what it costs wherever it is bound, so the tier travels WITH the metric rather
# than being assigned by whichever family binds it. `ClassVar` is what makes it
# unauthorable: pydantic ignores it, and `extra="forbid"` then refuses a YAML `slow:` key.
# No default, so a new metric cannot skip the decision — swept at import in
# `experiments.eval_config`, which also refuses a tier that is merely inherited.
# ---------------------------------------------------------------------------


class CIHiddenActsReconLossConfig(BaseConfig):
    slow: ClassVar[bool] = True
    type: Literal["CIHiddenActsReconLoss"] = "CIHiddenActsReconLoss"


class CIHistogramsConfig(BaseConfig):
    """`n_batches_accum=None` accumulates every batch in the eval pass. `density_heatmap_n_bins`
    opts into the per-token per-component CI density heatmap (an on-device bincount into that
    many log-spaced `[1e-9, 1]` bands sharing the same forward, accumulated over EVERY batch);
    `None` disables it."""

    slow: ClassVar[bool] = True
    type: Literal["CIHistograms"] = "CIHistograms"
    n_batches_accum: PositiveInt | None
    density_heatmap_n_bins: PositiveInt | None = None


class CI_L0Config(BaseConfig):
    """`groups` maps `{group_name: [fnmatch-style layer pattern, ...]}`.

    Matching layers' L0s are summed into the group and logged under the group's name.
    """

    slow: ClassVar[bool] = False
    type: Literal["CI_L0"] = "CI_L0"
    groups: dict[str, list[str]] | None
    ci_alive_threshold: float = 0.0


class CIMeanPerComponentConfig(BaseConfig):
    slow: ClassVar[bool] = True
    type: Literal["CIMeanPerComponent"] = "CIMeanPerComponent"


class ComponentActivationDensityConfig(BaseConfig):
    slow: ClassVar[bool] = True
    type: Literal["ComponentActivationDensity"] = "ComponentActivationDensity"
    ci_alive_threshold: float = 0.0


class IdentityCITargetSpec(BaseConfig):
    """A layer expected to produce an Identity CI pattern over `n_features` features."""

    layer_pattern: str
    n_features: PositiveInt


class DenseCITargetSpec(BaseConfig):
    """A layer expected to produce a Dense CI pattern with `k` active components."""

    layer_pattern: str
    k: PositiveInt


class IdentityCIErrorConfig(BaseConfig):
    """`identity_ci` / `dense_ci` list layers expected to produce Identity / Dense patterns."""

    slow: ClassVar[bool] = True
    type: Literal["IdentityCIError"] = "IdentityCIError"
    identity_ci: list[IdentityCITargetSpec] | None
    dense_ci: list[DenseCITargetSpec] | None


class _PermutationPlotsBaseConfig(BaseConfig):
    """fnmatch patterns for layers permuted to align with the corresponding target solution.

    `identity_patterns` and `dense_patterns` are matched separately against the model.
    """

    identity_patterns: list[str] | None
    dense_patterns: list[str] | None


class PermutedCIPlotsConfig(_PermutationPlotsBaseConfig):
    slow: ClassVar[bool] = True
    type: Literal["PermutedCIPlots"] = "PermutedCIPlots"


class UVPlotsConfig(_PermutationPlotsBaseConfig):
    slow: ClassVar[bool] = True
    type: Literal["UVPlots"] = "UVPlots"


# ---------------------------------------------------------------------------
# Top-level PD configs
# ---------------------------------------------------------------------------


class AdamWOptimizerConfig(BaseConfig):
    type: Literal["adamw"] = "adamw"
    lr_schedule: ScheduleConfig = Field(..., description="Learning rate schedule")
    weight_decay: NonNegativeFloat = Field(default=0.0, description="AdamW weight decay")
    betas: tuple[Probability, Probability] = Field(
        default=(0.9, 0.999), description="AdamW (beta1, beta2)"
    )
    grad_clip_norm: PositiveFloat | None = Field(
        default=None,
        description="If set, clip the grad norm of this group's parameters to this value",
    )


class MuonOptimizerConfig(BaseConfig):
    """Muon (`optax.contrib.muon`): Newton-Schulz-orthogonalized momentum for the group's
    matrix leaves; the rest fall back to Adam(0.9, 0.999) at the same LR. Experimental
    (non-canonical). Which leaves are matrices is per-group (`run_state.build_optimizers`):
    the V/U components tree is all-2D (fallback never fires); the chunkwise CI fn is
    per-chunk stacks, so its 3D leaves are muon'd over the trailing two axes (chunk axis
    batched) and its 2D bias stacks take the fallback; the MLP CI fns use the plain 2D rule."""

    type: Literal["muon"]
    lr_schedule: ScheduleConfig = Field(..., description="Learning rate schedule")
    beta: Probability = Field(
        default=0.95, description="Momentum decay for the orthogonalized update"
    )
    consistent_rms: PositiveFloat | None = Field(
        default=None,
        description=(
            "If set, scale updates by `sqrt(max(fan_in, fan_out)) * consistent_rms` so update"
            " RMS is shape-independent (0.2 ~ AdamW's empirical RMS, making the AdamW LR"
            " transferable). If None, optax's width scaling `sqrt(max(1, fan_out / fan_in))`."
        ),
    )
    weight_decay: NonNegativeFloat = Field(default=0.0, description="Weight decay")
    grad_clip_norm: PositiveFloat | None = Field(
        default=None,
        description="If set, clip the grad norm of this group's parameters to this value",
    )
    impl: Literal["optax", "stacked"] = Field(
        default="optax",
        description=(
            "NS implementation. `optax` = per-leaf `optax.contrib.muon` (the reference"
            " semantics, SPEC S20). `stacked` = same-shape leaves batched into"
            " one NS with the stack axis sharded over (replicate, fsdp) — device-local"
            " orthogonalization, no per-iteration collectives (`muon_stacked.py`); same"
            " trajectory up to float reassociation (the SPEC D4 tolerance class)."
        ),
    )
    ns_steps: PositiveInt = Field(
        default=5, description="Newton-Schulz iterations (optax default 5; fewer = cheaper/looser)"
    )
    ns_dtype: Literal["float32", "bfloat16"] = Field(
        default="float32",
        description=(
            "Dtype of the NS orthogonalization only (masters/momentum stay fp32 per N1);"
            " bfloat16 halves NS compute+comm (the Kimi recipe). `stacked` impl only."
        ),
    )


def _default_optimizer_type_adamw(data: object) -> object:
    """AdamW is the canonical optimizer, so a config without `type` (every config predating
    the muon gate, and the common case going forward) discriminates to it."""
    if isinstance(data, dict) and "type" not in data:
        return {**data, "type": "adamw"}
    return data


AnyOptimizerConfig = Annotated[
    AdamWOptimizerConfig | MuonOptimizerConfig,
    Discriminator("type"),
    BeforeValidator(_default_optimizer_type_adamw),
]


AnyLossMetricConfig = Annotated[
    ChunkwiseSubsetReconLossConfig
    | CIMaskedReconLayerwiseLossConfig
    | CIMaskedReconLossConfig
    | CIMaskedReconSubsetLossConfig
    | FaithfulnessLossConfig
    | ImportanceMinimalityLossConfig
    | MergedStochasticSubsetPPGDReconLossConfig
    | PersistentPGDReconLossConfig
    | PGDReconLayerwiseLossConfig
    | PGDReconLossConfig
    | PGDReconSubsetLossConfig
    | SmoothL0ImportanceMinimalityLossConfig
    | StochasticHiddenActsReconLossConfig
    | StochasticReconLayerwiseLossConfig
    | StochasticReconLossConfig
    | StochasticReconSubsetLossConfig
    | UnmaskedReconLossConfig,
    Discriminator("type"),
]


RuleConfig = dict[str, str | list[str] | None]
"""One placement row as configured: semantic axis name -> mesh axis, ordered mesh axes,
or null (replicate). Axis-name keys are free-form — semantic names are declared by the
code that owns each tensor, not enumerated here."""


class ParamsPlacementConfig(BaseConfig):
    """The trainable V/U placement rows of an explicit table (`placement.ParamsPlacement`)."""

    persist: RuleConfig
    zero1: RuleConfig | None = None
    """The OPT-IN row for shape groups whose stack does not tile the persist stack
    sharding. A bidirectional claim, checked at config build (`placement.from_config`):
    absence is strictness (a non-tiling group is a loud error), and declaring it when
    every group tiles is equally an error (a declared-but-unreachable arm)."""
    forward: RuleConfig


class PlacementTableConfig(BaseConfig):
    """An explicit placement table (`runtime.sharding`), mirroring the typed
    `placement.PlacementRules`. The row vocabulary is CLOSED (extra keys are a parse
    error); rule values are free-form axis-name -> mesh-axes mappings, where YAML list
    order is semantics (nested-axis linearization — PLACEMENT_DESIGN.md lesson 4)."""

    params: ParamsPlacementConfig
    activations: RuleConfig


ComponentUpdateScaling = Literal[
    "none",
    "c_covariant",
    "c_covariant_balanced",
    "function_covariant_balanced",
]


class ControllerObservableConfig(BaseConfig):
    """Affine units for the authored reconstruction constraint observable.

    ``metric_key`` names one scalar emitted by an authored eval operation. The controller
    reads ``(metric - offset) / scale``; both constants are fixed for the run, so the model
    cannot game a moving normalization. A TMS run may use a fixed zero-mask MSE ceiling as
    ``scale``; an LM may instead author raw excess KL with ``offset`` equal to its unmasked
    floor and ``scale=1``.
    """

    metric_key: str
    offset: float = 0.0
    scale: PositiveFloat


class ReconBudgetControlConfig(BaseConfig):
    """Slow host-side control of the combined minimality force.

    The fast primal/CI/adversary dynamics remain inside the ordinary train step. This
    config has no controller learning rate: settled windows bracket the largest feasible
    complexity scale, so changing model scale does not introduce another tuned gain.
    """

    observable: ControllerObservableConfig
    tau: float
    noise_margin: NonNegativeFloat
    initial_complexity_scale: PositiveFloat
    max_complexity_scale: PositiveFloat
    control_after_step: NonNegativeInt = 0
    """Number of completed primal-training steps before the controller may observe or
    update. The initial complexity scale remains fixed through this boundary, allowing
    authored schedules such as gamma/LR anneals to settle before the outer loop starts."""
    expand_factor: float = 4.0
    resolution_factor: float = 1.05
    dwell_windows: PositiveInt = 3
    plateau_rtol: NonNegativeFloat = 0.02
    probe_improvement_rtol: NonNegativeFloat = 0.02
    birth_improvement_rtol: NonNegativeFloat = 0.02
    probe_cooldown_windows: NonNegativeInt = 6
    max_rejected_probes: PositiveInt = 3
    settle_points: PositiveInt = 3
    settle_rtol: NonNegativeFloat = 0.02
    settle_atol: NonNegativeFloat = 1e-8
    protect_windows: PositiveInt = 3
    initial_active_slots: dict[str, PositiveInt] | None = None
    initial_split_copies: PositiveInt = 1
    """Experimental discovery falsifier: replace every authored initial SVD slot by
    this many identical rank-one children before the first train step. Their products
    sum exactly to the original slot and their CI heads are copied. This isolates
    whether function-preserving live-width expansion can escape the C-flat SVD basin;
    it is not the eventual settle-gated split controller."""
    birth_block_cap: PositiveInt = 1
    """Maximum scratch/birth slots priced per site in one lifecycle transaction. One
    preserves serial v1; larger values are a systems work cap, not a selected rank."""
    birth_validation_repeats: PositiveInt | None = None
    """Independent fresh-referee signs required by block birth. Required exactly when
    ``birth_block_cap > 1``; no raw singular-value floor enters the decision."""

    @model_validator(mode="after")
    def validate_multiplicative_steps(self) -> Self:
        assert self.expand_factor > 1.0, self.expand_factor
        assert self.resolution_factor > 1.0, self.resolution_factor
        assert self.settle_points >= 2, self.settle_points
        assert (self.birth_block_cap == 1) == (self.birth_validation_repeats is None), (
            self.birth_block_cap,
            self.birth_validation_repeats,
        )
        return self


class PDConfig(BaseConfig):
    """Algorithm specification: seed, losses, optimizers, faithfulness warmup.

    Domain-agnostic — the target-coupled apparatus (which sites to decompose + the CI-fn
    arch) lives in the per-domain `decomposition` section, not here. Flipping any field here
    changes what algorithm runs. Pair with `Cadence` (when to emit) and `RunSink` (where
    output goes) when running the trainer (`param_decomp.core.run`); the compute substrate
    reaches the engine unpacked into primitives, never as a config object.
    """

    # --- General ---
    seed: int = Field(
        default=0,
        description="Random seed for reproducibility, including LM dataset shuffling.",
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
    components_optimizer: AnyOptimizerConfig = Field(
        ..., description="Optimizer config for the component (LinearComponent etc.) parameters"
    )
    ci_fn_optimizer: AnyOptimizerConfig = Field(
        ..., description="Optimizer config for the CI function parameters"
    )
    component_update_scaling: ComponentUpdateScaling = Field(
        default="none",
        description=(
            "Post-optimizer scaling for the bilinear V/U factors. `c_covariant` multiplies "
            "V updates by sqrt(d_in/C) and U updates by d_in/C, so the first Adam step of "
            "the represented matrix is approximately invariant to overcomplete C. "
            "`c_covariant_balanced` fixes the exact V/U scale gauge at equal factor norms "
            "and scales both updates by (d_in/C)^(3/4). "
            "`function_covariant_balanced` uses a product-space LR: it balances the gauge "
            "and divides both updates by C^(3/4) times the derived matrix-shape factor, "
            "making the first represented-matrix Adam step covariant in C, width, and "
            "aspect ratio under canonical initialization."
        ),
    )
    recon_budget_control: ReconBudgetControlConfig | None = Field(
        default=None,
        description=(
            "Optional settled host controller over the combined importance+frequency "
            "minimality force. Its constraint observable is an authored eval scalar."
        ),
    )
    component_init: Literal["random", "svd_null_tail"] = Field(
        default="random",
        description=(
            "V/U + CI-head initialization. `svd_null_tail` starts the first min(d_in, d_out) "
            "slots at an exact SVD factorization of the target weight with CI pinned 1, and "
            "every extra slot as an exact null (U row zero, prefix-stable random V column, CI "
            "pinned 0) — so every C >= rank(W) starts from the same represented matrix, "
            "reconstruction, and L0 (the C-nesting property the random init lacks)."
        ),
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


class DenseLogPhase(BaseConfig):
    """Denser train-log period for the first `until_step` steps, then `Cadence`'s
    steady `train_log_every` takes over. Early training has the fastest dynamics
    (faithfulness warmup, the sharp initial loss drop), so denser sampling there is
    where the signal is; the per-log overhead is a few scalar cross-pool reductions,
    negligible against the step. `until_step` is exclusive."""

    every: PositiveInt
    until_step: PositiveInt


class KeepLastNCheckpoints(BaseConfig):
    """Keep only the `n` most recent `ckpts/<step>/` directories; every save deletes
    whatever falls out of that window."""

    kind: Literal["keep_last"] = "keep_last"
    n: PositiveInt


class KeepAllCheckpoints(BaseConfig):
    """Keep every checkpoint the run writes — the whole trajectory stays on disk."""

    kind: Literal["keep_all"] = "keep_all"


CheckpointRetention = Annotated[KeepLastNCheckpoints | KeepAllCheckpoints, Discriminator("kind")]
"""Which of a run's written checkpoints survive on disk. The trainer always checkpoints — on
`Cadence.save_every`, on SIGTERM, and at the final step — so this axis only prunes what has
already been written, and there is deliberately no keep-nothing arm."""


class Cadence(BaseConfig):
    """Rhythm of non-eval loop emissions: train-log and checkpoint periods.

    Held separately from `RunSink` so the sink only owns *where* output goes; `Cadence`
    owns *when* train logs and checkpoints fire. Eval timing lives on `EvalLoop`,
    alongside the runtime objects it depends on. The trainer (`param_decomp.core.run`) loop
    always checkpoints at the final step regardless of `save_every`.
    """

    train_log_every: PositiveInt
    save_every: PositiveInt
    checkpoint_retention: CheckpointRetention
    """Which orbax `ckpts/<step>/` checkpoints survive each checkpoint write. Under
    `keep_last` the retained window always contains the newest checkpoint, so the
    final-step one is never pruned."""
    dense_log_phase: DenseLogPhase | None = None
    """Optional denser logging for early training; `None` means a flat `train_log_every`."""

    def should_log_train(self, step: int) -> bool:
        if self.dense_log_phase is not None and step < self.dense_log_phase.until_step:
            return step % self.dense_log_phase.every == 0
        return step % self.train_log_every == 0


# ---------------------------------------------------------------------------
# Run-level config (logging + fine-tune lineage)
# ---------------------------------------------------------------------------


class WandbConfig(BaseConfig):
    """Wandb logging settings. Presence on `ExperimentConfig` opts in; omit to skip wandb."""

    project: str
    entity: str | None = None
    group: str | None = None
    """Wandb UI group; None = ungrouped."""
    tags: list[str] = Field(default_factory=list)
    """Wandb tags; empty = untagged."""


class ResumeProvenance(BaseConfig):
    """Fine-tune lineage: a fresh run initialized from a PARENT decomposition. Lives on
    `ExperimentConfig`.

    A fine-tune run gets its own `run_id` / `launch_config.yaml` / `ckpts/`; this records the
    parent it forked from. The JAX trainer (SPEC S33) loads the parent checkpoint's
    V/U + ci_fn onto a fresh reference state and trains a clean schedule from step 0
    (fresh optimizer / sources) under the new config — only when the run's own `ckpts/`
    is empty (a subsequent SLURM requeue resumes from the run's own dir, ignoring
    provenance). The structure (sites / C / ci-fn arch) must match the parent; only
    LR / coeffs / eps / seq / batch / steps may change. Provenance flows into
    `launch_config.yaml` and `wandb.config` so the lineage is visible in the wandb UI. A run with
    `resume_provenance is None` is a fresh-from-init run.
    """

    parent_run_dir: Path
    """Path to the parent run's directory (the dir that contains `ckpts/<parent_step>/`)."""

    parent_step: int
    """The parent's orbax `ckpts/<step>/` checkpoint step to initialize V/U + ci_fn from."""


# ---------------------------------------------------------------------------
# wandb.config shaping
# ---------------------------------------------------------------------------

METRIC_SHORT_NAMES: dict[str, str] = {
    "CIMaskedReconLayerwiseLoss": "CIMaskReconLayer",
    "CIMaskedReconLoss": "CIMaskRecon",
    "CIMaskedReconSubsetLoss": "CIMaskReconSub",
    "FaithfulnessLoss": "Faith",
    "ImportanceMinimalityLoss": "ImpMin",
    "SmoothL0ImportanceMinimalityLoss": "SmoothL0ImpMin",
    "PersistentPGDReconLoss": "PersistPGDRecon",
    "PGDReconLayerwiseLoss": "PGDReconLayer",
    "PGDReconLoss": "PGDRecon",
    "PGDReconSubsetLoss": "PGDReconSub",
    "StochasticHiddenActsReconLoss": "StochHiddenActRecon",
    "StochasticReconLayerwiseLoss": "StochReconLayer",
    "StochasticReconLoss": "StochRecon",
    "StochasticReconSubsetLoss": "StochReconSub",
    "UnmaskedReconLoss": "UnmaskedRecon",
    "ArithmeticCIGrid": "ArithCIGrid",
    "CEandKLLosses": "CEandKL",
    "CIHiddenActsReconLoss": "CIHiddenActRecon",
    "CIHistograms": "CIHist",
    "CI_L0": "CI_L0",
    "CIMaskedAttnPatternsReconLoss": "CIAttnRecon",
    "CIMeanPerComponent": "CIMeanPerComp",
    "ComponentActivationDensity": "CompActDens",
    "IdentityCIError": "IdCIErr",
    "PermutedCIPlots": "PermCIPlots",
    "StochasticAttnPatternsReconLoss": "StochAttnRecon",
    "UVPlots": "UVPlots",
}


def flatten_typed_lists(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested lists-of-typed-dicts (loss/eval metric lists) into queryable flat
    keys addressed by metric `short_name`, returning a copy with the raw lists dropped.

    Example: `pd.loss_metrics: [{type: "ImportanceMinimalityLoss", coeff: 0.1}]`
    becomes `pd.loss_metrics.ImpMin.coeff: 0.1`, and the raw `pd.loss_metrics` list is
    removed so wandb doesn't also log it as an opaque JSON blob. A metric type with no
    entry in `METRIC_SHORT_NAMES` falls back to its raw type string.
    """
    flattened: dict[str, Any] = {}

    def is_typed_list(obj: Any) -> bool:
        return (
            isinstance(obj, list)
            and len(obj) > 0
            and all(isinstance(x, dict) and "type" in x for x in obj)
        )

    def walk(obj: Any, path: str) -> None:
        if isinstance(obj, dict):
            for key in list(obj.keys()):
                child = obj[key]
                child_path = f"{path}.{key}" if path else key
                if is_typed_list(child):
                    for entry in child:
                        short = METRIC_SHORT_NAMES.get(entry["type"], entry["type"])
                        for k, v in entry.items():
                            if k == "type":
                                continue
                            flattened[f"{child_path}.{short}.{k}"] = v
                    del obj[key]
                else:
                    walk(child, child_path)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                walk(item, f"{path}.{i}")

    out = copy.deepcopy(config_dict)  # walk dels nested keys; never mutate the caller's dict
    walk(out, "")
    out.update(flattened)
    return out
