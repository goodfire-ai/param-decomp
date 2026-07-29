"""The torch-free pydantic config schema for the algorithm core.

Every algorithm-level config class lives here (or in the sibling `base_config` /
`schedule` modules): routing, the explicit (toy) site spec, loss-metric configs,
eval-metric configs, the top-level `PDConfig` / `RuntimeConfig` / `Cadence`, and the
`wandb.config` shaping helpers. Depends only on pydantic / numpy / pyyaml /
annotated-types (via `base_config`), so non-trainer consumers validate the same
YAML run configs without pulling jax/wandb.

Experiment-level schema (the `ExperimentConfig` base and its LM / TMS / ResidMLP
subclasses, each binding concrete `target`/`decomposition`/`data` sections) lives
lab-side under `param_decomp/experiments/` — including the authored
`decomposition.ci` configs AND the tiled LM site specs, which speak each domain's
vocabulary. Core carries only the RESOLVED CI-fn arches (`ci_fn.py`) and the
resolved flat sites.
"""

import copy
from pathlib import Path
from typing import Annotated, Any, Literal, Self

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

    `pnorm` is the exponent's full schedule (SPEC S9; canonical is linear `2.0 → 0.4`:
    `start_val=2.0, fn_type=linear, final_val_frac=0.2`; constant-`p` is
    `fn_type=constant`). Warmup is refused where the term is built — a `p` ramping from 0
    is never intended. `frequency` (when present) adds the batch-invariant
    frequency-minimality penalty over the same `(c + eps)^p` per-component sums.
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

    `gamma` is the width's full schedule (SPEC S9′); annealing it down (e.g.
    `fn_type=linear, final_val_frac < 1`) sharpens the count. Warmup is refused where
    the term is built.

    With `normalize_at_one`, `phi` is rescaled by `(1 + gamma^2)` so a fully-on component
    (`c = 1`) contributes exactly 1 regardless of `gamma`. Otherwise `phi(1) = 1/(1+gamma^2)`
    grows as `gamma` anneals, silently ramping the effective `coeff` on saturated components.
    """

    type: Literal["SmoothL0ImportanceMinimalityLoss"] = "SmoothL0ImportanceMinimalityLoss"
    gamma: ScheduleConfig
    frequency: FrequencyMinimalityConfig | None = None
    normalize_at_one: bool = False


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

    The JAX single-pool trainer implements this natively: `recon.build_loss_terms`
    maps this `type` onto `recon.subset_chunk_plan` (a parameterization of the one
    `chunkwise_plan` builder), and the jitted step runs the chunk forwards directly —
    no vendored `LMComponentModel` or lab recon-plan machinery is involved.
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


# Stored run configs carry older generations of the per-step field: `mask_scope` (with
# the original verbose value names before that). Alias exactly the forms that exist in
# stored data so old runs keep loading (`unique_per_datapoint` occurs only in LM runs,
# hence `bsc`). Delete once stored runs are migrated.
_LEGACY_MASK_SCOPE_VALUES = {
    "shared_across_batch": "c",
    "unique_per_datapoint": "bsc",
}


def _alias_legacy_mask_scope_field(data: Any) -> Any:
    if isinstance(data, dict) and "mask_scope" in data and "source_shape" not in data:
        data = dict(data)
        value = data.pop("mask_scope")
        data["source_shape"] = _LEGACY_MASK_SCOPE_VALUES.get(value, value)
    return data


class PGDConfig(LossMetricConfig):
    """Shared base for per-step PGD loss configs."""

    _alias_legacy_mask_scope = model_validator(mode="before")(_alias_legacy_mask_scope_field)

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


# Stored run configs carry two older generations of the field: `scope` as a nested
# `{type: ...}` object (`sc`/`bsc`) and before that verbose type names. Alias exactly
# the forms that exist in stored data so old runs keep loading. Delete once stored runs
# are migrated.
_LEGACY_SCOPE_VALUES = {
    "broadcast_across_batch": "sc",
    "per_batch_per_position": "bsc",
}


def _alias_legacy_scope_field(data: Any) -> Any:
    if isinstance(data, dict) and "scope" in data and "source_shape" not in data:
        data = dict(data)
        scope = data.pop("scope")
        if isinstance(scope, dict):
            scope = scope["type"]
        data["source_shape"] = _LEGACY_SCOPE_VALUES.get(scope, scope)
    return data


class PersistentPGDLossConfig(LossMetricConfig):
    """Shared adversary fields for the persistent-PGD loss terms (SPEC §4.4–4.5): the
    Adam-ascended source bundle's optimizer, stored shape, dtype, and warmup. Sources are
    clamped to `[0, 1]` after each step — the only implemented parameterization."""

    _alias_legacy_scope = model_validator(mode="before")(_alias_legacy_scope_field)

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

    @model_validator(mode="before")
    @classmethod
    def _strip_removed_fields(cls, data: object) -> object:
        # Shared-storage shim: stored run configs carry removed fields whose only
        # supported value was inlined. Strip them so those configs still load; any
        # other value never had an implementation -> reject.
        if isinstance(data, dict):
            if "use_sigmoid_parameterization" in data:
                assert not data.pop("use_sigmoid_parameterization"), (
                    "use_sigmoid_parameterization was removed (clamp-only)"
                )
            if "n_samples" in data:
                assert data.pop("n_samples") == 1, (
                    "n_samples was removed (route-all + one persistent source bundle make"
                    " every draw identical, so only 1 was ever meaningful)"
                )
        return data

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
        start = self.adv_fraction.start_val
        end = start * self.adv_fraction.final_val_frac
        if not (start <= 1.0 and end <= 1.0):
            raise ValueError(f"adv_fraction must stay within [0, 1]: {self.adv_fraction}")
        return self


# ---------------------------------------------------------------------------
# Eval-metric configs
# ---------------------------------------------------------------------------


class CEandKLLossesConfig(BaseConfig):
    """`rounding_threshold` binarises CI for the `*_rounded_masked` variant (`ci > threshold`)."""

    type: Literal["CEandKLLosses"] = "CEandKLLosses"
    rounding_threshold: float


class CIHiddenActsReconLossConfig(BaseConfig):
    type: Literal["CIHiddenActsReconLoss"] = "CIHiddenActsReconLoss"


class CIHistogramsConfig(BaseConfig):
    """`n_batches_accum=None` accumulates every batch in the eval pass. `density_heatmap_n_bins`
    opts into the per-token per-component CI density heatmap (an on-device bincount into that
    many log-spaced `[1e-9, 1]` bands sharing the same forward, accumulated over EVERY batch);
    `None` disables it."""

    type: Literal["CIHistograms"] = "CIHistograms"
    n_batches_accum: PositiveInt | None
    density_heatmap_n_bins: PositiveInt | None = None


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

    n_heads: PositiveInt
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
    n_mask_samples: PositiveInt = 1


class CIMeanPerComponentConfig(BaseConfig):
    type: Literal["CIMeanPerComponent"] = "CIMeanPerComponent"


class ComponentActivationDensityConfig(BaseConfig):
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
    type: Literal["PermutedCIPlots"] = "PermutedCIPlots"


class UVPlotsConfig(_PermutationPlotsBaseConfig):
    type: Literal["UVPlots"] = "UVPlots"


class ArithmeticCIGridConfig(BaseConfig):
    """Per-component causal-importance heatmaps over an `a x b` arithmetic operand grid.

    The probe is a SPEC, not a filesystem artifact: the `[a_range] x [b_range]` grid of
    `"<a><op><b>="` prompts is built in-memory at startup from the target's tokenizer
    (`experiments/lm/arithmetic_probe.py`), so configs stay cluster-portable. For each
    threshold in `thresholds`, every component alive on the probe (max CI over the grid
    `>` threshold) is counted in n_alive, and the `top_k` most-active (by max CI) get
    `a x b` CI + activation heatmaps. A figure tier — renders on `eval.slow_every`. Any
    alive beyond `top_k` are reported via n_dropped (not silently cut)."""

    type: Literal["ArithmeticCIGrid"] = "ArithmeticCIGrid"
    operation: Literal["add", "sub", "mul"] = "add"
    a_range: tuple[int, int] = (1, 100)
    b_range: tuple[int, int] = (1, 100)
    thresholds: list[float] = Field(default_factory=lambda: [0.1])
    top_k: PositiveInt = 24


AnyEvalMetricConfig = Annotated[
    ArithmeticCIGridConfig
    | CEandKLLossesConfig
    | CIHiddenActsReconLossConfig
    | CIHistogramsConfig
    | CI_L0Config
    | CIMaskedAttnPatternsReconLossConfig
    | CIMeanPerComponentConfig
    | ComponentActivationDensityConfig
    | IdentityCIErrorConfig
    | PermutedCIPlotsConfig
    | PGDReconLossConfig
    | StochasticAttnPatternsReconLossConfig
    | StochasticHiddenActsReconLossConfig
    | UVPlotsConfig,
    Discriminator("type"),
]


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


class ProfileConfig(BaseConfig):
    """Profiling/instrumentation toggles the trainer (`run.py`) reads DIRECTLY off the config.

    A profiling run is a CONFIG, not an env hack: the trainer parses its `launch_config.yaml` and
    reads these fields, so the pinned config records exactly which hooks ran. All hooks are
    DEFAULT-OFF; the empty `ProfileConfig()` enables nothing.

    `mem_profile` (static + runtime memory analysis, then exits), `time_steps` (per-step wall
    breakdown), `trace` (perfetto trace over `trace_start`..`trace_start+trace_steps`),
    `profile_max_events` (raise the perfetto GPU-activity event cap), `async_test`,
    `leaf_bench`, `no_checkpoint` (skip ALL saves — throwaway profiling only).
    """

    mem_profile: bool = False
    time_steps: bool = False
    trace: bool = False
    trace_start: PositiveInt | None = None
    """First step of the perfetto trace window; `None` lets `run.py` default it to the first
    post-warmup step. Only meaningful when `trace` is set."""
    trace_steps: PositiveInt | None = None
    """Number of steps to trace; `None` lets `run.py` default it (3). Only with `trace`."""
    profile_max_events: PositiveInt | None = None
    """Raise the perfetto GPU-activity event cap (`gpu_max_activity_api_events`) so a full
    step's kernels fit under the exporter's 1M-event limit. `None` leaves the default. Only
    with `trace`."""
    async_test: bool = False
    leaf_bench: bool = False
    no_checkpoint: bool = False


class LaunchEnv(BaseConfig):
    """The process-environment surface a SLURM-launched rank runs with — the XLA *client*
    knobs (mem fraction / allocator / host-memory limit), NCCL/glibc tuning, and a free-form
    env escape hatch — lifted into the run config so a run's `launch_config.yaml` fully captures its
    environment (tracking + repro), and A/B-ing a knob is a config edit, not a launcher edit.

    XLA *compiler* flags are NOT here — they go through `RuntimeConfig.compiler_options`
    (passed natively to each jit, no env round-trip; see that field). This class is only the
    env that must exist before the process starts (read at backend/NCCL init).

    The launcher (`experiments/lm/launch.py`) renders this into the rank env it exports;
    `LD_LIBRARY_PATH` is NOT here (it is computed at submit time from the workspace venv's
    nvidia libs — machine-specific, not a tracked decision). Defaults mirror the values the
    launcher used to hardcode; they are the single source of truth.
    """

    @model_validator(mode="before")
    @classmethod
    def _strip_xla_flags(cls, data: object) -> object:
        # Shared-storage shim: XLA compiler flags moved off `launch_env.xla_flags` (env) onto
        # `RuntimeConfig.compiler_options` (native, in-process). Drop the stored env-form key
        # so old run configs still load; the run picks up the current `compiler_options`.
        if isinstance(data, dict):
            data.pop("xla_flags", None)
        return data

    xla_python_client_mem_fraction: PositiveFloat = 0.92
    """`XLA_PYTHON_CLIENT_MEM_FRACTION` — the BFC pool cap as a fraction of HBM."""
    xla_python_client_allocator: str | None = None
    """`XLA_PYTHON_CLIENT_ALLOCATOR` — e.g. `platform` for the on-demand cudaMalloc allocator
    (avoids BFC fragmentation OOMs near the HBM cap, at some per-alloc cost). `None` leaves
    the XLA default (BFC)."""
    xla_pjrt_gpu_host_memory_limit_gb: PositiveInt = 1024
    """`XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB` — cap on XLA's pinned host-staging pool
    (allocated on demand)."""
    nccl_debug: str = "WARN"
    """`NCCL_DEBUG` — overrides the cluster default (INFO + SUBSYS=ALL), which logs every
    collective and bloats slurm logs to tens of GB per run."""
    malloc_arena_max: PositiveInt = 2
    """`MALLOC_ARENA_MAX` — caps glibc malloc arenas to bound host RSS under many threads."""
    env: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Arbitrary extra exports merged into the rank env LAST (after the typed knobs), "
            "so it can override any of them. The escape hatch for a one-off var without a "
            "schema field."
        ),
    )
    profile: ProfileConfig = Field(default_factory=ProfileConfig)

    def as_env(self) -> dict[str, str]:
        """Render the ordered `{VAR: value}` map the launcher exports (sans the
        submit-time-computed `LD_LIBRARY_PATH`). Only the env that must exist before
        backend/NCCL init — XLA *compiler* flags are passed natively via
        `RuntimeConfig.compiler_options`, not here. Later keys override earlier, so the
        free-form `env` block wins last."""
        rendered: dict[str, str] = {
            "NCCL_DEBUG": self.nccl_debug,
            "MALLOC_ARENA_MAX": str(self.malloc_arena_max),
            "XLA_PYTHON_CLIENT_MEM_FRACTION": str(self.xla_python_client_mem_fraction),
            "XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB": str(self.xla_pjrt_gpu_host_memory_limit_gb),
        }
        if self.xla_python_client_allocator is not None:
            rendered["XLA_PYTHON_CLIENT_ALLOCATOR"] = self.xla_python_client_allocator
        rendered |= self.env
        return rendered


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


class RuntimeConfig(BaseConfig):
    """Compute substrate: launch mode, world size, rematerialization, and the launch-time
    env/XLA-flag surface (`launch_env`).

    Perturbs numerics but doesn't change the algorithm.
    """

    @model_validator(mode="before")
    @classmethod
    def _strip_removed_torch_runtime_fields(cls, data: object) -> object:
        # Shared-storage shim (provenance): stored run config.yamls carry torch-trainer
        # runtime fields the JAX trainer no longer has (`device`, `autocast_bf16` — bf16 is
        # unconditional, device is JAX-managed). Strip them so existing runs still load;
        # reject a non-supported value loudly.
        if not isinstance(data, dict):
            return data
        data.pop("device", None)
        # `launch: slurm | inline` was deleted (bring-up derives from dp vs gpus_per_node;
        # submission is the wrapper's business) — stored pins from its one day still parse.
        if "launch" in data:
            assert data.pop("launch") in ("slurm", "inline"), data
        if "autocast_bf16" in data:
            assert data.pop("autocast_bf16") is True, (
                "autocast_bf16 was removed (the JAX trainer always computes in bf16)"
            )
        return data

    dp: PositiveInt = Field(
        description=(
            "World size — the total device count, THE single source of truth for topology, "
            "NEVER inferred from ambient env (`SLURM_PROCID` is present in every process on "
            "a SLURM box). Process bring-up DERIVES from it: `dp <= gpus_per_node` → ONE "
            "process over exactly `dp` local devices, asserted at startup "
            "(`sharding.assert_inline_topology`) — `dp: 1` is the single-device smoke, "
            "`dp: 8` a run inside an external scheduler's own whole-node job. "
            "`dp > gpus_per_node` → one process per node, brought up via `jax.distributed`'s "
            "own cluster auto-detection (`init_distributed` — the jax ecosystem's contract; "
            "SLURM/MPI/TPU), asserted against the realized `jax.device_count()`. Multiple "
            "processes on one node is deliberately unrepresentable. The batch shards "
            "data-parallel across all `dp` devices."
        ),
    )
    gpus_per_node: PositiveInt = Field(
        default=8,
        description=(
            "GPUs per node — the size of the intra-node NVLink group the mesh's `fsdp*tp` "
            "plane is carved from, and the launcher's node math (`nodes = dp / gpus_per_node`). "
            "A property of the cluster, carried in the config so the pinned launch_config "
            "fully determines the topology. Default 8 (H100/H200/B200 nodes)."
        ),
    )
    tp: int = Field(
        default=1,
        ge=1,
        le=8,
        description=(
            "Tensor-parallel (Megatron) degree, carved from the intra-node GPUs so "
            "`fsdp * tp = GPUS_PER_NODE` — both stay on NVLink. Shards the component C axis "
            "(V/U, CI-fn output heads) and the CI-fn MLP hidden, halving the per-layer weight "
            "all-gather. `tp = 1` (default) is the pure-HSDP layout (degenerate tp axis, "
            "behaviour-preserving). Must divide both the device count and GPUS_PER_NODE."
        ),
    )
    sharding: Literal["owner", "owner+zero1", "zero1", "ddp"] | PlacementTableConfig = Field(
        description=(
            "Placement policy for the trainable state (placement.py). REQUIRED, no "
            "default — a layout this consequential is written down per config. Presets: "
            "`zero1` = intra-matrix ZeRO-1 over the full data mesh "
            "(~equivalent comms to `owner` under "
            "elementwise optimizers); `owner` = whole-matrix ownership (stack ÷replicate, "
            "d ÷fsdp, C ÷tp) — the muon-motivated layout (Newton-Schulz stays "
            "node-local); STRICT — a shape group whose stack does "
            "not tile ÷replicate is an error; `owner+zero1` = `owner` plus the "
            "`params.zero1` opt-in row, ZeRO-1-ing exactly those non-tiling groups "
            "intra-matrix; `ddp` = fully replicated. Each value is a BIDIRECTIONAL claim "
            "checked at config build (placement.from_config, pre-submission for a submitted run): "
            "`owner` claims every group tiles; `owner+zero1` claims at least one does "
            "not — all-tiling under it is equally an error. Or an explicit "
            "`PlacementTableConfig` table (nested `params: {persist, zero1?, forward}` + "
            "`activations`, each row a semantic-axis -> mesh-axes rule; list order is "
            "semantics). Same math under every value — layouts differ only by float "
            "reassociation (SPEC D4)."
        ),
    )
    remat_recon_forwards: bool = Field(
        default=False,
        description=(
            "JAX trainer memory/compute trade: rematerialize the recon-loss masked "
            "forwards under the full model (deep targets need it to fit). Compute "
            "substrate knob, no algorithm effect."
        ),
    )
    remat_ci_fn: bool = Field(
        default=False,
        description=(
            "JAX trainer memory/compute trade: rematerialize the CI-fn forward "
            "(recompute it in the backward instead of storing its activations). The "
            "CI-fn activations scale with batch, so this is the main lever for larger "
            "batch on big targets. Compute substrate knob, no algorithm effect."
        ),
    )
    scan_unroll: PositiveInt = Field(
        default=1,
        description=(
            "`lax.scan(unroll=k)` over the layer block stack: emit k iterations as "
            "straight-line code so XLA can prefetch gather(L+1) under matmul(L) (the "
            "cross-iteration overlap a 1-layer while-body denies). Per-layer remat is "
            "unchanged, so it is memory-neutral. 1 = plain per-layer scan. Compute substrate."
        ),
    )
    gather_fp8: bool = Field(
        default=False,
        description=(
            "Quantized all-gather: cast the ÷fsdp compute V/U to fp8 before the per-layer "
            "÷fsdp→full gather (½ the bf16 bytes on the wire), dequantized to bf16 in the "
            "block. Documented net-negative at b128 (fp8 on the wire was slower); kept as a "
            "gated experiment. Compute substrate."
        ),
    )
    ascend_replicate: bool = Field(
        default=False,
        description=(
            "Replicate the ÷fsdp compute weights once before the adversary ascents so the "
            "n_warmup ascend forwards skip the per-layer ÷fsdp→full gather (mask-independent "
            "and detached, so the re-gather is pure redundancy). Numerics-identical. Trades "
            "the full V/U resident during the ascend phase for the eliminated re-gathers."
        ),
    )
    compiler_options: dict[str, bool | int | str] = Field(
        default_factory=lambda: {
            "xla_gpu_enable_latency_hiding_scheduler": True,
            "xla_gpu_enable_triton_gemm": False,
            "xla_gpu_enable_command_buffer": "",
            "xla_gpu_enable_highest_priority_async_stream": True,
            "xla_gpu_all_reduce_combine_threshold_bytes": 1073741824,
            "xla_gpu_all_gather_combine_threshold_bytes": 1073741824,
            "xla_gpu_reduce_scatter_combine_threshold_bytes": 134217728,
            "xla_gpu_enable_pipelined_all_gather": True,
            "xla_gpu_enable_pipelined_reduce_scatter": True,
            "xla_gpu_enable_pipelined_all_reduce": True,
            "xla_gpu_enable_while_loop_double_buffering": True,
            "xla_gpu_enable_all_gather_combine_by_dim": False,
            "xla_gpu_enable_reduce_scatter_combine_by_dim": False,
        },
        description=(
            "XLA compiler flags passed NATIVELY to every jit's `compiler_options` — no "
            "`XLA_FLAGS` env round-trip, and (unlike env) they ARE in the compile-cache key, "
            "so changing one actually recompiles. Full `xla_*` flag names, typed values "
            "(True/int/str, not 'true'). Default = the tuned MaxText set (latency-hiding "
            "scheduler + 1 GiB combine thresholds + pipelined collectives + double-buffering; "
            "`command_buffer:''` disables CUDA-graph capture, a correctness guard). Add "
            "`xla_disable_hlo_passes: rematerialization` to opt into the disable-XLA-remat win "
            "(validate save/resume first). On CPU (toys/tests) the GPU flags are ignored."
        ),
    )
    launch_env: LaunchEnv = Field(default_factory=LaunchEnv)
    """The pre-process env the SLURM launcher exports into each rank (XLA *client* / NCCL /
    glibc knobs — the env that must exist before backend init; NOT compiler flags, which go
    via `compiler_options`). Only a SLURM submitter renders it — a run inside an
    external scheduler's allocation inherits the caller's environment."""

    @property
    def distributed(self) -> bool:
        """Derived, never authored: a world larger than one node is multi-process (one per
        node); anything else is one process over `dp` local devices."""
        return self.dp > self.gpus_per_node

    @model_validator(mode="after")
    def validate_topology(self) -> Self:
        if self.dp > self.gpus_per_node:
            assert self.dp % self.gpus_per_node == 0, (
                f"a multi-node world allocates whole {self.gpus_per_node}-GPU nodes — "
                f"dp={self.dp} must be a multiple of gpus_per_node={self.gpus_per_node} "
                f"(a sub-node world runs as one process inside an existing allocation)"
            )
        return self

    @model_validator(mode="after")
    def validate_gather_reshape(self) -> Self:
        assert not (self.ascend_replicate and self.gather_fp8), (
            "ascend_replicate and gather_fp8 both re-pin the compute-weight gather — pick one"
        )
        return self


class PDConfig(BaseConfig):
    """Algorithm specification: seed, losses, optimizers, faithfulness warmup.

    Domain-agnostic — the target-coupled apparatus (which sites to decompose + the CI-fn
    arch) lives in the per-domain `decomposition` section, not here. Flipping any field here
    changes what algorithm runs. Pair with `RuntimeConfig` (substrate), `Cadence` (when to
    emit) and `RunSink` (where output goes) when running the trainer (`param_decomp.core.run`).
    """

    @model_validator(mode="before")
    @classmethod
    def _strip_removed_jax_unsupported_fields(cls, data: object) -> object:
        # Shared-storage shim (provenance): stored run config.yamls carry fields the JAX
        # trainer no longer has — each only ever had ONE supported value. Strip them so
        # existing runs still load (harvest / autointerp / fine-tune / run_metadata); reject
        # a non-supported value loudly.
        if not isinstance(data, dict):
            return data
        if "sigmoid_type" in data:
            assert data.pop("sigmoid_type") == "leaky_hard", (
                "sigmoid_type was removed (only leaky_hard is implemented)"
            )
        if "use_delta_component" in data:
            assert data.pop("use_delta_component") is True, (
                "use_delta_component was removed (always on in the JAX trainer)"
            )
        if "tied_weights" in data:
            assert not data.pop("tied_weights"), (
                "tied_weights was removed (obviated by the JAX design)"
            )
        if "identity_decomposition_targets" in data:
            assert not data.pop("identity_decomposition_targets"), (
                "identity_decomposition_targets was removed (identity insertion is not in the JAX trainer)"
            )
        if "sampling" in data:
            assert data.pop("sampling") == "continuous", (
                "sampling was removed (continuous-only); binomial mask sampling is gone"
            )
        if "n_mask_samples" in data:
            # `n_mask_samples` moved from a trainer-level knob onto the stochastic loss
            # configs that actually draw samples. Push the stored value down onto every
            # stochastic recon entry that does not set its own, so existing run configs
            # keep their sample count; entries with an explicit value win.
            n = data.pop("n_mask_samples")
            stochastic_types = {
                "StochasticReconLoss",
                "StochasticReconLayerwiseLoss",
                "StochasticReconSubsetLoss",
            }
            for entry in data.get("loss_metrics", []):
                if (
                    isinstance(entry, dict)
                    and entry.get("type") in stochastic_types
                    and "n_mask_samples" not in entry
                ):
                    entry["n_mask_samples"] = n
        return data

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


class Cadence(BaseConfig):
    """Rhythm of non-eval loop emissions: train-log and checkpoint periods.

    Held separately from `RunSink` so the sink only owns *where* output goes; `Cadence`
    owns *when* train logs and checkpoints fire. Eval timing lives on `EvalLoop`,
    alongside the runtime objects it depends on. The trainer (`param_decomp.core.run`) loop
    always checkpoints at the final step regardless of `save_every`.
    """

    train_log_every: PositiveInt
    dense_log_phase: DenseLogPhase | None = None
    """Optional denser logging for early training; `None` means a flat `train_log_every`."""
    save_every: PositiveInt | None = None
    keep_last_n_checkpoints: PositiveInt | None = None
    """How many of the most-recent orbax `ckpts/<step>/` checkpoints to keep on disk
    after each checkpoint write. `None` (the default) keeps all checkpoints — the
    conservative choice for research where prior steps may matter. Opt in to e.g. `3`
    for long jobs where disk pressure outweighs the value of intermediate checkpoints;
    the final-step checkpoint is always included in the retained set."""

    def should_log_train(self, step: int) -> bool:
        if self.dense_log_phase is not None and step < self.dense_log_phase.until_step:
            return step % self.dense_log_phase.every == 0
        return step % self.train_log_every == 0

    def should_save(self, step: int) -> bool:
        if self.save_every is None or step == 0:
            return False
        return step % self.save_every == 0


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
