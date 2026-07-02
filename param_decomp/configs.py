"""The torch-free pydantic config schema for the algorithm core.

Every algorithm-level config class lives here (or in the sibling `base_config` /
`schedule` modules): routing, decomposition targets, the CI-fn config tree,
loss-metric configs, eval-metric configs, the top-level `PDConfig` / `RuntimeConfig` /
`Cadence`, and the `wandb.config` shaping helpers. Depends only on pydantic / numpy /
pyyaml / annotated-types (via `base_config`), so non-trainer consumers validate the same
YAML run configs without pulling jax/wandb.

Experiment-level schema (the `ExperimentConfig[T, D]` generic and its LM / TMS / ResidMLP
subclasses) lives lab-side under `param_decomp_lab/experiments/`.
"""

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

from param_decomp.base_config import BaseConfig, Probability
from param_decomp.schedule import ScheduleConfig

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
# Decomposition target
# ---------------------------------------------------------------------------


class DecompositionTargetConfig(BaseConfig):
    module_pattern: str = Field(..., description="fnmatch-style pattern to match module names")
    C: PositiveInt = Field(
        ..., description="Number of components for modules matching this pattern"
    )


# ---------------------------------------------------------------------------
# Causal-importance function configs
# ---------------------------------------------------------------------------


class LayerwiseMlpCiConfig(BaseConfig):
    """Per-site MLP CI fn (positionless toys): one independent MLP per site."""

    type: Literal["layerwise_mlp"] = "layerwise_mlp"
    hidden_dims: list[PositiveInt] = Field(
        ..., min_length=1, description="Hidden dims of each per-site MLP"
    )


class GlobalMlpCiConfig(BaseConfig):
    """Single shared MLP over all sites jointly (positionless toys)."""

    type: Literal["global_mlp"] = "global_mlp"
    hidden_dims: list[PositiveInt] = Field(
        ..., min_length=1, description="Hidden dims of the shared global MLP"
    )


class ChunkwiseTransformerCiConfig(BaseConfig):
    """Chunkwise-transformer CI fn (LMs). Each chunk is `blocks_per_chunk` consecutive
    transformer blocks; its input is the residual stream entering the chunk and its output
    is CI for every matrix site in those blocks. `d_model`/`n_blocks`/`n_heads`/`mlp_hidden`
    size the per-chunk CI transformer (`d_model % n_heads == 0`; head_dim even for RoPE)."""

    type: Literal["chunkwise_transformer"] = "chunkwise_transformer"
    blocks_per_chunk: PositiveInt
    d_model: PositiveInt
    n_blocks: PositiveInt
    n_heads: PositiveInt
    mlp_hidden: PositiveInt

    @model_validator(mode="after")
    def validate_heads(self) -> Self:
        assert self.d_model % self.n_heads == 0, (self.d_model, self.n_heads)
        assert (self.d_model // self.n_heads) % 2 == 0, "head_dim must be even for RoPE"
        return self


# Flat discriminated union (by `type`): one self-contained config per CI fn.
CiConfig = Annotated[
    LayerwiseMlpCiConfig | GlobalMlpCiConfig | ChunkwiseTransformerCiConfig,
    Field(discriminator="type"),
]


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

    `pnorm` is the initial `p`, linearly annealed toward `p_anneal_final_p` between
    `p_anneal_start_frac` and `p_anneal_end_frac` of training (no-op when
    `p_anneal_final_p is None` or `p_anneal_start_frac == 1.0`). `frequency` (when present)
    adds the batch-invariant frequency-minimality penalty over the same `(c + eps)^p`
    per-component sums.
    """

    type: Literal["ImportanceMinimalityLoss"] = "ImportanceMinimalityLoss"
    pnorm: NonNegativeFloat
    frequency: FrequencyMinimalityConfig | None = None
    p_anneal_start_frac: Probability = 1.0
    p_anneal_final_p: NonNegativeFloat | None = None
    p_anneal_end_frac: Probability = 1.0
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

    `gamma` is annealed linearly toward `gamma_anneal_final_gamma` between
    `gamma_anneal_start_frac` and `gamma_anneal_end_frac` of training; annealing it down
    sharpens the count. A constant schedule is `gamma_anneal_final_gamma == gamma`.
    """

    type: Literal["SmoothL0ImportanceMinimalityLoss"] = "SmoothL0ImportanceMinimalityLoss"
    gamma: PositiveFloat
    frequency: FrequencyMinimalityConfig | None = None
    gamma_anneal_start_frac: Probability = 1.0
    gamma_anneal_final_gamma: PositiveFloat | None = None
    gamma_anneal_end_frac: Probability = 1.0


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
#
# Deliberately NOT unified with `PersistentPGDSourceScope` below: per-step PGD encodes
# its scope as a bare YAML string (this `Literal`), while persistent PGD encodes it as a
# nested config object (the `SCScope | BSCScope` discriminated union). The value spaces
# also differ — `bc` is per-step-only; `sc` is persistent-only. Converging them would
# change the stored YAML shape of one side and break old-run parsing.
MaskScopeLiteral = Literal["c", "bc", "bsc"]
MaskScope = Annotated[MaskScopeLiteral, BeforeValidator(_alias_legacy_mask_scope)]


class PGDConfig(LossMetricConfig):
    """Shared base for per-step PGD loss configs."""

    init: PGDInitStrategy
    step_size: PositiveFloat
    n_steps: NonNegativeInt
    mask_scope: MaskScope


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


class SCScope(BaseConfig):
    """PPGD source scope: `[seq, C]` sources shared across batch elements, free per position."""

    type: Literal["sc"] = "sc"


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
# (batch, seq, C). Only the two seq-bearing scopes are implemented for persistent PGD;
# both require a sequence axis and are illegal off-LM.
PersistentPGDSourceScope = Annotated[
    SCScope | BSCScope,
    Field(discriminator="type"),
    BeforeValidator(_alias_legacy_scope_type),
]


class PersistentPGDReconLossConfig(LossMetricConfig):
    """Persistent-PGD recon loss: adversarial mask sources persist across train steps,
    routed to all layers every forward.

    Sources are clamped to `[0, 1]` after each step — the only implemented
    parameterization. (A sigmoid parameterization was removed.)
    """

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
    optimizer: AdamPGDConfig
    scope: PersistentPGDSourceScope
    source_dtype: Literal["float32", "bfloat16"] = "float32"
    """Storage dtype for the persistent PPGD source tensors AND their Adam moments
    (`m`/`v`). `float32` (default) is SPEC N1 (fp32 SRC_STEP moments) and the only
    oracle-parity path. `bfloat16` halves the resident source+moment footprint (~21 GiB
    on the full-32L step, the dominant f32 transient there) at some numerical risk: the
    second-moment `v` accumulates squared grads, which can underflow in bf16 for small
    grads — opt in only as an experiment."""
    n_warmup_steps: NonNegativeInt = Field(
        default=0,
        description=(
            "Extra inner PGD source-optimization steps on each train batch before the final loss"
            " computation."
        ),
    )


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


AnyEvalMetricConfig = Annotated[
    CEandKLLossesConfig
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
    ChunkwiseSubsetReconLossConfig
    | CIMaskedReconLayerwiseLossConfig
    | CIMaskedReconLossConfig
    | CIMaskedReconSubsetLossConfig
    | FaithfulnessLossConfig
    | ImportanceMinimalityLossConfig
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
    """`XLA_PYTHON_CLIENT_MEM_FRACTION` — the BFC pool cap as a fraction of HBM. The XLA
    default 0.75 caps production steps too low (OOM, job 50644)."""
    xla_python_client_allocator: str | None = None
    """`XLA_PYTHON_CLIENT_ALLOCATOR` — e.g. `platform` for the on-demand cudaMalloc allocator
    (avoids BFC fragmentation OOMs near the HBM cap, at some per-alloc cost). `None` leaves
    the XLA default (BFC). Replaces the old `pd-lm --allocator` flag."""
    xla_pjrt_gpu_host_memory_limit_gb: PositiveInt = 1024
    """`XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB` — cap on XLA's pinned host-staging pool (allocated
    on demand). The 64 GB default is blown past right after faith warmup on the full-model
    step (job 127622); the b200 nodes carry ~2 TB RAM."""
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


class RuntimeConfig(BaseConfig):
    """Compute substrate: data-parallelism degree, rematerialization, and the launch-time
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
        if "autocast_bf16" in data:
            assert data.pop("autocast_bf16") is True, (
                "autocast_bf16 was removed (the JAX trainer always computes in bf16)"
            )
        return data

    dp: PositiveInt | None = Field(
        default=None,
        description=(
            "Distributed world size — the number of data-parallel workers (= nodes × 8 on "
            "the cluster). The SINGLE source of truth for distributedness: the launcher "
            "submits across `dp // 8` nodes and the trainer calls "
            "`init_distributed(dp)`, which asserts the realized `jax.process_count()` "
            "equals it. NEVER inferred from ambient SLURM env. None means a single device "
            "(the launcher runs the trainer inline, no jax.distributed). The batch is "
            "sharded data-parallel across the workers."
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
    via `compiler_options`). Ignored on the inline `dp is None` path (inherits the caller's
    environment)."""

    @model_validator(mode="after")
    def validate_dp(self) -> Self:
        if self.dp is not None:
            assert self.dp >= 2, "if set, dp must be at least 2 (pass None for single device)."
        return self

    @model_validator(mode="after")
    def validate_gather_reshape(self) -> Self:
        assert not (self.ascend_replicate and self.gather_fp8), (
            "ascend_replicate and gather_fp8 both re-pin the compute-weight gather — pick one"
        )
        return self


class PDConfig(BaseConfig):
    """Algorithm specification: seed, CI function, losses, optimizers, target modules.

    Flipping any field here changes what algorithm runs. Pair with `RuntimeConfig`
    (substrate), `Cadence` (when to emit) and `RunSink` (where output goes) when
    running the trainer (`param_decomp.run`).
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
    ci_config: CiConfig = Field(
        ...,
        discriminator="type",
        description="Configuration for the causal importance function.",
    )
    decomposition_targets: list[DecompositionTargetConfig] = Field(
        ...,
        description="List of module patterns with C values specifying which modules to decompose.",
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
    alongside the runtime objects it depends on. The trainer (`param_decomp.run`) loop
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
    """Wandb UI group (`pd-lm --group`); None = ungrouped."""
    tags: list[str] = Field(default_factory=list)
    """Wandb tags (`pd-lm --tags a,b,c`, comma-split); empty = untagged."""


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

    out = dict(config_dict)
    walk(out, "")
    out.update(flattened)
    return out
