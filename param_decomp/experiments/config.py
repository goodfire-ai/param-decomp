"""Shared config schema for in-repo experiment YAMLs, plus the shared validation /
run-identity helpers every experiment reuses.

Each experiment subclasses `ExperimentConfig` to fix the concrete `target` / `data` types.
The generic engine reads the pydantic `pd` / `cadence` / `runtime` DIRECTLY, so there is
no flattened mirror to build — `assert_canonical_algorithm_config` only VALIDATES that the
schema lives in the subspace the JAX trainer implements (cosine-to-0.1 LR, plain AdamW,
components-only grad clip, …), and `run_instance` / `ci_arch` resolve the run identity and
the CI-fn architecture; each experiment's `run.py` assembles the rest (target + data).
"""

import re
from typing import Annotated, Literal, Self

from pydantic import Field, PositiveInt, model_validator

from param_decomp.core.base_config import BaseConfig
from param_decomp.core.built_run import RunInstance
from param_decomp.core.ci_fn import CIFnArch, GlobalMLPCIArch, MLPCIArch
from param_decomp.core.configs import (
    AdamWOptimizerConfig,
    AnyEvalMetricConfig,
    Cadence,
    ExplicitCSpec,
    MuonOptimizerConfig,
    PDConfig,
    ResumeProvenance,
    RuntimeConfig,
    WandbConfig,
)
from param_decomp.core.schedule import ScheduleConfig
from param_decomp.infra.settings import ENV


class EvalConfig(BaseConfig):
    """Eval-pass settings consumed by `EvalLoop`. `slow_every` must be a multiple of `every`."""

    batch_size: PositiveInt
    n_steps: PositiveInt
    every: PositiveInt
    slow_every: PositiveInt
    slow_on_first_step: bool = True
    metrics: list[AnyEvalMetricConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_slow_every_multiple_of_every(self) -> Self:
        assert self.slow_every % self.every == 0, (
            f"slow_every ({self.slow_every}) must be a multiple of every ({self.every})"
        )
        return self


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


class ToyDecompositionConfig(BaseConfig):
    """Decomposition apparatus for the positionless toys (TMS / ResidMLP): arbitrary named
    sites + an MLP CI fn. Toys have no arch grid and the MLP CI fns have no chunk-homogeneity
    requirement, so `explicit` sites are safe here (the LM bundle forbids them)."""

    sites: ExplicitCSpec
    ci: Annotated[LayerwiseMlpCiConfig | GlobalMlpCiConfig, Field(discriminator="type")]


class ExperimentConfig(BaseConfig):
    """The domain-AGNOSTIC sections of an in-repo experiment YAML.

    The domain-specific, co-varying config — `target`, the `decomposition` apparatus
    (site-spec + CI-fn arch), and `data` — is declared as CONCRETE fields on each per-domain
    subclass (`LMExperimentConfig`, `TMSExperimentConfig`, …), NOT as generic type params:
    those three vary together as one axis (the domain), so binding them on the concrete
    subclass makes a cross-domain mismatch (e.g. LM target + toy CI fn) unrepresentable.

    Omit the `eval:` block to skip eval entirely; omit `wandb:` to skip wandb (the run
    still writes `config.yaml` + checkpoints locally).

    The run id is NOT a config field: it is minted by the launcher and passed to
    `run_instance` as an explicit argument. The run dir is a pure function of settings
    + id (`ENV.output_root/runs/<run_id>`).
    """

    @model_validator(mode="before")
    @classmethod
    def _strip_removed_run_identity_fields(cls, data: object) -> object:
        # Shared-storage shim: stored run config.yamls carry `run_id` (minted identity,
        # now passed to `run_instance` as an arg and derived from the run-dir name) and
        # `out_dir` (vestigial; the run dir is `ENV.output_root/runs/<run_id>`). Both
        # fields are removed; strip them so existing configs still load under extra=forbid.
        if not isinstance(data, dict):
            return data
        data.pop("run_id", None)
        data.pop("out_dir", None)
        return data

    run_name: str
    """Human-readable display name (the wandb run NAME)."""

    pd: PDConfig
    runtime: RuntimeConfig
    cadence: Cadence
    eval: EvalConfig | None = None
    wandb: WandbConfig | None = None
    resume_provenance: ResumeProvenance | None = None
    """Set on resumed runs (parent run dir + step); `None` for fresh runs. Lives on the
    config so it flows into `experiment_config.yaml` and `wandb.config` via `init_pd_run`,
    making a resumed run's lineage visible in the wandb UI."""


_RUN_ID_PATTERN = re.compile(r"^p-[0-9a-f]{8}$")


def ci_arch(ci_config: LayerwiseMlpCiConfig | GlobalMlpCiConfig) -> CIFnArch:
    """Config→arch for the positionless-toy MLP CI fns — the arch IS the pydantic config (strip
    `type`, list→tuple). The LM chunkwise arch resolves against the target and is built directly
    in `experiments.lm.config._resolve_chunkwise_ci_arch`, so it never routes through here."""
    match ci_config:
        case LayerwiseMlpCiConfig():
            return MLPCIArch(hidden_dims=tuple(ci_config.hidden_dims))
        case GlobalMlpCiConfig():
            return GlobalMLPCIArch(hidden_dims=tuple(ci_config.hidden_dims))


def _assert_cosine_to_tenth(schedule: ScheduleConfig, who: str) -> None:
    """The trainer honors the full `ScheduleConfig`; the METHOD's LR is cosine-to-0.1x
    with no warmup (SPEC S20), so the conversion gate pins that shape."""
    assert schedule.fn_type == "cosine", f"{who}: only cosine lr supported, got {schedule}"
    assert schedule.warmup_pct == 0.0, f"{who}: lr warmup unsupported, got {schedule}"
    assert schedule.final_val_frac == 0.1, f"{who}: final_val_frac must be 0.1, got {schedule}"


def _assert_canonical_optimizer(
    optimizer: AdamWOptimizerConfig | MuonOptimizerConfig, who: str
) -> None:
    """AdamW is canonical; Muon is a deliberate, config-gated experimental deviation (still
    constrained to the trainer's no-weight-decay subspace)."""
    if isinstance(optimizer, AdamWOptimizerConfig):
        assert optimizer.betas == (0.9, 0.999), f"{who}: betas must be (0.9, 0.999)"
    assert optimizer.weight_decay == 0.0, f"{who}: weight_decay must be 0"


def assert_canonical_algorithm_config(cfg: ExperimentConfig) -> None:
    """Assert the schema lives in the subspace the JAX trainer implements (the engine then
    reads `pd` / `cadence` DIRECTLY). The numerics-load-bearing constraints:
    cosine-to-0.1 LR with no warmup, plain AdamW (betas (0.9, 0.999), no weight decay) or
    the config-gated experimental Muon,
    a required components grad clip (CI-fn grad clip is optional), and a fully-specified
    checkpoint cadence. (Leaky-hard
    sigmoid, the always-built delta component, and no tied weights are now enforced by
    REMOVAL of those fields from `PDConfig` — `extra=forbid` rejects any attempt to set
    them.)"""
    vu_opt = cfg.pd.components_optimizer
    ci_opt = cfg.pd.ci_fn_optimizer
    _assert_cosine_to_tenth(vu_opt.lr_schedule, "components_optimizer")
    _assert_cosine_to_tenth(ci_opt.lr_schedule, "ci_fn_optimizer")
    _assert_canonical_optimizer(vu_opt, "components_optimizer")
    _assert_canonical_optimizer(ci_opt, "ci_fn_optimizer")
    assert vu_opt.grad_clip_norm is not None, "components grad clip is part of the method"

    cadence = cfg.cadence
    assert cadence.save_every is not None and cadence.keep_last_n_checkpoints is not None, cadence


def run_instance(cfg: ExperimentConfig, run_id: str) -> RunInstance:
    """The resolved run identity + logging lineage. `run_id` is minted by the launcher (a
    toy mints its own); the run dir is `ENV.output_root/runs/<run_id>`."""
    assert _RUN_ID_PATTERN.match(run_id), f"run_id must be p-<8hex>, got {run_id!r}"
    return RunInstance(
        run_name=cfg.run_name,
        run_id=run_id,
        out_dir=ENV.output_root / "runs",
        wandb=cfg.wandb,
        resume_provenance=cfg.resume_provenance,
    )
