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
from collections.abc import Callable
from typing import Any, Self

from pydantic import Field, NonNegativeFloat, PositiveInt, model_validator

from param_decomp.base_config import BaseConfig
from param_decomp.built_run import RunInstance
from param_decomp.ci_fn import (
    ChunkwiseTransformerCIArch,
    CIFnArch,
    GlobalMLPCIArch,
    MLPCIArch,
)
from param_decomp.configs import (
    AnyEvalMetricConfig,
    AnyLossMetricConfig,
    Cadence,
    ChunkwiseTransformerCiConfig,
    CiConfig,
    FaithfulnessLossConfig,
    GlobalMlpCiConfig,
    ImportanceMinimalityLossConfig,
    LayerwiseMlpCiConfig,
    OptimizerConfig,
    PDConfig,
    PersistentPGDReconLossConfig,
    PGDReconLossConfig,
    PGDReconSubsetLossConfig,
    ResumeProvenance,
    RuntimeConfig,
    SmoothL0ImportanceMinimalityLossConfig,
    StochasticHiddenActsReconLossConfig,
    UnmaskedReconLossConfig,
    WandbConfig,
)
from param_decomp.schedule import ScheduleConfig
from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR

# ── targeted PD (tPD) shared config surface (SPEC §11), reused across the per-domain
# experiment configs. ──

# Losses dropped from the NON-TARGET pass (SPEC S35): the non-target pass is stochastic-only,
# so both adversarial recons (persistent + fresh PGD) are excluded — the delta is forced on
# there, not adversarially probed; unmasked recon is meaningless with a forced-on delta;
# hidden-acts recon is a target-only diagnostic. FaithfulnessLoss is KEPT (inert, coeff 0) —
# the engine requires exactly one faith term.
EXCLUDED_NONTARGET_LOSS_CONFIGS: tuple[type, ...] = (
    PersistentPGDReconLossConfig,
    PGDReconLossConfig,
    PGDReconSubsetLossConfig,
    UnmaskedReconLossConfig,
    StochasticHiddenActsReconLossConfig,
)


class NontargetConfig[D: BaseConfig](BaseConfig):
    """The broad NON-TARGET stream + its per-pass settings (SPEC §11). `data` reuses the
    experiment's own data-config type (`TMSDataConfig`, `LMDataConfig`, …) at the full
    distribution (no `active_indices`). `impmin_coeff_ratio` scales the importance-minimality
    coeff on the non-target pass (the paper ~2, accounting for fewer active components
    off-target). The non-target pass forces the delta component fully on."""

    data: D
    batch_size: PositiveInt
    impmin_coeff_ratio: NonNegativeFloat = 1.0


def build_nontarget_loss_metrics(
    loss_metrics: list[AnyLossMetricConfig], impmin_coeff_ratio: float
) -> list[AnyLossMetricConfig]:
    """Derive the non-target-pass loss set from the target-pass losses: drop the excluded
    types (`EXCLUDED_NONTARGET_LOSS_CONFIGS`) and scale the importance-minimality coeff by
    `impmin_coeff_ratio`. Keeps a full-model recon loss + the inert faith term."""
    out: list[AnyLossMetricConfig] = []
    for cfg in loss_metrics:
        if isinstance(cfg, EXCLUDED_NONTARGET_LOSS_CONFIGS):
            continue
        is_impmin = isinstance(
            cfg, (ImportanceMinimalityLossConfig, SmoothL0ImportanceMinimalityLossConfig)
        )
        if is_impmin and cfg.coeff is not None:
            out.append(cfg.model_copy(update={"coeff": cfg.coeff * impmin_coeff_ratio}))
        else:
            out.append(cfg.model_copy())
    return out


def assert_targeted_faithfulness_off(pd: PDConfig) -> None:
    """The shared tPD invariant: faithfulness pressure OFF (it drives the delta -> 0, but the
    delta must stay nonzero to carry non-target behavior), while satisfying the engine's
    requirement of exactly one FaithfulnessLoss term (kept at coeff 0)."""
    assert pd.faithfulness_warmup_steps == 0, (
        "targeted PD needs a nonzero delta; a faithfulness warmup drives delta -> 0. "
        "Set pd.faithfulness_warmup_steps: 0."
    )
    faith = [m for m in pd.loss_metrics if isinstance(m, FaithfulnessLossConfig)]
    assert len(faith) == 1 and faith[0].coeff == 0.0, (
        "targeted PD requires exactly one FaithfulnessLoss with coeff: 0.0 — the engine needs "
        f"the term, tPD needs it inert so the delta stays nonzero. Got "
        f"{[(type(m).__name__, m.coeff) for m in faith]}."
    )


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


class ExperimentConfig[T: BaseConfig, D: BaseConfig](BaseConfig):
    """Full YAML schema for an in-repo experiment.

    Subclass with concrete `target` / `data` types per experiment:

        class LMExperimentConfig(ExperimentConfig[LMTargetConfig, LMDataConfig]):
            pass

    Omit the `eval:` block to skip eval entirely; omit `wandb:` to skip wandb (the run
    still writes `config.yaml` + checkpoints locally).

    The run id is NOT a config field: it is minted by the launcher and passed to
    `run_instance` as an explicit argument. The run dir is a pure function of settings
    + id (`PARAM_DECOMP_OUT_DIR/runs/<run_id>`).
    """

    @model_validator(mode="before")
    @classmethod
    def _strip_removed_run_identity_fields(cls, data: object) -> object:
        # Shared-storage shim: stored run config.yamls carry `run_id` (minted identity,
        # now passed to `run_instance` as an arg and derived from the run-dir name) and
        # `out_dir` (vestigial; the run dir is `PARAM_DECOMP_OUT_DIR/runs/<run_id>`). Both
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
    target: T
    data: D
    eval: EvalConfig | None = None
    wandb: WandbConfig | None = None
    resume_provenance: ResumeProvenance | None = None
    """Set on resumed runs (parent run dir + step); `None` for fresh runs. Lives on the
    config so it flows into `experiment_config.yaml` and `wandb.config` via `init_pd_run`,
    making a resumed run's lineage visible in the wandb UI."""


_RUN_ID_PATTERN = re.compile(r"^p-[0-9a-f]{8}$")


def ci_arch(
    ci_config: CiConfig,
    resolve_chunkwise: "Callable[[ChunkwiseTransformerCiConfig], ChunkwiseTransformerCIArch] | None",
) -> CIFnArch:
    """The single config→arch converter. The MLP/global archs ARE their pydantic config
    (strip `type`, list→tuple); the chunkwise arch RESOLVES against the LM target, so the
    caller supplies `resolve_chunkwise` (a closure binding the resolved target — the chunk
    generator + residual-width logic stays LM-side). The positionless toys never hit the
    chunkwise branch and pass `resolve_chunkwise=None`."""
    match ci_config:
        case LayerwiseMlpCiConfig():
            return MLPCIArch(hidden_dims=tuple(ci_config.hidden_dims))
        case GlobalMlpCiConfig():
            return GlobalMLPCIArch(hidden_dims=tuple(ci_config.hidden_dims))
        case ChunkwiseTransformerCiConfig():
            assert resolve_chunkwise is not None, (
                "chunkwise_transformer CI fn needs an LM target to resolve against; "
                "the positionless toys can't request it"
            )
            return resolve_chunkwise(ci_config)


def _assert_cosine_to_tenth(schedule: ScheduleConfig, who: str) -> None:
    """The trainer hardcodes optax cosine decay to 0.1x with no warmup (SPEC S19/S20)."""
    assert schedule.fn_type == "cosine", f"{who}: only cosine lr supported, got {schedule}"
    assert schedule.warmup_pct == 0.0, f"{who}: lr warmup unsupported, got {schedule}"
    assert schedule.final_val_frac == 0.1, f"{who}: final_val_frac must be 0.1, got {schedule}"


def _assert_plain_adamw(optimizer: OptimizerConfig, who: str) -> None:
    assert optimizer.betas == (0.9, 0.999), f"{who}: betas must be (0.9, 0.999)"
    assert optimizer.weight_decay == 0.0, f"{who}: weight_decay must be 0"


def assert_canonical_algorithm_config(cfg: "ExperimentConfig[Any, Any]") -> None:
    """Assert the schema lives in the subspace the JAX trainer implements (the engine then
    reads `pd` / `cadence` DIRECTLY). The numerics-load-bearing constraints:
    cosine-to-0.1 LR with no warmup, plain AdamW (betas (0.9, 0.999), no weight decay),
    a required components grad clip (CI-fn grad clip is optional), and a fully-specified
    checkpoint cadence. (Leaky-hard
    sigmoid, the always-built delta component, and no tied weights are now enforced by
    REMOVAL of those fields from `PDConfig` — `extra=forbid` rejects any attempt to set
    them.)"""
    vu_opt = cfg.pd.components_optimizer
    ci_opt = cfg.pd.ci_fn_optimizer
    _assert_cosine_to_tenth(vu_opt.lr_schedule, "components_optimizer")
    _assert_cosine_to_tenth(ci_opt.lr_schedule, "ci_fn_optimizer")
    _assert_plain_adamw(vu_opt, "components_optimizer")
    _assert_plain_adamw(ci_opt, "ci_fn_optimizer")
    assert vu_opt.grad_clip_norm is not None, "components grad clip is part of the method"

    # The persistent-PGD source LR is constant-after-warmup only — `source_lr` ignores
    # `fn_type` / `final_val_frac`, so a decaying source schedule would be silently flattened.
    for metric in cfg.pd.loss_metrics:
        if isinstance(metric, PersistentPGDReconLossConfig):
            sched = metric.optimizer.lr_schedule
            assert sched.fn_type == "constant", (
                f"persistent-PGD source LR is constant-only, got {sched.fn_type!r}"
            )

    cadence = cfg.cadence
    assert cadence.save_every is not None and cadence.keep_last_n_checkpoints is not None, cadence


def run_instance(cfg: "ExperimentConfig[Any, Any]", run_id: str) -> RunInstance:
    """The resolved run identity + logging lineage. `run_id` is minted by the launcher (a
    toy mints its own); the run dir is `PARAM_DECOMP_OUT_DIR/runs/<run_id>`."""
    assert _RUN_ID_PATTERN.match(run_id), f"run_id must be p-<8hex>, got {run_id!r}"
    return RunInstance(
        run_name=cfg.run_name,
        run_id=run_id,
        out_dir=PARAM_DECOMP_OUT_DIR / "runs",
        wandb=cfg.wandb,
        resume_provenance=cfg.resume_provenance,
    )
