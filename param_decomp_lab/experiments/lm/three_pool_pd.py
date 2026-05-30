"""3-pool-constrained `PDConfig` subclass + the typed `ThreePoolLosses` struct.

The 3-pool training path implements exactly one algorithm: continuous sampling, a
single stochastic mask, a delta component, no identity ops, and exactly the five
loss terms (faithfulness, importance-minimality, frequency-minimality, layerwise
stoch recon, persistent PGD recon). On the generic `PDConfig` those facts were
enforced by ~60 lines of
runtime asserts inside `ThreePoolTrainer.__init__`
(`_validate_pd_config_for_three_pool`), firing only after a multi-node launch
reached construction.

`ThreePoolConstrainedPDConfig` lifts those into the type: the fixed scalars become
frozen `Literal` defaults, and `loss_metrics` (a generic ordered list) is replaced
by the `ThreePoolLosses(faith, imp, freq, stoch, ppgd)` struct so "exactly these
five, each with a coeff" is unrepresentable-if-wrong. A `model_validator` derives the
inherited `loss_metrics` list from the struct so everything that still consumes a
`list[AnyLossMetricConfig]` (`ComponentModel` wiring, `validate_pgd_scope`,
snapshot serialization, eval) keeps working unchanged.

The residual cross-field checks (batch divisibility, rank-0 convention, site
coverage) that couple `pd` to the topology live on `ThreePoolLMExperimentConfig`
(see `three_pool_run.py`), since only there are both `pd` and `runtime.topology`
visible at load time.
"""

from typing import Any, Literal, Self

from pydantic import Field, model_validator

from param_decomp.base_config import BaseConfig
from param_decomp.configs import AnyLossMetricConfig, PDConfig
from param_decomp.metrics.faithfulness import FaithfulnessLossConfig
from param_decomp.metrics.importance_minimality import (
    FrequencyMinimalityLossConfig,
    ImportanceMinimalityLossConfig,
)
from param_decomp.metrics.persistent_pgd_recon import PersistentPGDReconLossConfig
from param_decomp.metrics.stochastic_recon_layerwise import StochasticReconLayerwiseLossConfig


class ThreePoolLosses(BaseConfig):
    """The exactly-these-five loss set the 3-pool path implements.

    Replaces `PDConfig.loss_metrics`'s generic `list[AnyLossMetricConfig]` so a
    missing / extra / wrong-typed loss is a parse error, not a runtime assert. Each
    field carries the metric's own params (`imp.pnorm`, `freq.reference_token_count`,
    `ppgd.optimizer`/`scope`/`n_warmup_steps`, …) plus its `coeff`. The trainer reads
    `pd.losses.faith` / `.imp` / `.freq` / `.stoch` / `.ppgd` directly — no `by_type`
    dict, no `isinstance` narrowing.
    """

    faith: FaithfulnessLossConfig
    imp: ImportanceMinimalityLossConfig
    freq: FrequencyMinimalityLossConfig
    stoch: StochasticReconLayerwiseLossConfig
    ppgd: PersistentPGDReconLossConfig

    @model_validator(mode="after")
    def validate_coeffs_present(self) -> Self:
        for name, cfg in (
            ("faith", self.faith),
            ("imp", self.imp),
            ("freq", self.freq),
            ("stoch", self.stoch),
            ("ppgd", self.ppgd),
        ):
            assert cfg.coeff is not None, f"losses.{name}.coeff is required for 3-pool training"
        assert self.ppgd.start_frac == 0.0, (
            "3-pool path does not implement PersistentPGDReconLoss.start_frac > 0; "
            "PPGD always runs from step 0."
        )
        # The CI step shares one set of per-component `(ci + eps)^p` sums between the
        # imp and freq terms (`step_ci._sparsity_losses`), so their p-power params must
        # match. Differing values would silently apply imp's power to the freq term.
        shared = ("pnorm", "eps", "p_anneal_start_frac", "p_anneal_final_p", "p_anneal_end_frac")
        for field in shared:
            imp_val, freq_val = getattr(self.imp, field), getattr(self.freq, field)
            assert imp_val == freq_val, (
                f"3-pool shares the (ci+eps)^p sums between imp and freq, so "
                f"losses.imp.{field} ({imp_val}) must equal losses.freq.{field} ({freq_val})"
            )
        return self


class ThreePoolConstrainedPDConfig(PDConfig):
    """`PDConfig` narrowed to what the 3-pool path can honour.

    Substitutable for `PDConfig` everywhere (`ThreePoolTrainer`, `load_component_model`
    consume inherited fields), but the fixed scalars are frozen `Literal` defaults and
    the loss set is the typed `ThreePoolLosses` struct. The inherited `loss_metrics`
    list is derived from `losses` so list-consuming code keeps working.
    """

    # These narrow inherited `PDConfig` fields to the single value the 3-pool path
    # honours. basedpyright flags the narrowing as an incompatible variable override
    # (pydantic class attrs read as mutable/invariant); the narrowing is the point.
    sampling: Literal["continuous"] = "continuous"  # pyright: ignore[reportIncompatibleVariableOverride]
    n_mask_samples: Literal[1] = 1  # pyright: ignore[reportIncompatibleVariableOverride]
    use_delta_component: Literal[True] = True  # pyright: ignore[reportIncompatibleVariableOverride]
    identity_decomposition_targets: None = None  # pyright: ignore[reportIncompatibleVariableOverride]

    losses: ThreePoolLosses

    # Derived from `losses` (see `derive_loss_metrics`); excluded from serialization
    # so a round-trip through `model_dump()` re-derives it rather than re-supplying it.
    loss_metrics: list[AnyLossMetricConfig] = Field(default_factory=list, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def derive_loss_metrics(cls, data: Any) -> Any:
        # `loss_metrics` is the inherited generic list consumed by ComponentModel
        # wiring, validate_pgd_scope, snapshot serialization, and eval. It's not
        # authored on this config (the struct is) — derive it from `losses` in the
        # canonical order before field validation, so the parent's
        # `validate_loss_metrics_have_coeff` (which runs on the populated list) and
        # downstream list consumers both see the four metrics. Authoring
        # `loss_metrics` directly is rejected — the struct is the single source.
        if not isinstance(data, dict):
            return data
        assert "loss_metrics" not in data, (
            "ThreePoolConstrainedPDConfig derives `loss_metrics` from `losses`; "
            "do not author `loss_metrics` directly."
        )
        assert "losses" in data, "ThreePoolConstrainedPDConfig requires a `losses` block"
        losses = ThreePoolLosses.model_validate(data["losses"])
        data = dict(data)
        data["loss_metrics"] = [
            losses.faith.model_dump(),
            losses.imp.model_dump(),
            losses.freq.model_dump(),
            losses.stoch.model_dump(),
            losses.ppgd.model_dump(),
        ]
        return data
