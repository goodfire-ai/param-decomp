"""The closed VPD training objectives — plain and targeted.

Authored loss metrics become explicit objective roles: the plain objective is exactly one
faithfulness term, one importance-minimality term, a non-empty ordered tuple of recon
terms, and at most one nonlinearity-locality term; the targeted (tPD, SPEC §11)
objective is a faithfulness-free target-pass surface plus a directly-authored
non-target pass (delta-pinned recon + importance-minimality at its own coefficient).
The recon vocabulary (routing samplers, mask-source strategies) lives in `recon.py`;
this module alone composes it with the other objective roles.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from jaxtyping import Array

from param_decomp.core.components import ComponentStacks, SiteSpec, nonlinearity_partitions
from param_decomp.core.configs import (
    AllRoutingConfig,
    AnyLossMetricConfig,
    AnyReconLossMetricConfig,
    CIMaskedReconLossConfig,
    CIMaskedReconSubsetLossConfig,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    LossCoeff,
    MergedStochasticSubsetPPGDReconLossConfig,
    NonlinearityLocalityLossConfig,
    NontargetConfig,
    NontargetReconLossMetricConfig,
    PersistentPGDReconLossConfig,
    PGDReconLossConfig,
    PGDReconSubsetLossConfig,
    StochasticReconLossConfig,
    StochasticReconSubsetLossConfig,
    SubsetRoutingType,
    TargetedLossMetricConfig,
    UnmaskedNoDeltaReconLossConfig,
    UnmaskedReconLossConfig,
)
from param_decomp.core.losses import coeff_at, nonlinearity_loss, scheduled_value_at
from param_decomp.core.nonlinearity import NonlinearityPartition, NonlinearityUnitKind
from param_decomp.core.recon import (
    AnyReconLossTerm,
    ConstantSources,
    FreshPGDSources,
    MaskSourceStrategy,
    MixedPersistentStochasticSources,
    PersistentSources,
    ReconLossTerm,
    StochasticSources,
    UnmaskedNoDeltaSources,
    routing_sampler_from_config,
)


@dataclass(frozen=True)
class FaithfulnessTerm:
    name: str
    coeff: LossCoeff


@dataclass(frozen=True)
class ImportanceMinimalityTerm:
    """CI-space importance-minimality plus optional frequency penalty."""

    name: str
    coeff: LossCoeff
    cfg: ImportanceMinimalityLossConfig


@dataclass(frozen=True)
class NonlinearityTerm:
    """Weight-space concentration over authored nonlinearity-facing output units."""

    name: str
    coeff: LossCoeff
    cfg: NonlinearityLocalityLossConfig


@dataclass(frozen=True)
class ResolvedNonlinearity:
    """A `NonlinearityTerm` joined with the target's declared partitions: the closed
    unit-kind check happens at `resolve`, once, and `None`-weighted kinds are filtered
    out here so no loss math ever sees an excluded kind."""

    term: NonlinearityTerm
    trained_partitions: dict[str, NonlinearityPartition]
    kind_coefficients: dict[NonlinearityUnitKind, float]

    @staticmethod
    def resolve(term: NonlinearityTerm, sites: tuple[SiteSpec, ...]) -> "ResolvedNonlinearity":
        partitions = nonlinearity_partitions(sites)
        assert partitions, "NonlinearityLocalityLoss needs a partitioned site"
        declared_kinds = {p.unit_kind for p in partitions.values()}
        authored = term.cfg.unit_kind_coefficients
        assert authored.keys() == declared_kinds, (
            f"unit_kind_coefficients must name exactly the target's partitioned kinds: "
            f"authored {sorted(authored)}, declared {sorted(declared_kinds)}"
        )
        kind_coefficients: dict[NonlinearityUnitKind, float] = {
            kind: w for kind, w in authored.items() if w is not None
        }
        return ResolvedNonlinearity(
            term,
            {name: p for name, p in partitions.items() if p.unit_kind in kind_coefficients},
            kind_coefficients,
        )

    def weighted_loss_and_metrics(
        self, train_frac: Array, components: ComponentStacks
    ) -> tuple[Array, dict[str, Array]]:
        """The term's coefficient-weighted step value plus its ready-to-log metrics."""
        threshold = scheduled_value_at(train_frac, self.term.cfg.relative_threshold)
        value, by_kind = nonlinearity_loss(
            components, self.trained_partitions, threshold, self.kind_coefficients
        )
        metrics: dict[str, Array] = {
            f"loss/{self.term.name}": value,
            "nonlinearity_relative_threshold": threshold,
            **{f"loss/{self.term.name}_{kind}": v for kind, v in by_kind.items()},
        }
        return coeff_at(train_frac, self.term.coeff) * value, metrics


@dataclass(frozen=True)
class LossSurface:
    """`L = c·faith + c·importance + Σ c·recon [+ c·nonlinearity]`."""

    faith: FaithfulnessTerm
    imp: ImportanceMinimalityTerm
    recon: tuple[AnyReconLossTerm, ...]
    nonlinearity: NonlinearityTerm | None


@dataclass(frozen=True)
class TargetPass:
    """The tPD target-pass surface (SPEC T3/T7): the full decomposition objective minus
    faithfulness — the delta is the off-target escape valve and must never be penalized,
    so a targeted objective has no faithfulness role at all."""

    imp: ImportanceMinimalityTerm
    recon: tuple[AnyReconLossTerm, ...]


@dataclass(frozen=True)
class NontargetPass:
    """The tPD non-target-pass surface, complete (SPEC T4/T5): with the delta mask pinned
    fully on, the broad stream judges only what the components must not disturb — so its
    whole objective is delta-pinned reconstruction against the frozen output plus
    importance-minimality at its own coefficient (T4's one enumerated exception is the
    unmasked-no-delta term, whose delta is pinned OFF). `imp.cfg` IS the target pass's
    config (penalty shape, anneal, frequency block shared by construction); only the
    coefficient is the non-target pass's own."""

    recon: tuple[ReconLossTerm[StochasticSources | ConstantSources | UnmaskedNoDeltaSources], ...]
    """The enumerated non-target strategies ONLY, in the type (SPEC T5): the delta-pinned
    stochastic/constant pair plus the delta-off unmasked arm — a term carrying an
    adversarial or mixed strategy is unrepresentable here, not filtered out."""
    impmin_coeff: LossCoeff
    """The non-target pass's importance-minimality COEFFICIENT — the penalty config
    (shape, anneal, frequency block) is the target pass's, structurally: this pass
    cannot carry its own (SPEC T6)."""


@dataclass(frozen=True)
class TargetedObjective:
    """The complete two-pass tPD objective; both passes sum into ONE backward (SPEC §11)."""

    target: TargetPass
    nontarget: NontargetPass


def _collect_terms(
    loss_metrics: Sequence[AnyLossMetricConfig],
    site_names: tuple[str, ...],
) -> tuple[
    FaithfulnessTerm | None,
    ImportanceMinimalityTerm | None,
    tuple[AnyReconLossTerm, ...],
    NonlinearityTerm | None,
]:
    """One pass over an authored loss list into its objective roles, names unique across
    all roles. Completeness (which roles must be present) is each objective builder's own
    claim, not this walk's.

    Recon-term order follows the authored list and is semantically load-bearing: per-term
    RNG keys derive from the recon index (SPEC R1).
    """
    faith: FaithfulnessTerm | None = None
    imp: ImportanceMinimalityTerm | None = None
    recon_terms: list[AnyReconLossTerm] = []
    nonlinearity: NonlinearityTerm | None = None

    def unique_name(cfg: AnyLossMetricConfig) -> str:
        # Only committed terms are in `taken`, so persistent terms may call this once for
        # their state key and again inside `recon` without colliding with themselves.
        name = cfg.name if cfg.name is not None else cfg.type
        taken = {term.name for term in recon_terms}
        if faith is not None:
            taken.add(faith.name)
        if imp is not None:
            taken.add(imp.name)
        if nonlinearity is not None:
            taken.add(nonlinearity.name)
        assert name not in taken, f"duplicate loss instance_key {name!r}"
        return name

    def recon(
        cfg: AnyReconLossMetricConfig,
        routing: SubsetRoutingType,
        sources: MaskSourceStrategy,
        n_samples: int,
    ) -> AnyReconLossTerm:
        # `sources` sits in parameter position, so the term is built width-erased
        # directly (storage is width-erased; narrower widths are the builders' concern).
        assert cfg.coeff is not None
        return ReconLossTerm(
            unique_name(cfg),
            cfg.coeff,
            routing_sampler_from_config(routing, site_names, n_samples),
            sources,
            cfg.hidden_acts_reconstruction,
        )

    for cfg in loss_metrics:
        assert cfg.coeff is not None, f"{cfg.type}: training losses need a coeff"
        match cfg:
            case FaithfulnessLossConfig():
                assert faith is None
                faith = FaithfulnessTerm(unique_name(cfg), cfg.coeff)
            case ImportanceMinimalityLossConfig():
                assert imp is None
                assert all(k.frac > 0 for k in cfg.gamma.points), (
                    f"gamma knots must all keep frac > 0, got {cfg.gamma.points}: a zero "
                    "width collapses the smooth-L0 threshold band the gradient lives on"
                )
                imp = ImportanceMinimalityTerm(unique_name(cfg), cfg.coeff, cfg)
            case UnmaskedReconLossConfig():
                recon_terms.append(
                    recon(cfg, AllRoutingConfig(), ConstantSources(1.0), n_samples=1)
                )
            case (
                CIMaskedReconLossConfig()
                | CIMaskedReconSubsetLossConfig()
                | StochasticReconLossConfig()
                | StochasticReconSubsetLossConfig()
            ):
                routing, sources, n_samples = _nontarget_recon_parts(cfg)
                recon_terms.append(recon(cfg, routing, sources, n_samples))
            case PGDReconLossConfig() | PGDReconSubsetLossConfig():
                sources = FreshPGDSources(cfg.init, cfg.n_steps, cfg.step_size, cfg.source_shape)
                routing = (
                    cfg.routing if isinstance(cfg, PGDReconSubsetLossConfig) else AllRoutingConfig()
                )
                recon_terms.append(recon(cfg, routing, sources, n_samples=1))
            case MergedStochasticSubsetPPGDReconLossConfig():
                key = unique_name(cfg)
                sources = MixedPersistentStochasticSources(state_key=key, cfg=cfg)
                recon_terms.append(recon(cfg, cfg.routing, sources, n_samples=1))
            case PersistentPGDReconLossConfig():
                key = unique_name(cfg)
                sources = PersistentSources(state_key=key, cfg=cfg)
                recon_terms.append(recon(cfg, AllRoutingConfig(), sources, n_samples=1))
            case NonlinearityLocalityLossConfig():
                assert nonlinearity is None
                nonlinearity = NonlinearityTerm(unique_name(cfg), cfg.coeff, cfg)

    return faith, imp, tuple(recon_terms), nonlinearity


def build_objective(
    loss_metrics: Sequence[AnyLossMetricConfig],
    site_names: tuple[str, ...],
) -> LossSurface:
    """Build the closed plain-VPD objective, rejecting incomplete authored surfaces."""
    faith, imp, recon_terms, nonlinearity = _collect_terms(loss_metrics, site_names)
    assert faith is not None and imp is not None, (
        f"need FaithfulnessLoss + ImportanceMinimalityLoss, got {[m.type for m in loss_metrics]}"
    )
    assert recon_terms, "no recon loss terms configured"
    return LossSurface(faith, imp, recon_terms, nonlinearity)


def build_recon_terms(
    loss_metrics: Sequence[AnyLossMetricConfig],
    site_names: tuple[str, ...],
) -> tuple[AnyReconLossTerm, ...]:
    """Just the recon Σ of an authored loss list — the persistent-source layout derives
    from these (`recon.persistent_configs`), so state init shares this walk with both
    objective builders instead of demanding one builder's completeness rules."""
    return _collect_terms(loss_metrics, site_names)[2]


def build_targeted_objective(
    loss_metrics: Sequence[TargetedLossMetricConfig],
    nontarget: NontargetConfig,
    site_names: tuple[str, ...],
) -> TargetedObjective:
    """Build the closed two-pass tPD objective (SPEC §11).

    `loss_metrics` authors the TARGET pass — typed by `TargetedLossMetricConfig`, which
    has no faithfulness member (T3: the delta is the unpenalized off-target escape valve,
    so a targeted config cannot spell a faithfulness role). The non-target pass is
    authored directly on `nontarget` — never derived from the target list — and its
    importance-minimality shares the target's penalty config (shape + anneal) by
    construction, at the non-target pass's own coefficient."""
    faith, imp, recon_terms, nonlinearity = _collect_terms(loss_metrics, site_names)
    # The library boundary for lists built outside the schema; unreachable for a parsed
    # TargetedPDConfig.
    assert faith is None, "a targeted loss list carried a FaithfulnessLossConfig (SPEC T3)"
    assert nonlinearity is None, (
        "a targeted loss list carried a NonlinearityLocalityLossConfig (SPEC S36/T3)"
    )
    assert imp is not None, f"need a ImportanceMinimalityLoss, got {[m.type for m in loss_metrics]}"
    assert imp.cfg.frequency is None or imp.cfg.frequency.ema_halflife_steps is None, (
        "frequency.ema_halflife_steps is not implemented for the targeted (tPD) objective "
        "(SPEC S8'' — plain PD only; the TargetedPDConfig validator carries the why)"
    )
    assert recon_terms, "no recon loss terms configured"

    nt_terms: list[ReconLossTerm[StochasticSources | ConstantSources | UnmaskedNoDeltaSources]] = []
    for cfg in nontarget.recon:
        assert cfg.coeff is not None  # non-None at parse (NontargetConfig); narrows the type
        name = cfg.name if cfg.name is not None else cfg.type
        assert name not in {t.name for t in nt_terms}, f"duplicate non-target loss {name!r}"
        routing, sources, n_samples = _nontarget_recon_parts(cfg)
        # `hidden_acts_reconstruction=None` structurally: this option is target-pass-only
        # (SPEC T5), refused at parse by `NontargetConfig`.
        nt_terms.append(
            ReconLossTerm(
                name,
                cfg.coeff,
                routing_sampler_from_config(routing, site_names, n_samples),
                sources,
                None,
            )
        )
    return TargetedObjective(
        target=TargetPass(imp=imp, recon=recon_terms),
        nontarget=NontargetPass(recon=tuple(nt_terms), impmin_coeff=nontarget.impmin_coeff),
    )


def _nontarget_recon_parts(
    cfg: NontargetReconLossMetricConfig,
) -> tuple[SubsetRoutingType, StochasticSources | ConstantSources | UnmaskedNoDeltaSources, int]:
    """The `(routing, sources, n_samples)` family of one recon config the non-target pass
    admits (SPEC T5): the stochastic/constant-source types — shared verbatim with the
    plain objective's arms, which widen through `recon` — plus the non-target-only
    unmasked-no-delta term (T4's one delta-off exception)."""
    match cfg:
        case CIMaskedReconLossConfig():
            return AllRoutingConfig(), ConstantSources(0.0), 1
        case CIMaskedReconSubsetLossConfig():
            return cfg.routing, ConstantSources(0.0), 1
        case StochasticReconLossConfig():
            return AllRoutingConfig(), StochasticSources(), cfg.n_mask_samples
        case StochasticReconSubsetLossConfig():
            return cfg.routing, StochasticSources(), cfg.n_mask_samples
        case UnmaskedNoDeltaReconLossConfig():
            return AllRoutingConfig(), UnmaskedNoDeltaSources(), 1
