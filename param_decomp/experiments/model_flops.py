"""Prepare architecture costs once, then evaluate useful FLOPs for each completed step."""

from bisect import bisect_left, bisect_right
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Protocol

from param_decomp.core.ci_fn.architecture import CIFnArchitectureFootprint
from param_decomp.core.components import SiteSpec
from param_decomp.core.configs import (
    MergedStochasticSubsetPooledPPGDReconLossConfig,
    MergedStochasticSubsetPPGDReconLossConfig,
    NontargetConfig,
    PDConfig,
    ReconstructionAuxiliariesMixin,
    TargetedPDConfig,
)
from param_decomp.core.flops.ci import ci_fn_flops
from param_decomp.core.flops.model import (
    FlopsTerm,
    ReconstructionPlan,
    StreamFlops,
    decomposition_step_flops,
    nontarget_reconstruction_plans,
    reconstruction_plans,
    targeted_step_flops,
)
from param_decomp.core.flops.optimizer import OptimizerFlops, prepare_optimizer_flops
from param_decomp.core.flops.target import GradientTarget, TargetPassFlops
from param_decomp.core.flops.types import ForwardBackwardFlops, StepFlops, UsefulFlops
from param_decomp.core.model import PositionAxis
from param_decomp.core.schedule import ScheduleConfig, get_scheduled_value


class TargetPassCalculator(Protocol):
    def __call__(
        self,
        sites: tuple[SiteSpec, ...],
        *,
        batch_size: int,
        gradients: GradientTarget,
        include_frozen_paths: bool,
        capture_keys: frozenset[str],
    ) -> TargetPassFlops: ...


@dataclass(frozen=True)
class StreamDescription:
    target: TargetPassCalculator
    sites: tuple[SiteSpec, ...]
    ci_fn_arch: CIFnArchitectureFootprint
    batch_size: int
    positions: PositionAxis
    n_selected_blocks_per_token: int | None
    prefix: str


def source_gradient_flops(stream: StreamDescription, plan: ReconstructionPlan) -> int:
    return stream.target(
        stream.sites,
        batch_size=stream.batch_size,
        gradients="sources",
        include_frozen_paths=plan.include_frozen_paths,
        capture_keys=plan.auxiliary_captures,
    ).additional_backward


def stream_flops(stream: StreamDescription, plans: tuple[ReconstructionPlan, ...]) -> StreamFlops:
    reconstructions: list[FlopsTerm] = []
    reused_components: dict[str, ForwardBackwardFlops] = {}
    for plan in plans:
        name = f"{stream.prefix}reconstruction/{plan.name}"
        main = stream.target(
            stream.sites,
            batch_size=stream.batch_size,
            gradients="components",
            include_frozen_paths=plan.include_frozen_paths,
            capture_keys=plan.auxiliary_captures,
        )
        reusable_forward, reusable_backward = main.shared_forward, 0
        for key, cost in main.reusable_components.items():
            previous = reused_components.get(key)
            if previous is not None:
                reusable_forward += previous.forward
                reusable_backward += min(previous.backward, cost.backward)
                cost = ForwardBackwardFlops(cost.forward, max(previous.backward, cost.backward))
            reused_components[key] = cost
        reconstructions.append(
            FlopsTerm(
                name,
                ForwardBackwardFlops(
                    main.flops.forward - reusable_forward,
                    main.flops.backward - reusable_backward,
                ),
                1,
            )
        )
        if plan.n_ascent_steps:
            ascent = stream.target(
                stream.sites,
                batch_size=stream.batch_size,
                gradients="sources",
                include_frozen_paths=plan.include_frozen_paths,
                capture_keys=plan.source_captures,
            )
            reusable = ascent.shared_forward + sum(
                cost.forward for cost in ascent.reusable_components.values()
            )
            reconstructions.append(
                FlopsTerm(
                    f"{name}/ascent",
                    ForwardBackwardFlops(ascent.flops.forward - reusable, ascent.flops.backward),
                    plan.n_ascent_steps,
                )
            )
        if plan.retake_source_gradient and plan.source_gradient_fraction > 0:
            reconstructions.append(
                FlopsTerm(
                    f"{name}/source_gradient",
                    ForwardBackwardFlops(0, source_gradient_flops(stream, plan)),
                    plan.source_gradient_fraction,
                )
            )
    captures = stream.ci_fn_arch.capture_keys.union(*(plan.auxiliary_captures for plan in plans))
    clean = stream.target(
        (),
        batch_size=stream.batch_size,
        gradients="none",
        include_frozen_paths=False,
        capture_keys=captures,
    ).flops.forward
    ci = ci_fn_flops(
        stream.ci_fn_arch,
        stream.sites,
        stream.batch_size,
        stream.positions,
        n_selected_blocks_per_token=stream.n_selected_blocks_per_token,
    )
    return StreamFlops(
        clean,
        ci,
        tuple(reconstructions),
    )


def _auxiliary_schedules(pd: PDConfig | TargetedPDConfig) -> tuple[ScheduleConfig, ...]:
    return tuple(
        auxiliary.coeff
        for loss in pd.loss_metrics
        if isinstance(loss, ReconstructionAuxiliariesMixin)
        for auxiliary in loss.auxiliaries
        if isinstance(auxiliary.coeff, ScheduleConfig)
    )


def _objective_change_steps(schedules: tuple[ScheduleConfig, ...], n_steps: int) -> tuple[int, ...]:
    """Find changes in objective activity without evaluating every training step."""
    changes = {0}
    for schedule in schedules:
        boundaries = (
            0,
            *(
                bisect_left(range(n_steps), knot.at, key=lambda step: step / max(n_steps - 1, 1))
                for knot in schedule.points[1:-1]
            ),
            n_steps,
        )
        for start, stop in zip(boundaries, boundaries[1:], strict=False):
            if start == stop:
                continue
            active_at_start = get_scheduled_value(start, n_steps, schedule) > 0
            if start and (get_scheduled_value(start - 1, n_steps, schedule) > 0) != active_at_start:
                changes.add(start)
            # Each knot interval is monotone, including floating-point plateaus at zero.
            if (get_scheduled_value(stop - 1, n_steps, schedule) > 0) != active_at_start:
                changes.add(
                    bisect_left(
                        range(n_steps),
                        True,
                        lo=start,
                        hi=stop,
                        key=lambda step: (get_scheduled_value(step, n_steps, schedule) > 0)
                        != active_at_start,
                    )
                )
    return tuple(sorted(changes))


@dataclass(frozen=True)
class _ScheduledSourceFlops:
    fraction: ScheduleConfig
    flops: int


@dataclass(frozen=True)
class _ModelPhase:
    start_step: int
    fixed_flops: float
    source_gradients: tuple[_ScheduledSourceFlops, ...]


def _source_fractions(pd: PDConfig | TargetedPDConfig) -> dict[str, ScheduleConfig]:
    return {
        loss.name or loss.type: loss.adv_fraction
        for loss in pd.loss_metrics
        if isinstance(
            loss,
            (
                MergedStochasticSubsetPPGDReconLossConfig,
                MergedStochasticSubsetPooledPPGDReconLossConfig,
            ),
        )
    }


def _prepare_stream_phase(
    stream: StreamDescription,
    plans: tuple[ReconstructionPlan, ...],
    fractions: dict[str, ScheduleConfig],
) -> tuple[StreamFlops, tuple[_ScheduledSourceFlops, ...]]:
    fixed_plans: list[ReconstructionPlan] = []
    source_gradients: list[_ScheduledSourceFlops] = []
    for plan in plans:
        fraction = fractions.get(plan.name)
        if plan.retake_source_gradient and fraction is not None:
            source_gradients.append(
                _ScheduledSourceFlops(fraction, source_gradient_flops(stream, plan))
            )
            plan = replace(plan, retake_source_gradient=False)
        fixed_plans.append(plan)
    return stream_flops(stream, tuple(fixed_plans)), tuple(source_gradients)


def _step_flops(
    phases: tuple[_ModelPhase, ...],
    optimizer_flops: Callable[[int], OptimizerFlops],
    n_steps: int,
) -> StepFlops:
    starts = tuple(phase.start_step for phase in phases)

    def compute(step: int) -> UsefulFlops:
        if not 0 <= step < n_steps:
            raise ValueError("Training step must be within the configured run")
        phase = phases[bisect_right(starts, step) - 1]
        model = phase.fixed_flops + sum(
            term.flops * get_scheduled_value(step, n_steps, term.fraction)
            for term in phase.source_gradients
        )
        return UsefulFlops(model, optimizer_flops(step).total)

    return compute


def prepare_ordinary_step_flops(
    pd: PDConfig,
    stream: StreamDescription,
) -> StepFlops:
    fractions = _source_fractions(pd)
    phases = []
    for step in _objective_change_steps(_auxiliary_schedules(pd), pd.steps):
        fixed, source_gradients = _prepare_stream_phase(
            stream, reconstruction_plans(pd, step), fractions
        )
        phases.append(
            _ModelPhase(
                step, decomposition_step_flops(pd, stream.sites, fixed).total, source_gradients
            )
        )
    return _step_flops(
        tuple(phases),
        prepare_optimizer_flops(pd, stream.sites, stream.ci_fn_arch, stream.positions),
        pd.steps,
    )


def prepare_targeted_step_flops(
    pd: TargetedPDConfig,
    nontarget: NontargetConfig,
    target_stream: StreamDescription,
    nontarget_stream: StreamDescription,
) -> StepFlops:
    fractions = _source_fractions(pd)
    phases = []
    nontarget_flops = stream_flops(nontarget_stream, nontarget_reconstruction_plans(nontarget))
    for step in _objective_change_steps(_auxiliary_schedules(pd), pd.steps):
        target, source_gradients = _prepare_stream_phase(
            target_stream, reconstruction_plans(pd, step), fractions
        )
        phases.append(
            _ModelPhase(step, targeted_step_flops(target, nontarget_flops).total, source_gradients)
        )
    return _step_flops(
        tuple(phases),
        prepare_optimizer_flops(
            pd, target_stream.sites, target_stream.ci_fn_arch, target_stream.positions
        ),
        pd.steps,
    )
