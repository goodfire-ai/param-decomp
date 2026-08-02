"""Host-side recon-budget controller + capacity-lifecycle state machine (task #811).

Pure logic over settled window metrics — no jax state, no trainer coupling: the loop
feeds it one `WindowSummary` per controller window and applies the returned action
(`StepControls` fields + a lifecycle event). Spec:
lore 2026-08-02--design--reconstruction-budget-control-and-demand-triggered-capacity-birth
+ lore 2026-08-02--addendum--capacity-lifecycle-needs-column-generation-not-only-feasibility-birth.

Coordinates: the controller moves `log c` where `c` multiplies the COMBINED imp+freq
complexity force (the reciprocal of the reconstruction dual — violation drives `c`
DOWN, i.e. the feasibility boundary sits BELOW a violating `log_c` and ABOVE a slack
one). The search is a true bracket, so one-sided sign sequences terminate: slack at
`x` raises the feasible endpoint `lo`; violation at `x` lowers the infeasible endpoint
`hi`; with both, bisect; with only `hi`, probe OFF (c = 0) — OFF-slack re-enters with
a downward expansion toward `hi`, OFF-violation is the feasibility-birth condition;
with only `lo`, expand upward under a hard guard. `c = 0` is an explicit phase, not a
tuned positive floor. Gain-free: no controller learning rate exists to sweep."""

import math
from dataclasses import dataclass, replace
from enum import Enum


@dataclass(frozen=True)
class ControllerConfig:
    tau: float
    """Dimensionless recon-damage tolerance: violation is `r_adv > tau`, with
    `r_adv = (R - R_unmasked) / (R_zero - R_unmasked)` computed by the caller."""
    noise_margin: float
    """|r_adv - tau| below this reads as on-target: neither violation nor slack."""
    max_log_c: float
    """Hard guard on upward expansion (a sanity rail against unbounded growth under
    permanent slack, not an operating point)."""
    expand_log_step: float = math.log(4.0)
    resolution: float = math.log(1.05)
    """Bracket widths below this terminate the search: hold at `lo` (the largest
    known-feasible complexity scale)."""
    dwell_windows: int = 3
    """Consecutive qualifying windows required for any lifecycle event."""
    plateau_rtol: float = 0.02
    """Relative settled-complexity change below which complexity counts as plateaued
    (the column-generation trigger, with recon on-budget)."""
    probe_cooldown_windows: int = 6
    """Windows to suppress plateau-triggered probes after a rejected COLUMN_PROBE."""
    max_rejected_probes: int = 3
    """Rejected probes before the plateau is declared terminal (NO_IMPROVING_COLUMN)."""


class Phase(Enum):
    CONTROL = "control"
    OFF = "off"  # complexity force fully off (c = 0); the only phase feasibility birth fires from
    BIRTH_PROTECTED = "birth_protected"  # feasibility trial; exits only via explicit verdict
    PROBE_PENDING = "probe_pending"  # a COLUMN_PROBE trial is out; exits ONLY via a verdict


class Event(Enum):
    NONE = "none"
    FEASIBILITY_BIRTH = "feasibility_birth"
    """Recon violated for `dwell_windows` with complexity OFF: the active set cannot
    represent the target — open capacity (GradMax slot)."""
    COLUMN_PROBE = "column_probe"
    """Recon on-budget and settled complexity plateaued for `dwell_windows`: trial a
    protected slot, ACCEPTED by the caller only if settled complexity falls at matched
    recon, else rolled back + reported via `probe_rejected` (this event alone makes
    physical C a true upper bound)."""
    CAPACITY_EXHAUSTED = "capacity_exhausted"
    """A birth was demanded but no inactive slot exists — reportable, never a silent clip."""
    NO_IMPROVING_COLUMN = "no_improving_column"
    """`max_rejected_probes` consecutive probe rejections at the same plateau: stop
    probing; the operating point stands."""


@dataclass(frozen=True)
class WindowSummary:
    """Settled means over one controller window, computed by the caller."""

    r_adv: float  # normalized recon damage, in tau's units
    complexity: float  # the combined S = imp_lp + beta_freq * freq (unscaled by c)
    spare_slot_exists: bool


@dataclass(frozen=True)
class ControllerState:
    phase: Phase
    log_c: float  # meaningful only outside OFF
    lo: float | None  # highest log_c OBSERVED slack/on-target (feasible endpoint)
    hi: float | None  # lowest log_c OBSERVED violating (infeasible endpoint)
    dwell: int
    prev_complexity: float | None
    protect_windows_left: int
    probe_cooldown: int
    rejected_probes: int

    @staticmethod
    def initial(log_c: float) -> "ControllerState":
        return ControllerState(
            phase=Phase.CONTROL,
            log_c=log_c,
            lo=None,
            hi=None,
            dwell=0,
            prev_complexity=None,
            protect_windows_left=0,
            probe_cooldown=0,
            rejected_probes=0,
        )

    @property
    def c(self) -> float:
        """BIRTH_PROTECTED is reachable only via FEASIBILITY_BIRTH (probes have their own
        phase), and that birth fired because complexity-OFF was still infeasible —
        reapplying pressure mid-growth would kill the newborn, so complexity stays OFF
        through protected settling."""
        if self.phase in (Phase.OFF, Phase.BIRTH_PROTECTED):
            return 0.0
        return math.exp(self.log_c)


@dataclass(frozen=True)
class Action:
    state: ControllerState
    event: Event


def birth_accepted(state: ControllerState, cfg: ControllerConfig) -> ControllerState:
    """Caller accepts a protected feasibility birth after it became load-bearing and
    improved the violated reconstruction constraint. The landscape changed: clear the
    stale bracket/plateau lineage and re-enter from a gentler coefficient than the last
    known violation.
    """
    assert state.phase is Phase.BIRTH_PROTECTED, state.phase
    return replace(
        state,
        phase=Phase.CONTROL,
        log_c=state.log_c - cfg.expand_log_step,
        lo=None,
        hi=None,
        dwell=0,
        prev_complexity=None,
        protect_windows_left=0,
        probe_cooldown=0,
        rejected_probes=0,
    )


def birth_rejected(state: ControllerState) -> ControllerState:
    """Caller rolled a failed feasibility birth back to the pre-trial state. Complexity
    remains OFF because the pre-birth active set is still known infeasible; a caller-side
    direction/site cooldown must prevent immediately retrying the identical column.
    """
    assert state.phase is Phase.BIRTH_PROTECTED, state.phase
    return replace(state, phase=Phase.OFF, dwell=0, protect_windows_left=0)


def probe_rejected(state: ControllerState, cfg: ControllerConfig) -> ControllerState:
    """Caller reports a rolled-back COLUMN_PROBE: start the cooldown and count it.
    The trial slot is back to exact null, so control resumes where it left off (the
    plateau is the SAME, so prev_complexity is retained)."""
    assert state.phase is Phase.PROBE_PENDING, state.phase
    return replace(
        state,
        phase=Phase.CONTROL,
        protect_windows_left=0,
        probe_cooldown=cfg.probe_cooldown_windows,
        rejected_probes=state.rejected_probes + 1,
    )


def probe_accepted(state: ControllerState) -> ControllerState:
    """Caller reports an accepted COLUMN_PROBE: the landscape changed — fresh bracket,
    rejection count cleared (a new plateau earns new probes)."""
    assert state.phase is Phase.PROBE_PENDING, state.phase
    return replace(
        state,
        phase=Phase.CONTROL,
        lo=None,
        hi=None,
        dwell=0,
        prev_complexity=None,
        protect_windows_left=0,
        probe_cooldown=0,
        rejected_probes=0,
    )


def _next_log_c(state: ControllerState, cfg: ControllerConfig) -> float | None:
    """The bracket search's next coefficient, or None to transition OFF. Requires the
    endpoints already updated for the current window."""
    match state.lo, state.hi:
        case lo, hi if lo is not None and hi is not None:
            if hi - lo < cfg.resolution:
                return lo  # converged: hold at the largest known-feasible scale
            return (lo + hi) / 2.0
        case None, hi if hi is not None:
            return None  # never seen feasible at c > 0: probe OFF
        case lo, None if lo is not None:
            return min(lo + cfg.expand_log_step, cfg.max_log_c)  # expand under the guard
    raise AssertionError("unreachable: a window always sets one endpoint")


def controller_update(
    state: ControllerState, window: WindowSummary, cfg: ControllerConfig, protect_windows: int
) -> Action:
    """One controller decision from one settled window. The caller applies `state.c` as
    the next windows' complexity_scale and executes the event (birth surgery / probe /
    report); probe verdicts come back via `probe_accepted` / `probe_rejected`."""
    g = window.r_adv - cfg.tau
    sign = 0 if abs(g) <= cfg.noise_margin else (1 if g > 0 else -1)

    match state.phase:
        case Phase.PROBE_PENDING:
            # a trial is out: the caller drives it and reports via probe_accepted /
            # probe_rejected — time NEVER ages a probe back into control.
            return Action(state, Event.NONE)

        case Phase.BIRTH_PROTECTED:
            # Time is not evidence that a feasibility column became useful. The caller
            # holds its gate open, releases/resettles it, then reports an explicit verdict
            # through birth_accepted / birth_rejected.
            return Action(state, Event.NONE)

        case Phase.OFF:
            if sign > 0:  # violating with complexity fully off -> demand capacity
                dwell = state.dwell + 1
                if dwell < cfg.dwell_windows:
                    return Action(replace(state, dwell=dwell), Event.NONE)
                if not window.spare_slot_exists:
                    return Action(replace(state, dwell=0), Event.CAPACITY_EXHAUSTED)
                return Action(
                    replace(
                        state,
                        phase=Phase.BIRTH_PROTECTED,
                        dwell=0,
                        protect_windows_left=protect_windows,
                    ),  # fmt: skip
                    Event.FEASIBILITY_BIRTH,
                )
            # OFF is feasible: re-enter control expanding DOWNWARD toward hi (feasibility
            # exists somewhere below it; OFF itself is not a finite bracket endpoint)
            assert state.hi is not None, "OFF is only reachable after a violation set hi"
            return Action(
                replace(
                    state,
                    phase=Phase.CONTROL,
                    log_c=state.hi - cfg.expand_log_step,
                    dwell=0,
                ),
                Event.NONE,
            )

        case Phase.CONTROL:
            cooled = replace(state, probe_cooldown=max(0, state.probe_cooldown - 1))
            converged_at_lo = (
                state.lo is not None
                and state.hi is not None
                and state.hi - state.lo < cfg.resolution
                and state.log_c == state.lo
            )
            # A converged bracket holds at feasible `lo`, whose sampled point is
            # typically slack by more than the deadband (finite resolution) — the
            # plateau/column-probe logic must engage there too, or case-B probes
            # starve forever behind a permanent sign<0.
            if sign == 0 or (sign < 0 and converged_at_lo):
                plateaued = cooled.prev_complexity is not None and abs(
                    window.complexity - cooled.prev_complexity
                ) <= cfg.plateau_rtol * abs(cooled.prev_complexity)
                dwell = cooled.dwell + 1 if plateaued else 0
                next_state = replace(
                    cooled,
                    lo=max(cooled.lo, cooled.log_c) if cooled.lo is not None else cooled.log_c,
                    dwell=dwell,
                    prev_complexity=window.complexity,
                )
                if dwell < cfg.dwell_windows or state.probe_cooldown > 0:
                    return Action(next_state, Event.NONE)
                if next_state.rejected_probes >= cfg.max_rejected_probes:
                    return Action(replace(next_state, dwell=0), Event.NO_IMPROVING_COLUMN)
                if not window.spare_slot_exists:
                    return Action(replace(next_state, dwell=0), Event.CAPACITY_EXHAUSTED)
                return Action(
                    replace(next_state, phase=Phase.PROBE_PENDING, dwell=0),
                    Event.COLUMN_PROBE,
                )
            if sign > 0:
                updated = replace(
                    cooled,
                    hi=min(cooled.hi, cooled.log_c) if cooled.hi is not None else cooled.log_c,
                    dwell=0,
                    prev_complexity=window.complexity,
                )
            else:
                updated = replace(
                    cooled,
                    lo=max(cooled.lo, cooled.log_c) if cooled.lo is not None else cooled.log_c,
                    dwell=0,
                    prev_complexity=window.complexity,
                )
            target = _next_log_c(updated, cfg)
            if target is None:
                return Action(replace(updated, phase=Phase.OFF), Event.NONE)
            return Action(replace(updated, log_c=target), Event.NONE)


@dataclass(frozen=True)
class SettlementConfig:
    points: int
    rtol: float
    atol: float

    def __post_init__(self) -> None:
        assert self.points >= 2, self.points
        assert self.rtol >= 0.0, self.rtol
        assert self.atol >= 0.0, self.atol


@dataclass(frozen=True)
class ControllerObservation:
    """One authored-referee read paired with the primal window that preceded it.

    ``qualified`` is the composition root's adversary-strength/validity gate. An
    under-powered, failed, or stale referee read resets settling rather than being fed to
    the outer controller as evidence of feasibility.
    """

    r_adv: float
    complexity: float
    spare_slot_exists: bool
    qualified: bool = True


@dataclass(frozen=True)
class SettlementState:
    observations: tuple[ControllerObservation, ...]

    @staticmethod
    def initial() -> "SettlementState":
        return SettlementState(())


def authored_observable(
    record: dict[str, float], metric_key: str, offset: float, scale: float
) -> float:
    """Read an explicitly-authored scalar in fixed affine units.

    Missing/non-finite values are bugs, not a reason to silently reuse the last referee
    read. ``scale`` is fixed config state (never a moving model-produced denominator).
    """
    assert scale > 0.0, scale
    assert metric_key in record, (metric_key, sorted(record))
    value = float(record[metric_key])
    assert math.isfinite(value), (metric_key, value)
    return (value - offset) / scale


def _settled(values: tuple[float, ...], cfg: SettlementConfig) -> bool:
    mean = sum(values) / len(values)
    return max(values) - min(values) <= cfg.atol + cfg.rtol * abs(mean)


def observe_settled_window(
    state: SettlementState,
    observation: ControllerObservation,
    cfg: SettlementConfig,
) -> tuple[SettlementState, WindowSummary | None]:
    """Accumulate independent referee/primal reads and emit only a settled window.

    The buffer slides while either reconstruction or declared complexity is moving, then
    clears after emission so controller dwell counts independent settled windows. Capacity
    availability is lifecycle state, not a noisy metric; changing it mid-window is a caller
    bug (the caller must reset settling after every birth/rollback).
    """
    for value in (observation.r_adv, observation.complexity):
        assert math.isfinite(value), value
    if not observation.qualified:
        return SettlementState.initial(), None

    observations = (*state.observations, observation)[-cfg.points :]
    next_state = SettlementState(observations)
    if len(observations) < cfg.points:
        return next_state, None
    spare_values = {o.spare_slot_exists for o in observations}
    assert len(spare_values) == 1, "capacity changed without resetting the settlement buffer"
    r_values = tuple(o.r_adv for o in observations)
    complexity_values = tuple(o.complexity for o in observations)
    if not (_settled(r_values, cfg) and _settled(complexity_values, cfg)):
        return next_state, None
    return SettlementState.initial(), WindowSummary(
        r_adv=sum(r_values) / len(r_values),
        complexity=sum(complexity_values) / len(complexity_values),
        spare_slot_exists=observation.spare_slot_exists,
    )
