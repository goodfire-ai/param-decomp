"""Host-side recon-budget controller + capacity-lifecycle state machine (task #811).

Pure logic over settled window metrics — no jax state, no trainer coupling: the loop
feeds it one `WindowSummary` per controller window and applies the returned action
(`StepControls` fields + a lifecycle event). Spec:
lore 2026-08-02--design--reconstruction-budget-control-and-demand-triggered-capacity-birth
+ lore 2026-08-02--addendum--capacity-lifecycle-needs-column-generation-not-only-feasibility-birth.

Coordinates: the controller moves `log c` where `c` multiplies the COMBINED imp+freq
complexity force (the reciprocal of the reconstruction dual — violation drives `c`
DOWN). Gain-free by construction: multiplicative bracketing halves the step on every
violation/slack sign flip, and `c = 0` is an explicit OFF state rather than a tuned
positive floor — feasibility birth requires violation with complexity fully off.
"""

import math
from dataclasses import dataclass, replace
from enum import Enum


@dataclass(frozen=True)
class ControllerConfig:
    tau: float
    """Dimensionless recon-damage tolerance: violation is `r_adv > tau`, with
    `r_adv = (R - R_unmasked) / (R_zero - R_unmasked)` computed by the caller."""
    noise_margin: float
    """|EMA(g)| below this reads as on-target: neither violation nor slack."""
    initial_log_step: float = math.log(2.0)
    min_log_step: float = math.log(1.05)
    """Bracketing resolution: at a smaller step, persistent violation transitions to OFF
    and persistent slack holds (further complexity increase is noise-chasing)."""
    dwell_windows: int = 3
    """Consecutive qualifying windows required for any lifecycle event."""
    plateau_rtol: float = 0.02
    """Relative settled-complexity change below which complexity counts as plateaued
    (the column-generation trigger, with recon on-budget)."""


class Phase(Enum):
    CONTROL = "control"
    OFF = "off"  # complexity force fully off (c = 0); the only state feasibility birth can fire from
    BIRTH_PROTECTED = "birth_protected"  # a newborn slot is gate-protected; controller frozen


class Event(Enum):
    NONE = "none"
    FEASIBILITY_BIRTH = "feasibility_birth"
    """Recon violated for `dwell_windows` with complexity OFF: the active set cannot
    represent the target — open capacity (GradMax slot)."""
    COLUMN_PROBE = "column_probe"
    """Recon on-budget and settled complexity plateaued for `dwell_windows`: trial a
    protected slot, to be ACCEPTED only if settled complexity falls at matched recon,
    else rolled back (this event alone makes physical C a true upper bound; without it
    the controller can oscillate dense-SVD <-> pruned-SVD and never use the tail)."""
    CAPACITY_EXHAUSTED = "capacity_exhausted"
    """A birth was demanded but no inactive slot exists — a reportable terminal state,
    never a silent clip."""


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
    log_step: float
    last_sign: int  # sign of the previous window's g (0 before the first move)
    dwell: int  # consecutive qualifying windows toward the phase's pending event
    prev_complexity: float | None
    protect_windows_left: int

    @staticmethod
    def initial(log_c: float, cfg: ControllerConfig) -> "ControllerState":
        return ControllerState(
            phase=Phase.CONTROL,
            log_c=log_c,
            log_step=cfg.initial_log_step,
            last_sign=0,
            dwell=0,
            prev_complexity=None,
            protect_windows_left=0,
        )

    @property
    def c(self) -> float:
        return 0.0 if self.phase is Phase.OFF else math.exp(self.log_c)


@dataclass(frozen=True)
class Action:
    state: ControllerState
    event: Event


def controller_update(
    state: ControllerState, window: WindowSummary, cfg: ControllerConfig, protect_windows: int
) -> Action:
    """One controller decision from one settled window. The caller applies `state.c` as
    the next windows' complexity_scale, executes the event (birth surgery / probe /
    report), and on birth re-enters via the returned BIRTH_PROTECTED state."""
    g = window.r_adv - cfg.tau
    sign = 0 if abs(g) <= cfg.noise_margin else (1 if g > 0 else -1)

    match state.phase:
        case Phase.BIRTH_PROTECTED:
            left = state.protect_windows_left - 1
            if left > 0:
                return Action(replace(state, protect_windows_left=left), Event.NONE)
            # protection expires -> resume control; bracketing restarts (the newborn
            # changed the landscape, the old bracket is stale)
            return Action(
                replace(
                    state,
                    phase=Phase.CONTROL,
                    log_step=cfg.initial_log_step,
                    last_sign=0,
                    dwell=0,
                    protect_windows_left=0,
                ),
                Event.NONE,
            )

        case Phase.OFF:
            if sign > 0:  # still violating with complexity fully off -> demand capacity
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
                    ),
                    Event.FEASIBILITY_BIRTH,
                )
            if sign < 0:  # slack returned -> re-enter control at the last coefficient
                return Action(
                    replace(
                        state,
                        phase=Phase.CONTROL,
                        log_step=cfg.initial_log_step,
                        last_sign=0,
                        dwell=0,
                    ),
                    Event.NONE,
                )
            return Action(replace(state, dwell=0), Event.NONE)

        case Phase.CONTROL:
            if sign == 0:
                # on-target: complexity-plateau check drives column generation
                plateaued = (
                    state.prev_complexity is not None
                    and abs(window.complexity - state.prev_complexity)
                    <= cfg.plateau_rtol * abs(state.prev_complexity)
                )
                dwell = state.dwell + 1 if plateaued else 0
                next_state = replace(state, dwell=dwell, prev_complexity=window.complexity)
                if dwell < cfg.dwell_windows:
                    return Action(next_state, Event.NONE)
                if not window.spare_slot_exists:
                    return Action(replace(next_state, dwell=0), Event.CAPACITY_EXHAUSTED)
                return Action(
                    replace(
                        next_state,
                        phase=Phase.BIRTH_PROTECTED,
                        dwell=0,
                        protect_windows_left=protect_windows,
                    ),
                    Event.COLUMN_PROBE,
                )
            # bracketing move: violation -> c down, slack -> c up; halve on sign flip
            log_step = (
                state.log_step / 2.0
                if state.last_sign != 0 and sign != state.last_sign
                else state.log_step
            )
            if log_step < cfg.min_log_step and sign > 0:
                # bracket exhausted while still violating: complexity force OFF
                return Action(
                    replace(state, phase=Phase.OFF, dwell=0, last_sign=0), Event.NONE
                )
            if log_step < cfg.min_log_step:
                return Action(  # exhausted with slack: hold (chasing noise upward)
                    replace(state, last_sign=sign, dwell=0, prev_complexity=window.complexity),
                    Event.NONE,
                )
            return Action(
                replace(
                    state,
                    log_c=state.log_c - sign * log_step,
                    log_step=log_step,
                    last_sign=sign,
                    dwell=0,
                    prev_complexity=window.complexity,
                ),
                Event.NONE,
            )
