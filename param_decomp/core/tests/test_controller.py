"""Recon-budget controller state machine: sign correctness, bracketing, OFF state,
the two lifecycle events, capacity-exhausted reporting, and protection expiry."""

import math

from param_decomp.core.controller import (
    Action,
    ControllerConfig,
    ControllerState,
    Event,
    Phase,
    WindowSummary,
    controller_update,
)

CFG = ControllerConfig(tau=0.1, noise_margin=0.01, dwell_windows=2, plateau_rtol=0.02)


def window(r_adv: float, complexity: float = 100.0, spare: bool = True) -> WindowSummary:
    return WindowSummary(r_adv=r_adv, complexity=complexity, spare_slot_exists=spare)


def drive(state: ControllerState, windows: list[WindowSummary]) -> list[Action]:
    actions = []
    for w in windows:
        action = controller_update(state, w, CFG, protect_windows=3)
        actions.append(action)
        state = action.state
    return actions


def test_violation_drives_c_down_and_slack_up() -> None:
    s0 = ControllerState.initial(math.log(1.0), CFG)
    violated = controller_update(s0, window(r_adv=0.5), CFG, protect_windows=3).state
    assert violated.log_c < s0.log_c
    slack = controller_update(s0, window(r_adv=0.0), CFG, protect_windows=3).state
    assert slack.log_c > s0.log_c


def test_sign_flip_halves_the_bracketing_step() -> None:
    s0 = ControllerState.initial(math.log(1.0), CFG)
    a1 = controller_update(s0, window(0.5), CFG, 3)  # violation
    a2 = controller_update(a1.state, window(0.0), CFG, 3)  # slack -> flip
    assert a2.state.log_step == a1.state.log_step / 2


def test_exhausted_bracket_under_violation_goes_off_then_feasibility_birth() -> None:
    state = ControllerState.initial(math.log(1.0), CFG)
    actions = drive(state, [window(0.5), window(0.0)] * 8 + [window(0.5)] * 6)
    phases = [a.state.phase for a in actions]
    assert Phase.OFF in phases
    events = [a.event for a in actions]
    assert Event.FEASIBILITY_BIRTH in events
    birth_idx = events.index(Event.FEASIBILITY_BIRTH)
    assert phases[birth_idx] is Phase.BIRTH_PROTECTED
    # dwell was respected: the windows immediately before the birth were OFF + violating
    assert phases[birth_idx - 1] is Phase.OFF


def test_no_birth_while_constraint_is_slack() -> None:
    state = ControllerState.initial(math.log(1.0), CFG)
    actions = drive(state, [window(0.0)] * 20)
    assert all(a.event in (Event.NONE, Event.COLUMN_PROBE) for a in actions)
    assert Event.FEASIBILITY_BIRTH not in [a.event for a in actions]


def test_on_target_plateau_fires_column_probe() -> None:
    state = ControllerState.initial(math.log(1.0), CFG)
    # r_adv exactly on tau (within noise margin), complexity flat -> plateau dwell -> probe
    actions = drive(state, [window(CFG.tau, complexity=50.0)] * 4)
    assert Event.COLUMN_PROBE in [a.event for a in actions]


def test_capacity_exhausted_is_reported_not_clipped() -> None:
    state = ControllerState.initial(math.log(1.0), CFG)
    actions = drive(
        state, [window(CFG.tau, complexity=50.0, spare=False)] * 6
    )
    assert Event.CAPACITY_EXHAUSTED in [a.event for a in actions]
    assert all(a.state.phase is not Phase.BIRTH_PROTECTED for a in actions)


def test_protection_expires_back_to_control_with_fresh_bracket() -> None:
    protected = ControllerState(
        phase=Phase.BIRTH_PROTECTED,
        log_c=0.0,
        log_step=CFG.min_log_step / 4,
        last_sign=1,
        dwell=0,
        prev_complexity=None,
        protect_windows_left=2,
    )
    a1 = controller_update(protected, window(0.5), CFG, 3)
    assert a1.state.phase is Phase.BIRTH_PROTECTED and a1.event is Event.NONE
    a2 = controller_update(a1.state, window(0.5), CFG, 3)
    assert a2.state.phase is Phase.CONTROL
    assert a2.state.log_step == CFG.initial_log_step  # bracket restarted


def test_off_state_c_is_exactly_zero_and_slack_reenters_control() -> None:
    off = ControllerState(
        phase=Phase.OFF, log_c=-3.0, log_step=CFG.initial_log_step,
        last_sign=0, dwell=0, prev_complexity=None, protect_windows_left=0,
    )  # fmt: skip
    assert off.c == 0.0
    back = controller_update(off, window(0.0), CFG, 3).state
    assert back.phase is Phase.CONTROL and back.c == math.exp(-3.0)
