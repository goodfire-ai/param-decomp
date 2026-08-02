"""Recon-budget controller: bracket termination on one-sided signs, bisection, OFF and
the two lifecycle events, probe cooldown/terminal, capacity-exhausted reporting."""

import math

from param_decomp.core.controller import (
    Action,
    ControllerConfig,
    ControllerState,
    Event,
    Phase,
    WindowSummary,
    controller_update,
    probe_accepted,
    probe_rejected,
)

CFG = ControllerConfig(
    tau=0.1, noise_margin=0.01, max_log_c=math.log(1e4), dwell_windows=2,
    probe_cooldown_windows=3, max_rejected_probes=2,
)  # fmt: skip


def window(r_adv: float, complexity: float = 100.0, spare: bool = True) -> WindowSummary:
    return WindowSummary(r_adv=r_adv, complexity=complexity, spare_slot_exists=spare)


def drive(state: ControllerState, windows: list[WindowSummary]) -> list[Action]:
    actions = []
    for w in windows:
        action = controller_update(state, w, CFG, protect_windows=3)
        actions.append(action)
        state = action.state
    return actions


def test_all_violation_reaches_off_then_feasibility_birth_in_finite_windows() -> None:
    state = ControllerState.initial(math.log(1.0))
    actions = drive(state, [window(0.5)] * 8)
    phases = [a.state.phase for a in actions]
    events = [a.event for a in actions]
    # first violating window: hi set, no lo -> immediate OFF probe
    assert phases[0] is Phase.OFF
    # OFF + sustained violation -> dwell -> birth, all within a handful of windows
    assert Event.FEASIBILITY_BIRTH in events
    assert phases[events.index(Event.FEASIBILITY_BIRTH)] is Phase.BIRTH_PROTECTED


def test_monotone_slack_is_bounded_by_the_guard() -> None:
    state = ControllerState.initial(math.log(1.0))
    actions = drive(state, [window(0.0, complexity=c) for c in range(100, 200)])
    assert max(a.state.log_c for a in actions) <= CFG.max_log_c
    assert Event.FEASIBILITY_BIRTH not in [a.event for a in actions]


def test_bracket_bisects_to_the_boundary() -> None:
    # true boundary at log_c = 0: violation iff log_c > 0
    state = ControllerState.initial(math.log(1.0) + 2.0)
    for _ in range(40):
        r = 0.5 if state.log_c > 0 else 0.0
        out = controller_update(state, window(r), CFG, 3)
        state = out.state
        if state.phase is Phase.OFF:  # transient: first window has no lo yet
            state = controller_update(state, window(0.0), CFG, 3).state
    assert state.lo is not None and state.hi is not None
    assert abs(state.lo) < 1.0  # converged near the boundary from either side


def test_off_slack_reenters_below_hi() -> None:
    state = ControllerState.initial(2.0)
    a1 = controller_update(state, window(0.5), CFG, 3)  # violation, no lo -> OFF
    assert a1.state.phase is Phase.OFF and a1.state.hi == 2.0
    a2 = controller_update(a1.state, window(0.0), CFG, 3)  # OFF feasible -> re-enter
    assert a2.state.phase is Phase.CONTROL
    assert a2.state.log_c < a1.state.hi


def test_on_target_plateau_fires_column_probe_and_rejection_cools_down() -> None:
    state = ControllerState.initial(0.0)
    actions = drive(state, [window(CFG.tau, complexity=50.0)] * 4)
    events = [a.event for a in actions]
    assert Event.COLUMN_PROBE in events
    probed = actions[events.index(Event.COLUMN_PROBE)].state
    # rejection -> cooldown suppresses the immediate re-probe
    after = probe_rejected(probed, CFG)
    actions2 = drive(after, [window(CFG.tau, complexity=50.0)] * CFG.probe_cooldown_windows)
    assert Event.COLUMN_PROBE not in [a.event for a in actions2]
    # second rejection at the same plateau -> terminal
    actions3 = drive(actions2[-1].state, [window(CFG.tau, complexity=50.0)] * 10)
    events3 = [a.event for a in actions3]
    if Event.COLUMN_PROBE in events3:
        after2 = probe_rejected(actions3[events3.index(Event.COLUMN_PROBE)].state, CFG)
        actions4 = drive(after2, [window(CFG.tau, complexity=50.0)] * 12)
        assert Event.NO_IMPROVING_COLUMN in [a.event for a in actions4]
        assert Event.COLUMN_PROBE not in [a.event for a in actions4]


def test_probe_accepted_resets_bracket_and_rejections() -> None:
    state = ControllerState(
        phase=Phase.BIRTH_PROTECTED, log_c=0.0, lo=-1.0, hi=1.0, dwell=0,
        prev_complexity=50.0, protect_windows_left=2, probe_cooldown=0, rejected_probes=1,
    )  # fmt: skip
    accepted = probe_accepted(state)
    assert accepted.phase is Phase.CONTROL
    assert accepted.lo is None and accepted.hi is None and accepted.rejected_probes == 0


def test_capacity_exhausted_is_reported_not_clipped() -> None:
    state = ControllerState.initial(0.0)
    actions = drive(state, [window(CFG.tau, complexity=50.0, spare=False)] * 6)
    assert Event.CAPACITY_EXHAUSTED in [a.event for a in actions]
    assert all(a.state.phase is not Phase.BIRTH_PROTECTED for a in actions)


def test_no_feasibility_birth_while_slack() -> None:
    state = ControllerState.initial(0.0)
    actions = drive(state, [window(0.0)] * 20)
    assert Event.FEASIBILITY_BIRTH not in [a.event for a in actions]
