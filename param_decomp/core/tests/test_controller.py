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
    assert probed.phase is Phase.PROBE_PENDING
    # time never ages a pending probe back to control
    aged = drive(probed, [window(CFG.tau, complexity=50.0)] * 10)
    assert all(a.state.phase is Phase.PROBE_PENDING for a in aged)
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


def test_probe_accepted_resets_bracket_plateau_and_rejections() -> None:
    state = ControllerState(
        phase=Phase.PROBE_PENDING, log_c=0.0, lo=-1.0, hi=1.0, dwell=0,
        prev_complexity=50.0, protect_windows_left=0, probe_cooldown=0, rejected_probes=1,
    )  # fmt: skip
    accepted = probe_accepted(state)
    assert accepted.phase is Phase.CONTROL
    assert accepted.lo is None and accepted.hi is None and accepted.rejected_probes == 0
    assert accepted.prev_complexity is None


def test_verdicts_assert_phase_legality() -> None:
    import pytest

    control = ControllerState.initial(0.0)
    with pytest.raises(AssertionError):
        probe_accepted(control)
    with pytest.raises(AssertionError):
        probe_rejected(control, CFG)


def test_feasibility_protection_expiry_clears_plateau_and_lineage() -> None:
    state = ControllerState(
        phase=Phase.BIRTH_PROTECTED, log_c=0.0, lo=-1.0, hi=1.0, dwell=0,
        prev_complexity=50.0, protect_windows_left=1, probe_cooldown=2, rejected_probes=2,
    )  # fmt: skip
    out = controller_update(state, window(0.5), CFG, 3).state
    assert out.phase is Phase.CONTROL
    assert out.prev_complexity is None and out.rejected_probes == 0 and out.probe_cooldown == 0


def test_capacity_exhausted_is_reported_not_clipped() -> None:
    state = ControllerState.initial(0.0)
    actions = drive(state, [window(CFG.tau, complexity=50.0, spare=False)] * 6)
    assert Event.CAPACITY_EXHAUSTED in [a.event for a in actions]
    assert all(a.state.phase is not Phase.BIRTH_PROTECTED for a in actions)


def test_no_feasibility_birth_while_slack() -> None:
    state = ControllerState.initial(0.0)
    actions = drive(state, [window(0.0)] * 20)
    assert Event.FEASIBILITY_BIRTH not in [a.event for a in actions]


def test_converged_slack_bracket_still_fires_column_probe() -> None:
    # boundary between lo and hi with the sampled lo strictly slack (beyond the deadband):
    # a converged bracket must route into plateau logic, not reset dwell forever
    state = ControllerState(
        phase=Phase.CONTROL, log_c=0.0, lo=0.0, hi=0.0 + CFG.resolution / 2, dwell=0,
        prev_complexity=None, protect_windows_left=0, probe_cooldown=0, rejected_probes=0,
    )  # fmt: skip
    actions = drive(state, [window(0.0, complexity=50.0)] * 5)  # r_adv far below tau: slack
    assert Event.COLUMN_PROBE in [a.event for a in actions]


def test_complexity_stays_off_through_feasibility_birth_and_protection() -> None:
    state = ControllerState.initial(math.log(1.0))
    actions = drive(state, [window(0.5)] * 10)
    events = [a.event for a in actions]
    birth_idx = events.index(Event.FEASIBILITY_BIRTH)
    assert actions[birth_idx].state.c == 0.0
    for a in actions[birth_idx + 1 :]:
        if a.state.phase is Phase.BIRTH_PROTECTED:
            assert a.state.c == 0.0
    # protection expiry re-enters control BELOW the last known-violating coefficient
    expiry = next(a.state for a in actions[birth_idx + 1 :] if a.state.phase is Phase.CONTROL)
    assert expiry.log_c < actions[birth_idx].state.log_c


def test_authored_observable_uses_fixed_affine_units_and_fails_closed() -> None:
    import pytest

    from param_decomp.core.controller import authored_observable

    assert authored_observable({"eval/recon": 0.5}, "eval/recon", 0.1, 0.2) == 2.0
    with pytest.raises(AssertionError):
        authored_observable({}, "eval/recon", 0.0, 1.0)
    with pytest.raises(AssertionError):
        authored_observable({"eval/recon": float("nan")}, "eval/recon", 0.0, 1.0)


def test_settlement_gate_emits_independent_stable_windows_only() -> None:
    from param_decomp.core.controller import (
        ControllerObservation,
        SettlementConfig,
        SettlementState,
        observe_settled_window,
    )

    cfg = SettlementConfig(points=3, rtol=0.02, atol=1e-8)
    state = SettlementState.initial()
    for r, c in [(0.20, 100.0), (0.15, 90.0), (0.10, 80.0)]:
        state, settled = observe_settled_window(state, ControllerObservation(r, c, True), cfg)
        assert settled is None
    # Sliding reads become stationary; the first stable triple emits and clears.
    emitted = []
    for r, c in [(0.100, 80.0), (0.101, 80.5), (0.1005, 80.2)]:
        state, settled = observe_settled_window(state, ControllerObservation(r, c, True), cfg)
        if settled is not None:
            emitted.append(settled)
    assert len(emitted) == 1
    assert len(state.observations) == 1  # the post-emission read starts the next independent window
    assert abs(emitted[0].r_adv - (0.1 + 0.1 + 0.101) / 3) < 1e-12


def test_unqualified_referee_resets_settlement_and_capacity_change_asserts() -> None:
    import pytest

    from param_decomp.core.controller import (
        ControllerObservation,
        SettlementConfig,
        SettlementState,
        observe_settled_window,
    )

    cfg = SettlementConfig(points=2, rtol=1.0, atol=0.0)
    state, _ = observe_settled_window(
        SettlementState.initial(), ControllerObservation(0.1, 10.0, True), cfg
    )
    state, settled = observe_settled_window(
        state, ControllerObservation(0.1, 10.0, True, qualified=False), cfg
    )
    assert state == SettlementState.initial() and settled is None

    state, _ = observe_settled_window(state, ControllerObservation(0.1, 10.0, True), cfg)
    with pytest.raises(AssertionError, match="capacity changed"):
        observe_settled_window(state, ControllerObservation(0.1, 10.0, False), cfg)
