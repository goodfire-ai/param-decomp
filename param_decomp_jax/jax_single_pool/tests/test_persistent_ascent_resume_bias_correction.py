"""The first post-resume persistent ascent must apply Adam bias-correction at count
N+1, not reset to 1 (SPEC S22/S13/S15/S23; EDGES E13).

`SourcesAdamState.step_count` is an fp32 scalar incremented +1.0 per ascent
(`sources_adam_ascend_project`) and round-trips through the checkpoint. torch keeps
it as an int and likewise increments-then-bias-corrects
(`param_decomp/metrics/persistent_pgd_state.py`). If resume reset the count, the
first post-resume source step would be mis-scaled (its bias-correction divisor
`1 - beta**count` would use count 1 instead of N+1).

These tests pin that: a "save then restore" of the Adam state (the round-trip the
checkpoint performs on the `step_count` leaf) must make the (N+1)th ascent's source
delta IDENTICAL to an uninterrupted run's (N+1)th delta — and that delta must differ
from a count-reset variant, so the test actually discriminates N+1 from 1.
"""

import jax
import jax.numpy as jnp

from jax_single_pool.adversary import (
    SourcesAdamState,
    init_sources_adam_state,
    sources_adam_ascend_project,
)
from param_decomp_config.losses import AdamPGDConfig
from param_decomp_config.schedule import ScheduleConfig


def _adam() -> AdamPGDConfig:
    return AdamPGDConfig(
        beta1=0.8, beta2=0.9, eps=1e-8, lr_schedule=ScheduleConfig(start_val=0.05, warmup_pct=0.0)
    )


def _grad_for_ascent(ascent_idx: int) -> dict[str, jax.Array]:
    # Distinct per ascent so the Adam moments are non-degenerate and the bias-correction
    # divisor genuinely matters (constant grads would converge m/v to the grad and shrink
    # the count-1-vs-count-N+1 gap).
    return {"site": jnp.sin(jnp.arange(6.0).reshape(2, 3) + float(ascent_idx))}


def _roundtrip(adam_state: SourcesAdamState) -> SourcesAdamState:
    """Mimic the checkpoint save/restore of the Adam state: every leaf (including the
    fp32 `step_count`) goes out to arrays and comes back, with no in-flight count reset."""
    return SourcesAdamState(
        m={k: jnp.asarray(v) for k, v in adam_state.m.items()},
        v={k: jnp.asarray(v) for k, v in adam_state.v.items()},
        step_count=jnp.asarray(adam_state.step_count),
    )


def _ascend_n(
    n: int, lr: jax.Array, adam: AdamPGDConfig
) -> tuple[dict[str, jax.Array], SourcesAdamState]:
    sources = {"site": jnp.full((2, 3), 0.5)}
    adam_state = init_sources_adam_state(sources)
    for ascent_idx in range(n):
        sources, adam_state = sources_adam_ascend_project(
            sources, _grad_for_ascent(ascent_idx), adam_state, lr, adam
        )
    return sources, adam_state


def test_first_post_resume_ascent_uses_count_n_plus_1():
    adam = _adam()
    lr = jnp.asarray(0.05)
    n = 4

    # Uninterrupted: run N ascents, then the (N+1)th and capture its source delta.
    sources_n, adam_state_n = _ascend_n(n, lr, adam)
    uninterrupted_next, adam_state_n1 = sources_adam_ascend_project(
        sources_n, _grad_for_ascent(n), adam_state_n, lr, adam
    )
    uninterrupted_delta = uninterrupted_next["site"] - sources_n["site"]

    # Resumed: round-trip the post-N Adam state through the checkpoint, then run the
    # (N+1)th ascent from the SAME sources/grad.
    resumed_state = _roundtrip(adam_state_n)
    assert float(resumed_state.step_count) == float(n)  # SPEC S22: count N survives resume
    resumed_next, resumed_state_n1 = sources_adam_ascend_project(
        sources_n, _grad_for_ascent(n), resumed_state, lr, adam
    )
    resumed_delta = resumed_next["site"] - sources_n["site"]

    assert jnp.allclose(resumed_delta, uninterrupted_delta, atol=0.0, rtol=0.0)
    assert float(resumed_state_n1.step_count) == float(n + 1)
    assert jnp.array_equal(resumed_next["site"], uninterrupted_next["site"])
    assert jnp.array_equal(adam_state_n1.step_count, resumed_state_n1.step_count)


def test_count_reset_would_mis_scale_first_post_resume_ascent():
    """Discriminating check: had resume reset `step_count` to 0 (so the first ascent
    bias-corrects at count 1), the (N+1)th source delta would DIFFER. Confirms the
    above test is sensitive to the N+1 invariant, not vacuously true."""
    adam = _adam()
    lr = jnp.asarray(0.05)
    n = 4

    sources_n, adam_state_n = _ascend_n(n, lr, adam)
    correct_next, _ = sources_adam_ascend_project(
        sources_n, _grad_for_ascent(n), adam_state_n, lr, adam
    )

    reset_state = SourcesAdamState(m=adam_state_n.m, v=adam_state_n.v, step_count=jnp.zeros(()))
    reset_next, reset_state_after = sources_adam_ascend_project(
        sources_n, _grad_for_ascent(n), reset_state, lr, adam
    )

    assert float(reset_state_after.step_count) == 1.0
    assert not jnp.allclose(reset_next["site"], correct_next["site"])
