"""Routing samplers: per-position subset laws and the scheduled-p step dependence (S11)."""

import jax.numpy as jnp
import pytest
from jax import random

from param_decomp.configs import (
    FixedKSubsetRoutingConfig,
    ScheduledProbabilityRoutingConfig,
    StaticProbabilityRoutingConfig,
    UniformKSubsetRoutingConfig,
)
from param_decomp.recon import routing_sampler_from_config
from param_decomp.schedule import ScheduleConfig, get_scheduled_value

SITES = tuple(f"s{i}" for i in range(8))
LEADING = (16, 32)
STEP0 = jnp.zeros((), jnp.float32)


def _routed_count_per_position(routes: dict[str, jnp.ndarray]) -> jnp.ndarray:
    return jnp.stack([routes[s] for s in SITES]).sum(axis=0)


@pytest.mark.parametrize("k", [1, 3, 8])
def test_fixed_k_routes_exactly_k_per_position(k: int):
    sampler = routing_sampler_from_config(
        FixedKSubsetRoutingConfig(k=k), SITES, n_draws=2, total_steps=100
    )
    for routes in sampler(random.PRNGKey(0), LEADING, STEP0):
        assert routes is not None
        assert bool((_routed_count_per_position(routes) == k).all())


def test_fixed_k_refuses_k_above_n_sites():
    with pytest.raises(AssertionError):
        routing_sampler_from_config(
            FixedKSubsetRoutingConfig(k=9), SITES, n_draws=1, total_steps=100
        )


def test_uniform_k_spans_1_to_n_per_position():
    sampler = routing_sampler_from_config(
        UniformKSubsetRoutingConfig(), SITES, n_draws=1, total_steps=100
    )
    (routes,) = sampler(random.PRNGKey(0), LEADING, STEP0)
    assert routes is not None
    counts = _routed_count_per_position(routes)
    assert counts.min() >= 1 and counts.max() <= len(SITES)


def test_static_probability_matches_p():
    sampler = routing_sampler_from_config(
        StaticProbabilityRoutingConfig(p=0.25), SITES, n_draws=1, total_steps=100
    )
    (routes,) = sampler(random.PRNGKey(0), (64, 64), STEP0)
    assert routes is not None
    mean = jnp.stack([routes[s] for s in SITES]).mean()
    assert abs(float(mean) - 0.25) < 0.02


def test_scheduled_probability_tracks_schedule():
    total_steps = 100
    schedule = ScheduleConfig(start_val=0.1, fn_type="linear", final_val_frac=10.0)
    sampler = routing_sampler_from_config(
        ScheduledProbabilityRoutingConfig(p=schedule), SITES, n_draws=1, total_steps=total_steps
    )
    for step in (0, 50, 99):
        expected = get_scheduled_value(step, total_steps, schedule)
        (routes,) = sampler(random.PRNGKey(0), (64, 64), jnp.float32(step))
        assert routes is not None
        mean = float(jnp.stack([routes[s] for s in SITES]).mean())
        assert abs(mean - expected) < 0.02, (step, mean, expected)
    assert abs(get_scheduled_value(99, total_steps, schedule) - 1.0) < 1e-6


def test_scheduled_probability_refuses_endpoints_above_1():
    with pytest.raises(ValueError):
        ScheduledProbabilityRoutingConfig(
            p=ScheduleConfig(start_val=0.5, fn_type="linear", final_val_frac=4.0)
        )
    with pytest.raises(ValueError):
        ScheduledProbabilityRoutingConfig(p=ScheduleConfig(start_val=1.5, fn_type="constant"))
