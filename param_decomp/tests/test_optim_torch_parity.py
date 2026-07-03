"""Component/CI optimizer seams match torch's formulas exactly (SPEC S19, S20).

- cosine LR uses torch's `step / (total_steps - 1)` denominator, NOT optax's
  `cosine_decay_schedule` `count / total_steps` (reaches `0.1×` one step later).
- global-norm grad clip uses torch's `clip(max_norm / (norm + 1e-6), max=1)`, NOT
  optax's eps-free `clip_by_global_norm`.
"""

import jax
import jax.numpy as jnp
import optax
import pytest
from jax.typing import ArrayLike
from jaxtyping import Array

from param_decomp.run_state import clip_by_global_norm_with_eps, torch_cosine_schedule


def _scalar(value: ArrayLike) -> float:
    """`optax.Schedule` returns the wide `ArrayLike` union; narrow a scalar schedule
    output to a Python float for comparison."""
    return float(jnp.asarray(value))


def _clip(transform: optax.GradientTransformation, grads: dict[str, Array]) -> dict[str, Array]:
    """Run a `GradientTransformation` over a concrete grad dict and recover the result
    with that same `dict[str, Array]` structure — `optax.Updates` is a wide pytree union
    the type-checker can't index, but the transform preserves the input tree."""
    out, _ = transform.update(grads, transform.init(grads))
    _, treedef = jax.tree.flatten(grads)
    return jax.tree.unflatten(treedef, jax.tree.leaves(out))


def torch_cosine_reference(peak_lr: float, total_steps: int, alpha: float, step: int) -> float:
    import math

    progress = step / (total_steps - 1)
    return peak_lr * (alpha + (1 - alpha) * 0.5 * (1 + math.cos(math.pi * progress)))


def test_cosine_schedule_matches_torch_denominator():
    peak_lr = 1.5e-4
    total_steps = 400_000
    alpha = 0.1
    sched = torch_cosine_schedule(peak_lr, total_steps, alpha)
    for step in (0, total_steps // 2, total_steps - 1):
        jax_value = _scalar(sched(jnp.int32(step)))
        torch_value = torch_cosine_reference(peak_lr, total_steps, alpha, step)
        assert jax_value == pytest.approx(torch_value, rel=1e-7), f"step {step}"
    assert _scalar(sched(jnp.int32(total_steps - 1))) == pytest.approx(alpha * peak_lr, rel=1e-6)


def test_cosine_schedule_differs_from_optax():
    """Torch's `step / (total_steps - 1)` denominator reaches `alpha·peak` one step
    earlier than optax's `count / total_steps`: at `total_steps - 1` ours is already at
    the floor while optax still has a full step of decay left. The gap is largest with
    few steps (with 400k it flattens into fp noise at the endpoints — SPEC S19)."""
    peak_lr = 1.5e-4
    total_steps = 10
    optax_sched = optax.cosine_decay_schedule(peak_lr, total_steps, alpha=0.1)
    ours = torch_cosine_schedule(peak_lr, total_steps, alpha=0.1)
    endpoint = total_steps - 1
    assert _scalar(ours(jnp.int32(endpoint))) == pytest.approx(0.1 * peak_lr, rel=1e-6)
    assert _scalar(optax_sched(jnp.int32(endpoint))) != pytest.approx(0.1 * peak_lr, rel=1e-6)
    assert _scalar(optax_sched(jnp.int32(total_steps))) == pytest.approx(0.1 * peak_lr, rel=1e-6)


def test_grad_clip_matches_torch_eps():
    max_norm = 0.01
    eps = 1e-6
    clip = clip_by_global_norm_with_eps(max_norm, eps)
    grads = {"a": jnp.array([3.0, 4.0]), "b": jnp.array([0.0])}
    global_norm = 5.0
    out = _clip(clip, grads)
    torch_coef = min(max_norm / (global_norm + eps), 1.0)
    assert float(out["a"][0]) == pytest.approx(3.0 * torch_coef, rel=1e-7)
    assert float(out["a"][1]) == pytest.approx(4.0 * torch_coef, rel=1e-7)


def test_grad_clip_differs_from_optax_when_clipping():
    max_norm = 0.01
    grads = {"a": jnp.array([3.0, 4.0])}
    out_ours = _clip(clip_by_global_norm_with_eps(max_norm, eps=1e-6), grads)
    out_optax = _clip(optax.clip_by_global_norm(max_norm), grads)
    assert float(out_ours["a"][0]) != pytest.approx(float(out_optax["a"][0]), rel=1e-9)


def test_grad_clip_noop_below_threshold():
    max_norm = 100.0
    clip = clip_by_global_norm_with_eps(max_norm, eps=1e-6)
    grads = {"a": jnp.array([3.0, 4.0])}
    out = _clip(clip, grads)
    assert float(out["a"][0]) == pytest.approx(3.0)
    assert float(out["a"][1]) == pytest.approx(4.0)
