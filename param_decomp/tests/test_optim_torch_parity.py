"""Component/CI optimizer seams match torch's formulas exactly (SPEC S19, S20).

- cosine LR uses torch's `step / (total_steps - 1)` denominator, NOT optax's
  `cosine_decay_schedule` `count / total_steps` (reaches `0.1×` one step later).
- global-norm grad clip uses torch's `clip(max_norm / (norm + 1e-6), max=1)`, NOT
  optax's eps-free `clip_by_global_norm`.
"""

from typing import Literal

import jax
import jax.numpy as jnp
import optax
import pytest
from jax.typing import ArrayLike
from jaxtyping import Array

from param_decomp.configs import OptimizerConfig
from param_decomp.run_state import _adamw_with_clip, clip_by_global_norm_with_eps, optax_schedule
from param_decomp.schedule import ScheduleConfig


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
    config = ScheduleConfig(start_val=peak_lr, fn_type="cosine", final_val_frac=alpha)
    sched = optax_schedule(config, total_steps)
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
    ours = optax_schedule(
        ScheduleConfig(start_val=peak_lr, fn_type="cosine", final_val_frac=0.1), total_steps
    )
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


def _adamw(moments_dtype: Literal["float32", "bfloat16"]) -> optax.GradientTransformation:
    opt = OptimizerConfig(
        lr_schedule=ScheduleConfig(start_val=1e-3, fn_type="constant"),
        moments_dtype=moments_dtype,
    )
    return _adamw_with_clip(opt, optax_schedule(opt.lr_schedule, total_steps=100))


def _run_steps(
    optimizer: optax.GradientTransformation, n_steps: int, key: Array
) -> tuple[dict[str, Array], optax.OptState]:
    params = {"w": jnp.ones((4, 8), jnp.float32)}
    state = optimizer.init(params)
    for step in range(n_steps):
        grads = {"w": jax.random.normal(jax.random.fold_in(key, step), (4, 8)) * 1e-3}
        updates, state = optimizer.update(grads, state, params)
        new_params, treedef = jax.tree.flatten(optax.apply_updates(params, updates))
        params = jax.tree.unflatten(treedef, new_params)
    return params, state


def test_bf16_moments_dtype_persists_across_updates():
    _, state = _run_steps(_adamw("bfloat16"), n_steps=3, key=jax.random.key(0))
    assert isinstance(state, tuple)
    adam_state = state[0]
    assert isinstance(adam_state, optax.ScaleByAdamState)
    assert all(m.dtype == jnp.bfloat16 for m in jax.tree.leaves(adam_state.mu))
    assert all(v.dtype == jnp.bfloat16 for v in jax.tree.leaves(adam_state.nu))
    assert all(bool(jnp.any(v != 0)) for v in jax.tree.leaves(adam_state.nu))


def test_bf16_moments_track_fp32_trajectory():
    key = jax.random.key(1)
    params_f32, _ = _run_steps(_adamw("float32"), n_steps=20, key=key)
    params_bf16, _ = _run_steps(_adamw("bfloat16"), n_steps=20, key=key)
    divergence = jnp.max(jnp.abs(params_f32["w"] - params_bf16["w"]))
    total_movement = jnp.max(jnp.abs(params_f32["w"] - 1.0))
    assert float(divergence) < 0.02 * float(total_movement)
    assert bool(jnp.any(params_f32["w"] != params_bf16["w"]))


def test_fp32_moments_default_matches_plain_optax_adamw():
    opt = OptimizerConfig(lr_schedule=ScheduleConfig(start_val=1e-3, fn_type="constant"))
    assert opt.moments_dtype == "float32"
    schedule = optax_schedule(opt.lr_schedule, total_steps=100)
    ours = _adamw_with_clip(opt, schedule)
    plain = optax.adamw(schedule, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0)
    key = jax.random.key(2)
    params_ours, _ = _run_steps(ours, n_steps=5, key=key)
    params_plain, _ = _run_steps(plain, n_steps=5, key=key)
    assert bool(jnp.all(params_ours["w"] == params_plain["w"]))
