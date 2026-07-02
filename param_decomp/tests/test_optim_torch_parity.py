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
from pydantic import TypeAdapter

from param_decomp.configs import AdamWOptimizerConfig, AnyOptimizerConfig, MuonOptimizerConfig
from param_decomp.run_state import (
    _optimizer_with_clip,
    clip_by_global_norm_with_eps,
    torch_cosine_schedule,
)
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


def test_muon_orthogonalizes_2d_leaves_and_adam_falls_back_elsewhere():
    """`type: muon` (SPEC S20 amendment): a 2D leaf's update is NS-orthogonalized (flat
    singular values), a non-2D leaf falls back to Adam; default `type: adamw` keeps the
    canonical optimizer so existing configs are untouched."""
    muon_cfg = MuonOptimizerConfig(
        type="muon",
        lr_schedule=ScheduleConfig(
            fn_type="cosine", start_val=1e-3, final_val_frac=0.1, warmup_pct=0.0
        ),
        grad_clip_norm=0.01,
    )
    lr = 1e-3
    opt = _optimizer_with_clip(muon_cfg, lambda count: jnp.float32(lr))
    key = jax.random.key(0)
    params = {"V": jnp.zeros((16, 8)), "scale": jnp.zeros((8,))}
    grads = {
        "V": jax.random.normal(key, (16, 8)),
        "scale": jax.random.normal(jax.random.fold_in(key, 1), (8,)),
    }
    updates, _ = opt.update(grads, opt.init(params), params)
    _, treedef = jax.tree.flatten(grads)
    updates = jax.tree.unflatten(treedef, jax.tree.leaves(updates))

    grad_sv = jnp.linalg.svd(grads["V"], compute_uv=False)
    update_sv = jnp.linalg.svd(updates["V"], compute_uv=False)
    grad_flatness = float(grad_sv.max() / grad_sv.min())
    update_flatness = float(update_sv.max() / update_sv.min())
    assert update_flatness < 2.0 and update_flatness < grad_flatness / 2, (
        "muon update on a 2D leaf must be near-orthogonal (5-step NS is approximate, so the"
        f" spectrum is flat-ish, not exactly flat): grad {grad_flatness:.2f} ->"
        f" update {update_flatness:.2f}"
    )
    scale_update_magnitude = float(jnp.abs(updates["scale"]).max())
    assert bool(jnp.all(jnp.isfinite(updates["scale"])))
    assert 0.3 * lr < scale_update_magnitude < 3 * lr, (
        f"non-2D leaf takes an Adam-fallback step of O(lr), got {scale_update_magnitude}"
    )


def test_optimizer_config_type_discriminator():
    schedule = {"fn_type": "cosine", "start_val": 5e-5, "final_val_frac": 0.1}
    adapter = TypeAdapter(AnyOptimizerConfig)
    default = adapter.validate_python({"lr_schedule": schedule})
    assert isinstance(default, AdamWOptimizerConfig), "untyped configs stay canonical AdamW"
    muon = adapter.validate_python({"type": "muon", "lr_schedule": schedule})
    assert isinstance(muon, MuonOptimizerConfig)
    assert muon.beta == 0.95 and muon.consistent_rms is None
