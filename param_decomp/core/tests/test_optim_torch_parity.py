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

from param_decomp.core.configs import AdamWOptimizerConfig, AnyOptimizerConfig, MuonOptimizerConfig
from param_decomp.core.run_state import (
    _optimizer_with_clip,
    clip_by_global_norm_with_eps,
    optax_schedule,
    stacked_muon_dimension_numbers,
)
from param_decomp.core.schedule import ScheduleConfig


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
    opt = _optimizer_with_clip(muon_cfg, lambda count: jnp.float32(lr), None, mesh=None)
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


def test_muon_chunk_stacked_dimension_numbers_orthogonalize_3d_and_adam_2d_bias_stacks():
    """SPEC S20 amendment (2026-07-11): under `stacked_muon_dimension_numbers` a 3D
    `[n_chunks, d_in, d_out]` matrix stack is NS-orthogonalized per chunk slice, while a 2D
    `[n_chunks, d]` bias stack takes the Adam fallback — the reverse of optax's default 2D
    rule, which on the chunkwise CI-fn tree would orthogonalize the bias stacks."""
    muon_cfg = MuonOptimizerConfig(
        type="muon",
        lr_schedule=ScheduleConfig(
            fn_type="cosine", start_val=1e-3, final_val_frac=0.1, warmup_pct=0.0
        ),
        grad_clip_norm=None,
    )
    lr = 1e-3
    opt = _optimizer_with_clip(
        muon_cfg, lambda count: jnp.float32(lr), stacked_muon_dimension_numbers, mesh=None
    )
    key = jax.random.key(0)
    n_chunks = 3
    params = {"w": jnp.zeros((n_chunks, 16, 8)), "b": jnp.zeros((n_chunks, 8))}
    grads = {
        "w": jax.random.normal(key, (n_chunks, 16, 8)),
        "b": jax.random.normal(jax.random.fold_in(key, 1), (n_chunks, 8)),
    }
    updates, _ = opt.update(grads, opt.init(params), params)
    _, treedef = jax.tree.flatten(grads)
    updates = jax.tree.unflatten(treedef, jax.tree.leaves(updates))

    for chunk in range(n_chunks):
        grad_sv = jnp.linalg.svd(grads["w"][chunk], compute_uv=False)
        update_sv = jnp.linalg.svd(updates["w"][chunk], compute_uv=False)
        grad_flatness = float(grad_sv.max() / grad_sv.min())
        update_flatness = float(update_sv.max() / update_sv.min())
        assert update_flatness < 2.0 and update_flatness < grad_flatness / 2, (
            f"chunk {chunk}: 3D stack slice must be near-orthogonal:"
            f" grad {grad_flatness:.2f} -> update {update_flatness:.2f}"
        )
    bias_update_magnitude = float(jnp.abs(updates["b"]).max())
    assert bool(jnp.all(jnp.isfinite(updates["b"])))
    assert 0.3 * lr < bias_update_magnitude < 3 * lr, (
        f"2D bias stack takes an Adam-fallback step of O(lr), got {bias_update_magnitude}"
    )


def test_stacked_muon_update_matches_optax_muon():
    """SPEC S20 `impl: stacked`: the stack-by-shape batched NS produces the same updates as
    per-leaf `optax.contrib.muon` (same momentum, same partition, same post-NS chain) up to
    float reassociation — on a tree mixing 2D matrices (shared-shape group + a transposed
    member), a 3D chunk stack, and Adam-fallback leaves."""
    schedule = ScheduleConfig(fn_type="cosine", start_val=1e-3, final_val_frac=0.1, warmup_pct=0.0)
    lr = lambda count: jnp.float32(1e-3)
    key = jax.random.key(7)
    params = {
        "w_a": jnp.zeros((16, 24)),
        "w_b": jnp.zeros((24, 16)),
        "stack": jnp.zeros((3, 16, 24)),
        "bias_stack": jnp.zeros((3, 24)),
    }
    grads = {
        name: jax.random.normal(jax.random.fold_in(key, i), p.shape)
        for i, (name, p) in enumerate(params.items())
    }
    dim_nums = stacked_muon_dimension_numbers

    from typing import Literal

    def cfg(impl: Literal["optax", "stacked"]) -> MuonOptimizerConfig:
        return MuonOptimizerConfig(
            type="muon", lr_schedule=schedule, grad_clip_norm=0.01, consistent_rms=0.2, impl=impl
        )

    def two_steps(impl: Literal["optax", "stacked"]):
        opt = _optimizer_with_clip(cfg(impl), lr, dim_nums, mesh=None)
        state = opt.init(params)
        p = params
        for _ in range(2):
            updates, state = opt.update(grads, state, p)
            p = jax.tree.map(lambda x, u: x + u, p, updates)
        return p

    optax_p, stacked_p = two_steps("optax"), two_steps("stacked")
    for k in params:
        assert jnp.allclose(optax_p[k], stacked_p[k], rtol=1e-4, atol=1e-6), (
            f"{k}: stacked impl diverged from optax beyond reassociation tolerance"
        )


@pytest.mark.multidevice
def test_stacked_muon_sharded_matches_unsharded():
    """The stack-axis sharding constraint is layout-only: the sharded stacked NS
    reproduces the mesh=None updates (and preserves finiteness). Needs >1 device to be
    non-vacuous — at 4 sim devices the (16, 24) canonical group holds 5 matrices, so
    `_pad_to_multiple` genuinely pads 5 -> 8 and each device owns a real sub-stack; at
    1 device the constraint and the padding both collapse to no-ops."""
    from jax.sharding import Mesh

    assert jax.device_count() >= 4, "needs 4 sim devices; run via `make test-multidevice`"

    from param_decomp.core.muon_stacked import stacked_muon
    from param_decomp.core.sharding import hsdp_mesh

    key = jax.random.key(3)
    params = {
        "w": jnp.zeros((4, 16, 24)),
        "v": jnp.zeros((24, 16)),
        "b": jnp.zeros((4, 24)),
    }
    grads = {
        name: jax.random.normal(jax.random.fold_in(key, i), p.shape)
        for i, (name, p) in enumerate(params.items())
    }
    dim_nums = stacked_muon_dimension_numbers

    def one_update(mesh: "Mesh | None"):
        opt = stacked_muon(
            lambda count: jnp.float32(1e-3),
            beta=0.95,
            weight_decay=0.0,
            consistent_rms=0.2,
            muon_weight_dimension_numbers=dim_nums,
            ns_steps=5,
            ns_dtype=jnp.dtype(jnp.float32),
            mesh=mesh,
        )
        updates, _ = jax.jit(opt.update)(grads, opt.init(params), params)
        _, treedef = jax.tree.flatten(grads)
        return jax.tree.unflatten(treedef, jax.tree.leaves(updates))

    unsharded, sharded = one_update(None), one_update(hsdp_mesh())
    for k in params:
        assert bool(jnp.all(jnp.isfinite(sharded[k]))), k
        assert jnp.allclose(unsharded[k], sharded[k], rtol=1e-5, atol=1e-7), (
            f"{k}: sharded stacked NS diverged from unsharded"
        )


def test_stacked_muon_bf16_ns_is_sane():
    """`ns_dtype: bfloat16` (the stacked-only Kimi recipe, SPEC N1: masters and momentum
    stay fp32 — only the NS iteration itself runs half-precision). This pins "the fast
    path is not fp16-degenerate and not garbage", NOT parity: bf16 NS genuinely drifts a
    few percent from fp32 NS, so the update-norm comparison is a loose ~10% sanity bound.
    The exactness claims live in the fp32-NS tests above."""
    from typing import Literal

    from param_decomp.core.muon_stacked import stacked_muon

    key = jax.random.key(11)
    params = {
        "stack": jnp.zeros((3, 16, 24)),
        "w": jnp.zeros((24, 16)),
        "bias_stack": jnp.zeros((3, 24)),
    }
    grads = {
        name: jax.random.normal(jax.random.fold_in(key, i), p.shape)
        for i, (name, p) in enumerate(params.items())
    }

    _, treedef = jax.tree.flatten(params)

    def run(ns_dtype: Literal["float32", "bfloat16"]) -> tuple[dict[str, Array], optax.OptState]:
        opt = stacked_muon(
            lambda count: jnp.float32(1e-3),
            beta=0.95,
            weight_decay=0.0,
            consistent_rms=0.2,
            muon_weight_dimension_numbers=stacked_muon_dimension_numbers,
            ns_steps=5,
            ns_dtype=jnp.dtype(ns_dtype),
            mesh=None,
        )
        state = opt.init(params)
        p = params
        for _ in range(2):
            raw, state = opt.update(grads, state, p)
            p = jax.tree.map(
                lambda x, u: x + u, p, jax.tree.unflatten(treedef, jax.tree.leaves(raw))
            )
        raw, state = opt.update(grads, state, p)
        return jax.tree.unflatten(treedef, jax.tree.leaves(raw)), state

    bf16_updates, bf16_state = run("bfloat16")
    fp32_updates, _ = run("float32")

    for leaf in jax.tree.leaves((bf16_updates, bf16_state)):
        assert bool(jnp.all(jnp.isfinite(leaf)))
        assert leaf.dtype != jnp.bfloat16, "bf16 must stay inside NS: updates + state are fp32"
    for k in params:
        bf16_norm = float(jnp.linalg.norm(bf16_updates[k]))
        fp32_norm = float(jnp.linalg.norm(fp32_updates[k]))
        assert abs(bf16_norm - fp32_norm) < 0.1 * fp32_norm, (
            f"{k}: bf16-NS update norm {bf16_norm} vs fp32 {fp32_norm}"
        )
    assert not bool(jnp.array_equal(bf16_updates["stack"], fp32_updates["stack"])), (
        "bit-identical updates: the ns_dtype knob did not reach the NS iteration"
    )


def test_stacked_muon_dim_numbers_fail_closed():
    """SPEC S20 cross-impl promise: `impl: stacked` executes hardcoded trailing-two matrix
    axes and DISCARDS the declared dim numbers, so a muon leaf honestly declaring any other
    layout must die at optimizer build — `impl: optax` would honor the declaration and the
    two impls would silently diverge. Conforming declarations (trailing-two, negative or
    positive indices) build fine."""
    from param_decomp.core.muon_stacked import stacked_muon

    def build(params: dict[str, Array], spec: optax.contrib.MuonDimensionNumbers):
        opt = stacked_muon(
            lambda count: jnp.float32(1e-3),
            beta=0.95,
            weight_decay=0.0,
            consistent_rms=0.2,
            muon_weight_dimension_numbers=lambda p: jax.tree.map(lambda _: spec, p),
            ns_steps=5,
            ns_dtype=jnp.dtype(jnp.float32),
            mesh=None,
        )
        return opt.init(params)

    stack_3d = {"w": jnp.zeros((3, 16, 24))}
    for reduction, output in ((-2, -1), (1, 2)):
        build(stack_3d, optax.contrib.MuonDimensionNumbers(reduction, output))
    matrix_2d = {"v": jnp.zeros((16, 24))}
    for reduction, output in ((0, 1), (-2, -1)):
        build(matrix_2d, optax.contrib.MuonDimensionNumbers(reduction, output))

    # A 3D leaf whose stack axis is LAST, honestly declared: correct under `impl: optax`,
    # silently wrong under stacked — must be refused, naming the escape paths.
    stack_last = {"w": jnp.zeros((16, 24, 3))}
    with pytest.raises(AssertionError, match=r"impl: optax"):
        build(stack_last, optax.contrib.MuonDimensionNumbers(reduction_axis=0, output_axis=1))
    # Swapped orientation: optax's width scaling (consistent_rms=None) is directional.
    with pytest.raises(AssertionError, match=r"impl: optax"):
        build(matrix_2d, optax.contrib.MuonDimensionNumbers(reduction_axis=1, output_axis=0))
    # A 4D conv-style kernel lands on the ndim gate.
    with pytest.raises(AssertionError, match=r"impl: optax"):
        build(
            {"k": jnp.zeros((3, 3, 8, 16))},
            optax.contrib.MuonDimensionNumbers(reduction_axis=-2, output_axis=-1),
        )


def test_optimizer_config_type_discriminator():
    schedule = {"fn_type": "cosine", "start_val": 5e-5, "final_val_frac": 0.1}
    adapter = TypeAdapter(AnyOptimizerConfig)
    default = adapter.validate_python({"lr_schedule": schedule})
    assert isinstance(default, AdamWOptimizerConfig), "untyped configs stay canonical AdamW"
    muon = adapter.validate_python({"type": "muon", "lr_schedule": schedule})
    assert isinstance(muon, MuonOptimizerConfig)
    assert muon.beta == 0.95 and muon.consistent_rms is None
