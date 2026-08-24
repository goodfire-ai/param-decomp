"""Component/CI optimizer seams match torch's formulas exactly (SPEC S19, S20).

- cosine LR uses torch's `step / (total_steps - 1)` denominator, NOT optax's
  `cosine_decay_schedule` `count / total_steps` (reaches `0.1×` one step later).
- global-norm grad clip uses torch's `clip(max_norm / (norm + 1e-6), max=1)`, NOT
  optax's eps-free `clip_by_global_norm`.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
import pytest
from jax.sharding import AxisType
from jax.typing import ArrayLike
from jaxtyping import Array
from pydantic import TypeAdapter

from param_decomp.core.configs import (
    AdamWOptimizerConfig,
    AnyOptimizerConfig,
    MuonOptimizerConfig,
    PlacementTableConfig,
)
from param_decomp.core.run_state import (
    _optimizer_with_clip,
    clip_by_global_norm_with_eps,
    optax_schedule,
    stacked_muon_dimension_numbers,
)
from param_decomp.core.schedule import Knot, ScheduleConfig


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
    config = ScheduleConfig(
        max_val=peak_lr, points=(Knot(at=0.0, frac=1.0), Knot(at=1.0, frac=alpha, interp="cosine"))
    )
    sched = optax_schedule(config, total_steps)
    for step in (0, total_steps // 2, total_steps - 1):
        jax_value = _scalar(sched(jnp.int32(step)))
        torch_value = torch_cosine_reference(peak_lr, total_steps, alpha, step)
        # rel 1e-6: the traced evaluator runs in fp32 and associates the cosine
        # interpolation differently than torch's float64 formula; the S20 contract is
        # the step placement (endpoints exact below), not mid-curve bit-parity.
        assert jax_value == pytest.approx(torch_value, rel=1e-6), f"step {step}"
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
        ScheduleConfig(
            max_val=peak_lr,
            points=(Knot(at=0.0, frac=1.0), Knot(at=1.0, frac=0.1, interp="cosine")),
        ),
        total_steps,
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
            max_val=1e-3, points=(Knot(at=0.0, frac=1.0), Knot(at=1.0, frac=0.1, interp="cosine"))
        ),
        grad_clip_norm=0.01,
    )
    lr = 1e-3
    opt = _optimizer_with_clip(muon_cfg, lambda count: jnp.float32(lr), None, waypoints=None)
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
            max_val=1e-3, points=(Knot(at=0.0, frac=1.0), Knot(at=1.0, frac=0.1, interp="cosine"))
        ),
        grad_clip_norm=None,
    )
    lr = 1e-3
    opt = _optimizer_with_clip(
        muon_cfg,
        lambda count: jnp.float32(lr),
        stacked_muon_dimension_numbers,
        waypoints=None,
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
    schedule = ScheduleConfig(
        max_val=1e-3, points=(Knot(at=0.0, frac=1.0), Knot(at=1.0, frac=0.1, interp="cosine"))
    )
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
        opt = _optimizer_with_clip(cfg(impl), lr, dim_nums, waypoints=None)
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
    """The staging reshards are layout-only: the placed stacked NS reproduces the
    mesh=None updates (and preserves finiteness). The tree mixes a 4-stack, a 2D
    matrix (canonical g=1), and an adam-fallback bias stack, so per-kind staging
    covers every canonicalization arm."""
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

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

    def replicated_waypoints(mesh: "Mesh") -> Callable[[optax.Updates], optax.Updates]:
        sharding = NamedSharding(mesh, P(None, None, None))
        return lambda tree: jax.tree.map(lambda _: sharding, tree)

    def one_update(mesh: "Mesh | None"):
        opt = stacked_muon(
            lambda count: jnp.float32(1e-3),
            beta=0.95,
            weight_decay=0.0,
            consistent_rms=0.2,
            muon_weight_dimension_numbers=dim_nums,
            ns_steps=5,
            ns_dtype=jnp.dtype(jnp.float32),
            waypoints=None if mesh is None else replicated_waypoints(mesh),
        )
        import contextlib

        with jax.set_mesh(mesh) if mesh is not None else contextlib.nullcontext():
            updates, _ = jax.jit(opt.update)(grads, opt.init(params), params)
        _, treedef = jax.tree.flatten(grads)
        return jax.tree.unflatten(treedef, jax.tree.leaves(updates))

    unsharded, sharded = one_update(None), one_update(hsdp_mesh(1, jax.device_count(), 1))
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
            waypoints=None,
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
            waypoints=None,
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
    schedule = {
        "max_val": 5e-5,
        "points": [{"at": 0.0, "frac": 1.0}, {"at": 1.0, "frac": 0.1, "interp": "cosine"}],
    }
    adapter = TypeAdapter(AnyOptimizerConfig)
    default = adapter.validate_python({"lr_schedule": schedule})
    assert isinstance(default, AdamWOptimizerConfig), "untyped configs stay canonical AdamW"
    muon = adapter.validate_python({"type": "muon", "lr_schedule": schedule})
    assert isinstance(muon, MuonOptimizerConfig)
    assert muon.beta == 0.95 and muon.consistent_rms is None


@pytest.mark.multidevice
@pytest.mark.parametrize("ns_dtype", [jnp.float32, jnp.bfloat16])
def test_grouped_ns_owner_waypoint_matches_replicated_on_stack_owned_leaves(
    ns_dtype: jnp.dtype, capfd: pytest.CaptureFixture[str]
):
    """The stack-owner `ns_compute` waypoint (`{stack: replicate}` staging) is the same
    math as the replicated waypoint — the declared rows change only where the entry
    reshards happen (SPEC D4 tolerance class) — and NEITHER staging may trigger the SPMD
    partitioner's involuntary-full-rematerialization fallback. The check reads the
    partitioner's warning off fd 2 (`capfd`) — the same signal the production log grep
    uses. bf16 is a separate arm because an unpinned `convert_element_type` is its own
    fallback trigger."""
    import numpy as np
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from param_decomp.core.muon_stacked import _staged_newton_schulz

    assert jax.device_count() >= 4, "needs 4 sim devices; run via `make test-multidevice`"
    mesh = Mesh(
        np.array(jax.devices()[:4]).reshape(2, 2),
        ("replicate", "fsdp"),
        axis_types=(AxisType.Explicit,) * 2,
    )
    owner = NamedSharding(mesh, P("replicate", "fsdp", None))

    key = jax.random.key(0)
    k1, k2 = jax.random.split(key)
    tree = {
        "a": jax.device_put(jax.random.normal(k1, (4, 8, 16), jnp.float32), owner),
        "b": jax.device_put(jax.random.normal(k2, (4, 16, 8), jnp.float32), owner),
    }
    coeffs = jnp.array([(3.4445, -4.7750, 2.0315)] * 5, jnp.float32)

    def compiled(waypoint: NamedSharding):
        waypoints = lambda t: jax.tree.map(lambda _: waypoint, t)
        fn = jax.jit(lambda t: _staged_newton_schulz(t, coeffs, 5, jnp.dtype(ns_dtype), waypoints))
        return fn.lower(tree).compile()

    fallback_warning = "Involuntary full rematerialization"

    capfd.readouterr()
    replicated_exe = compiled(NamedSharding(mesh, P(None, None, None)))
    assert fallback_warning not in capfd.readouterr().err, (
        "replicated staging regressed to the fallback lowering"
    )
    owner_exe = compiled(NamedSharding(mesh, P("replicate", None, None)))
    assert fallback_warning not in capfd.readouterr().err, (
        "stack-owner staging regressed to the fallback lowering"
    )

    replicated, owner_staged = replicated_exe(tree), owner_exe(tree)
    tol = 1e-5 if ns_dtype == jnp.float32 else 5e-2
    for name in tree:
        assert jnp.max(jnp.abs(replicated[name] - owner_staged[name])) < tol, name


@pytest.mark.multidevice
def test_stacked_ns_is_invariant_to_tensor_parallel_partition():
    import numpy as np
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from param_decomp.core.muon_stacked import _staged_newton_schulz

    assert jax.device_count() >= 4
    value = jax.random.normal(jax.random.key(0), (4, 8, 6))
    coeffs = jnp.array([(3.4445, -4.7750, 2.0315)] * 5)

    def run(tp: int) -> Array:
        mesh = Mesh(
            np.asarray(jax.devices()[:4]).reshape(1, 4 // tp, tp),
            ("replicate", "fsdp", "tp"),
            axis_types=(AxisType.Explicit,) * 3,
        )
        persistent = NamedSharding(mesh, P(None, "fsdp", ("tp", "replicate")))
        replicated = NamedSharding(mesh, P(None, None, None))
        waypoints = lambda t: jax.tree.map(lambda _: replicated, t)
        tree = {"value": jax.device_put(value, persistent)}
        with jax.set_mesh(mesh):
            result = jax.jit(lambda x: _staged_newton_schulz(x, coeffs, 5, jnp.float32, waypoints))(
                tree
            )
        return result["value"]

    # Host-side: the two runs commit to different meshes; explicit-mode ops refuse
    # cross-mesh operands.
    assert np.allclose(np.asarray(run(1)), np.asarray(run(2)), rtol=2e-6, atol=1e-6)


def _explicit_table(
    components: dict[str, object], ci_ns: dict[str, object]
) -> PlacementTableConfig:
    """A minimal valid explicit table around parametrized components rows: replicated
    CI/target rows, the shared activation waist."""
    ci_family = {"optimizer_state": {}, "compute_weights": {}, "operands": {}, "ns_compute": ci_ns}
    return PlacementTableConfig.model_validate(
        {
            "components": components,
            "ci_fn": {
                "attention": ci_family,
                "ffn": ci_family,
                "input": ci_family,
                "output": ci_family,
                "vectors": {},
                "activations": {},
            },
            "activations": {
                "external": {"batch": ["replicate", "fsdp"]},
                "component": {"batch": ["replicate", "fsdp"]},
            },
            "target": {
                "embedding": {"persist": {}, "operand": {}},
                "normalization": {},
                "position_encoding": {},
                "column": {
                    "persist": {},
                    "operand": {},
                    "input": "external",
                    "output": "intermediate",
                },
                "row": {
                    "persist": {},
                    "operand": {},
                    "input": "intermediate",
                    "output": "external",
                },
                "output": {"persist": {}, "operand": {}},
                "intermediate": {},
                "component": {"input": "external", "output": "external"},
            },
        }
    )


def test_ns_compute_waypoints_are_declared_rows():
    """The muon-NS staging is placement-table DATA, not derived geometry: every preset
    declares the same node-axis stack split (`{stack: replicate}` — owner's ingress is
    the identity on the stack axis, zero1/ddp hop to it), any stack-sharded persistence
    is placeable once its waypoint is declared (no owner-geometry sniffing survives),
    and the row language fails closed — a matrix-axis assignment refuses at table build
    (the batched NS needs whole matrices per device), while a stack split some kind
    cannot tile refuses at the stacked-muon consumer's claim (nothing hunts for an
    alternative split, and non-muon runs never consume the row)."""
    import jax.sharding
    from pydantic import ValidationError

    from param_decomp.core.components import SiteSpec
    from param_decomp.core.placement import from_config

    mesh = jax.sharding.AbstractMesh((4, 8, 1), ("replicate", "fsdp", "tp"))
    tiling = tuple(SiteSpec(f"t.{i}", 64, 32, 8, "t") for i in range(4))
    mixed = tiling + (SiteSpec("odd.0", 128, 64, 8, "odd"),)

    for preset in ("owner", "zero1", "ddp"):
        rules = from_config(preset, mesh, tiling)
        assert dict(rules.components.ns_compute.rule) == {"stack": "replicate"}
        assert dict(rules.ci_fn.ffn.ns_compute.rule) == {"stack": "replicate"}

    # Only a stacked-muon optimizer consumes the ns_compute rows, so a non-tiling group
    # builds fine (zero1 keeps its any-stack-length universality for adamw runs) and the
    # refusal fires at the muon consumer's claim instead.
    from param_decomp.core.placement import (
        assert_stacked_muon_ci_staging,
        assert_stacked_muon_component_staging,
    )

    mixed_rules = from_config("zero1", mesh, mixed)
    with pytest.raises(AssertionError, match="stack lengths do not tile"):
        assert_stacked_muon_component_staging(mixed_rules)
    with pytest.raises(AssertionError, match="stack lengths do not tile"):
        assert_stacked_muon_ci_staging(mixed_rules, n_chunks=1)
    assert_stacked_muon_component_staging(from_config("zero1", mesh, tiling))
    assert_stacked_muon_ci_staging(mixed_rules, n_chunks=8)

    # A stack-sharded persistence that is NOT the owner geometry simply builds: its
    # NS staging is whatever row it declares, not a derived hop path.
    swapped = _explicit_table(
        {
            "optimizer_state": {"stack": "fsdp", "d_in": "replicate", "d_out": "replicate"},
            "compute_weights": {"d_in": "fsdp", "d_out": "fsdp"},
            "faithfulness_weights": {"stack": "fsdp", "d_in": "replicate", "d_out": "replicate"},
            "faithfulness_deltas": {"stack": "fsdp", "d_out": "replicate"},
            "operands": {},
            "ns_compute": {},
        },
        {},
    )
    swapped_rules = from_config(
        swapped, mesh, tuple(SiteSpec(f"sw.{i}", 64, 32, 8, "sw") for i in range(8))
    )
    assert dict(swapped_rules.components.ns_compute.rule) == {}

    ddp_components = {
        "optimizer_state": {},
        "compute_weights": {},
        "faithfulness_weights": {},
        "faithfulness_deltas": {},
        "operands": {},
    }
    with pytest.raises(AssertionError, match="may assign only `stack`"):
        from_config(
            _explicit_table({**ddp_components, "ns_compute": {"d_in": "fsdp"}}, {}), mesh, tiling
        )
    # `rows`/`cols` died with the canonical-shape grouping: not a semantic axis at all,
    # so an explicit table naming one refuses at parse, before table build.
    with pytest.raises(ValidationError):
        _explicit_table({**ddp_components, "ns_compute": {}}, {"rows": "fsdp"})


def test_ns_staging_sharding_is_the_row_verbatim():
    """The waypoint is the `ns_compute` row's stack split verbatim — declared value,
    no search; a non-stack key is refused (the batched NS needs whole matrices)."""
    from jax.sharding import AbstractMesh
    from jax.sharding import PartitionSpec as P

    from param_decomp.core.placement import PlacedRule, ns_staging_sharding

    mesh = AbstractMesh((2, 2, 1), ("replicate", "fsdp", "tp"))
    row = PlacedRule(mesh=mesh, label="ns", rule={"stack": "replicate"})
    assert ns_staging_sharding(row, mesh).spec == P("replicate", None, None)
    replicated = PlacedRule(mesh=mesh, label="ns", rule={})
    assert ns_staging_sharding(replicated, mesh).spec == P(None, None, None)
    with pytest.raises(AssertionError):
        ns_staging_sharding(PlacedRule(mesh=mesh, label="bad", rule={"d_in": "fsdp"}), mesh)


def test_staging_hops_move_one_axis_per_reshard():
    """`staging_hops` reaches the waypoint one mesh-axis move per reshard — the spelling
    that avoids GSPMD's involuntary-full-rematerialization fallback — ending exactly at
    the waypoint, with matrix-dim (and surplus-stack) drops in one trailing reshard."""
    from jax.sharding import PartitionSpec as P

    from param_decomp.core.muon_stacked import staging_hops

    # owner: masters already stack-split — one trailing drop of the matrix axes.
    assert staging_hops(P("replicate", "fsdp", "tp"), ("replicate",)) == [
        P(("replicate",), None, None)
    ]
    # zero1 V: move replicate off the C dim, then drop the rest.
    assert staging_hops(P(None, "fsdp", ("tp", "replicate")), ("replicate",)) == [
        P(("replicate",), ("fsdp",), ("tp",)),
        P(("replicate",), None, None),
    ]
    # CI ffn: the whole mesh on one matrix dim — the pair that remats as one reshard.
    assert staging_hops(P(None, None, ("tp", "fsdp", "replicate")), ("replicate",)) == [
        P(("replicate",), None, ("tp", "fsdp")),
        P(("replicate",), None, None),
    ]
    # ddp: replicated masters — the split is comms-free, still one hop.
    assert staging_hops(P(None, None, None), ("replicate",)) == [P(("replicate",), None, None)]
    # replicated waypoint from replicated master: the identity hop.
    assert staging_hops(P(None, None, None), ()) == [P(None, None, None)]


def _hlo_computations(hlo: str) -> dict[str, str]:
    """HLO text split into named computations (module-level `name (...) -> ... {` blocks)."""
    import re

    computations: dict[str, str] = {}
    name, lines = None, []
    for line in hlo.splitlines():
        started = re.match(r"^(ENTRY )?%?([\w\.\-]+) \(.*\) -> .* \{$", line)
        if started and name is None:
            name, lines = started.group(2), [line]
        elif name is not None:
            lines.append(line)
            if line.startswith("}"):
                computations[name] = "\n".join(lines)
                name, lines = None, []
    return computations


def _collective_count(hlo_text: str) -> int:
    import re

    return len(
        re.findall(
            r"=\s+\S+\s+(?:all-reduce|all-gather|all-to-all|collective-permute|"
            r"reduce-scatter|collective-broadcast)(?:-start)?\(",
            hlo_text,
        )
    )


def _while_body_collective_count(hlo: str) -> int:
    """Collectives reachable from any while-loop body, fusions/calls included."""
    import re

    computations = _hlo_computations(hlo)
    reachable: set[str] = set()
    frontier = list(set(re.findall(r"body=%?([\w\.\-]+)", hlo)))
    assert frontier, "no while loop found — the NS iteration should lower as one"
    while frontier:
        name = frontier.pop()
        if name in reachable or name not in computations:
            continue
        reachable.add(name)
        body = computations[name]
        frontier += re.findall(r"(?:calls=|to_apply=)%?([\w\.\-]+)", body)
        frontier += re.findall(r"%([\w\.\-]+)\s*\}?\s*,?\s*kind=", body)
    return sum(_collective_count(computations[name]) for name in reachable)


def _collective_result_shapes(hlo_text: str) -> list[tuple[int, ...]]:
    """Per-device result shapes of every collective in the partitioned module."""
    import re

    shapes: list[tuple[int, ...]] = []
    for m in re.finditer(
        r"=\s+\S*?\[([\d,]+)\]\S*\s+(?:all-reduce|all-gather|all-to-all|"
        r"collective-permute|reduce-scatter|collective-broadcast)(?:-start)?\(",
        hlo_text,
    ):
        shapes.append(tuple(int(d) for d in m.group(1).split(",")))
    return shapes


@pytest.mark.multidevice
def test_per_kind_ns_census_zero_collectives_inside_the_ns_loop(
    capfd: pytest.CaptureFixture[str],
):
    """The declared-waypoint choreography keeps every collective at entry/exit: each
    kind's NS while-loop lowers with ZERO collectives inside its body — the staging
    layout holds whole matrices per device, so the Gram matmuls are local — and the
    lowering must not fall back to involuntary full rematerialization. Both preset
    geometries are pinned: owner (stack-owned masters, `{stack: replicate}` staging)
    and zero1 (intra-matrix masters hopping to the same split), the second with the
    receipt-class guard — no collective may materialize a full fp32 kind-stack per
    device; every transition stays shard-to-shard. The module-level collective count
    is the positive control that the census is parsing real HLO."""
    import math

    import numpy as np
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from param_decomp.core.muon_stacked import _staged_newton_schulz
    from param_decomp.core.placement import PlacedRule, ns_staging_sharding

    assert jax.device_count() >= 4, "needs 4 sim devices; run via `make test-multidevice`"
    mesh = Mesh(
        np.array(jax.devices()[:4]).reshape(2, 2),
        ("replicate", "fsdp"),
        axis_types=(AxisType.Explicit,) * 2,
    )
    coeffs = jnp.array([(3.4445, -4.7750, 2.0315)] * 5, jnp.float32)

    def compiled_census(tree: dict[str, Array], row: PlacedRule) -> str:
        waypoint = ns_staging_sharding(row, mesh)
        waypoints = lambda t: jax.tree.map(lambda _: waypoint, t)
        capfd.readouterr()
        compiled = (
            jax.jit(
                lambda t: _staged_newton_schulz(t, coeffs, 5, jnp.dtype(jnp.float32), waypoints)
            )
            .lower(tree)
            .compile()
        )
        assert "Involuntary full rematerialization" not in capfd.readouterr().err
        hlo = compiled.as_text()
        assert hlo is not None
        assert _collective_count(hlo) > 0, "entry/exit transitions must exist on a real mesh"
        assert _while_body_collective_count(hlo) == 0, (
            "a collective lowered INSIDE the NS while-loop — the staging layout no"
            " longer keeps the Gram matmuls device-local"
        )
        return hlo

    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    owner = NamedSharding(mesh, P("replicate", "fsdp", None))
    owner_tree = {
        "a": jax.device_put(jax.random.normal(k1, (4, 8, 16), jnp.float32), owner),
        "b": jax.device_put(jax.random.normal(k2, (2, 16, 8), jnp.float32), owner),
    }
    compiled_census(owner_tree, PlacedRule(mesh=mesh, label="ns", rule={"stack": "replicate"}))

    # zero1 geometry: intra-matrix masters hopping to the same node-axis stack split.
    # The receipt-class guard: replicated staging used to materialize whole fp32 stacks
    # on every device — no collective may produce a full-kind-stack-sized (or larger)
    # per-device buffer.
    zero1 = NamedSharding(mesh, P(None, "fsdp", "replicate"))
    zero1_tree = {
        "a": jax.device_put(jax.random.normal(k1, (4, 8, 16), jnp.float32), zero1),
        "b": jax.device_put(jax.random.normal(k3, (6, 8, 16), jnp.float32), zero1),
    }
    hlo = compiled_census(
        zero1_tree, PlacedRule(mesh=mesh, label="ns", rule={"stack": "replicate"})
    )
    smallest_kind_elems = min(math.prod(v.shape) for v in zero1_tree.values())
    for shape in _collective_result_shapes(hlo):
        assert math.prod(shape) < smallest_kind_elems, (
            f"a collective materializes a per-device buffer of shape {shape} — the"
            " whole-stack replication the stack-split staging exists to prevent"
        )

    # The CI-ffn master layout (the whole mesh on ONE matrix dim, reversed axis order)
    # is the pair that trips the SPMD fallback under any single direct reshard — the
    # cell `staging_hops` exists for.
    assert jax.device_count() >= 8, "needs 8 sim devices; run via `make test-multidevice`"
    mesh3 = Mesh(
        np.array(jax.devices()[:8]).reshape(2, 2, 2),
        ("replicate", "fsdp", "tp"),
        axis_types=(AxisType.Explicit,) * 3,
    )
    ffn_master = NamedSharding(mesh3, P(None, None, ("tp", "fsdp", "replicate")))
    ffn_tree = {"w1": jax.device_put(jax.random.normal(k1, (4, 8, 16), jnp.float32), ffn_master)}
    row = PlacedRule(mesh=mesh3, label="ns", rule={"stack": "replicate"})
    waypoint3 = ns_staging_sharding(row, mesh3)
    waypoints = lambda t: jax.tree.map(lambda _: waypoint3, t)
    capfd.readouterr()
    compiled = (
        jax.jit(lambda t: _staged_newton_schulz(t, coeffs, 5, jnp.dtype(jnp.float32), waypoints))
        .lower(ffn_tree)
        .compile()
    )
    assert "Involuntary full rematerialization" not in capfd.readouterr().err, (
        "the hop-chain staging regressed to the fallback lowering on the CI-ffn layout"
    )
    hlo = compiled.as_text()
    assert hlo is not None
    assert _while_body_collective_count(hlo) == 0
    for shape in _collective_result_shapes(hlo):
        assert math.prod(shape) < math.prod(ffn_tree["w1"].shape)
