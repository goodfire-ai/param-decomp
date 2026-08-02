import jax
import jax.numpy as jnp

from param_decomp.core.adversary import componentwise_uniform
from param_decomp.core.ci_fn import (
    LayerwiseMLPCIArch,
    LayerwiseMLPCIFn,
    build_ci_fn,
    pin_ci_head_random_prefix_null_tail,
)
from param_decomp.core.components import (
    SiteSpec,
    component_stacks_from_sites,
    init_stack_arrays_random_prefix_null_tail,
)
from param_decomp.core.run_state import scale_component_updates_function_covariant_balanced


def _site(c: int) -> tuple[SiteSpec, ...]:
    return (SiteSpec("s", 7, 5, c),)


def _vu(c: int, k: int):
    sites = _site(c)
    stacks = init_stack_arrays_random_prefix_null_tail(sites, {"s": k}, jax.random.key(1))
    shape = (7, 5, c)
    return stacks[shape][0][0], stacks[shape][1][0]


def test_random_prefix_uses_logical_scale_and_is_physical_width_invariant() -> None:
    v100, u100 = _vu(100, 100)
    v800, u800 = _vu(800, 100)
    assert jnp.array_equal(v100, v800[:, :100])
    assert jnp.array_equal(u100, u800[:100])
    assert jnp.all(u800[100:] == 0.0)

    _, u800_full = _vu(800, 800)
    assert jnp.allclose(u800[:100], u800_full[:100] * jnp.sqrt(8.0))


def test_random_prefix_ci_head_is_physical_width_invariant() -> None:
    arch = LayerwiseMLPCIArch(hidden_dims=(11,), has_position_axis=False)
    key = jax.random.key(2)
    ci100 = pin_ci_head_random_prefix_null_tail(
        build_ci_fn(arch, _site(100), key), _site(100), {"s": 100}, key
    )
    ci800 = pin_ci_head_random_prefix_null_tail(
        build_ci_fn(arch, _site(800), key), _site(800), {"s": 100}, key
    )
    assert isinstance(ci100, LayerwiseMLPCIFn) and isinstance(ci800, LayerwiseMLPCIFn)
    m100, m800 = ci100.site_mlps["s"], ci800.site_mlps["s"]
    assert jnp.array_equal(m100.weights[0], m800.weights[0])
    assert jnp.array_equal(m100.weights[-1], m800.weights[-1][:, :100])
    assert jnp.all(m800.weights[-1][:, 100:] == 0.0)


def test_function_covariant_rule_reads_logical_not_physical_width() -> None:
    def update(c: int):
        stacks = component_stacks_from_sites({"s": (jnp.ones((7, c)), jnp.ones((c, 5)))})
        transform = scale_component_updates_function_covariant_balanced({"s": 100})
        state = transform.init(stacks)
        scaled, _ = transform.update(stacks, state)
        return scaled.site("s")

    dv100, du100 = update(100)
    dv800, du800 = update(800)
    assert jnp.array_equal(dv100, dv800[:, :100])
    assert jnp.array_equal(du100, du800[:100])


def test_componentwise_uniform_prefix_and_delta_ignore_physical_width() -> None:
    key = jax.random.key(3)
    x100 = componentwise_uniform(key, (4, 2), 100, with_delta=True)
    x800 = componentwise_uniform(key, (4, 2), 800, with_delta=True)
    assert jnp.array_equal(x100[..., :-1], x800[..., :100])
    assert jnp.array_equal(x100[..., -1], x800[..., -1])
