"""Per-site values read through per-group stacks."""

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from param_decomp.core.components import SiteCI, SiteStackIndices, SiteValueStacks, site_ci_values

_INDICES: SiteStackIndices = tuple(
    (f"layers.{layer}.{kind}", kind, layer) for layer in range(4) for kind in ("q", "down")
)


def _values() -> dict[str, jax.Array]:
    return {
        name: jax.random.normal(jax.random.fold_in(jax.random.key(0), i), (2, 3, 5))
        for i, (name, _, _) in enumerate(_INDICES)
    }


def test_site_reads_return_each_sites_value_in_site_order():
    values = _values()
    stacks = SiteValueStacks.of(values, _INDICES)
    assert list(stacks) == [name for name, _, _ in _INDICES]
    assert len(stacks) == len(values)
    assert {
        group: [leaf.shape for leaf in jax.tree.leaves(stack)]
        for group, stack in stacks.stacks.items()
    } == {"q": [(4, 2, 3, 5)], "down": [(4, 2, 3, 5)]}
    for name, value in values.items():
        np.testing.assert_array_equal(np.asarray(stacks[name]), np.asarray(value))


def test_tree_maps_act_on_the_stacks_and_keep_the_view():
    stacks = SiteValueStacks.of(_values(), _INDICES)
    doubled = jax.tree.map(lambda leaf: 2 * leaf, stacks)
    assert isinstance(doubled, SiteValueStacks)
    assert len(jax.tree.leaves(doubled)) == 2
    for name in stacks:
        np.testing.assert_array_equal(np.asarray(doubled[name]), 2 * np.asarray(stacks[name]))


def test_a_loop_body_reading_every_site_closes_over_one_stack_per_group():
    """The memory property: a scan body slicing its sites from the view carries the
    group stacks as its operands, where a per-site dict would carry every site."""

    def scan_constant_count(values: Mapping[str, SiteCI]) -> int:
        def run(values: Mapping[str, SiteCI]) -> jax.Array:
            def body(total: jax.Array, _: None) -> tuple[jax.Array, None]:
                reads = (jnp.sum(site_ci_values(values[name])) for name, _, _ in _INDICES)
                return total + sum(reads), None

            return jax.lax.scan(body, jnp.zeros(()), None, length=2)[0]

        (scan,) = [eqn for eqn in jax.make_jaxpr(run)(values).eqns if eqn.primitive.name == "scan"]
        return len(scan.invars) - 1  # every operand but the carry is a closed-over constant

    assert scan_constant_count(SiteValueStacks.of(_values(), _INDICES)) == 2
    assert scan_constant_count(_values()) == len(_INDICES)
