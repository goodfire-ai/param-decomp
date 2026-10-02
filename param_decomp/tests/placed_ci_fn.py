"""Seeded CI fns initialized directly into their declared storage."""

import equinox as eqx
import jax
from jax.sharding import Mesh
from jaxtyping import PRNGKeyArray

from param_decomp.core.ci_fn.architecture import CIFnArchitecture
from param_decomp.core.ci_fn.interface import CIFn
from param_decomp.core.components import SiteSpec
from param_decomp.core.placement import PlacementRules


def placed_ci_fn[Conditioning](
    arch: CIFnArchitecture[Conditioning],
    sites: tuple[SiteSpec, ...],
    key: PRNGKeyArray,
    mesh: Mesh,
    rules: PlacementRules,
) -> CIFn[Conditioning]:
    init = lambda k: arch.initialize(sites, rules, k)
    shardings = eqx.filter_eval_shape(init, key).shardings(mesh)
    return jax.reshard(jax.jit(init, out_shardings=shardings)(key), shardings)
