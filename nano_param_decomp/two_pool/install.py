"""Layout-aware model construction.

`install_components_for_layout`: only installs ComponentLinear at sites this rank
actually has — on pool A, just the owned sites; on pool B, every site. Non-installed
paths keep their original `nn.Linear`. No phantom V/U on pool A.

`build_ci_fns_for_layout`: returns the per-module CI fns this rank actually owns.
Empty dict on pool B.
"""

from typing import cast

import torch.nn as nn
from torch import Tensor

from nano_param_decomp.run import ComponentLinear
from nano_param_decomp.two_pool.layout import TwoPoolLayout
from nano_param_decomp.two_pool_stage2 import ModuleCIFn, build_ci_fns


def install_components_for_layout(
    target: nn.Module,
    layout: TwoPoolLayout,
    c_per_site: dict[str, int],
) -> dict[str, ComponentLinear]:
    """Install ComponentLinear at the sites this rank has; freeze the rest of the target.

    Pool A: install only at `layout.my_owned_sites`. Other site paths keep their
    original `nn.Linear` — no V/U allocated for non-owned sites.

    Pool B: install at every site. Pool B needs all V/U replicated for the
    full-model PPGD forward, so every site needs a ComponentLinear wrapper.

    Returns a dict keyed only by the sites actually installed on this rank.
    """
    paths = layout.my_owned_sites if layout.my_pool == "a" else layout.world.all_sites

    for p in target.parameters():
        p.requires_grad_(False)

    wrappers: dict[str, ComponentLinear] = {}
    for path in paths:
        C = c_per_site[path]
        parent_path, _, attr = path.rpartition(".")
        parent = target.get_submodule(parent_path) if parent_path else target
        linear = target.get_submodule(path)
        assert isinstance(linear, nn.Linear), f"{path} is not nn.Linear"
        wrapper = ComponentLinear(linear, C)
        setattr(parent, attr, wrapper)
        wrappers[path] = wrapper
    return wrappers


def build_ci_fns_for_layout(
    layout: TwoPoolLayout,
    wrappers: dict[str, ComponentLinear],
    c_per_site: dict[str, int],
    hidden: int,
    leaky_alpha: float,
) -> dict[str, ModuleCIFn]:
    """Per-site CI fns — pool A only, owned sites only. Empty on pool B."""
    if layout.my_pool != "a":
        return {}
    d_in_per_site = {
        s: int(cast(Tensor, wrappers[s].W_target).shape[1]) for s in layout.my_owned_sites
    }
    owned_c = {s: c_per_site[s] for s in layout.my_owned_sites}
    return build_ci_fns(d_in_per_site, owned_c, hidden=hidden, leaky_alpha=leaky_alpha)
