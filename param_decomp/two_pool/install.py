"""Layout-aware helpers for installing ComponentModel under a 2-pool layout.

The existing `ComponentModel.__init__` takes a `module_path_info: list[ModulePathInfo]`
and installs components at exactly those module paths. The 2-pool integration is
therefore non-invasive: we just compute the right per-pool subset of
`ModulePathInfo` and hand it to `ComponentModel`.

  - Pool A rank: only its owned sites' ModulePathInfo
    → ComponentModel only allocates V/U + CI fn at owned sites.
    → Non-owned site paths remain as their original `nn.Linear`/`nn.Embedding`/`Conv1D`.

  - Pool B rank: every site's ModulePathInfo
    → ComponentModel installs V/U at every site for the full-model PPGD forward.
    → CI fn is also built here in the current implementation; in practice pool B
      doesn't use it (we send CI values across the wire) — call sites should ensure
      pool B's CI fn params are excluded from its (nonexistent) optimizer.
"""

from param_decomp.two_pool.layout import BlockDDPLayout, TwoPoolLayout
from param_decomp.utils.module_utils import ModulePathInfo


def build_pool_a_module_path_info(
    layout: TwoPoolLayout | BlockDDPLayout,
    c_per_site: dict[str, int],
) -> list[ModulePathInfo]:
    """For a pool-A rank, the subset of ModulePathInfo for its owned sites only."""
    assert layout.my_pool == "a"
    return [ModulePathInfo(module_path=s, C=c_per_site[s]) for s in layout.my_owned_sites]


def build_pool_b_module_path_info(
    layout: TwoPoolLayout | BlockDDPLayout,
    c_per_site: dict[str, int],
) -> list[ModulePathInfo]:
    """For a pool-B rank, the full ModulePathInfo for every site (replicated V/U)."""
    assert layout.my_pool == "b"
    return [
        ModulePathInfo(module_path=s, C=c_per_site[s])
        for s in layout.world.all_sites
    ]
