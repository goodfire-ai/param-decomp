"""Layout-aware helpers for installing ComponentModel under a 2-pool layout.

The existing `ComponentModel.__init__` takes a `module_path_info: list[DecompositionTarget]`
and installs components at exactly those module paths. The 2-pool integration is
therefore non-invasive: we just compute the right per-pool subset of
`DecompositionTarget` and hand it to `ComponentModel`.

  - Pool A rank: only its owned sites' DecompositionTarget
    → ComponentModel only allocates V/U + CI fn at owned sites.
    → Non-owned site paths remain as their original `nn.Linear`/`nn.Embedding`/`Conv1D`.

  - Pool B rank: every site's DecompositionTarget
    → ComponentModel installs V/U at every site for the full-model PPGD forward.
    → CI fn is also built here in the current implementation; in practice pool B
      doesn't use it (we send CI values across the wire) — call sites should ensure
      pool B's CI fn params are excluded from its (nonexistent) optimizer.
"""

from param_decomp.decomposition_targets import DecompositionTarget
from param_decomp.two_pool.layout import BlockDDPLayout, TwoPoolLayout


def build_pool_a_module_path_info(
    layout: TwoPoolLayout | BlockDDPLayout,
    c_per_site: dict[str, int],
) -> list[DecompositionTarget]:
    """For a pool-A rank, the subset of DecompositionTarget for its owned sites only."""
    assert layout.my_pool == "a"
    return [DecompositionTarget(module_path=s, C=c_per_site[s]) for s in layout.my_owned_sites]


def build_pool_b_module_path_info(
    layout: TwoPoolLayout | BlockDDPLayout,
    c_per_site: dict[str, int],
) -> list[DecompositionTarget]:
    """For a pool-B rank, the full DecompositionTarget for every site (replicated V/U)."""
    assert layout.my_pool == "b"
    return [DecompositionTarget(module_path=s, C=c_per_site[s]) for s in layout.world.all_sites]
