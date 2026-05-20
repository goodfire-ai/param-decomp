"""Serializable config for 2-pool training (`optimize_two_pool`).

Parallels `PDConfig` for the single-pool path. Carries only the bits that
make sense to specify in YAML: topology, loss coefficients, and the PPGD
config. Runtime objects (PDTarget callables) come from the driver like any
other run.

Other bits we deliberately don't duplicate here:
  - ``ci_config``  — comes from ``PDConfig.ci_config``.
  - ``autocast_bf16`` — comes from ``RuntimeConfig.autocast_bf16``.
  - Optimizer LR for V/U & CI fn — comes from ``PDConfig.components_optimizer``
    / ``PDConfig.ci_fn_optimizer``.
  - ``module_info`` / ``c_per_site`` — derived from ``PDConfig.module_info``
    at runtime against the loaded target model.
"""

from typing import Self

from pydantic import Field, model_validator

from param_decomp.base_config import BaseConfig
from param_decomp.configs import PersistentPGDReconLossConfig


class BlockGroupSpec(BaseConfig):
    """One block-DDP group: the ranks that replicate V/U + CI fn for a shared
    set of sites. The first rank is the block leader (canonical actor for
    cross-pool sends). Within a group, in-block all-reduce keeps the replicas
    in sync after each optimizer step.

    Serializable mirror of `param_decomp.two_pool.layout.BlockGroup`. The
    layout module's dataclass is constructed from this at runtime.
    """

    ranks: list[int] = Field(..., description="Ranks that replicate V/U for `owned_sites`.")
    owned_sites: list[str] = Field(
        ..., description="Module paths in the target that this group owns."
    )


class TwoPoolConfig(BaseConfig):
    """Topology + loss coefficients for 2-pool training.

    Set ``RunConfig.two_pool = TwoPoolConfig(...)`` to dispatch ``run_pd`` to
    ``optimize_two_pool`` instead of the single-pool ``optimize``.

    Topology is fully explicit — every pool-A block group lists its ranks and
    owned sites, and pool-B ranks are an explicit list. Convenience builders
    (e.g. attn/mlp split, one-site-per-rank) belong elsewhere; this is the
    serializable substrate.
    """

    block_groups: list[BlockGroupSpec] = Field(
        ...,
        description="Pool-A block groups. Each group's ranks replicate V/U + CI fn for "
        "the group's owned sites.",
    )
    pool_b_ranks: list[int] = Field(
        ...,
        description="Ranks assigned to pool B (stateless PPGD replica with DP across the batch).",
    )
    batch_global: int = Field(
        ...,
        description="Global batch size. Must be divisible by ``len(pool_b_ranks)`` and by "
        "``len(block_groups[0].ranks)`` (intra-block DDP).",
    )

    # ── Loss coefficients ──
    coeff_faith: float = 1e6
    coeff_imp: float = 1e-4
    coeff_stoch: float = 0.5
    coeff_ppgd: float = 0.5

    # ── PPGD config (separate from pd.loss_metrics — 2-pool drives PPGD
    #    directly via PersistentPGDState rather than through the metric loop) ──
    ppgd: PersistentPGDReconLossConfig

    # ── Optimizer LRs (V1 — to be merged with pd.components_optimizer /
    #    pd.ci_fn_optimizer in a later cleanup) ──
    lr_components: float = 5e-5
    lr_ci_fn: float = 5e-5

    @model_validator(mode="after")
    def validate_topology(self) -> Self:
        n_per_block = len(self.block_groups[0].ranks)
        for bg in self.block_groups:
            assert len(bg.ranks) == n_per_block, (
                f"all block groups must have the same N_per_block; got "
                f"{n_per_block} vs {len(bg.ranks)} for sites {bg.owned_sites[:2]}..."
            )
        assert self.batch_global % n_per_block == 0, (
            f"batch_global ({self.batch_global}) must be divisible by N_per_block ({n_per_block})"
        )
        assert self.batch_global % len(self.pool_b_ranks) == 0, (
            f"batch_global ({self.batch_global}) must be divisible by N_pool_b ({len(self.pool_b_ranks)})"
        )
        pool_a = {r for bg in self.block_groups for r in bg.ranks}
        pool_b = set(self.pool_b_ranks)
        assert pool_a.isdisjoint(pool_b), (
            f"pool A and pool B ranks must be disjoint; overlap: {pool_a & pool_b}"
        )
        return self
