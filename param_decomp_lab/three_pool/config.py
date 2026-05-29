"""Serializable config for 3-pool training topology.

``ThreePoolConfig`` carries only what's genuinely 3-pool-specific: which ranks
form the CI pool (replicated CI fn + DP across batch), the Layerwise pool's
block groups (replicated V/U + shared owned sites within a group), and the
PPGD pool (stateless full V/U replica + DP across batch).

Everything else (batch size, loss coefficients, optimizer LRs, PPGD config,
faithfulness warmup, LR schedules, autocast, ci_config, sigmoid_type,
decomposition_targets) lives on the regular ``PDConfig`` / ``RuntimeConfig``
that the 3-pool path consumes — so a 3-pool training run is configured like a
normal SPD run plus this topology block.

See ``DESIGN.md`` for the per-step dependency graph and the rationale for
splitting the CI fn into its own pool (a dedicated unsharded CI pool enables
global shared transformer CI fns that span all sites).

Topology integrity checks (rank disjointness, uniform N_per_block, CI-pool
cross-divisibility with Layerwise/PPGD arities) run at validation time; checks
that depend on ``pd.batch_size`` (divisibility) + the rank-0 convention run on
``ThreePoolLMExperimentConfig`` (in ``experiments.lm.three_pool_run``) since they
need the paired ``ThreePoolConstrainedPDConfig`` to evaluate.
"""

from typing import Self

from pydantic import Field, model_validator

from param_decomp.base_config import BaseConfig


class LayerwiseBlockGroupSpec(BaseConfig):
    """One block-DDP group on the Layerwise pool: ranks that replicate V/U for
    a shared set of sites. The first rank is the block leader (canonical actor
    for cross-pool sends). Within a group, in-block all-reduce keeps the
    replicas in sync after each optimizer step.

    Serializable mirror of ``param_decomp_lab.three_pool.layout.LayerwiseBlockGroup``.
    The layout module's dataclass is constructed from this at runtime.
    """

    ranks: list[int] = Field(..., description="Ranks that replicate V/U for `owned_sites`.")
    owned_sites: list[str] = Field(
        ..., description="Module paths in the target that this group owns."
    )


class ThreePoolConfig(BaseConfig):
    """Topology for 3-pool training. Pairs with a regular ``PDConfig``.

    Every Layerwise-pool rank lives in exactly one ``LayerwiseBlockGroupSpec``.
    The CI pool and PPGD pool ranks are listed explicitly. The three pools
    must be rank-disjoint.

    Batch-split divisibility (see ``DESIGN.md`` open Q1):

      * ``N_ci`` and ``N_per_block_layerwise`` must be cross-divisible — one
        divides the other. Whichever is smaller owns the coarser batch slice;
        the finer pool's shards each nest in exactly one coarse shard, so the
        CI↔LW exchange is a clean one-to-K fan-out (and K-to-one reduction)
        in either direction.
      * ``N_ci`` and ``N_ppgd`` must be cross-divisible for the same reason on
        the CI↔PPGD side.

    These two constraints keep every batch-slice overlap a whole, aligned
    sub-slice (uniform integer fan-in/out), avoiding ragged many-to-many
    routing. The validator rejects any other arrangement loudly.
    """

    ci_ranks: list[int] = Field(
        ...,
        description="Ranks assigned to the CI pool (replicated CI fn, DP across batch). "
        "Must be cross-divisible with both N_per_block_layerwise and N_ppgd "
        "(see class docstring).",
    )
    layerwise_block_groups: list[LayerwiseBlockGroupSpec] = Field(
        ...,
        description="Layerwise-pool block groups. Each group's ranks replicate V/U for "
        "the group's owned sites.",
    )
    ppgd_ranks: list[int] = Field(
        ...,
        description="Ranks assigned to the PPGD pool (stateless full V/U replica, DP "
        "across batch).",
    )
    use_fused_kl: bool = Field(
        default=True,
        description="If True (default), Layerwise + PPGD bypass the LM head and "
        "compute KL via the chunked fused linear+KL kernel — never materializes "
        "[b_local, seq, vocab] tensors. If False, the unfused path is used "
        "(materialize logits + standard recon_loss). "
        "Defaults to fused because the kernel reliably cuts peak memory ~50%% on "
        "large-vocab targets with negligible step-time cost.",
    )

    @model_validator(mode="after")
    def validate_topology(self) -> Self:
        assert self.ci_ranks, "ci_ranks must be non-empty"
        assert self.layerwise_block_groups, "layerwise_block_groups must be non-empty"
        assert self.ppgd_ranks, "ppgd_ranks must be non-empty"

        # Per-block uniformity + non-empty owned sites
        n_per_block = len(self.layerwise_block_groups[0].ranks)
        assert n_per_block > 0, "layerwise_block_groups[0].ranks must be non-empty"
        for bg in self.layerwise_block_groups:
            assert len(bg.ranks) == n_per_block, (
                f"all layerwise block groups must have the same N_per_block; got "
                f"{n_per_block} vs {len(bg.ranks)} for sites {bg.owned_sites[:2]}..."
            )
            assert bg.owned_sites, "layerwise_block_group must own at least one site"

        # Rank disjointness across all three pools
        ci_set = set(self.ci_ranks)
        assert len(ci_set) == len(self.ci_ranks), "duplicate rank in ci_ranks"

        lw_flat = [r for bg in self.layerwise_block_groups for r in bg.ranks]
        lw_set = set(lw_flat)
        assert len(lw_set) == len(lw_flat), (
            "the same rank appears in multiple layerwise block groups"
        )

        pgd_set = set(self.ppgd_ranks)
        assert len(pgd_set) == len(self.ppgd_ranks), "duplicate rank in ppgd_ranks"

        ci_lw_overlap = sorted(ci_set & lw_set)
        ci_pgd_overlap = sorted(ci_set & pgd_set)
        lw_pgd_overlap = sorted(lw_set & pgd_set)
        assert not ci_lw_overlap, (
            f"CI pool and Layerwise pool must be rank-disjoint; overlap: {ci_lw_overlap}"
        )
        assert not ci_pgd_overlap, (
            f"CI pool and PPGD pool must be rank-disjoint; overlap: {ci_pgd_overlap}"
        )
        assert not lw_pgd_overlap, (
            f"Layerwise pool and PPGD pool must be rank-disjoint; overlap: {lw_pgd_overlap}"
        )

        # Batch-split divisibility (see DESIGN.md open Q1). Each cross-pool edge
        # (CI↔LW, CI↔PPGD) must be a clean one-to-K fan-out in EITHER direction:
        # one arity divides the other, so every batch-slice overlap is a whole,
        # aligned sub-slice. (The batch-divisibility of each arity itself is
        # enforced against pd.batch_size on ThreePoolLMExperimentConfig.)
        n_ci = len(self.ci_ranks)
        n_ppgd = len(self.ppgd_ranks)
        assert n_per_block % n_ci == 0 or n_ci % n_per_block == 0, (
            f"N_ci ({n_ci}) and N_per_block_layerwise ({n_per_block}) must be "
            f"cross-divisible (one divides the other) so each batch-slice overlap "
            f"is a whole sub-slice. Tip: make one a multiple of the other."
        )
        assert n_ppgd % n_ci == 0 or n_ci % n_ppgd == 0, (
            f"N_ci ({n_ci}) and N_ppgd ({n_ppgd}) must be cross-divisible (one "
            f"divides the other) so each batch-slice overlap is a whole sub-slice. "
            f"Tip: make one a multiple of the other."
        )
        return self
