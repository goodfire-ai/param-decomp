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
splitting the CI fn into its own pool (enables global shared transformer CI
fns again — under 2-pool, sites are sharded across pool-A ranks, which
structurally rules out CI fns that span all sites).

Topology integrity checks (rank disjointness, uniform N_per_block, CI-pool
divisibility into Layerwise/PPGD batch shards) run at validation time; checks
that depend on ``pd.batch_size`` (divisibility) run in
``_validate_pd_config_for_three_pool`` since they need the paired ``PDConfig``
to evaluate.
"""

from typing import Self

from pydantic import Field, model_validator

from param_decomp.base_config import BaseConfig


class LayerwiseBlockGroupSpec(BaseConfig):
    """One block-DDP group on the Layerwise pool: ranks that replicate V/U for
    a shared set of sites. The first rank is the block leader (canonical actor
    for cross-pool sends). Within a group, in-block all-reduce keeps the
    replicas in sync after each optimizer step.

    Serializable mirror of ``param_decomp.three_pool.layout.LayerwiseBlockGroup``.
    The layout module's dataclass is constructed from this at runtime.

    Identical shape to ``two_pool.config.BlockGroupSpec`` — duplicated rather
    than imported so each subsystem owns its own config surface.
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

      * ``N_ci`` must divide ``N_per_block_layerwise`` so each Layerwise rank's
        batch slice fits inside exactly one CI rank's slice (one-to-many
        fan-out for CI values; many-to-one reduction for CI grads).
      * ``N_ci`` must divide ``N_ppgd`` for the same reason on the CI↔PPGD
        side.

    These two constraints reduce the otherwise many-to-many batch routing to
    the simpler one-to-many / many-to-one shape that the comms layer
    implements. The validator rejects any other arrangement loudly.
    """

    ci_ranks: list[int] = Field(
        ...,
        description="Ranks assigned to the CI pool (replicated CI fn, DP across batch). "
        "Must divide both N_per_block_layerwise and N_ppgd (see class docstring).",
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
        "(materialize logits + standard recon_loss). Same lever as in 2-pool; "
        "defaults to fused because the kernel reliably cuts peak memory ~50%% on "
        "large-vocab targets with negligible step-time cost.",
    )
    defer_vu_opt: bool = Field(
        default=False,
        description="If True, defer the Layerwise pool's V/U AdamW step + the "
        "V/U ship-back to PPGD from end-of-step-T to start-of-step-T+1, so they "
        "hide behind T+1's CI fn forward window. Requires symmetric deferral on "
        "the PPGD pool (recv_updated_vu also moves to start-of-step-T+1) — "
        "otherwise deadlock. Sync and deferred modes are mathematically "
        "equivalent (the deferred tail still uses step-T's grads with step-T's "
        "LR via `get_scheduled_value(step-1, ...)`); the toggle is purely a "
        "wall-clock perf knob. A/B by flipping the flag and comparing "
        "`perf/step_ms` traces.",
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

        # Batch-split divisibility (MVP constraint — see DESIGN.md open Q1).
        n_ci = len(self.ci_ranks)
        n_ppgd = len(self.ppgd_ranks)
        assert n_per_block % n_ci == 0, (
            f"N_ci ({n_ci}) must divide N_per_block_layerwise ({n_per_block}) so each "
            f"Layerwise rank's batch slice fits inside exactly one CI rank's slice. "
            f"Tip: choose a smaller CI pool, or grow the layerwise block-DDP factor."
        )
        assert n_ppgd % n_ci == 0, (
            f"N_ci ({n_ci}) must divide N_ppgd ({n_ppgd}) so each PPGD rank's batch "
            f"slice fits inside exactly one CI rank's slice. "
            f"Tip: choose a smaller CI pool, or grow the PPGD pool."
        )
        return self
