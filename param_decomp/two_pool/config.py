"""Serializable config for 2-pool training topology.

``TwoPoolConfig`` carries only what's genuinely 2-pool-specific: which ranks
form pool A's block groups (replicated V/U + CI fn + shared owned sites) and
which ranks form pool B (stateless PPGD replica).

Everything else (batch size, loss coefficients, optimizer LRs, PPGD config,
faithfulness warmup, LR schedules, autocast, ci_config, sigmoid_type,
module_info) lives on the regular ``RunConfig`` / ``PDConfig`` /
``RuntimeConfig`` that the 2-pool path consumes — so a 2-pool training run is
configured like a normal SPD run, plus this topology block.

``validate_run_cfg_for_two_pool`` in ``driver_entry.py`` checks that the
RunConfig's ``pd.loss_metrics`` contains the four loss types the 2-pool path
implements (FaithfulnessLoss, ImportanceMinimalityLoss,
StochasticReconLayerwiseLoss, PersistentPGDReconLoss) and that none of the
unsupported loss types are silently set.
"""

from typing import Self

from pydantic import Field, model_validator

from param_decomp.base_config import BaseConfig


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
    """Topology for 2-pool training. Pairs with a regular ``RunConfig``.

    Every pool-A rank lives in exactly one ``BlockGroupSpec``. Pool-B ranks
    are listed explicitly. Topology integrity checks (rank disjointness,
    uniform N_per_block) run at validation time; checks that depend on
    ``pd.batch_size`` (divisibility) run in ``validate_run_cfg_for_two_pool``
    since they need the paired ``RunConfig`` to evaluate.
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
    use_fused_kl: bool = Field(
        default=True,
        description="If True (default), pool A + pool B bypass the LM head and "
        "compute KL via the chunked fused linear+KL kernel — never materializes "
        "[b_local, seq, vocab] tensors. If False, the unfused path is used "
        "(materialize logits + standard recon_loss). Toggle for ablation of "
        "the kernel vs pure topology levers; defaults to fused because the "
        "kernel reliably cuts pool A peak ~50%% at b=64 s=2048 with negligible "
        "step-time cost.",
    )

    @model_validator(mode="after")
    def validate_topology(self) -> Self:
        assert self.block_groups, "block_groups must be non-empty"
        n_per_block = len(self.block_groups[0].ranks)
        assert n_per_block > 0, "block_groups[0].ranks must be non-empty"
        for bg in self.block_groups:
            assert len(bg.ranks) == n_per_block, (
                f"all block groups must have the same N_per_block; got "
                f"{n_per_block} vs {len(bg.ranks)} for sites {bg.owned_sites[:2]}..."
            )
            assert bg.owned_sites, "block_group must own at least one site"
        pool_a_flat = [r for bg in self.block_groups for r in bg.ranks]
        assert len(set(pool_a_flat)) == len(pool_a_flat), (
            "the same rank appears in multiple block groups"
        )
        pool_b = set(self.pool_b_ranks)
        pool_a = set(pool_a_flat)
        assert pool_a.isdisjoint(pool_b), (
            f"pool A and pool B ranks must be disjoint; overlap: {sorted(pool_a & pool_b)}"
        )
        return self
