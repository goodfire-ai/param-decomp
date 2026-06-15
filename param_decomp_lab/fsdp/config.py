"""`FsdpRuntimeConfig`: the core `RuntimeConfig` plus FSDP2 + torch.compile knobs."""

from pydantic import Field, PositiveInt

from param_decomp_config.pd import RuntimeConfig


class FsdpRuntimeConfig(RuntimeConfig):
    """Compute substrate for the single-pool FSDP2 LM path: core runtime + FSDP/compile knobs.

    Inherited `dp` is the FSDP world size here: the model (incl. the frozen target, when
    `shard_frozen_target`) is sharded across these ranks and the batch is data-parallel
    across them — not a DDP replica count.
    """

    dp: PositiveInt | None = Field(
        default=None,
        description=(
            "FSDP world size — ranks the model is sharded across, with the batch "
            "data-parallel over them. None means a single device."
        ),
    )

    compile_model: bool = Field(
        default=True,
        description="torch.compile the vendored target model (masked forward).",
    )
    compile_ci_fn: bool = Field(
        default=True,
        description="torch.compile the causal-importance function.",
    )
    checkpoint_blocks: bool = Field(
        default=True,
        description="Per-block activation checkpointing on the target model.",
    )
    shard_frozen_target: bool = Field(
        default=True,
        description="Convert frozen target buffers to no-grad params so FSDP2 shards the target "
        "(else they are replicated on every rank).",
    )
