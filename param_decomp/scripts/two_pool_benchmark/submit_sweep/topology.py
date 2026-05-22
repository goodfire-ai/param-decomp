"""Render a ``TwoPoolConfig`` topology yaml from a :class:`TopologySpec`.

Each transformer layer's 7 sites (q/k/v/o + gate/up/down) are partitioned into
1, 2, or 7 site-groups by the spec's ``grouping``. ``blocks_per_group`` fuses
N consecutive layers' partitions into the same group. Each resulting group is
assigned ``ddp`` ranks. Pool B size auto-pads to align world to 8 (= GPUs/node)
unless explicitly set.
"""

from param_decomp.scripts.two_pool_benchmark.submit_sweep.schema import TopologySpec

SITE_KINDS_BY_GROUPING: dict[str, list[list[str]]] = {
    "fused": [
        [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ]
    ],
    "attn_mlp": [
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"],
        ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
    ],
    "per_site": [
        [s]
        for s in [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ]
    ],
}


def _auto_pool_b(n_pool_a: int) -> int:
    """Smallest pool B size that aligns total world to 8 (= GPUs per node)."""
    rem = n_pool_a % 8
    return 8 - rem if rem != 0 else 8


def render_topology(spec: TopologySpec, n_layers: int) -> tuple[str, int, int, int]:
    """Render the topology yaml.

    Returns ``(yaml_text, world_size, n_nodes, pool_b_size)``.
    """
    site_groups = SITE_KINDS_BY_GROUPING[spec.grouping]
    bpg = spec.blocks_per_group
    assert n_layers % bpg == 0, f"n_layers={n_layers} not divisible by blocks_per_group={bpg}"
    groups: list[tuple[list[int], list[str]]] = []
    next_rank = 0
    # Iterate over layer-chunks of size `blocks_per_group`. Within each chunk,
    # `site_groups` partitions the chunk's sites. For fused (the only grouping
    # supporting bpg>1), all 7 sites of every layer in the chunk go into one
    # group → 7*bpg sites/group.
    for layer_start in range(0, n_layers, bpg):
        chunk_layers = list(range(layer_start, layer_start + bpg))
        for site_kinds in site_groups:
            ranks = list(range(next_rank, next_rank + spec.ddp))
            sites = [f"model.layers.{layer}.{k}" for layer in chunk_layers for k in site_kinds]
            groups.append((ranks, sites))
            next_rank += spec.ddp
    pool_b = spec.pool_b if spec.pool_b is not None else _auto_pool_b(next_rank)
    pool_b_ranks = list(range(next_rank, next_rank + pool_b))
    world = next_rank + pool_b
    assert world % 8 == 0, (
        f"world={world} not a multiple of 8 (one node per 8 GPUs). Set pool_b explicitly to align."
    )
    n_nodes = world // 8

    lines = [
        f"# Auto-generated topology. grouping={spec.grouping} ddp={spec.ddp} "
        f"use_fused_kl={spec.use_fused_kl}\n",
        f"# Layout: {world - pool_b}A + {pool_b}B = {world} GPUs across {n_nodes} nodes.\n",
        "block_groups:\n",
    ]
    for ranks, sites in groups:
        sites_inline = ", ".join(f"'{s}'" for s in sites)
        lines.append(f"  - {{ ranks: {ranks!r}, owned_sites: [{sites_inline}] }}\n")
    lines.append(f"pool_b_ranks: {pool_b_ranks!r}\n")
    lines.append(f"use_fused_kl: {str(spec.use_fused_kl).lower()}\n")
    return "".join(lines), world, n_nodes, pool_b


def topology_label(spec: TopologySpec) -> str:
    """Short human-readable label used in dir/job names and view_meta.

    Encodes the fused-KL toggle and blocks_per_group so distinct topologies
    at the same (batch, seq, ci) don't share a dir / W&B name.
    """
    grouping_short = {"fused": "fused", "attn_mlp": "split", "per_site": "site"}[spec.grouping]
    blk_prefix = "" if spec.blocks_per_group == 1 else f"{spec.blocks_per_group}block-"
    kl_suffix = "" if spec.use_fused_kl else "-nofkl"
    return f"{blk_prefix}{grouping_short}-{spec.ddp}ddp{kl_suffix}"
