"""Analytical 1-pool vs 3-pool ImpMin under hardcoded L0/C values.

Skips synthetic-tensor generation. For each site we assume a specific shape of
``per_component_sums`` vector (the [C]-length tensor that _finalize consumes)
and per-rank distribution, then plug DIRECTLY into the _finalize formula —
both for 1-pool's local-with-W-multiplier path and 3-pool's
global-with-world_size=1 path.

This makes the underlying distributional assumption explicit and lets us
trace where the 3× could come from.
"""

from __future__ import annotations

import math

# Observed per-site (L0, C) from p-93bf5f42.
SITES: list[tuple[str, float, int]] = [
    # (site_name, observed_L0_per_position, n_components_C)
    ("h.0.attn.q_proj", 6.60, 512),
    ("h.0.attn.k_proj", 6.40, 512),
    ("h.0.attn.v_proj", 6.45, 1024),
    ("h.0.attn.o_proj", 11.79, 1024),
    ("h.0.mlp.c_fc", 44.58, 3072),
    ("h.0.mlp.down_proj", 43.25, 3584),
    ("h.1.attn.q_proj", 4.17, 512),
    ("h.1.attn.k_proj", 4.99, 512),
    ("h.1.attn.v_proj", 12.11, 1024),
    ("h.1.attn.o_proj", 22.66, 1024),
    ("h.1.mlp.c_fc", 31.53, 3072),
    ("h.1.mlp.down_proj", 27.45, 3584),
    ("h.2.attn.q_proj", 13.18, 512),
    ("h.2.attn.k_proj", 12.58, 512),
    ("h.2.attn.v_proj", 19.00, 1024),
    ("h.2.attn.o_proj", 33.17, 1024),
    ("h.2.mlp.c_fc", 35.48, 3072),
    ("h.2.mlp.down_proj", 33.27, 3584),
    ("h.3.attn.q_proj", 7.38, 512),
    ("h.3.attn.k_proj", 7.21, 512),
    ("h.3.attn.v_proj", 18.27, 1024),
    ("h.3.attn.o_proj", 27.83, 1024),
    ("h.3.mlp.c_fc", 53.30, 3072),
    ("h.3.mlp.down_proj", 65.45, 3584),
]

BATCH_GLOBAL = 64
SEQ_LEN = 512
W_DP = 8
N_CI = 4
BETA = 0.5
# Sanity: at step ~13k, p_anneal_final_p=0.4, p_anneal_end_frac=1.0 → roughly p=2 still.
PNORM = 2.0  # try p=2.0; loss should be insensitive to this for binary CI


def site_loss_uniform_per_component(
    L0: float, C: int, batch_local: int, seq: int, *, world_size: int, beta: float
) -> float:
    """Loss contribution of one site under "every component has identical
    per-rank activation rate" assumption.

    Under that assumption, every component's local_sum is equal (= L0/C *
    batch_local * seq). _finalize sums over C components: layer_loss =
    C * (mean + beta * mean * log2(1 + sum * world_size)).
    """
    n_local = batch_local * seq
    per_comp_sum = (L0 / C) * n_local
    per_comp_mean = per_comp_sum / n_local  # = L0/C, independent of n_local
    log_term = math.log2(1 + per_comp_sum * world_size)
    layer_loss = C * (per_comp_mean + beta * per_comp_mean * log_term)
    return layer_loss


def total_loss_1pool(beta: float = BETA, pnorm_irrelevant_for_binary: float = PNORM) -> float:
    """Per-rank local loss under uniform assumption; 1-pool reports the avg
    across W ranks, but under uniform that's the same value."""
    del pnorm_irrelevant_for_binary
    batch_local = BATCH_GLOBAL // W_DP  # 8
    total = 0.0
    for _, L0, C in SITES:
        total += site_loss_uniform_per_component(
            L0, C, batch_local, SEQ_LEN, world_size=W_DP, beta=beta
        )
    return total


def total_loss_3pool(beta: float = BETA, pnorm_irrelevant_for_binary: float = PNORM) -> float:
    """3-pool computes on global sums, world_size=1."""
    del pnorm_irrelevant_for_binary
    total = 0.0
    for _, L0, C in SITES:
        # 3-pool: pretend we have global_sum = L0/C * batch_global * seq
        n_global = BATCH_GLOBAL * SEQ_LEN
        per_comp_sum = (L0 / C) * n_global
        per_comp_mean = per_comp_sum / n_global  # = L0/C
        log_term = math.log2(1 + per_comp_sum * 1)
        total += C * (per_comp_mean + beta * per_comp_mean * log_term)
    return total


def site_breakdown_uniform() -> None:
    print(f"{'site':<26} {'L0':>7} {'C':>5} {'1pool':>14} {'3pool':>14} {'ratio':>7}")
    one_pool_total = 0.0
    three_pool_total = 0.0
    for site, L0, C in SITES:
        # 1-pool: per-rank loss with world_size=W
        op = site_loss_uniform_per_component(
            L0, C, BATCH_GLOBAL // W_DP, SEQ_LEN, world_size=W_DP, beta=BETA
        )
        # 3-pool: global with world_size=1
        tp_n_local = BATCH_GLOBAL * SEQ_LEN
        tp_sum = (L0 / C) * tp_n_local
        tp_mean = tp_sum / tp_n_local
        tp_log = math.log2(1 + tp_sum * 1)
        tp = C * (tp_mean + BETA * tp_mean * tp_log)
        one_pool_total += op
        three_pool_total += tp
        ratio = op / tp if tp > 0 else float("nan")
        print(f"{site:<26} {L0:>7.2f} {C:>5} {op:>14.4g} {tp:>14.4g} {ratio:>7.4f}")
    print(
        f"{'TOTAL':<26} {'':>7} {'':>5} {one_pool_total:>14.4g} "
        f"{three_pool_total:>14.4g} {one_pool_total / three_pool_total:>7.4f}"
    )


def site_breakdown_extreme_concentration() -> None:
    """Now assume each component is active on EXACTLY 1 of W=8 batch elements
    (the worst-case cross-rank variance) and all positions of that batch
    element are active.
    """
    print(
        "\nAssuming each component active on 1 of 8 batch elements (all positions in that element):"
    )
    print(f"{'site':<26} {'L0':>7} {'C':>5} {'1pool':>14} {'3pool':>14} {'ratio':>7}")
    one_pool_total = 0.0
    three_pool_total = 0.0
    batch_local = BATCH_GLOBAL // W_DP  # 8
    seq = SEQ_LEN
    for site, L0, C in SITES:
        # Total active positions for a component = (L0/C) * BATCH_GLOBAL * seq.
        # Concentrated on 1 batch element → that element has all positions active.
        # Probability that this batch element falls on rank r: 1/W
        # When on rank r: local_sum_c = seq (the full element)
        # When not: local_sum_c = 0
        # Avg over r of 1-pool's per-rank loss:
        #   Per rank r contributing component c: (active prob = 1/W) * f(seq, W) + (1-1/W)*f(0,W)
        #   f(s, W) = s/n_local + beta * s/n_local * log2(1 + s*W)
        # Sum over C: each component contributes 1/W * f(seq, W)
        # Times C → C/W * f(seq, W)
        # But not every component is necessarily active — only the ones whose L0/C is non-trivial.
        # Total active components per layer = expected # = L0 (sum of probs).
        # Wait the modelling here is: each component active on exactly 1 batch
        # element with prob equal to its (L0/C) "rate". To keep L0 per position
        # consistent, we need...
        # Actually let me just say: of C components, K = L0 are "active" at all;
        # each active component picks 1 random batch element to live on.
        K_active = L0  # fractional ok for math
        per_comp_sum_when_on = seq  # all positions in that batch element
        per_rank_loss_per_active_comp = (1.0 / W_DP) * (
            (per_comp_sum_when_on / (batch_local * seq))
            + BETA
            * (per_comp_sum_when_on / (batch_local * seq))
            * math.log2(1 + per_comp_sum_when_on * W_DP)
        )  # the (1-1/W)*0 case contributes 0
        op = K_active * per_rank_loss_per_active_comp
        # 3-pool: global_sum for active comp = seq (one batch element worth)
        # global_n = BATCH_GLOBAL * seq
        tp_per_active = (per_comp_sum_when_on / (BATCH_GLOBAL * seq)) + BETA * (
            per_comp_sum_when_on / (BATCH_GLOBAL * seq)
        ) * math.log2(1 + per_comp_sum_when_on)
        tp = K_active * tp_per_active
        one_pool_total += op
        three_pool_total += tp
        ratio = op / tp if tp > 0 else float("nan")
        print(f"{site:<26} {L0:>7.2f} {C:>5} {op:>14.4g} {tp:>14.4g} {ratio:>7.4f}")
    print(
        f"{'TOTAL':<26} {'':>7} {'':>5} {one_pool_total:>14.4g} "
        f"{three_pool_total:>14.4g} {one_pool_total / three_pool_total:>7.4f}"
    )


def main() -> None:
    print(f"Config: batch_global={BATCH_GLOBAL}, seq={SEQ_LEN}, W_DP={W_DP}, N_CI={N_CI}, β={BETA}")
    print()
    print("Assumption 1: uniform — every component has same per-position activation rate L0/C")
    site_breakdown_uniform()
    site_breakdown_extreme_concentration()

    print()
    print("=== Actual observed (wandb at step 13800) ===")
    print("  1-pool train/loss/ImportanceMinimalityLoss ≈ 2143")
    print("  3-pool train/loss/imp (c4p4) ≈ 814")
    print("  3-pool train/loss/imp (c8p8) ≈ 735")
    print(f"  Observed ratio (1p/c4p4): {2143 / 814:.4f}")
    print(f"  Observed ratio (1p/c8p8): {2143 / 735:.4f}")


if __name__ == "__main__":
    main()
