"""Simulation: 1-pool vs 3-pool ImportanceMinimality semantics on identical global batches.

The goal is to faithfully replicate the EXACT semantics of both code paths and see
under what data distributions the reported loss values differ. We reuse the actual
``_per_component_sums`` and ``_finalize`` from ``param_decomp.metrics.importance_minimality``
so any bug in those is preserved by construction.

What we emulate:

* 1-pool live loss (what gets logged at train_log time):
  Each of W DDP ranks computes ``_finalize(local_sums, local_n_examples, beta, world_size=W)``
  on its slice of the global batch. The logged value is the average across the W ranks
  (matches ``avg_metrics_across_ranks`` divisor).

* 3-pool live loss (what gets logged from CI pool):
  Each of N_ci ranks computes local ``per_component_sums``; the CI pool SUM-reduces
  ``per_component_sums`` across the subgroup and multiplies ``n_examples`` by N_ci;
  then calls ``_finalize(global_sums, global_n_examples, beta, world_size=1)``.

Both paths see the SAME global batch — we just slice it differently.
"""

from __future__ import annotations

import torch

from param_decomp.metrics.importance_minimality import _finalize, _per_component_sums

# Config from p-93bf5f42 / our 4L equivalence test.
BATCH_GLOBAL = 64
SEQ_LEN = 512
W_DP = 8  # 1-pool DDP world size (the baseline at dp=8)
N_CI = 4  # 3-pool CI pool size
PNORM = 2.0  # initial p (anneals to 0.4 over training; at step 0 this is full p=2.0)
BETA = 0.5
EPS = 1e-12

# Per-site C from the 4L decomposition_targets.
SITE_C_PER_LAYER = {
    "attn.q_proj": 512,
    "attn.k_proj": 512,
    "attn.v_proj": 1024,
    "attn.o_proj": 1024,
    "mlp.c_fc": 3072,
    "mlp.down_proj": 3584,
}

# Per-site L0 (mean active components per position) from p-93bf5f42 eval/l0/0.0_*.
# Captured at run step ~210000.
OBSERVED_L0 = {
    "h.0.attn.q_proj": 6.60,
    "h.0.attn.k_proj": 6.40,
    "h.0.attn.v_proj": 6.45,
    "h.0.attn.o_proj": 11.79,
    "h.0.mlp.c_fc": 44.58,
    "h.0.mlp.down_proj": 43.25,
    "h.1.attn.q_proj": 4.17,
    "h.1.attn.k_proj": 4.99,
    "h.1.attn.v_proj": 12.11,
    "h.1.attn.o_proj": 22.66,
    "h.1.mlp.c_fc": 31.53,
    "h.1.mlp.down_proj": 27.45,
    "h.2.attn.q_proj": 13.18,
    "h.2.attn.k_proj": 12.58,
    "h.2.attn.v_proj": 19.00,
    "h.2.attn.o_proj": 33.17,
    "h.2.mlp.c_fc": 35.48,
    "h.2.mlp.down_proj": 33.27,
    "h.3.attn.q_proj": 7.38,
    "h.3.attn.k_proj": 7.21,
    "h.3.attn.v_proj": 18.27,
    "h.3.attn.o_proj": 27.83,
    "h.3.mlp.c_fc": 53.30,
    "h.3.mlp.down_proj": 65.45,
}

SITE_C: dict[str, int] = {
    site: SITE_C_PER_LAYER[site.split(".", 2)[2]] for site in OBSERVED_L0
}


def gen_global_ci_uniform_binary(
    batch: int = BATCH_GLOBAL,
    seq: int = SEQ_LEN,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Each component independently Bernoulli(L0/C) per position.

    This is the "expected per-position activation is L0" regime with iid components.
    Variance across batch positions is bounded; cross-rank variance is small.
    """
    g = torch.Generator().manual_seed(seed)
    out: dict[str, torch.Tensor] = {}
    for site, C in SITE_C.items():
        p_active = OBSERVED_L0[site] / C
        out[site] = torch.bernoulli(
            torch.full((batch, seq, C), p_active), generator=g
        ).float()
    return out


def gen_global_ci_position_concentrated(
    batch: int = BATCH_GLOBAL,
    seq: int = SEQ_LEN,
    seed: int = 0,
    cluster_frac: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Activations concentrated on a small fraction of positions.

    Of every batch's positions, only ``cluster_frac`` activate a given component;
    when they do, multiple components activate together. This creates much higher
    cross-rank variance — a small DP rank may see no activations of some components
    while another rank sees many.
    """
    g = torch.Generator().manual_seed(seed)
    out: dict[str, torch.Tensor] = {}
    for site, C in SITE_C.items():
        L0 = OBSERVED_L0[site]
        # Expected positions per component activating it: cluster_frac * batch*seq.
        # To keep mean L0 = (L0/C), boost the per-active-position rate accordingly.
        # P(active at position | position is "in cluster") * cluster_frac = L0/C
        # → P_in_cluster = (L0/C) / cluster_frac (clamp to <= 1).
        p_in_cluster = min(L0 / C / cluster_frac, 1.0)
        # Mask: which positions are "in cluster" for each component independently.
        cluster_mask = torch.bernoulli(
            torch.full((batch, seq, C), cluster_frac), generator=g
        )
        # Given in-cluster, draw Bernoulli with p_in_cluster.
        active = torch.bernoulli(
            torch.full((batch, seq, C), p_in_cluster), generator=g
        )
        out[site] = (cluster_mask * active).float()
    return out


def gen_global_ci_batchwise_concentrated(
    batch: int = BATCH_GLOBAL,
    seq: int = SEQ_LEN,
    seed: int = 0,
    active_batch_frac: float = 0.25,
) -> dict[str, torch.Tensor]:
    """Activations concentrated on a fraction of BATCH ELEMENTS (not positions).

    A subset of the batch dim "uses" each component, the rest don't. Within a
    batch element, all positions activate it. This makes 1-pool's slicing along
    the batch dim hit the worst case: some ranks see all activations, others see
    none.
    """
    g = torch.Generator().manual_seed(seed)
    out: dict[str, torch.Tensor] = {}
    for site, C in SITE_C.items():
        L0 = OBSERVED_L0[site]
        # P(active per position | batch element uses it) = L0 / C / active_batch_frac.
        p_when_active = min(L0 / C / active_batch_frac, 1.0)
        # Which batch elements use which components:
        batch_use = torch.bernoulli(
            torch.full((batch, 1, C), active_batch_frac), generator=g
        )
        per_pos = torch.bernoulli(
            torch.full((batch, seq, C), p_when_active), generator=g
        )
        out[site] = (batch_use * per_pos).float()
    return out


def slice_dim0(ci: dict[str, torch.Tensor], sl: slice) -> dict[str, torch.Tensor]:
    return {k: v[sl] for k, v in ci.items()}


def compute_1pool_logged(
    ci_global: dict[str, torch.Tensor],
    *,
    world_size: int = W_DP,
    pnorm: float = PNORM,
    beta: float = BETA,
    eps: float = EPS,
) -> tuple[float, list[float]]:
    """Replicate 1-pool live loss reporting at log step.

    Each of ``world_size`` ranks computes ``_finalize(local, world_size=W)`` on
    its slice. The logged value is the AVERAGE across ranks (since
    ``avg_metrics_across_ranks`` divides the SUM by world_size).
    """
    assert BATCH_GLOBAL % world_size == 0
    chunk = BATCH_GLOBAL // world_size
    per_rank: list[float] = []
    for r in range(world_size):
        ci_local = slice_dim0(ci_global, slice(r * chunk, (r + 1) * chunk))
        per_component_sums, n_examples = _per_component_sums(ci_local, pnorm, eps)
        local_loss = _finalize(per_component_sums, n_examples, beta, world_size).item()
        per_rank.append(local_loss)
    return sum(per_rank) / world_size, per_rank


def compute_3pool_logged(
    ci_global: dict[str, torch.Tensor],
    *,
    n_ci: int = N_CI,
    pnorm: float = PNORM,
    beta: float = BETA,
    eps: float = EPS,
) -> float:
    """Replicate 3-pool live loss reporting from CI pool.

    Each of ``n_ci`` ranks computes local ``per_component_sums`` on its slice;
    we SUM across the CI pool and multiply ``n_examples`` by ``n_ci``; then
    ``_finalize`` with ``world_size=1`` because sums are now global.
    """
    assert BATCH_GLOBAL % n_ci == 0
    chunk = BATCH_GLOBAL // n_ci
    summed: dict[str, torch.Tensor] = {}
    n_examples_total = 0
    for r in range(n_ci):
        ci_local = slice_dim0(ci_global, slice(r * chunk, (r + 1) * chunk))
        per_component_sums, n_examples = _per_component_sums(ci_local, pnorm, eps)
        for k, v in per_component_sums.items():
            summed[k] = summed.get(k, torch.zeros_like(v)) + v
        n_examples_total += n_examples
    # 3-pool multiplies n_examples by n_ci_pool (instead of summing the per-rank n).
    # Verify these match here so the emulation is faithful even though the values
    # come out identical when chunks are equal-sized.
    expected_n = (BATCH_GLOBAL // n_ci) * SEQ_LEN * n_ci
    assert n_examples_total == expected_n, (n_examples_total, expected_n)
    return _finalize(summed, n_examples_total, beta, world_size=1).item()


def per_site_breakdown(
    ci_global: dict[str, torch.Tensor],
    *,
    world_size: int = W_DP,
    n_ci: int = N_CI,
    pnorm: float = PNORM,
    beta: float = BETA,
    eps: float = EPS,
) -> list[tuple[str, float, float, float, float]]:
    """For each site, isolate its contribution under both paths.

    Returns a list of ``(site, l0_observed, 1pool_logged, 3pool_logged, ratio)``.
    Computed by running the full pipeline on a one-site dict (single site at a
    time). This isolates the per-site impact without changing the formula.
    """
    rows: list[tuple[str, float, float, float, float]] = []
    for site in ci_global:
        single = {site: ci_global[site]}
        one_pool, _ = compute_1pool_logged(
            single, world_size=world_size, pnorm=pnorm, beta=beta, eps=eps
        )
        three_pool = compute_3pool_logged(
            single, n_ci=n_ci, pnorm=pnorm, beta=beta, eps=eps
        )
        ratio = one_pool / three_pool if three_pool > 0 else float("nan")
        rows.append((site, OBSERVED_L0[site], one_pool, three_pool, ratio))
    return rows


def run_scenario(name: str, ci_global: dict[str, torch.Tensor]) -> None:
    print(f"\n{'=' * 78}\nScenario: {name}\n{'=' * 78}")
    one_pool_avg, per_rank = compute_1pool_logged(ci_global)
    three_pool = compute_3pool_logged(ci_global)
    ratio = one_pool_avg / three_pool if three_pool > 0 else float("nan")
    print(f"1-pool (avg over W={W_DP} ranks): {one_pool_avg:.6g}")
    print(f"3-pool (N_ci={N_CI}, world_size=1): {three_pool:.6g}")
    print(f"Ratio (1pool / 3pool): {ratio:.4f}")
    print(f"Per-rank 1-pool: {[round(x, 4) for x in per_rank]}")

    print("\nPer-site breakdown:")
    print(f"  {'site':<26} {'L0':>7} {'1pool':>14} {'3pool':>14} {'ratio':>8}")
    rows = per_site_breakdown(ci_global)
    for site, l0, op, tp, r in rows:
        print(f"  {site:<26} {l0:>7.2f} {op:>14.4g} {tp:>14.4g} {r:>8.4f}")


def gen_global_ci_one_batch_element_each(
    batch: int = BATCH_GLOBAL,
    seq: int = SEQ_LEN,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Each component is active on exactly ONE batch element (random). Worst-case
    for cross-rank variance: each component's local_sum is concentrated entirely
    on one rank.
    """
    g = torch.Generator().manual_seed(seed)
    out: dict[str, torch.Tensor] = {}
    for site, C in SITE_C.items():
        L0 = OBSERVED_L0[site]
        # Each component picks one batch element. When active (= on that element),
        # activates with prob L0/C * batch / seq … actually simpler: when on its
        # chosen batch element, every position activates with prob L0/C * batch.
        p_when_on = min(L0 / C * batch, 1.0)
        chosen = torch.randint(0, batch, (C,), generator=g)
        out_t = torch.zeros((batch, seq, C))
        for c in range(C):
            out_t[chosen[c], :, c] = torch.bernoulli(
                torch.full((seq,), p_when_on), generator=g
            )
        out[site] = out_t
    return out


def gen_global_ci_heavy_tail_continuous(
    batch: int = BATCH_GLOBAL,
    seq: int = SEQ_LEN,
    seed: int = 0,
    high_frac: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Continuous CI values with a heavy upper tail.

    Most values are near 0 (background, small Uniform); a small fraction are
    near 1 (active, Uniform(0.5, 1.0)). Squaring amplifies the high values.
    """
    g = torch.Generator().manual_seed(seed)
    out: dict[str, torch.Tensor] = {}
    for site, C in SITE_C.items():
        L0 = OBSERVED_L0[site]
        # Choose high_frac < L0/C: that fraction has high values.
        p_high = min(L0 / C, high_frac)
        high_mask = torch.bernoulli(torch.full((batch, seq, C), p_high), generator=g)
        high_vals = 0.5 + 0.5 * torch.rand((batch, seq, C), generator=g)  # ~ U(0.5, 1.0)
        low_vals = 0.05 * torch.rand((batch, seq, C), generator=g)        # ~ U(0, 0.05)
        out[site] = (high_mask * high_vals + (1 - high_mask) * low_vals).float()
    return out


def main() -> None:
    print(f"Config: BATCH_GLOBAL={BATCH_GLOBAL}, SEQ_LEN={SEQ_LEN}, "
          f"W_DP={W_DP}, N_CI={N_CI}, p={PNORM}, β={BETA}")

    run_scenario("uniform binary (iid Bernoulli per position)",
                 gen_global_ci_uniform_binary())
    run_scenario("position-concentrated (10% of positions activate components)",
                 gen_global_ci_position_concentrated(cluster_frac=0.1))
    run_scenario("batchwise-concentrated (25% of batch elements active per comp)",
                 gen_global_ci_batchwise_concentrated(active_batch_frac=0.25))
    run_scenario("batchwise-concentrated, more extreme (12.5% active)",
                 gen_global_ci_batchwise_concentrated(active_batch_frac=0.125))
    run_scenario("worst-case: each component active on exactly 1 batch element",
                 gen_global_ci_one_batch_element_each())
    run_scenario("heavy-tail continuous (most near 0, 5% near 1)",
                 gen_global_ci_heavy_tail_continuous(high_frac=0.05))
    run_scenario("heavy-tail continuous (most near 0, 1% near 1)",
                 gen_global_ci_heavy_tail_continuous(high_frac=0.01))


if __name__ == "__main__":
    main()
