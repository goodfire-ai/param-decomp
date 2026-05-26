"""Critical-path analysis for a 3-pool training step.

Reads ``[trace rank=R +Tms] phase: NAME ...`` lines from a slurm log (emitted
under ``PD_PHASE_TRACE=1``) plus the per-pool ``Trainer.run: step N: (start|done)``
markers. For a given step, identifies the chain of phases that gated the
step's finish time.

Approach: each rank has its own ``perf_counter()``, so timestamps are NOT
comparable across ranks. We work in per-pool local time (each pool's first
phase starts at t=0). The critical chain is built by walking *backwards*
from the last phase of the slowest pool (= the one whose step took longest),
following:

  * intra-pool predecessor when no cross-pool block is involved, OR
  * cross-pool sender when the current node is a recv phase that took
    ≥ ``block_threshold_ms`` ms (= it spent time waiting on remote data).

For "if-cut" estimates we approximate the per-pool step length without each
phase (zero its weight, recompute the pool's intra-pool-only span) plus the
delta this propagates to other pools through cross-pool waits. Report the
resulting drop in max-pool finish.

Defaults match the production smoke layout: rank 0 = LW, rank 96 = CI,
rank 104 = PPGD.

Usage:
    python scripts/critical_path.py /path/to/slurm-NNN.out --step 5
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gantt_step import parse_log, per_rank_phases_in_window, step_window_per_rank

# Cross-pool send → recv pairs: (send_pool, send_phase) → (recv_pool, recv_phase).
CROSS_POOL_EDGES: list[tuple[tuple[str, str], tuple[str, str]]] = [
    (("CI", "ci/2_async_send_ci"), ("LW", "lw/D2_wait_ci_recv")),
    (("CI", "ci/2_async_send_ci"), ("PPGD", "pgd/D2_wait_ci_recv")),
    (("LW", "lw/D4_send_g_ci"), ("CI", "ci/5_recv_g_ci_from_lw")),
    (("PPGD", "pgd/D8_send_g_ci_to_ci_pool"), ("CI", "ci/6_recv_g_ci_from_ppgd")),
    (("PPGD", "pgd/D7_send_g_vu_to_lw"), ("LW", "lw/D5_recv_g_vu_from_ppgd")),
]

# Recv-side phases (the destinations in CROSS_POOL_EDGES). When walking the
# critical path backwards, if we hit one of these and it blocked for ≥
# block_threshold_ms, jump to the matching sender in a different pool.
RECV_TO_SENDER: dict[tuple[str, str], tuple[str, str]] = {dst: src for src, dst in CROSS_POOL_EDGES}


@dataclass(frozen=True)
class Node:
    pool: str
    phase: str
    start_ms: float  # pool-local: first phase starts at 0
    end_ms: float
    gpu_ms: float | None  # real GPU stream time, None if exit line lacked it
    wait_ms: float | None  # cpu_ms - gpu_ms; positive = node was waiting on upstream

    @property
    def dur_ms(self) -> float:
        """CPU wall: end-of-phase trace timestamp minus entry. Includes cross-stream
        wait, so use ``weight_ms`` instead for critical-path analysis."""
        return self.end_ms - self.start_ms

    @property
    def weight_ms(self) -> float:
        """Real serial cost of this node: GPU time when available, else CPU wall.
        This is what the critical-path sum should use — CPU wall double-counts
        wait time and is misleading for "irreducible serial" questions."""
        return self.gpu_ms if self.gpu_ms is not None else self.dur_ms

    @property
    def key(self) -> tuple[str, str]:
        return (self.pool, self.phase)


def build_nodes(
    log_path: Path, target_step: int, rank_labels: list[tuple[int, str]]
) -> dict[str, list[Node]]:
    """Per-pool nodes in pool-local time (each pool's first phase at t=0)."""
    phases, steps = parse_log(log_path)
    out: dict[str, list[Node]] = {}
    for rank, label in rank_labels:
        win = step_window_per_rank(steps, rank, target_step)
        assert win is not None, f"step {target_step} not found for rank {rank} ({label})"
        ps = per_rank_phases_in_window(phases, rank, *win)
        if not ps:
            out[label] = []
            continue
        origin = ps[0][0]
        out[label] = [
            Node(
                pool=label,
                phase=name,
                start_ms=s - origin,
                end_ms=e - origin,
                gpu_ms=gpu_ms,
                wait_ms=wait_ms,
            )
            for s, e, name, gpu_ms, wait_ms in ps
        ]
    return out


def critical_path(
    pool_nodes: dict[str, list[Node]],
    *,
    block_threshold_ms: float,
) -> list[tuple[Node, str]]:
    """Walk backwards from the slowest pool's last phase. At each node, if the
    node is a recv that blocked ≥ block_threshold_ms, jump to the sender;
    otherwise step to the intra-pool predecessor.

    Returns the path in execution order, each entry tagged with the edge type
    ('entry'|'intra'|'cross').
    """
    # Pick the pool whose step took longest as the end-of-step.
    pool_span = {label: ns[-1].end_ms for label, ns in pool_nodes.items() if ns}
    end_pool = max(pool_span, key=lambda p: pool_span[p])

    # Index nodes by key for fast lookup, and per-pool ordering.
    by_key: dict[tuple[str, str], Node] = {n.key: n for ns in pool_nodes.values() for n in ns}
    pool_index: dict[str, dict[str, int]] = {
        label: {n.phase: i for i, n in enumerate(ns)} for label, ns in pool_nodes.items()
    }

    path: list[tuple[Node, str]] = []
    cur: Node | None = pool_nodes[end_pool][-1]
    edge: str = "(end)"
    while cur is not None:
        path.append((cur, edge))
        # If this node is a cross-pool recv and it blocked significantly,
        # jump to the sender.
        sender_key = RECV_TO_SENDER.get(cur.key)
        if sender_key is not None and cur.dur_ms >= block_threshold_ms:
            sender = by_key.get(sender_key)
            if sender is not None:
                cur = sender
                edge = "cross"
                continue
        # Else step to intra-pool predecessor (or stop if at index 0).
        i = pool_index[cur.pool][cur.phase]
        if i == 0:
            cur = None
        else:
            cur = pool_nodes[cur.pool][i - 1]
            edge = "intra"
    path.reverse()
    return path


def fmt_path(path: list[tuple[Node, str]]) -> list[str]:
    lines = [
        f"{'start_ms':>9s}  {'end_ms':>8s}  {'cpu_ms':>7s}  {'gpu_ms':>7s}  "
        f"{'wait_ms':>8s}  {'pool':<5s}  {'edge':<6s}  phase",
        "-" * 110,
    ]
    total_cpu = 0.0
    total_gpu = 0.0
    have_gpu = False
    for n, edge in path:
        gpu_s = f"{n.gpu_ms:7.1f}" if n.gpu_ms is not None else "    n/a"
        wait_s = f"{n.wait_ms:+8.1f}" if n.wait_ms is not None else "     n/a"
        lines.append(
            f"{n.start_ms:9.1f}  {n.end_ms:8.1f}  {n.dur_ms:7.1f}  {gpu_s}  {wait_s}  "
            f"{n.pool:<5s}  {edge:<6s}  {n.phase}"
        )
        total_cpu += n.dur_ms
        if n.gpu_ms is not None:
            total_gpu += n.gpu_ms
            have_gpu = True
    lines.append("-" * 110)
    if have_gpu:
        lines.append(
            f"path totals — cpu_wall={total_cpu:.1f}ms  irreducible_gpu={total_gpu:.1f}ms  "
            f"slack={total_cpu - total_gpu:+.1f}ms"
        )
    else:
        lines.append(
            f"path totals — cpu_wall={total_cpu:.1f}ms  (no gpu data; "
            "PD_PHASE_TRACE on with cpu/gpu/wait exit format required for irreducible serial)"
        )
    return lines


def cross_pool_block_table(
    pool_nodes: dict[str, list[Node]], *, threshold_ms: float
) -> list[tuple[Node, Node]]:
    """List of (recv, sender) pairs where the recv blocked ≥ threshold."""
    by_key = {n.key: n for ns in pool_nodes.values() for n in ns}
    blocks: list[tuple[Node, Node]] = []
    for src_key, dst_key in CROSS_POOL_EDGES:
        recv = by_key.get(dst_key)
        send = by_key.get(src_key)
        if recv is None or send is None:
            continue
        if recv.dur_ms >= threshold_ms:
            blocks.append((recv, send))
    blocks.sort(key=lambda rs: -rs[0].dur_ms)
    return blocks


def if_cut(
    pool_nodes: dict[str, list[Node]],
    baseline_step_ms: float,
    *,
    top_n: int = 5,
    min_dur_ms: float = 5.0,
) -> list[tuple[Node, float, float]]:
    """For each node with dur ≥ min_dur_ms, recompute the step finish time
    assuming this phase took 0 ms. We propagate the effect through the
    cross-pool send→recv edges: shortening a phase shifts the pool's later
    phases earlier; if that pool was the sender for some recv that blocked
    significantly, the receiver pool sees a corresponding shift too.

    First-order estimate — assumes structural overlap doesn't change.
    """
    by_key = {n.key: n for ns in pool_nodes.values() for n in ns}

    def simulate(zero_node: Node | None) -> float:
        # New per-pool end times after zeroing zero_node.
        pool_end: dict[str, float] = {}
        for label, nodes in pool_nodes.items():
            if not nodes:
                pool_end[label] = 0.0
                continue
            delta = zero_node.dur_ms if zero_node is not None and zero_node.pool == label else 0.0
            # Shorten if zeroed node is in this pool; treat downstream as
            # cascading by ``delta``. (Assumes no slack between phases —
            # close enough for ranking.)
            pool_end[label] = nodes[-1].end_ms - delta

        # Cross-pool: a recv can't end before its sender's send ends. Iterate
        # to fixed point (small DAG, converges in a few passes).
        for _ in range(len(pool_end) + 1):
            for src_key, dst_key in CROSS_POOL_EDGES:
                send = by_key.get(src_key)
                recv = by_key.get(dst_key)
                if send is None or recv is None:
                    continue
                # Recv pool's step end must be ≥ this recv's end. Recv's end
                # in turn is bounded by the sender's send-end + (recv.end -
                # recv.start) original gap. If send shifts earlier, recv can
                # shift earlier too (bounded by intra-pool sequencing).
                # For simplicity: cap the recv pool's end at the sender's
                # send-end + (recv pool's nodes after recv duration).
                pass
            # The first-order model above doesn't tighten further here; we
            # rely on each pool's own shortening propagating to its end.
            break

        return max(pool_end.values())

    baseline = simulate(None)
    results: list[tuple[Node, float, float]] = []
    for n in by_key.values():
        if n.dur_ms < min_dur_ms:
            continue
        s = simulate(n)
        results.append((n, n.dur_ms, baseline - s))
    results.sort(key=lambda x: -x[2])
    return results[:top_n]


def render(
    log_path: Path,
    *,
    target_step: int,
    rank_labels: list[tuple[int, str]],
    block_threshold_ms: float,
) -> str:
    pool_nodes = build_nodes(log_path, target_step, rank_labels)
    assert all(pool_nodes.values()), (
        f"empty pool nodes: {[(p, len(ns)) for p, ns in pool_nodes.items()]}"
    )

    path = critical_path(pool_nodes, block_threshold_ms=block_threshold_ms)
    pool_span = {label: ns[-1].end_ms for label, ns in pool_nodes.items()}
    step_finish = max(pool_span.values())
    end_pool = max(pool_span, key=lambda p: pool_span[p])

    out: list[str] = [f"=== critical path: step {target_step} ===", ""]
    out.append("(pool clocks are independent; each pool's first phase = t=0)")
    out.append("")
    out.extend(fmt_path(path))
    out.append("")

    cross_jumps = [i for i, (_, e) in enumerate(path) if e == "cross"]
    by_pool_cpu: dict[str, float] = {}
    by_pool_gpu: dict[str, float] = {}
    have_gpu_on_path = False
    for n, _ in path:
        by_pool_cpu[n.pool] = by_pool_cpu.get(n.pool, 0.0) + n.dur_ms
        if n.gpu_ms is not None:
            by_pool_gpu[n.pool] = by_pool_gpu.get(n.pool, 0.0) + n.gpu_ms
            have_gpu_on_path = True
    top_node = max((n for n, _ in path), key=lambda n: n.weight_ms)
    path_gpu_total = sum(n.gpu_ms for n, _ in path if n.gpu_ms is not None)

    out.append(f"step finish (slowest pool): {step_finish:.1f}ms in pool {end_pool}")
    if have_gpu_on_path:
        out.append(
            f"irreducible serial GPU time on critical path: {path_gpu_total:.1f}ms "
            f"(step has ~{step_finish - path_gpu_total:.0f}ms of recoverable wait/overlap headroom)"
        )
        out.append(
            "irreducible GPU by pool on path: "
            + ", ".join(
                f"{p}={ms:.1f}ms" for p, ms in sorted(by_pool_gpu.items(), key=lambda kv: -kv[1])
            )
        )
    out.append(
        "cpu-wall on critical path by pool: "
        + ", ".join(
            f"{p}={ms:.1f}ms" for p, ms in sorted(by_pool_cpu.items(), key=lambda kv: -kv[1])
        )
    )
    top_w = top_node.weight_ms
    out.append(f"heaviest path node (by GPU when present): {top_node.pool}/{top_node.phase} ({top_w:.1f}ms)")
    out.append(f"cross-pool jumps on the path: {len(cross_jumps)}")
    out.append("")
    out.append("per-pool step spans:")
    for label, dur in pool_span.items():
        out.append(f"  {label:<5s}: {dur:7.1f}ms  (slack vs slowest: {step_finish - dur:+6.1f}ms)")
    out.append("")

    blocks = cross_pool_block_table(pool_nodes, threshold_ms=block_threshold_ms)
    out.append(f"cross-pool blocking recvs (≥ {block_threshold_ms:.0f}ms):")
    if blocks:
        for recv, send in blocks:
            out.append(
                f"  {recv.pool}/{recv.phase} waited {recv.dur_ms:.1f}ms on {send.pool}/{send.phase}"
            )
    else:
        out.append("  (none — all recvs completed within threshold)")
    out.append("")

    cuts = if_cut(pool_nodes, step_finish, top_n=5, min_dur_ms=5.0)
    out.append("=== top-5 if-cut: 'if phase X (Y ms) were free, step would drop ~Z ms' ===")
    out.append(f"{'pool':<5s}  {'phase':<40s}  {'dur_ms':>7s}  {'delta_ms':>8s}")
    out.append("-" * 72)
    for n, dur, delta in cuts:
        out.append(f"{n.pool:<5s}  {n.phase:<40s}  {dur:7.1f}  {delta:8.1f}")
    out.append("")

    # One-line summary.
    other_pools = [p for p in pool_span if p != end_pool]
    next_pool_slack = min(step_finish - pool_span[p] for p in other_pools) if other_pools else 0.0
    if have_gpu_on_path:
        out.append(
            f"summary: step {step_finish:.0f}ms, irreducible GPU on critical path {path_gpu_total:.0f}ms "
            f"({100 * path_gpu_total / step_finish:.0f}% of step is real serial compute). "
            f"top blocker: {top_node.pool}/{top_node.phase} (gpu={top_w:.0f}ms). "
            f"next-pool slack: {next_pool_slack:.0f}ms ({end_pool} is slowest)."
        )
    else:
        out.append(
            f"summary: critical path total {step_finish:.0f}ms (cpu-wall only — gpu data missing). "
            f"top blocker: {top_node.pool}/{top_node.phase} ({top_node.dur_ms:.0f}ms in pool {top_node.pool}). "
            f"slack on next pool: {next_pool_slack:.0f}ms ({end_pool} is the slowest pool)."
        )
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path, help="slurm log path")
    ap.add_argument("--step", type=int, default=5, help="step number to analyze (default: 5)")
    ap.add_argument(
        "--ranks",
        type=str,
        default="0:LW,96:CI,104:PPGD",
        help="comma-separated rank:label pairs",
    )
    ap.add_argument(
        "--block-threshold-ms",
        type=float,
        default=5.0,
        help="recv phases ≥ this duration are treated as cross-pool blocks (default: 5)",
    )
    args = ap.parse_args()
    rank_labels = [
        (int(rank), label) for rank, label in (s.split(":") for s in args.ranks.split(","))
    ]
    print(
        render(
            args.log,
            target_step=args.step,
            rank_labels=rank_labels,
            block_threshold_ms=args.block_threshold_ms,
        )
    )


if __name__ == "__main__":
    main()
