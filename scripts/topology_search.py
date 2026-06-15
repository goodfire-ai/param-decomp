"""Search 3-pool topologies for the throughput-optimal rank allocation.

A generic screen over rank allocations, given a per-pool compute calibration
(supplied as INPUT — never baked in). Model (per-sample compute):

  compute_ci   ≈ k_ci   * (B / n_ci)              # CI fn fwd/bwd, DP over batch
  compute_ppgd ≈ k_ppgd * (B / n_ppgd)            # PPGD inner loop, DP over batch
  compute_lw   ≈ k_lw_total * (B / n_lw)          # all sites split over n_lw ranks
  step         ≈ max(compute_ci, compute_lw, compute_ppgd) + overhead
  throughput = B / step   (samples / ms; comparable across B)

Calibrate from a real run: per-pool compute means from a torch.profiler trace
(`scripts/analyze_3pool_trace.py`) + the run's logged step_ms (step wall). Supply
the numbers as a JSON file (`--calibration`) or build a `Calibration` and call
`report()`. A worked, reproducible example is `scripts/repro_big512_topology_search.py`.

CAVEATS (why this is a SCREEN, not a verdict):
  * Per-rank compute is assumed LINEAR in local batch. In practice LW is markedly
    SUBLINEAR — a large fixed per-site cost (the serial per-site recon loop) means
    doubling bl_lw far less than doubles compute_lw, so the LW term over-credits
    adding LW ranks. Calibrate near your operating bl_lw; trust the screen only there.
  * The model is LW-SHAPE-BLIND: compute_lw depends only on n_lw, not on
    n_per_block / sites_per_block, so it cannot distinguish thin vs fat LW blocks
    (which differ in practice). Use a real sweep to pick LW shape.
  * `overhead` (non-overlapped cross-pool comm/idle/sync) is treated as additive &
    constant; it actually grows with rank count, so calibrate it at your scale.
  * Memory is flagged crudely: batch_local_ci above the calibrated max risks OOM
    without checkpointing.
"""

import argparse
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path


@dataclass(frozen=True)
class Calibration:
    n_sites: int
    k_ci: float  # ms / sample
    k_ppgd: float  # ms / sample
    k_lw_total: float  # ms (per-(site·sample) cost × n_sites)
    overhead: float  # ms (non-overlapped cross-pool cost)
    batch_local_ci_max: int  # OOM-risk threshold (the calibrated bl_ci)

    @classmethod
    def from_measurements(
        cls,
        *,
        n_sites: int,
        ci: tuple[float, int],
        ppgd: tuple[float, int],
        lw: tuple[float, int],
        lw_sites_per_block: int,
        step_wall_ms: float,
    ) -> "Calibration":
        """ci / ppgd / lw are each (compute_ms, batch_local) measured per-rank for that pool."""
        return cls(
            n_sites=n_sites,
            k_ci=ci[0] / ci[1],
            k_ppgd=ppgd[0] / ppgd[1],
            k_lw_total=(lw[0] / (lw_sites_per_block * lw[1])) * n_sites,
            overhead=step_wall_ms - max(ci[0], ppgd[0], lw[0]),
            batch_local_ci_max=ci[1],
        )

    @classmethod
    def from_json(cls, path: Path) -> "Calibration":
        d = json.loads(path.read_text())
        return cls.from_measurements(
            n_sites=d["n_sites"],
            ci=(d["ci_compute_ms"], d["ci_batch_local"]),
            ppgd=(d["ppgd_compute_ms"], d["ppgd_batch_local"]),
            lw=(d["lw_compute_ms"], d["lw_batch_local"]),
            lw_sites_per_block=d["lw_sites_per_block"],
            step_wall_ms=d["step_wall_ms"],
        )


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def _div_ok(a: int, b: int, *, relaxed: bool) -> bool:
    """current: a | b.  relaxed: a | b OR b | a."""
    return (b % a == 0) or (relaxed and a % b == 0)


@dataclass(frozen=True)
class Topo:
    n_ci: int
    n_ppgd: int
    n_blocks: int
    n_per_block: int
    batch: int

    @property
    def n_lw(self) -> int:
        return self.n_blocks * self.n_per_block

    def sites_per_block(self, n_sites: int) -> int:
        assert n_sites % self.n_blocks == 0
        return n_sites // self.n_blocks

    def compute_ci(self, cal: Calibration) -> float:
        return cal.k_ci * self.batch / self.n_ci

    def compute_ppgd(self, cal: Calibration) -> float:
        return cal.k_ppgd * self.batch / self.n_ppgd

    def compute_lw(self, cal: Calibration) -> float:
        return cal.k_lw_total * self.batch / self.n_lw

    def step_ms(self, cal: Calibration) -> float:
        return (
            max(self.compute_ci(cal), self.compute_ppgd(cal), self.compute_lw(cal)) + cal.overhead
        )

    def throughput(self, cal: Calibration) -> float:  # samples / ms
        return self.batch / self.step_ms(cal)

    def bottleneck(self, cal: Calibration) -> str:
        return max(
            (
                ("ci", self.compute_ci(cal)),
                ("ppgd", self.compute_ppgd(cal)),
                ("lw", self.compute_lw(cal)),
            ),
            key=lambda kv: kv[1],
        )[0]

    @property
    def batch_local_ci(self) -> int:
        return self.batch // self.n_ci

    def ci_oom_risk(self, cal: Calibration) -> bool:
        return self.batch_local_ci > cal.batch_local_ci_max

    def needs_relaxation(self) -> bool:
        cur = _div_ok(self.n_ci, self.n_per_block, relaxed=False) and _div_ok(
            self.n_ci, self.n_ppgd, relaxed=False
        )
        return not cur


def enumerate_topos(batches: list[int], *, budget: int, n_sites: int, relaxed: bool) -> list[Topo]:
    out = []
    for B in batches:
        bdivs = _divisors(B)  # n_ci, n_ppgd, n_per_block must each divide B
        for n_blocks in _divisors(n_sites):
            for n_ci, n_ppgd, n_per_block in product(bdivs, bdivs, bdivs):
                n_lw = n_blocks * n_per_block
                if n_ci + n_ppgd + n_lw != budget:
                    continue
                if not _div_ok(n_ci, n_per_block, relaxed=relaxed):
                    continue
                if not _div_ok(n_ci, n_ppgd, relaxed=relaxed):
                    continue
                out.append(Topo(n_ci, n_ppgd, n_blocks, n_per_block, B))
    return out


def fmt(t: Topo, cal: Calibration) -> str:
    flags = []
    if t.needs_relaxation():
        flags.append("RELAX")
    if t.ci_oom_risk(cal):
        flags.append(f"CI-OOM?(bl={t.batch_local_ci})")
    return (
        f"B={t.batch:<4} ci={t.n_ci:<2} ppgd={t.n_ppgd:<2} "
        f"lw={t.n_blocks}x{t.n_per_block}={t.n_lw:<3} | "
        f"step~{t.step_ms(cal):6.1f}ms thru={t.throughput(cal):6.4f} "
        f"(ci {t.compute_ci(cal):5.0f} | lw {t.compute_lw(cal):5.0f} | ppgd {t.compute_ppgd(cal):5.0f}, "
        f"bottleneck={t.bottleneck(cal):<4}) {' '.join(flags)}"
    )


def report(
    cal: Calibration,
    *,
    budget: int,
    baseline: Topo | None,
    batch_groups: list[tuple[list[int], str]],
) -> None:
    print(
        f"calibration: K_CI={cal.k_ci:.1f} K_PPGD={cal.k_ppgd:.1f} "
        f"K_LW_TOTAL={cal.k_lw_total:.0f} OVERHEAD={cal.overhead:.0f}ms "
        f"(n_sites={cal.n_sites}, budget={budget}; ms, per-sample where applicable)"
    )
    if baseline is not None:
        print(f"\nBASELINE: {fmt(baseline, cal)}")
        print(f"  baseline throughput = {baseline.throughput(cal):.4f} samples/ms")

    for batches, label in batch_groups:
        print("\n" + "=" * 100)
        print(f"SEARCH: {label}")
        print("=" * 100)
        cur = sorted(
            enumerate_topos(batches, budget=budget, n_sites=cal.n_sites, relaxed=False),
            key=lambda t: -t.throughput(cal),
        )
        rel = sorted(
            enumerate_topos(batches, budget=budget, n_sites=cal.n_sites, relaxed=True),
            key=lambda t: -t.throughput(cal),
        )
        rel_only = [t for t in rel if t.needs_relaxation()]

        print("\n-- best under CURRENT constraints --")
        for t in cur[:6]:
            print("  " + fmt(t, cal))
        print("\n-- best that REQUIRE the relaxation (relaxed-only) --")
        if rel_only:
            for t in sorted(rel_only, key=lambda t: -t.throughput(cal))[:6]:
                print("  " + fmt(t, cal))
        else:
            print("  (none)")

        best_cur = cur[0].throughput(cal) if cur else 0.0
        best_rel = rel[0].throughput(cal) if rel else 0.0
        verdict = (
            "RELAXATION HELPS"
            if best_rel > best_cur * 1.005
            else "relaxation does NOT beat current-constraint best"
        )
        if baseline is not None:
            base = baseline.throughput(cal)
            print(
                f"\n  best CURRENT: {best_cur:.4f} ({100 * (best_cur / base - 1):+.1f}% vs baseline)"
            )
            print(
                f"  best RELAXED: {best_rel:.4f} "
                f"({100 * (best_rel / base - 1):+.1f}% vs baseline)  →  {verdict}"
            )
        else:
            print(f"\n  best CURRENT: {best_cur:.4f}   best RELAXED: {best_rel:.4f}  →  {verdict}")


def _parse_topo(s: str) -> Topo:
    n_ci, n_ppgd, n_blocks, n_per_block, batch = (int(x) for x in s.split(","))
    return Topo(n_ci, n_ppgd, n_blocks, n_per_block, batch)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generic 3-pool topology throughput screen.")
    ap.add_argument(
        "--calibration",
        type=Path,
        required=True,
        help="JSON calibration (see Calibration.from_json)",
    )
    ap.add_argument("--budget", type=int, required=True, help="total ranks to search over")
    ap.add_argument("--batches", default="512", help="comma-separated batch sizes (e.g. 256,512)")
    ap.add_argument(
        "--baseline", default=None, help="n_ci,n_ppgd,n_blocks,n_per_block,batch to compare against"
    )
    args = ap.parse_args()
    cal = Calibration.from_json(args.calibration)
    batches = [int(b) for b in args.batches.split(",")]
    baseline = _parse_topo(args.baseline) if args.baseline else None
    report(cal, budget=args.budget, baseline=baseline, batch_groups=[(batches, args.batches)])


if __name__ == "__main__":
    main()
