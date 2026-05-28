"""Compare single-pool vs heterogeneous-3-pool loss curves for the CI-fn-grad diagnostic.

Reads ``metrics.jsonl`` from each run, aligns the four loss terms per step, and
reports per-seed and seed-averaged final-value ratios (3pool / single-pool).

The headline question: do imp and/or pgd diverge (3-pool under-optimizing them)
while stoch tracks? That pattern confirms the suspected CI-fn-grad scaling bug;
roughly-tracking curves refute it.

Single-pool logs ``train/loss/<ClassName>``; 3-pool logs ``train/loss/<short>``.
We map them onto a common term name.
"""

import json
import sys

from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR

DECOMP = PARAM_DECOMP_OUT_DIR / "decompositions"

# term -> (single-pool key, 3-pool key)
TERM_KEYS = {
    "faith": ("train/loss/FaithfulnessLoss", "train/loss/faith"),
    "stoch": ("train/loss/StochasticReconLayerwiseLoss", "train/loss/stoch"),
    "imp": ("train/loss/ImportanceMinimalityLoss", "train/loss/imp"),
    "ppgd": ("train/loss/PersistentPGDReconLoss", "train/loss/ppgd"),
}


def load_curves(run_id: str, key_idx: int) -> dict[str, dict[int, float]]:
    """Return ``{term: {step: value}}`` for one run. ``key_idx`` selects the
    single-pool (0) or 3-pool (1) key from ``TERM_KEYS``."""
    path = DECOMP / run_id / "metrics.jsonl"
    assert path.exists(), f"missing {path}"
    rows = [json.loads(line) for line in path.open()]
    out: dict[str, dict[int, float]] = {t: {} for t in TERM_KEYS}
    for r in rows:
        step = r.get("step")
        if step is None:
            continue
        for term, keys in TERM_KEYS.items():
            k = keys[key_idx]
            if k in r:
                out[term][step] = r[k]
    return out


def aligned_final(curve: dict[int, float]) -> tuple[int, float]:
    last_step = max(curve)
    return last_step, curve[last_step]


def main(seeds: list[int]) -> None:
    per_seed_ratios: dict[str, list[float]] = {t: [] for t in TERM_KEYS}

    for seed in seeds:
        sp_id = f"eq-1p-s{seed}"
        tp_id = f"eq-3p-s{seed}"
        sp = load_curves(sp_id, 0)
        tp = load_curves(tp_id, 1)
        print(f"\n===== seed {seed} =====")
        print(f"{'term':>6} | {'step':>5} | {'single':>14} | {'3pool':>14} | {'3p/1p':>10}")
        print("-" * 64)
        for term in TERM_KEYS:
            if not sp[term] or not tp[term]:
                print(f"{term:>6} | (missing data: sp={len(sp[term])} tp={len(tp[term])})")
                continue
            s_step, s_val = aligned_final(sp[term])
            t_step, t_val = aligned_final(tp[term])
            ratio = t_val / s_val if s_val != 0 else float("nan")
            per_seed_ratios[term].append(ratio)
            print(
                f"{term:>6} | {min(s_step, t_step):>5} | "
                f"{s_val:>14.6g} | {t_val:>14.6g} | {ratio:>10.4f}"
            )

    print("\n===== seed-averaged final-value ratio (3pool / single-pool) =====")
    print(f"{'term':>6} | {'mean ratio':>12} | {'n seeds':>8}")
    print("-" * 34)
    for term, ratios in per_seed_ratios.items():
        if not ratios:
            print(f"{term:>6} | {'(none)':>12} |")
            continue
        mean = sum(ratios) / len(ratios)
        print(f"{term:>6} | {mean:>12.4f} | {len(ratios):>8}")

    # Step-averaged ratio (less sensitive to single-step noise than the final step)
    # plus an early-vs-late drift check: if a grad-scaling bug accumulates, the
    # 3p/1p ratio should drift systematically across training, not just jitter.
    print("\n===== step-averaged 3p/1p ratio + early/late drift (per seed) =====")
    print(
        f"{'seed':>4} | {'term':>6} | {'mean(all)':>10} | {'early':>8} | {'late':>8} | {'drift':>8}"
    )
    print("-" * 60)
    for seed in seeds:
        sp = load_curves(f"eq-1p-s{seed}", 0)
        tp = load_curves(f"eq-3p-s{seed}", 1)
        for term in TERM_KEYS:
            steps = sorted(set(sp[term]) & set(tp[term]))
            if not steps:
                continue
            ratios = [tp[term][st] / sp[term][st] for st in steps if sp[term][st] != 0]
            half = len(ratios) // 2
            early = sum(ratios[:half]) / max(half, 1)
            late = sum(ratios[half:]) / max(len(ratios) - half, 1)
            mean_all = sum(ratios) / len(ratios)
            print(
                f"{seed:>4} | {term:>6} | {mean_all:>10.4f} | "
                f"{early:>8.4f} | {late:>8.4f} | {late - early:>+8.4f}"
            )

    # Per-step trajectory for the two diagnostic terms (imp, pgd) and stoch.
    print("\n===== per-step trajectories (seed 0) =====")
    sp0 = load_curves(f"eq-1p-s{seeds[0]}", 0)
    tp0 = load_curves(f"eq-3p-s{seeds[0]}", 1)
    for term in ("stoch", "imp", "ppgd"):
        print(f"\n-- {term} --   step:  1p  ->  3p   (ratio)")
        steps = sorted(set(sp0[term]) & set(tp0[term]))
        for st in steps:
            s, t = sp0[term][st], tp0[term][st]
            r = t / s if s != 0 else float("nan")
            print(f"   step {st:>4}: {s:>12.6g} -> {t:>12.6g}  ({r:.4f})")


if __name__ == "__main__":
    seeds = [int(x) for x in sys.argv[1:]] or [0, 1, 2]
    main(seeds)
