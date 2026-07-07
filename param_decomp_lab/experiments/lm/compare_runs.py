"""Cross-run decomposition similarity: weight-space + CI-behavior component matching.

Quantifies how similar two finished runs' decompositions of the SAME frozen target are:
per site, components are matched Hungarian-style on (a) the Frobenius cosine of their
rank-1 matrices (which factorizes as `cos(V_a,V_b) * cos(U_a,U_b)` and is invariant to
the joint `(-V, -U)` gauge flip) and (b) the Pearson correlation of their CI values over
identical token batches. Reports matched-similarity distributions restricted to alive
components, per-position active-set Jaccard under the weight matching, and the agreement
between the two matchings — all against two calibration baselines: each run's
self-similarity across its last two checkpoints (near-upper bound) and pairing-broken
nulls (a column-shuffled run B is VACUOUS — the Hungarian optimum is invariant to column
permutations — so the weight null shuffles B's U rows against its V columns, and the CI
null rolls B's CI along the batch axis).
"""

import gc
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fire
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array, Float, Int
from scipy.optimize import linear_sum_assignment

from param_decomp.built_run import DataConfig
from param_decomp.checkpoint import make_checkpoint_manager
from param_decomp.ci_fn import CIFn, lower_leaky_hard_sigmoid
from param_decomp.data import BatchSchedule, ShardServer, scan_shards
from param_decomp.jit_util import filter_jit
from param_decomp.lm import DecomposedModel
from param_decomp.sharding import hsdp_mesh
from param_decomp.train import COMPUTE_DT, cast_floating
from param_decomp_lab.experiments.lm.config import load_run_dir_config
from param_decomp_lab.experiments.lm.load_run import open_jax_run
from param_decomp_lab.experiments.lm.replay_eval import _scalar_summary, _write_results
from param_decomp_lab.experiments.lm.run import _global_token_batch

_HIGHEST = jax.lax.Precision.HIGHEST


@dataclass(frozen=True)
class SlimRun:
    """The two trained pieces a comparison needs, slimmed from a restored TrainState
    (dropping Adam m/v and the persistent sources before the next restore)."""

    run_id: str
    step: int
    vu: dict[str, tuple[np.ndarray, np.ndarray]]
    ci_fn: CIFn


def _load_slim(run_dir: Path, step: int | None) -> tuple[SlimRun, DecomposedModel]:
    loaded = open_jax_run(run_dir, step)
    ci_fn = loaded._state.ci_fn
    assert isinstance(ci_fn, CIFn), "compare_runs is the transformer-CI-fn (LM) path only"
    vu = {
        site: (
            np.asarray(loaded._state.components.site(site)[0], np.float32),
            np.asarray(loaded._state.components.site(site)[1], np.float32),
        )
        for site in loaded.site_names
    }
    slim = SlimRun(run_id=loaded.run_id, step=loaded.step, vu=vu, ci_fn=ci_fn)
    lm = loaded.lm
    del loaded
    gc.collect()
    return slim, lm


def _checkpoint_steps(run_dir: Path, keep_last: int) -> tuple[int, ...]:
    return tuple(sorted(make_checkpoint_manager(run_dir / "ckpts", keep_last).all_steps()))


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PairIncrement:
    """One batch's sufficient statistics for the cross-run CI comparison, per site.
    `fire_*` / `joint` carry a leading thresholds axis; `sum_ab_null` pairs run A with a
    batch-rolled run B (the pairing-broken CI null)."""

    sum_a: dict[str, Float[Array, " C"]]
    sum_a2: dict[str, Float[Array, " C"]]
    sum_b: dict[str, Float[Array, " C"]]
    sum_b2: dict[str, Float[Array, " C"]]
    sum_ab: dict[str, Float[Array, "C C"]]
    sum_ab_null: dict[str, Float[Array, "C C"]]
    fire_a: dict[str, Float[Array, "K C"]]
    fire_b: dict[str, Float[Array, "K C"]]
    joint: dict[str, Float[Array, "K C C"]]
    n_positions: Array


PairStep = Callable[[DecomposedModel, CIFn, CIFn, Int[Array, "B T"]], PairIncrement]


def _make_ci_pair_step(
    lm: DecomposedModel,
    thresholds: tuple[float, ...],
    compiler_options: dict[str, bool | int | str] | None,
) -> PairStep:
    site_names = lm.site_names

    def ci_pair_step(
        model: DecomposedModel,
        ci_fn_a: CIFn,
        ci_fn_b: CIFn,
        token_ids: Int[Array, "B T"],
    ) -> PairIncrement:
        taps = {
            k: x.astype(COMPUTE_DT)
            for k, x in model.read_activations(token_ids, ci_fn_a.input_names).items()
        }

        def lower_of(ci_fn: CIFn) -> dict[str, Array]:
            # bf16 CI fn -> fp32 logits -> fp32 squash (the slow_eval route): the trained
            # bf16 readout, but a `> 0` firing predicate that isn't bf16-rounded.
            logits = cast_floating(ci_fn, COMPUTE_DT)(taps, remat=False).logits
            return {s: lower_leaky_hard_sigmoid(logits[s].astype(jnp.float32)) for s in site_names}

        lower_a, lower_b = lower_of(ci_fn_a), lower_of(ci_fn_b)

        sum_a, sum_a2, sum_b, sum_b2 = {}, {}, {}, {}
        sum_ab, sum_ab_null, fire_a, fire_b, joint = {}, {}, {}, {}, {}
        for site in site_names:
            c = lower_a[site].shape[-1]
            a = lower_a[site].reshape(-1, c)
            b = lower_b[site].reshape(-1, c)
            b_null = jnp.roll(lower_b[site], 1, axis=0).reshape(-1, c)
            sum_a[site] = a.sum(0)
            sum_a2[site] = jnp.square(a).sum(0)
            sum_b[site] = b.sum(0)
            sum_b2[site] = jnp.square(b).sum(0)
            sum_ab[site] = jnp.matmul(a.T, b, precision=_HIGHEST)
            sum_ab_null[site] = jnp.matmul(a.T, b_null, precision=_HIGHEST)
            fired_a = tuple((a > t).astype(jnp.float32) for t in thresholds)
            fired_b = tuple((b > t).astype(jnp.float32) for t in thresholds)
            fire_a[site] = jnp.stack([f.sum(0) for f in fired_a])
            fire_b[site] = jnp.stack([f.sum(0) for f in fired_b])
            joint[site] = jnp.stack(
                [
                    jnp.matmul(fa.T, fb, precision=_HIGHEST)
                    for fa, fb in zip(fired_a, fired_b, strict=True)
                ]
            )
        n_positions = jnp.asarray(math.prod(token_ids.shape), jnp.int32)
        return PairIncrement(
            sum_a=sum_a,
            sum_a2=sum_a2,
            sum_b=sum_b,
            sum_b2=sum_b2,
            sum_ab=sum_ab,
            sum_ab_null=sum_ab_null,
            fire_a=fire_a,
            fire_b=fire_b,
            joint=joint,
            n_positions=n_positions,
        )

    return filter_jit(ci_pair_step, compiler_options=compiler_options)


@dataclass
class HostPairStats:
    """Float64 accumulation of `PairIncrement`s (per-batch fp32 increments are exact
    enough; a cross-batch fp32 running sum is not)."""

    sums: dict[str, dict[str, np.ndarray]]
    n_positions: int

    @staticmethod
    def accumulate(
        step: PairStep,
        lm: DecomposedModel,
        ci_fn_a: CIFn,
        ci_fn_b: CIFn,
        batches: tuple[Array, ...],
        label: str,
    ) -> "HostPairStats":
        sums: dict[str, dict[str, np.ndarray]] = {}
        n_positions = 0
        for batch_idx, batch in enumerate(batches):
            increment = step(lm, ci_fn_a, ci_fn_b, batch)
            for field in (
                "sum_a",
                "sum_a2",
                "sum_b",
                "sum_b2",
                "sum_ab",
                "sum_ab_null",
                "fire_a",
                "fire_b",
                "joint",
            ):
                for site, value in getattr(increment, field).items():
                    host = np.asarray(value, np.float64)
                    sums.setdefault(field, {})
                    if site in sums[field]:
                        sums[field][site] += host
                    else:
                        sums[field][site] = host
            n_positions += int(increment.n_positions)
            print(f"[{label}] CI stats batch {batch_idx + 1}/{len(batches)}", flush=True)
        return HostPairStats(sums=sums, n_positions=n_positions)


def _pearson_from_stats(
    sum_a: Float[np.ndarray, " Ca"],
    sum_a2: Float[np.ndarray, " Ca"],
    sum_b: Float[np.ndarray, " Cb"],
    sum_b2: Float[np.ndarray, " Cb"],
    sum_ab: Float[np.ndarray, "Ca Cb"],
    n: int,
) -> Float[np.ndarray, "Ca Cb"]:
    """Pearson r per component pair; r := 0 where either side has zero variance (dead or
    constant components — deliberately excluded from alive stats anyway)."""
    cov = sum_ab / n - np.outer(sum_a, sum_b) / n**2
    var_a = np.maximum(sum_a2 / n - (sum_a / n) ** 2, 0.0)
    var_b = np.maximum(sum_b2 / n - (sum_b / n) ** 2, 0.0)
    denom = np.sqrt(np.outer(var_a, var_b))
    return np.where(denom > 0, cov / np.where(denom > 0, denom, 1.0), 0.0)


def _weight_similarity(
    vu_a: tuple[np.ndarray, np.ndarray],
    vu_b: tuple[np.ndarray, np.ndarray],
    u_perm_b: np.ndarray | None = None,
) -> Float[np.ndarray, "Ca Cb"]:
    """`S = (V̂_aᵀV̂_b) ⊙ (Û_aÛ_bᵀ)` — the Frobenius cosine of the rank-1 component
    matrices, exact via the factorization and invariant to the joint `(-V, -U)` flip.
    `u_perm_b` pairs B's V columns with permuted U rows: the pairing-broken null."""

    def normalized(vu: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        v = np.asarray(vu[0], np.float64)
        u = np.asarray(vu[1], np.float64)
        v_norm = np.linalg.norm(v, axis=0)
        u_norm = np.linalg.norm(u, axis=1)
        assert v_norm.min() > 0 and u_norm.min() > 0
        return v / v_norm, u / u_norm[:, None]

    v_a, u_a = normalized(vu_a)
    v_b, u_b = normalized(vu_b)
    cos_u = u_a @ (u_b if u_perm_b is None else u_b[u_perm_b]).T
    return (v_a.T @ v_b) * cos_u


def _hungarian_max(score: Float[np.ndarray, "Ca Cb"]) -> tuple[np.ndarray, np.ndarray]:
    """`(perm, matched)` maximizing `score[i, perm[i]].sum()`."""
    rows, cols = linear_sum_assignment(-score)
    assert (rows == np.arange(score.shape[0])).all()
    return cols, score[rows, cols]


def _jaccard_under_perm(
    fire_a: Float[np.ndarray, " C"],
    fire_b: Float[np.ndarray, " C"],
    joint: Float[np.ndarray, "C C"],
    perm: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-component active-set Jaccard of `(a_c, b_perm[c])` plus the analytic
    independence baseline `J = d_a d_b / (d_a + d_b - d_a d_b)`."""
    c_range = np.arange(len(perm))
    intersection = joint[c_range, perm]
    union = fire_a + fire_b[perm] - intersection
    jaccard = np.where(union > 0, intersection / np.where(union > 0, union, 1.0), 0.0)
    d_a, d_b = fire_a / n, fire_b[perm] / n
    indep_union = d_a + d_b - d_a * d_b
    independence = np.where(
        indep_union > 0, d_a * d_b / np.where(indep_union > 0, indep_union, 1.0), 0.0
    )
    return jaccard, independence


def _summary_or_none(values: np.ndarray) -> dict[str, float] | None:
    return _scalar_summary(values.tolist()) if values.size else None


def _site_report(
    site: str,
    s_weight: np.ndarray,
    s_weight_null: np.ndarray,
    stats: HostPairStats,
    threshold_idx_for_jaccard: int,
    alive_density_threshold: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    n = stats.n_positions
    sums = stats.sums
    pearson = _pearson_from_stats(
        sums["sum_a"][site],
        sums["sum_a2"][site],
        sums["sum_b"][site],
        sums["sum_b2"][site],
        sums["sum_ab"][site],
        n,
    )
    pearson_null = _pearson_from_stats(
        sums["sum_a"][site],
        sums["sum_a2"][site],
        sums["sum_b"][site],
        sums["sum_b2"][site],
        sums["sum_ab_null"][site],
        n,
    )
    perm_w, matched_cos = _hungarian_max(s_weight)
    perm_w_null, matched_cos_null = _hungarian_max(s_weight_null)
    perm_ci, matched_r = _hungarian_max(pearson)
    perm_ci_null, matched_r_null = _hungarian_max(pearson_null)

    density_a = sums["fire_a"][site][0] / n
    density_b = sums["fire_b"][site][0] / n
    alive_a = density_a > alive_density_threshold
    alive_b = density_b > alive_density_threshold
    both_alive_w = alive_a & alive_b[perm_w]
    both_alive_ci = alive_a & alive_b[perm_ci]

    r_of_weight_pairs = pearson[np.arange(len(perm_w)), perm_w]
    jaccard, independence = _jaccard_under_perm(
        sums["fire_a"][site][threshold_idx_for_jaccard],
        sums["fire_b"][site][threshold_idx_for_jaccard],
        sums["joint"][site][threshold_idx_for_jaccard],
        perm_w,
        n,
    )

    inverse_alive_b_unmatched = int(alive_b.sum() - (alive_b[perm_w] & alive_a).sum())
    report = {
        "n_alive_a": int(alive_a.sum()),
        "n_alive_b": int(alive_b.sum()),
        "weight": {
            "matched_cos_alive": _summary_or_none(matched_cos[both_alive_w]),
            "matched_cos_all": _scalar_summary(matched_cos.tolist()),
            "frac_gt_0p8_alive": float((matched_cos[both_alive_w] > 0.8).mean())
            if both_alive_w.any()
            else None,
            "frac_gt_0p9_alive": float((matched_cos[both_alive_w] > 0.9).mean())
            if both_alive_w.any()
            else None,
            "frac_negative_alive": float((matched_cos[both_alive_w] < 0).mean())
            if both_alive_w.any()
            else None,
            "null_matched_cos_alive": _summary_or_none(
                matched_cos_null[alive_a & alive_b[perm_w_null]]
            ),
        },
        "ci": {
            "matched_r_alive": _summary_or_none(matched_r[both_alive_ci]),
            "null_matched_r_alive": _summary_or_none(
                matched_r_null[alive_a & alive_b[perm_ci_null]]
            ),
            "r_of_weight_pairs_alive": _summary_or_none(r_of_weight_pairs[both_alive_w]),
        },
        "jaccard": {
            "matched_w_alive": _summary_or_none(jaccard[both_alive_w]),
            "independence_alive": _summary_or_none(independence[both_alive_w]),
        },
        "agreement_alive_frac": float((perm_w[alive_a] == perm_ci[alive_a]).mean())
        if alive_a.any()
        else None,
        "unmatched_alive_a": int((alive_a & ~alive_b[perm_w]).sum()),
        "unmatched_alive_b": inverse_alive_b_unmatched,
    }
    arrays = {
        f"{site}:perm_w": perm_w,
        f"{site}:perm_ci": perm_ci,
        f"{site}:matched_cos": matched_cos,
        f"{site}:matched_cos_null": matched_cos_null,
        f"{site}:matched_r": matched_r,
        f"{site}:matched_r_null": matched_r_null,
        f"{site}:r_of_weight_pairs": r_of_weight_pairs,
        f"{site}:jaccard_w": jaccard,
        f"{site}:jaccard_independence": independence,
        f"{site}:density_a": density_a,
        f"{site}:density_b": density_b,
    }
    return report, arrays


def _compare_pair(
    pair_step: PairStep,
    lm: DecomposedModel,
    run_a: SlimRun,
    run_b: SlimRun,
    batches: tuple[Array, ...],
    threshold_idx_for_jaccard: int,
    alive_density_threshold: float,
    null_rng: np.random.Generator,
    label: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    stats = HostPairStats.accumulate(pair_step, lm, run_a.ci_fn, run_b.ci_fn, batches, label)

    per_site: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    pooled_cos, pooled_r, pooled_agree, pooled_alive = [], [], [], []
    for site in lm.site_names:
        s_weight = _weight_similarity(run_a.vu[site], run_b.vu[site])
        u_perm = null_rng.permutation(run_b.vu[site][1].shape[0])
        s_weight_null = _weight_similarity(run_a.vu[site], run_b.vu[site], u_perm_b=u_perm)
        report, site_arrays = _site_report(
            site,
            s_weight,
            s_weight_null,
            stats,
            threshold_idx_for_jaccard,
            alive_density_threshold,
        )
        per_site[site] = report
        arrays.update(site_arrays)
        alive_a = site_arrays[f"{site}:density_a"] > alive_density_threshold
        alive_b = site_arrays[f"{site}:density_b"] > alive_density_threshold
        perm_w = site_arrays[f"{site}:perm_w"]
        both = alive_a & alive_b[perm_w]
        pooled_cos.append(site_arrays[f"{site}:matched_cos"][both])
        pooled_r.append(site_arrays[f"{site}:r_of_weight_pairs"][both])
        pooled_agree.append(
            (perm_w[alive_a] == site_arrays[f"{site}:perm_ci"][alive_a]).astype(np.float64)
        )
        pooled_alive.append((int(alive_a.sum()), int(alive_b.sum())))

    all_cos = np.concatenate(pooled_cos)
    all_r = np.concatenate(pooled_r)
    all_agree = np.concatenate(pooled_agree)
    overall = {
        "weight_matched_cos_alive": _summary_or_none(all_cos),
        "ci_r_of_weight_pairs_alive": _summary_or_none(all_r),
        "frac_gt_0p9_alive": float((all_cos > 0.9).mean()) if all_cos.size else None,
        "agreement_alive_frac": float(all_agree.mean()) if all_agree.size else None,
        "n_alive_a": int(sum(a for a, _ in pooled_alive)),
        "n_alive_b": int(sum(b for _, b in pooled_alive)),
    }
    result = {
        "run_ids": [run_a.run_id, run_b.run_id],
        "steps": [run_a.step, run_b.step],
        "overall": overall,
        "per_site": per_site,
    }
    return result, arrays


_FIGURES = (
    ("matched_cos", "matched weight cosine", "matched_cos_hist.png"),
    ("matched_r", "matched CI Pearson r", "matched_r_hist.png"),
)


def _render_figures(out_dir: Path, arrays_by_pair: dict[str, dict[str, np.ndarray]]) -> None:
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    reference = next(iter(arrays_by_pair.values()))
    sites = sorted({key.split(":")[0] for key in reference})
    n_cols = 6
    n_rows = math.ceil(len(sites) / n_cols)
    bins = np.linspace(-1.0, 1.0, 41).tolist()

    for array_key, x_label, filename in _FIGURES:
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(3.2 * n_cols, 2.6 * n_rows), squeeze=False
        )
        for site_idx, site in enumerate(sites):
            ax = axes[site_idx // n_cols][site_idx % n_cols]
            for pair_name, arrays in arrays_by_pair.items():
                ax.hist(
                    arrays[f"{site}:{array_key}"],
                    bins=bins,
                    histtype="step",
                    density=True,
                    label=pair_name,
                )
                ax.hist(
                    arrays[f"{site}:{array_key}_null"],
                    bins=bins,
                    histtype="step",
                    density=True,
                    linestyle="dashed",
                    label=f"{pair_name} null",
                )
            ax.set_title(site, fontsize=8)
            ax.set_yscale("log")
        for extra_idx in range(len(sites), n_rows * n_cols):
            axes[extra_idx // n_cols][extra_idx % n_cols].axis("off")
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower right", fontsize=8)
        fig.supxlabel(x_label)
        fig.tight_layout()
        fig.savefig(figures_dir / filename, dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(1, len(arrays_by_pair), figsize=(5.5 * len(arrays_by_pair), 4.5))
    for ax, (pair_name, arrays) in zip(np.atleast_1d(axes), arrays_by_pair.items(), strict=True):
        cos = np.concatenate([arrays[f"{site}:matched_cos"] for site in sites])
        r = np.concatenate([arrays[f"{site}:r_of_weight_pairs"] for site in sites])
        density = np.concatenate([arrays[f"{site}:density_a"] for site in sites])
        alive = density > 0
        scatter = ax.scatter(
            cos[alive],
            r[alive],
            c=np.log10(density[alive]),
            s=2,
            alpha=0.3,
            cmap="viridis",
        )
        ax.set_xlabel("matched weight cosine")
        ax.set_ylabel("CI Pearson r of the same pair")
        ax.set_title(pair_name, fontsize=9)
        fig.colorbar(scatter, ax=ax, label="log10 density (run A)")
    fig.tight_layout()
    fig.savefig(figures_dir / "ci_corr_vs_weight_cos.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    jaccard_bins = np.linspace(0.0, 1.0, 41).tolist()
    for pair_name, arrays in arrays_by_pair.items():
        jaccard = np.concatenate([arrays[f"{site}:jaccard_w"] for site in sites])
        indep = np.concatenate([arrays[f"{site}:jaccard_independence"] for site in sites])
        ax.hist(jaccard, bins=jaccard_bins, histtype="step", density=True, label=pair_name)
        ax.hist(
            indep,
            bins=jaccard_bins,
            histtype="step",
            density=True,
            linestyle="dashed",
            label=f"{pair_name} independence",
        )
    ax.set_xlabel("active-set Jaccard (weight matching)")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "jaccard_hist.png", dpi=150)
    plt.close(fig)


def compare(
    *,
    run_dir_a: str,
    run_dir_b: str,
    out_dir: str,
    step_a: int | None = None,
    step_b: int | None = None,
    n_batches: int = 32,
    batch_size: int = 128,
    thresholds: str | tuple[float, ...] = "0.0,0.1",
    alive_density_threshold: float = 1e-5,
    self_baselines: bool = True,
    null_seed: int = 0,
) -> None:
    """Compare two runs' decompositions on identical batches; write JSON + npz + figures."""
    path_a, path_b, out = Path(run_dir_a), Path(run_dir_b), Path(out_dir)
    threshold_values = (
        tuple(float(value) for value in thresholds.split(","))
        if isinstance(thresholds, str)
        else tuple(thresholds)
    )
    assert len(threshold_values) >= 2, "need thresholds (density, jaccard)"
    jax.config.update("jax_compilation_cache_dir", str(out / "xla_cache"))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    assert jax.process_count() == 1, "compare_runs is single-process"

    cfg_a = load_run_dir_config(path_a)
    cfg_b = load_run_dir_config(path_b)
    assert isinstance(cfg_a.data, DataConfig) and isinstance(cfg_b.data, DataConfig)
    assert (cfg_a.data.dir, cfg_a.data.seq_len) == (cfg_b.data.dir, cfg_b.data.seq_len)
    assert tuple((s.name, s.C) for s in cfg_a.target.sites) == tuple(
        (s.name, s.C) for s in cfg_b.target.sites
    )
    assert type(cfg_a.target) is type(cfg_b.target)
    assert cfg_a.ci_fn == cfg_b.ci_fn, "CI fn archs must match for tap sharing"
    assert cfg_a.pd.seed == cfg_b.pd.seed, "shared eval stream is seeded off pd.seed"
    assert cfg_a.cadence.keep_last_n_checkpoints is not None
    assert cfg_b.cadence.keep_last_n_checkpoints is not None

    mesh = hsdp_mesh()
    assert mesh.devices.size <= batch_size and batch_size % mesh.devices.size == 0
    schedule = BatchSchedule(scan_shards(cfg_a.data.dir), batch_size, cfg_a.pd.seed + 1)
    server = ShardServer(schedule, cfg_a.data.seq_len, process_index=0, process_count=1)
    batches = tuple(
        _global_token_batch(server.local_batch(batch_idx), mesh, batch_size)
        for batch_idx in range(n_batches)
    )

    run_a, lm = _load_slim(path_a, step_a)
    run_b, _ = _load_slim(path_b, step_b)
    assert set(run_a.vu) == set(run_b.vu) == set(lm.site_names)

    pair_step = _make_ci_pair_step(lm, threshold_values, cfg_a.runtime.compiler_options)
    null_rng = np.random.default_rng(null_seed)

    pairs: list[tuple[str, SlimRun, SlimRun]] = [("a_vs_b", run_a, run_b)]
    if self_baselines:
        for tag, path, run, keep_last in (
            ("a_selfsim", path_a, run_a, cfg_a.cadence.keep_last_n_checkpoints),
            ("b_selfsim", path_b, run_b, cfg_b.cadence.keep_last_n_checkpoints),
        ):
            steps = _checkpoint_steps(path, keep_last)
            earlier = tuple(s for s in steps if s < run.step)
            assert earlier, f"no earlier checkpoint than {run.step} in {steps} for {tag}"
            prev_run, _ = _load_slim(path, earlier[-1])
            pairs.append((tag, prev_run, run))

    results: dict[str, Any] = {
        "config": {
            "run_dir_a": str(path_a),
            "run_dir_b": str(path_b),
            "step_a": run_a.step,
            "step_b": run_b.step,
            "n_batches": n_batches,
            "batch_size": batch_size,
            "thresholds": threshold_values,
            "alive_density_threshold": alive_density_threshold,
            "null_seed": null_seed,
        },
        "pairs": {},
    }
    out.mkdir(parents=True, exist_ok=True)
    arrays_by_pair: dict[str, dict[str, np.ndarray]] = {}
    for pair_name, first, second in pairs:
        report, arrays = _compare_pair(
            pair_step,
            lm,
            first,
            second,
            batches,
            threshold_idx_for_jaccard=1,
            alive_density_threshold=alive_density_threshold,
            null_rng=null_rng,
            label=pair_name,
        )
        results["pairs"][pair_name] = report
        arrays_by_pair[pair_name] = arrays
        np.savez_compressed(out / f"matching_{pair_name}.npz", allow_pickle=False, **arrays)
        _write_results(out / "compare.json", results)
        print(f"[{pair_name}] done: {json.dumps(report['overall'], indent=2)}", flush=True)

    _render_figures(out, arrays_by_pair)


def cli() -> None:
    fire.Fire(compare)


if __name__ == "__main__":
    cli()
