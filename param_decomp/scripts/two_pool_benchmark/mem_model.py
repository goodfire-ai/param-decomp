"""MVP memory cost model for 2-pool training on Qwen3-1.7B.

Goal: predict pool A peak memory given a config (B, S, bpg, ddp, ci_d, ci_n,
use_fused_kl). Decomposes into independently-scaling terms, then fits two free
coefficients + a slack constant against our existing measurements.

Run via:
    python -m param_decomp.scripts.two_pool_benchmark.mem_model

Prints: per-data-point predicted vs actual, residuals, fit quality.

This is a smoke test of the modeling approach — not production-quality.
Hardcoded data points; refit if you add measurements.
"""

from dataclasses import dataclass

import numpy as np

# ──────────────── Qwen3-1.7B constants ────────────────
N_LAYERS = 28
D_MODEL = 2048
D_MLP = 6144
VOCAB = 151936
SITES_PER_LAYER = 7
C = 32  # components per site
BF16 = 2
OPT_BYTES_PER_PARAM = 16  # fp32 params + grad + Adam m + Adam v

TARGET_MODEL_GB = 1.7e9 * BF16 / 1e9  # ~3.4 GB
KL_CHUNK_SIZE = 1024  # default in fused_linear_kl_div


# ──────────────── Analytic per-site formulas ────────────────
def vu_params_for_site(in_dim: int, out_dim: int) -> int:
    """V[in, C] + U[out, C] for r=1."""
    return (in_dim + out_dim) * C


def site_dims(kind: str) -> tuple[int, int]:
    """(in_dim, out_dim) by site kind. Qwen3 GQA: kv_heads=8 (half q_heads)."""
    if kind in ("q_proj", "o_proj"):
        return (D_MODEL, D_MODEL)
    if kind in ("k_proj", "v_proj"):
        return (D_MODEL, D_MODEL // 2)  # GQA
    if kind in ("gate_proj", "up_proj"):
        return (D_MODEL, D_MLP)
    if kind == "down_proj":
        return (D_MLP, D_MODEL)
    raise ValueError(kind)


SITE_KINDS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def vu_params_per_block() -> int:
    return sum(vu_params_for_site(*site_dims(k)) for k in SITE_KINDS)


def ci_fn_params(d_ci: int, n_ci: int, in_dim: int) -> int:
    """LayerwiseTransformerCiFn per site. Body: n_ci × (4·d² + 2·d·(4d)) = n_ci × 12·d²."""
    body = n_ci * 12 * d_ci**2
    input_proj = in_dim * d_ci
    output_head = d_ci * C
    return body + input_proj + output_head


def avg_ci_params_per_site(d_ci: int, n_ci: int) -> int:
    """CI fn params averaged over the 7 site kinds (input_dim varies for down_proj)."""
    return int(np.mean([ci_fn_params(d_ci, n_ci, site_dims(k)[0]) for k in SITE_KINDS]))


# ──────────────── Memory contributors ────────────────
def predict_pool_a_components(
    batch: int,
    seq: int,
    bpg: int,
    ddp: int,
    ci_d: int,
    ci_n: int,
    use_fused_kl: bool,
) -> dict[str, float]:
    """Return per-contributor predicted GB. Sum = total predicted peak."""
    b_local = batch // ddp
    sites_per_rank = bpg * SITES_PER_LAYER

    target = TARGET_MODEL_GB
    vu = sites_per_rank * vu_params_per_block() / SITES_PER_LAYER * OPT_BYTES_PER_PARAM / 1e9
    ci_params = sites_per_rank * avg_ci_params_per_site(ci_d, ci_n) * OPT_BYTES_PER_PARAM / 1e9
    cached_hidden = b_local * seq * D_MODEL * BF16 / 1e9
    per_iter = 2 * b_local * seq * D_MODEL * BF16 / 1e9

    if use_fused_kl:
        # Fused kernel work area: chunk × vocab × bytes × ~3 (logits + softmax + grad)
        kl_work = KL_CHUNK_SIZE * VOCAB * BF16 * 3 / 1e9
    else:
        # Unfused: materialize [b_local, seq, vocab] tensors. Empirically ~3 of them live.
        kl_work = b_local * seq * VOCAB * BF16 * 3 / 1e9

    return {
        "target_model": target,
        "vu_state": vu,
        "ci_params": ci_params,
        "cached_hidden": cached_hidden,
        "per_iter_predgrad": per_iter,
        "kl_work": kl_work,
    }


def x_features(batch: int, seq: int, ddp: int, ci_d: int, ci_n: int) -> tuple[float, float]:
    """Return (X_act, X_ci) — the two regression features. Coefficients c_act, c_ci
    will be fit on these. Both in GB units so coefficients are dimensionless."""
    b_local = batch // ddp
    x_act = b_local * seq * D_MODEL * N_LAYERS * BF16 / 1e9  # target backbone acts
    x_ci = b_local * seq * ci_d * ci_n * BF16 / 1e9  # CI fn forward acts
    return x_act, x_ci


# ──────────────── Measured data points ────────────────


@dataclass(frozen=True)
class MemMeasurement:
    """One measured (config, pool_a_peak_gb) data point."""

    batch: int
    seq: int
    bpg: int
    ddp: int
    ci_d: int
    ci_n: int
    use_fused_kl: bool
    observed_pool_a_gb: float


MEASUREMENTS: list[MemMeasurement] = [
    # 1block-Nddp sweep (old path, unfused KL)
    MemMeasurement(8, 1024, 1, 1, 128, 2, False, 59.8),
    MemMeasurement(16, 1024, 1, 1, 128, 2, False, 108.9),
    MemMeasurement(8, 1024, 1, 2, 128, 2, False, 37.1),
    MemMeasurement(16, 1024, 1, 2, 128, 2, False, 63.6),
    MemMeasurement(32, 1024, 1, 2, 128, 2, False, 116.5),
    MemMeasurement(8, 1024, 1, 4, 128, 2, False, 25.8),
    MemMeasurement(16, 1024, 1, 4, 128, 2, False, 40.9),
    MemMeasurement(32, 1024, 1, 4, 128, 2, False, 71.2),
    MemMeasurement(64, 1024, 1, 4, 128, 2, False, 131.8),
    MemMeasurement(8, 1024, 1, 8, 128, 2, False, 20.1),
    MemMeasurement(16, 1024, 1, 8, 128, 2, False, 29.6),
    # Fused-KL pool A only / both pools (pool A peak identical either way)
    MemMeasurement(8, 1024, 1, 4, 128, 2, True, 20.9),  # 32166
    MemMeasurement(64, 1024, 1, 4, 128, 2, True, 62.3),  # 32169
    MemMeasurement(64, 2048, 1, 4, 128, 2, True, 112.5),  # 32170
    # Ablation: 1block-8ddp b=64 s=2048
    MemMeasurement(64, 2048, 1, 8, 128, 2, True, 73.4),  # 32175 fused on
    MemMeasurement(64, 2048, 1, 8, 128, 2, False, 162.3),  # 32176 fused off
]


# ──────────────── Fit + report ────────────────
def fit_and_report() -> None:
    x_rows: list[list[float]] = []  # [x_act, x_ci, 1] per row
    y_rows: list[float] = []  # residual (observed - constant) per row
    consts: list[float] = []
    x_feat_per_row: list[tuple[float, float]] = []

    for m in MEASUREMENTS:
        comps = predict_pool_a_components(
            m.batch,
            m.seq,
            m.bpg,
            m.ddp,
            m.ci_d,
            m.ci_n,
            m.use_fused_kl,
        )
        const = sum(comps.values())
        x_act, x_ci = x_features(m.batch, m.seq, m.ddp, m.ci_d, m.ci_n)
        consts.append(const)
        x_feat_per_row.append((x_act, x_ci))
        x_rows.append([x_act, x_ci, 1.0])
        y_rows.append(m.observed_pool_a_gb - const)

    x_arr = np.array(x_rows)
    y_arr = np.array(y_rows)
    # Solve y = X @ [c_act, c_ci, k_overhead]
    coeffs, *_ = np.linalg.lstsq(x_arr, y_arr, rcond=None)
    c_act, c_ci, k_overhead = coeffs

    print("=== fit ===")
    print(f"  c_act        = {c_act:.3f}  (target activation density)")
    print(f"  c_ci         = {c_ci:.3f}   (CI fn activation density)")
    print(f"  k_overhead   = {k_overhead:.2f} GB  (slack constant)")
    print()

    print(
        f"{'batch':>5} {'seq':>5} {'bpg':>4} {'ddp':>4} {'ci_d':>5} {'ci_n':>4} {'fused':>6} "
        f"{'observed':>9} {'predicted':>10} {'resid':>7} {'%err':>6}"
    )
    print("-" * 100)
    total_sq_err = 0.0
    for m, const, (x_act, x_ci) in zip(MEASUREMENTS, consts, x_feat_per_row, strict=True):
        batch, seq, bpg, ddp, ci_d, ci_n, fused, observed = (
            m.batch,
            m.seq,
            m.bpg,
            m.ddp,
            m.ci_d,
            m.ci_n,
            m.use_fused_kl,
            m.observed_pool_a_gb,
        )
        pred = const + c_act * x_act + c_ci * x_ci + k_overhead
        resid = observed - pred
        pct = 100 * resid / observed
        total_sq_err += resid**2
        print(
            f"{batch:>5} {seq:>5} {bpg:>4} {ddp:>4} {ci_d:>5} {ci_n:>4} {fused!s:>6} "
            f"{observed:>9.1f} {pred:>10.1f} {resid:>+7.1f} {pct:>+5.1f}%"
        )
    rms = np.sqrt(total_sq_err / len(MEASUREMENTS))
    print(f"\nRMS error: {rms:.2f} GB")


if __name__ == "__main__":
    fit_and_report()
