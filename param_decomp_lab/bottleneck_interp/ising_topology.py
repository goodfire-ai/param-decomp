"""Ising-model topology of the bottleneck code's support structure.

Adapted from sp-viz's `manifold_topology.py` (Bhalla, Fel et al., "Do Sparse
Autoencoders Capture Concept Manifolds?", arXiv 2604.28119, Appendix E). The analysis
core (FISTA L1 pseudo-likelihood IsingFit, EBIC model selection, Louvain communities,
PCA-gap validation, capture/shatter/dilution classification) is ported verbatim; the
sp-viz model/patch-loading and atom-grid plots are dropped. Input is the harvested code
matrix from `harvest_codes.py`.

Pipeline:
  1. binarise codes to spins. Our double-sided JumpReLU code is *signed sparse*, so the
     natural binary variable is ACTIVITY: s_i = 2*1[z_i != 0] - 1 (default). Bhalla's
     one-sided `1[z_i > 0]` is available via `--binarize positive` for parity; sign is
     separate information for a signed code.
  2. drop dims with firing rate outside [freq_floor, freq_ceiling]
  3. fit p(s) by symmetric L1 PLM (FISTA, GPU); select lambda by EBIC (gamma=0.5)
  4. Louvain communities on |J|
  5. validate each community by a sharp PCA spectral gap in the continuous codes
  6. label capture (J_intra>0) / shatter (J_intra<0) / dilution; unvalidated otherwise
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

from param_decomp_lab.bottleneck_interp.geometry import load_codes, twonn_intrinsic_dim


def _smooth_threshold(x: Tensor, tau: float) -> Tensor:
    """Soft-thresholding (L1 prox): sign(x)·max(|x|-tau, 0)."""
    return torch.sign(x) * torch.clamp(x.abs() - tau, min=0.0)


def fit_plm_l1_fista(
    S: Tensor,
    target: Tensor,
    lam: float,
    n_iter: int = 300,
    lr: float = 0.05,
    J_init: Tensor | None = None,
    h_init: Tensor | None = None,
) -> tuple[Tensor, Tensor, float]:
    """Symmetric L1-regularised pseudo-likelihood by FISTA.

    Minimises nll(J,h) + lam·Σ_{i<j}|J_ij| over symmetric J (zero diagonal) and h.
    Symmetry is re-enforced after each step; h is unregularised. Returns (J, h, nll).
    """
    device = S.device
    _, m = S.shape
    eye = torch.eye(m, device=device, dtype=torch.bool)

    J = (J_init.clone() if J_init is not None else torch.zeros(m, m, device=device)).requires_grad_(
        True
    )
    h = (h_init.clone() if h_init is not None else torch.zeros(m, device=device)).requires_grad_(
        True
    )
    y_j = J.detach().clone()
    y_h = h.detach().clone()
    J_prev = J.detach().clone()
    h_prev = h.detach().clone()
    t_prev = 1.0

    def nll_per_sample(J_in: Tensor, h_in: Tensor) -> Tensor:
        field_local = S @ J_in + h_in
        bce_elem = torch.nn.functional.binary_cross_entropy_with_logits(
            2 * field_local, target, reduction="none"
        )
        return bce_elem.sum(dim=1).mean()

    for _ in range(n_iter):
        with torch.no_grad():
            J.copy_(y_j)
            h.copy_(y_h)
            J[eye] = 0
        loss = nll_per_sample(J, h)
        loss.backward()
        assert J.grad is not None and h.grad is not None
        with torch.no_grad():
            J_step = J - lr * J.grad
            h_step = h - lr * h.grad
            J_step = 0.5 * (J_step + J_step.T)
            J_step[eye] = 0
            J_new = _smooth_threshold(J_step, lr * lam)
            J_new[eye] = 0
            h_new = h_step

            t_new = (1.0 + (1.0 + 4.0 * t_prev * t_prev) ** 0.5) / 2.0
            momentum = (t_prev - 1.0) / t_new
            y_j = J_new + momentum * (J_new - J_prev)
            y_h = h_new + momentum * (h_new - h_prev)
            y_j[eye] = 0
            y_j = 0.5 * (y_j + y_j.T)

            J_prev = J_new
            h_prev = h_new
            t_prev = t_new
            J.grad.zero_()
            h.grad.zero_()

    with torch.no_grad():
        J_prev[eye] = 0
        final_nll = nll_per_sample(J_prev, h_prev).item()
    return J_prev, h_prev, final_nll


def ebic(
    J: Tensor, h: Tensor, S: Tensor, target: Tensor, gamma: float = 0.5
) -> tuple[float, int, float]:
    """Extended BIC: -2·logL + |E|·logN + 4γ·|E|·logM. Returns (ebic, n_edges, logL)."""
    n, m = S.shape
    with torch.no_grad():
        field = S @ J + h
        bce_sum = torch.nn.functional.binary_cross_entropy_with_logits(
            2 * field, target, reduction="sum"
        )
        log_lik = -bce_sum.item()
        triu = torch.triu(J.abs(), diagonal=1)
        n_edges = int((triu > 1e-8).sum().item())
    ebic_val = -2 * log_lik + n_edges * np.log(n) + 4.0 * gamma * n_edges * np.log(m)
    return ebic_val, n_edges, log_lik


def fit_ising_ebic(
    spins: Tensor,
    lambdas: list[float],
    n_iter: int = 300,
    lr: float = 0.05,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Sweep lambda (warm-started largest→smallest), score by EBIC, return best J,h + sweep."""
    S = spins.float().to(device)
    target = ((S + 1) / 2).to(device)
    n, m = S.shape
    print(f"  lambda sweep over {len(lambdas)} values, N={n}, M={m}")
    J_warm: Tensor | None = None
    h_warm: Tensor | None = None
    results: list[dict[str, Any]] = []
    for lam in sorted(lambdas, reverse=True):
        t0 = time.time()
        J, h, final_nll = fit_plm_l1_fista(
            S, target, lam, n_iter=n_iter, lr=lr, J_init=J_warm, h_init=h_warm
        )
        ebic_val, n_edges, log_lik = ebic(J, h, S, target)
        print(
            f"    lambda={lam:8.5f}  nll={final_nll:.5f}  |E|={n_edges:>6}  "
            f"EBIC={ebic_val:>12.0f}  ({time.time() - t0:.1f}s)"
        )
        results.append(
            {
                "lambda": float(lam),
                "edges": n_edges,
                "log_lik": log_lik,
                "ebic": ebic_val,
                "nll": final_nll,
                "J": J.cpu().numpy(),
                "h": h.cpu().numpy(),
            }
        )
        J_warm, h_warm = J, h
    best = min(results, key=lambda r: r["ebic"])
    print(f"  best lambda by EBIC: {best['lambda']:.5f} (|E|={best['edges']})")
    return best["J"], best["h"], results


def cluster_atoms_louvain(J: np.ndarray, resolution: float = 1.0, seed: int = 42) -> np.ndarray:
    """Louvain community detection on |J|; returns (M,) integer label array."""
    m = J.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(m))
    iu, ju = np.triu_indices(m, k=1)
    w = np.abs(J[iu, ju])
    mask = w > 1e-8
    G.add_weighted_edges_from(
        (int(i), int(j), float(wij)) for i, j, wij in zip(iu[mask], ju[mask], w[mask], strict=False)
    )
    communities = nx.community.louvain_communities(
        G, weight="weight", resolution=resolution, seed=seed
    )
    labels = np.full(m, -1, dtype=int)
    for c_idx, community in enumerate(communities):
        for atom in community:
            labels[atom] = c_idx
    assert (labels >= 0).all(), "Louvain left some atoms unassigned"
    return labels


def validate_cluster_pca(
    z: np.ndarray, atom_indices: np.ndarray, gap_threshold: float = 1.5
) -> dict[str, Any]:
    """SVD on the (N,k) continuous-code submatrix for a cluster; report spectral gap."""
    k = len(atom_indices)
    sub = z[:, atom_indices].astype(np.float32, copy=False)
    sub = sub - sub.mean(axis=0, keepdims=True)
    if k == 1:
        return {
            "eff_rank": 1.0,
            "gap_idx": 1,
            "gap_ratio": float("inf"),
            "has_sharp_gap": False,
            "singular_values": [float(np.linalg.norm(sub))],
        }
    sv = np.linalg.svd(sub, full_matrices=False)[1].astype(np.float64)
    p = sv**2
    p = p / max(p.sum(), 1e-30)
    eff_rank = float(np.exp(-(p * np.log(p + 1e-30)).sum()))
    ratios = sv[:-1] / np.clip(sv[1:], 1e-30, None)
    gap_idx = int(np.argmax(ratios)) + 1
    gap_ratio = float(ratios[gap_idx - 1])
    has_sharp_gap = (gap_ratio > gap_threshold) and (gap_idx < k)
    return {
        "eff_rank": eff_rank,
        "gap_idx": gap_idx,
        "gap_ratio": gap_ratio,
        "has_sharp_gap": has_sharp_gap,
        "singular_values": sv.tolist(),
    }


def community_nonlinear_id(
    z_kept: np.ndarray, atom_indices: np.ndarray, sample: int
) -> float | None:
    """TwoNN intrinsic dim of the continuous codes for a community, on positions where
    the community is active. Returns None if too few active positions.

    Bhalla's PCA-gap validation tests for *linear* low-rank subspaces; our codes lie on
    a curved manifold (global TwoNN << PCA dim), so a community can be a genuine low-dim
    manifold while failing the linear test. This is the curvature-aware counterpart:
    `eff_rank / twonn_id >> 1` marks a curved community.
    """
    sub = torch.from_numpy(z_kept[:, atom_indices].astype(np.float32))
    active = (sub != 0).any(dim=1)
    sub_a = sub[active]
    if sub_a.shape[0] < 2000:
        return None
    torch.manual_seed(0)
    return float(twonn_intrinsic_dim(sub_a, min(sample, sub_a.shape[0]))["twonn_id"])


def classify_clusters(
    J: np.ndarray,
    labels: np.ndarray,
    z_kept: np.ndarray,
    min_size: int = 3,
    capture_frac_pos: float = 0.85,
    shatter_frac_pos: float = 0.15,
    pca_gap_threshold: float = 1.5,
    nonlinear_id_sample: int = 15000,
) -> list[dict[str, Any]]:
    """Validate each community by PCA gap, then label capture/shatter/dilution/unvalidated.

    Also reports the curvature-aware nonlinear intrinsic dim per community (see
    `community_nonlinear_id`): PCA-`unvalidated` communities are often curved low-dim
    manifolds, flagged here by `nonlinear_low_dim`.
    """
    out: list[dict[str, Any]] = []
    for c in sorted(set(labels)):
        idx = np.where(labels == c)[0]
        k = len(idx)
        if k < 2:
            mean_intra, abs_intra, frac_pos, n_nonzero = 0.0, 0.0, 0.5, 0
        else:
            sub = J[np.ix_(idx, idx)]
            offdiag = sub[~np.eye(k, dtype=bool)]
            nz = offdiag[np.abs(offdiag) > 1e-6]
            mean_intra = float(offdiag.mean())
            abs_intra = float(np.abs(offdiag).mean())
            n_nonzero = int(len(nz))
            frac_pos = float((nz > 0).mean()) if n_nonzero > 0 else 0.5
        if k >= min_size:
            pca = validate_cluster_pca(z_kept, idx, gap_threshold=pca_gap_threshold)
            twonn_id = community_nonlinear_id(z_kept, idx, nonlinear_id_sample)
        else:
            pca = {"eff_rank": float(k), "gap_idx": k, "gap_ratio": 1.0, "has_sharp_gap": False}
            twonn_id = None
        validated = pca["has_sharp_gap"] and k >= min_size
        if not validated:
            regime = "unvalidated"
        elif frac_pos >= capture_frac_pos:
            regime = "capture"
        elif frac_pos <= shatter_frac_pos:
            regime = "shatter"
        else:
            regime = "dilution"
        nonlinear_low_dim = (
            twonn_id is not None and pca["eff_rank"] > 3 * twonn_id and twonn_id < 0.5 * k
        )
        out.append(
            {
                "cluster": int(c),
                "size": k,
                "atoms": idx.tolist(),
                "mean_intra_J": mean_intra,
                "abs_intra_J": abs_intra,
                "frac_pos": frac_pos,
                "n_nonzero_intra": n_nonzero,
                "regime": regime,
                "validated": validated,
                "eff_rank": pca["eff_rank"],
                "twonn_id": twonn_id,
                "nonlinear_low_dim": nonlinear_low_dim,
                "gap_idx": pca["gap_idx"],
                "gap_ratio": pca["gap_ratio"],
            }
        )
    return out


def plot_J_heatmap(J: np.ndarray, labels: np.ndarray, out_path: Path) -> None:
    order = np.argsort(labels)
    J_sorted = J[np.ix_(order, order)]
    vmax = float(np.percentile(np.abs(J), 99)) or float(np.abs(J).max()) or 1.0
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(J_sorted, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    for b in np.where(np.diff(labels[order]) != 0)[0] + 0.5:
        ax.axhline(b, color="black", linewidth=0.3, alpha=0.5)
        ax.axvline(b, color="black", linewidth=0.3, alpha=0.5)
    ax.set_title(f"Ising couplings J (Louvain reorder) - {len(set(labels))} communities")
    plt.colorbar(im, ax=ax, label="J_ij")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_ebic_curve(results: list[dict[str, Any]], out_path: Path) -> None:
    lams = [r["lambda"] for r in results]
    ebics = [r["ebic"] for r in results]
    edges = [r["edges"] for r in results]
    best = min(range(len(results)), key=lambda i: ebics[i])
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4))
    axL.plot(lams, ebics, "o-")
    axL.axvline(lams[best], color="red", linestyle="--", label=f"best lambda={lams[best]:.4f}")
    axL.set_xscale("log")
    axL.set(xlabel="lambda", ylabel="EBIC", title="EBIC vs lambda")
    axL.legend()
    axR.plot(lams, edges, "o-", color="darkgreen")
    axR.axvline(lams[best], color="red", linestyle="--")
    axR.set_xscale("log")
    axR.set_yscale("log")
    axR.set(xlabel="lambda", ylabel="|E|", title="sparsity vs lambda")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_regime_composition(clusters: list[dict[str, Any]], n_kept: int, out_path: Path) -> None:
    totals = {"capture": 0, "shatter": 0, "dilution": 0, "unvalidated": 0}
    for c in clusters:
        totals[c["regime"]] += c["size"]
    colour = {
        "capture": "#1f7a1f",
        "shatter": "#b31f1f",
        "dilution": "#d97500",
        "unvalidated": "#888888",
    }
    fig, ax = plt.subplots(figsize=(8, 2.5))
    left = 0
    for r in ("capture", "dilution", "shatter", "unvalidated"):
        n = totals[r]
        if n == 0:
            continue
        ax.barh([0], [n], left=left, color=colour[r], edgecolor="white", linewidth=1.5)
        if n / n_kept > 0.04:
            ax.text(
                left + n / 2,
                0,
                f"{r}\n{n} ({100 * n / n_kept:.1f}%)",
                ha="center",
                va="center",
                fontsize=10,
                color="white",
                fontweight="bold",
            )
        left += n
    ax.set_xlim(0, n_kept)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_title(f"Regime composition - {n_kept} kept dims")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def binarise(
    codes: Float[Tensor, "n D"], mode: Literal["activity", "positive"]
) -> Float[Tensor, "n D"]:
    """Codes -> spins in {-1,+1}. 'activity': fired vs not (natural for signed sparse
    codes). 'positive': Bhalla's one-sided 1[z>0] (parity baseline)."""
    if mode == "activity":
        return 2.0 * (codes != 0).float() - 1.0
    return 2.0 * (codes > 0).float() - 1.0


def run(
    code_dir: Path,
    out_dir: Path,
    binarize: Literal["activity", "positive"],
    n_samples: int,
    lambdas: list[float],
    freq_floor: float,
    freq_ceiling: float,
    n_iter: int,
    device: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    codes = load_codes(code_dir, n_samples)[:n_samples]
    print(f"loaded codes {tuple(codes.shape)}")

    firing_rate = (codes != 0).float().mean(dim=0)
    keep = ((firing_rate >= freq_floor) & (firing_rate <= freq_ceiling)).nonzero().flatten()
    print(f"kept {len(keep)}/{codes.shape[1]} dims in freq [{freq_floor}, {freq_ceiling}]")
    z_kept = codes[:, keep]
    spins = binarise(z_kept, binarize)

    J, _, results = fit_ising_ebic(spins, lambdas, n_iter=n_iter, device=device)
    labels = cluster_atoms_louvain(J)
    clusters = classify_clusters(J, labels, z_kept.numpy())

    plot_J_heatmap(J, labels, out_dir / "J_heatmap.png")
    plot_ebic_curve(results, out_dir / "ebic_curve.png")
    plot_regime_composition(clusters, len(keep), out_dir / "regime_composition.png")

    regime_counts: dict[str, int] = {}
    for c in clusters:
        regime_counts[c["regime"]] = regime_counts.get(c["regime"], 0) + 1
    summary = {
        "code_dir": str(code_dir),
        "binarize": binarize,
        "n_samples": int(codes.shape[0]),
        "n_kept_dims": int(len(keep)),
        "kept_dims": keep.tolist(),
        "best_lambda": float(min(results, key=lambda r: r["ebic"])["lambda"]),
        "n_communities": int(len(set(labels))),
        "regime_counts": regime_counts,
        "clusters": [{k: v for k, v in c.items() if k != "atoms"} for c in clusters],
    }
    (out_dir / "topology_summary.json").write_text(json.dumps(summary, indent=2))
    np.save(out_dir / "J.npy", J)
    np.save(out_dir / "labels.npy", labels)
    print(f"communities: {len(set(labels))}  regimes: {regime_counts}")
    print(f"wrote {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--binarize", choices=["activity", "positive"], default="activity")
    ap.add_argument("--n_samples", type=int, default=200_000)
    ap.add_argument("--n_iter", type=int, default=300)
    ap.add_argument("--freq_floor", type=float, default=0.02)
    ap.add_argument("--freq_ceiling", type=float, default=0.98)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[0.005, 0.01, 0.02, 0.04, 0.08, 0.16],
    )
    args = ap.parse_args()
    run(
        code_dir=args.codes,
        out_dir=args.out,
        binarize=args.binarize,
        n_samples=args.n_samples,
        lambdas=args.lambdas,
        freq_floor=args.freq_floor,
        freq_ceiling=args.freq_ceiling,
        n_iter=args.n_iter,
        device=args.device,
    )


if __name__ == "__main__":
    main()
