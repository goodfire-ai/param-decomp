"""First-pass geometry of the harvested bottleneck codes.

Adjudicates the central question: are the realized codes a low-dimensional manifold
(intrinsic dim << active-L0) or a genuine high-dim feature code? Reports:
  - support structure: active-set sizes, support entropy vs an independent-Bernoulli
    baseline (are dims correlated in firing?), always-on backbone
  - intrinsic dimension: TwoNN estimator (Facco et al. 2017) on a sample
  - linear curvature proxy: PCA variance-explained curve (how many linear dims for
    90/95/99% variance) vs the nonlinear TwoNN estimate
"""

import argparse
import json
from pathlib import Path

import torch
from jaxtyping import Float
from torch import Tensor


def load_codes(code_dir: Path, max_positions: int) -> Float[Tensor, "n D"]:
    chunks: list[Tensor] = []
    total = 0
    for f in sorted(code_dir.glob("codes_*.pt")):
        c = torch.load(f).float()
        chunks.append(c)
        total += c.shape[0]
        if total >= max_positions:
            break
    codes = torch.cat(chunks, dim=0)[:max_positions]
    return codes


def support_structure(codes: Float[Tensor, "n D"]) -> dict[str, float]:
    fired = codes != 0
    n, dim = fired.shape
    l0 = fired.sum(dim=1).float()
    firing_rate = fired.float().mean(dim=0)
    backbone = (firing_rate > 0.99).sum().item()
    dead = (firing_rate == 0).sum().item()

    # Support entropy vs independent-Bernoulli baseline. Real entropy is estimated over
    # observed unique supports; baseline is sum of per-dim Bernoulli entropies (bits).
    eps = 1e-12
    p = firing_rate.clamp(eps, 1 - eps)
    bernoulli_bits = float((-(p * p.log2() + (1 - p) * (1 - p).log2())).sum())
    # Empirical support entropy via hashing active-sets (bits), on a subsample.
    sub = fired[: min(n, 200_000)]
    packed = [hash(tuple(row.nonzero().flatten().tolist())) for row in sub]
    from collections import Counter

    counts = torch.tensor(list(Counter(packed).values()), dtype=torch.float64)
    probs = counts / counts.sum()
    empirical_bits = float(-(probs * probs.log2()).sum())
    n_unique = len(counts)

    return {
        "n_positions": float(n),
        "dim": float(dim),
        "l0_mean": float(l0.mean()),
        "l0_std": float(l0.std()),
        "backbone_dims_>99pct": float(backbone),
        "dead_dims": float(dead),
        "support_entropy_bits_empirical": empirical_bits,
        "support_entropy_bits_bernoulli_indep": bernoulli_bits,
        "n_unique_supports_in_subsample": float(n_unique),
        "subsample_size": float(sub.shape[0]),
    }


def twonn_intrinsic_dim(codes: Float[Tensor, "n D"], sample: int) -> dict[str, float]:
    """TwoNN (Facco et al. 2017): ID from the ratio of 2nd/1st nearest-neighbor distances.

    mu_i = r2_i / r1_i has CDF 1 - mu^-d, so d = -mean(log mu) fit via the linear
    relation log(1 - F(mu)) = -d log mu. We use the closed-form MLE d = N / sum(log mu).
    """
    idx = torch.randperm(codes.shape[0])[:sample]
    x = codes[idx]
    # pairwise distances in chunks to bound memory
    r1 = torch.full((x.shape[0],), float("inf"))
    r2 = torch.full((x.shape[0],), float("inf"))
    block = 2048
    for i in range(0, x.shape[0], block):
        d = torch.cdist(x[i : i + block], x)  # [b, N]; self-distance is 0 (smallest)
        vals, _ = d.topk(3, dim=1, largest=False)
        # col 0 is self (0); 1st/2nd NN are cols 1,2
        r1[i : i + block] = vals[:, 1]
        r2[i : i + block] = vals[:, 2]
    valid = (r1 > 0) & torch.isfinite(r2)
    mu = (r2[valid] / r1[valid]).clamp(min=1 + 1e-9)
    logmu = mu.log()
    # discard top 10% of mu (TwoNN robustness trick)
    keep = logmu <= logmu.quantile(0.9)
    logmu = logmu[keep]
    d_mle = float(logmu.shape[0] / logmu.sum())
    return {"twonn_id": d_mle, "twonn_sample": float(int(valid.sum()))}


def pca_curve(codes: Float[Tensor, "n D"], sample: int) -> dict[str, float]:
    idx = torch.randperm(codes.shape[0])[:sample]
    x = codes[idx]
    x = x - x.mean(dim=0, keepdim=True)
    # SVD on centered data; variance explained from singular values
    s = torch.linalg.svdvals(x)
    var = s**2
    cum = var.cumsum(0) / var.sum()
    out = {}
    for thresh in (0.90, 0.95, 0.99):
        out[f"pca_dims_{int(thresh * 100)}pct"] = float((cum < thresh).sum() + 1)
    out["pca_total_dims"] = float(len(s))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", required=True, type=Path)
    ap.add_argument("--max_positions", type=int, default=2_000_000)
    ap.add_argument("--id_sample", type=int, default=20_000)
    ap.add_argument("--pca_sample", type=int, default=100_000)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    codes = load_codes(args.codes, args.max_positions)
    print(f"loaded codes {tuple(codes.shape)}")

    result = {}
    result["support"] = support_structure(codes)
    print("support:", json.dumps(result["support"], indent=2))
    result["intrinsic_dim"] = twonn_intrinsic_dim(codes, args.id_sample)
    print("intrinsic_dim:", json.dumps(result["intrinsic_dim"], indent=2))
    result["pca"] = pca_curve(codes, args.pca_sample)
    print("pca:", json.dumps(result["pca"], indent=2))

    out = args.out or (args.codes / "geometry.json")
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
