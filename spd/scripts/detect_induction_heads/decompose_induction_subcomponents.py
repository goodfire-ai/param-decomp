"""Decompose induction-head attention into per-subcomponent Q,K contributions.

For repeated random sequences [t_0..t_{L-1} | t_0..t_{L-1}] the induction pattern
attends from position L+k to position k+1 (head 4 of layer 2 carries it in run
`s-55ea3f9b`; see `induction_scores.png` in the same out dir).

This script decomposes the pre-softmax attention logit on the induction diagonal at
the chosen (layer, head) into per-(Q-subcomp, K-subcomp) contributions, summed over
all (batch, diagonal-position) pairs. RoPE is folded into a single
`compute_qk_rope_coefficients` + `evaluate_qk_at_offsets` call at offset = L-1.

Component activations are gated by causal importance, so the ranking reflects the
subcomponents that actually carry the induction signal in the live model.

Usage:
    python -m spd.scripts.detect_induction_heads.decompose_induction_subcomponents \
        wandb:goodfire/spd/runs/s-55ea3f9b
"""

import math
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.models.components import LinearComponents
from spd.scripts.rope_aware_qk import compute_qk_rope_coefficients, evaluate_qk_at_offsets
from spd.spd_types import ModelPath
from spd.utils.wandb_utils import parse_wandb_run_path

SCRIPT_DIR = Path(__file__).parent
DEFAULT_N_BATCHES = 50
DEFAULT_BATCH_SIZE = 32
DEFAULT_HALF_SEQ_LEN = 256
DEFAULT_LAYER_IDX = 2
DEFAULT_HEAD_IDX = 4
DEFAULT_TOP_N = 30


def _make_repeated_inputs(
    batch_size: int, half_len: int, vocab_size: int, device: torch.device, seed: int
) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    half = torch.randint(0, vocab_size, (batch_size, half_len), generator=g)
    return torch.cat([half, half], dim=1).to(device)


def _accumulate_diagonal_weights(
    model: ComponentModel,
    input_ids: torch.Tensor,
    layer_idx: int,
    half_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """For one batch, sum CI-gated (q_act_i, k_act_j) products across the induction diagonal.

    Returns:
        weights: (C_q, C_k) — sum over (b, m) of q_gated[b, L+m, i] * k_gated[b, m+1, j]
        q_alive_acc: (C_q,) — count of (b, m) with q-mask alive at L+m
        k_alive_acc: (C_k,) — count of (b, m) with k-mask alive at m+1
    """
    q_path = f"h.{layer_idx}.attn.q_proj"
    k_path = f"h.{layer_idx}.attn.k_proj"

    with torch.no_grad():
        out = model(input_ids, cache_type="input")
        comp_acts = model.get_all_component_acts(out.cache)
        ci = model.calc_causal_importances(
            pre_weight_acts=out.cache,
            sampling="continuous",
            detach_inputs=True,
        ).lower_leaky

    q_acts = comp_acts[q_path].float()  # (B, T, C_q)
    k_acts = comp_acts[k_path].float()  # (B, T, C_k)
    q_ci = ci[q_path].float()  # (B, T, C_q)
    k_ci = ci[k_path].float()  # (B, T, C_k)

    q_gated = q_acts * q_ci
    k_gated = k_acts * k_ci

    # Induction diagonal positions: t_q = L + m, t_k = m + 1, m in [0, L-2]
    L = half_len
    dst = torch.arange(L, 2 * L - 1, device=q_gated.device)
    src = torch.arange(1, L, device=k_gated.device)

    q_diag = q_gated[:, dst, :]  # (B, L-1, C_q)
    k_diag = k_gated[:, src, :]  # (B, L-1, C_k)

    # Sum over (B, L-1) -> (C_q, C_k)
    weights = torch.einsum("blq,blk->qk", q_diag, k_diag)

    # Alive counts (CI > 0.001) for diagnostics
    q_alive_count = (q_ci[:, dst, :] > 0.001).float().sum(dim=(0, 1))
    k_alive_count = (k_ci[:, src, :] > 0.001).float().sum(dim=(0, 1))

    return weights.cpu(), q_alive_count.cpu(), k_alive_count.cpu()


def _compute_rope_qk_at_offset(
    model: ComponentModel,
    layer_idx: int,
    head_idx: int,
    offset: int,
) -> torch.Tensor:
    """Compute the RoPE-rotated QK pair-coefficient W[i, j] at a single offset for one head.

    Returns shape (C_q, C_k).
    """
    q_path = f"h.{layer_idx}.attn.q_proj"
    k_path = f"h.{layer_idx}.attn.k_proj"

    q_component = model.components[q_path]
    k_component = model.components[k_path]
    assert isinstance(q_component, LinearComponents)
    assert isinstance(k_component, LinearComponents)

    target_block = model.target_model._h[layer_idx]
    attn = target_block.attn
    n_q_heads = attn.n_head
    n_kv_heads = attn.n_key_value_heads
    head_dim = attn.head_dim
    g = n_q_heads // n_kv_heads

    C_q = q_component.U.shape[0]
    C_k = k_component.U.shape[0]
    U_q = q_component.U.float().reshape(C_q, n_q_heads, head_dim)
    U_k = k_component.U.float().reshape(C_k, n_kv_heads, head_dim)
    U_k_expanded = U_k.repeat_interleave(g, dim=1)  # (C_k, n_q_heads, head_dim)

    A, B = compute_qk_rope_coefficients(U_q[:, head_idx, :], U_k_expanded[:, head_idx, :])
    W = evaluate_qk_at_offsets(A, B, attn.rotary_cos, attn.rotary_sin, (offset,))
    return W.squeeze(0).cpu()  # (C_q, C_k)


def _plot_top_pairs(
    contributions: NDArray[np.floating],
    top_pairs: list[tuple[int, int]],
    run_id: str,
    layer_idx: int,
    head_idx: int,
    out_path: Path,
) -> None:
    rows = [(qi, kj, contributions[qi, kj]) for qi, kj in top_pairs]
    labels = [f"Q.{qi} × K.{kj}" for qi, kj, _ in rows]
    values = [v for _, _, v in rows]
    colors = ["#1f77b4" if v >= 0 else "#d62728" for v in values]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(rows))))
    ax.barh(range(len(rows)), values, color=colors)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=8, family="monospace")
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Pre-softmax logit contribution (sum over induction diagonal)")
    ax.set_title(
        f"{run_id} | L{layer_idx}H{head_idx} induction subcomponents | top {len(rows)} (Q, K) pairs"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main(
    model_path: ModelPath,
    layer_idx: int = DEFAULT_LAYER_IDX,
    head_idx: int = DEFAULT_HEAD_IDX,
    n_batches: int = DEFAULT_N_BATCHES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    half_seq_len: int = DEFAULT_HALF_SEQ_LEN,
    top_n: int = DEFAULT_TOP_N,
    out_dir: str | None = None,
    seed: int = 0,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading SPD run {model_path} on {device}")

    run_info = SPDRunInfo.from_path(model_path)
    model = ComponentModel.from_run_info(run_info).to(device)
    model.eval()
    for block in model.target_model._h:
        block.attn.flash_attention = False

    run_id = (
        parse_wandb_run_path(model_path)[2] if str(model_path).startswith("wandb:") else "local"
    )
    out_path = Path(out_dir) if out_dir is not None else SCRIPT_DIR / "out" / run_id
    out_path.mkdir(parents=True, exist_ok=True)

    vocab_size = model.target_model.wte.weight.shape[0]
    L = half_seq_len
    offset = L - 1

    q_path = f"h.{layer_idx}.attn.q_proj"
    k_path = f"h.{layer_idx}.attn.k_proj"
    q_comp = model.components[q_path]
    k_comp = model.components[k_path]
    assert isinstance(q_comp, LinearComponents) and isinstance(k_comp, LinearComponents)
    C_q = q_comp.U.shape[0]
    C_k = k_comp.U.shape[0]
    logger.info(f"layer {layer_idx}: C_q={C_q}, C_k={C_k}")

    weight_acc = torch.zeros(C_q, C_k)
    q_alive_acc = torch.zeros(C_q)
    k_alive_acc = torch.zeros(C_k)

    for b in range(n_batches):
        input_ids = _make_repeated_inputs(batch_size, L, vocab_size, device, seed + b)
        w, qa, ka = _accumulate_diagonal_weights(model, input_ids, layer_idx, L)
        weight_acc += w
        q_alive_acc += qa
        k_alive_acc += ka
        if (b + 1) % 10 == 0:
            logger.info(f"  batch {b + 1}/{n_batches}")

    n_diag_positions = n_batches * batch_size * (L - 1)
    logger.info(f"accumulated over {n_diag_positions} (batch, diagonal-position) pairs")

    head_dim = model.target_model._h[layer_idx].attn.head_dim
    scale = 1.0 / math.sqrt(head_dim)

    W_qk = _compute_rope_qk_at_offset(model, layer_idx, head_idx, offset)  # (C_q, C_k)
    contributions = (scale * W_qk * weight_acc).detach().numpy()  # (C_q, C_k)

    # Rank by absolute contribution
    flat = np.argsort(np.abs(contributions).ravel())[::-1][:top_n]
    top_pairs = [tuple(int(x) for x in divmod(int(idx), C_k)) for idx in flat]

    # Save full grid + alive counts
    np.savez(
        out_path / f"l{layer_idx}h{head_idx}_induction_decomp.npz",
        contributions=contributions,
        weight=weight_acc.numpy(),
        W_qk=W_qk.detach().numpy(),
        q_alive_count=q_alive_acc.numpy(),
        k_alive_count=k_alive_acc.numpy(),
        n_diag_positions=np.array([n_diag_positions]),
        layer_idx=np.array([layer_idx]),
        head_idx=np.array([head_idx]),
        offset=np.array([offset]),
    )
    logger.info(f"saved decomp to {out_path}/l{layer_idx}h{head_idx}_induction_decomp.npz")

    # Pretty print top pairs
    print(
        f"\nTop {top_n} (Q, K) subcomponent pairs driving L{layer_idx}H{head_idx} "
        f"induction (mean per-diag-pos contribution):"
    )
    print(
        f"{'rank':<5}{'Q':<8}{'K':<8}{'sum contrib':>14}{'mean/pos':>12}"
        f"{'q_alive%':>10}{'k_alive%':>10}"
    )
    for rank, (qi, kj) in enumerate(top_pairs, 1):
        total = contributions[qi, kj]
        mean = total / max(n_diag_positions, 1)
        q_pct = 100.0 * q_alive_acc[qi].item() / max(n_diag_positions, 1)
        k_pct = 100.0 * k_alive_acc[kj].item() / max(n_diag_positions, 1)
        print(
            f"{rank:<5}Q.{qi:<6}K.{kj:<6}{total:>+14.3f}{mean:>+12.4f}{q_pct:>9.1f}%{k_pct:>9.1f}%"
        )

    # Per-subcomp marginals (top contributors regardless of partner)
    q_marginal = contributions.sum(axis=1)
    k_marginal = contributions.sum(axis=0)
    print("\nTop 10 Q subcomponents (sum across K):")
    for qi in np.argsort(np.abs(q_marginal))[::-1][:10]:
        print(f"  Q.{int(qi):<6}{q_marginal[qi]:>+12.3f}")
    print("\nTop 10 K subcomponents (sum across Q):")
    for kj in np.argsort(np.abs(k_marginal))[::-1][:10]:
        print(f"  K.{int(kj):<6}{k_marginal[kj]:>+12.3f}")

    _plot_top_pairs(
        contributions,
        top_pairs,
        run_id,
        layer_idx,
        head_idx,
        out_path / f"l{layer_idx}h{head_idx}_induction_top_pairs.png",
    )
    logger.info(f"saved plot to {out_path}/l{layer_idx}h{head_idx}_induction_top_pairs.png")


if __name__ == "__main__":
    fire.Fire(main)
