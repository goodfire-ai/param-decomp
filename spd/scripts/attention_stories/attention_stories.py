"""Generate PDF reports tracing Q -> K -> V attention component interaction chains.

For each layer, produces a multi-page PDF:
  - Page 1: Layer overview with Q->K attention contributions at multiple RoPE offsets
    and K->V CI co-occurrence heatmaps
  - Pages 2+: Individual Q component "stories" with positive and negative attention separated
    into two columns, showing which K components the Q looks for vs avoids, and what V
    information those K components carry forward

The Q->K attention contribution is a weight-only measure (V-norm-scaled U dot products
with RoPE applied at specified relative position offsets, normalized per-head and averaged
across heads so each head's pair contributions sum to 1). The K->V
association uses CI co-occurrence counts (number of tokens where both components are
causally important).

Usage:
    python -m spd.scripts.attention_stories.attention_stories \
        wandb:goodfire/spd/runs/<run_id>
"""

import textwrap
from dataclasses import dataclass
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from numpy.typing import NDArray

from spd.autointerp.repo import InterpRepo
from spd.autointerp.schemas import InterpretationResult
from spd.harvest.repo import HarvestRepo
from spd.harvest.schemas import ComponentSummary
from spd.harvest.storage import CorrelationStorage
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.models.components import LinearComponents
from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLP
from spd.scripts.rope_aware_qk import compute_qk_rope_coefficients, evaluate_qk_at_offsets
from spd.spd_types import ModelPath
from spd.utils.wandb_utils import parse_wandb_run_path

SCRIPT_DIR = Path(__file__).parent
MIN_MEAN_CI = 0.01
N_STORIES_PER_LAYER = 10
TOP_K_PER_SIDE = 5
TOP_V_PER_K = 3
N_K_TEXT_PER_SIDE = 3
TEXT_WRAP_WIDTH = 75
LINE_HEIGHT = 0.013
STORY_OFFSETS = [0, 1, 2, 4, 8]


@dataclass
class ComponentInfo:
    idx: int
    causal_importance: float
    label: str | None
    reasoning: str | None


@dataclass
class KPartner:
    info: ComponentInfo
    attention_contribution: float  # peak W(Δ) value (at offset with max |W|)
    contributions_by_offset: list[tuple[int, float]]  # (offset, value) for each offset
    v_partners: list[tuple[ComponentInfo, float]]  # (v_info, cooccurrence_count)


def _get_alive_indices(summary: dict[str, ComponentSummary], module_path: str) -> list[int]:
    """Return component indices sorted by CI descending, filtered to alive."""
    components = [
        (s.component_idx, s.mean_activations["causal_importance"])
        for s in summary.values()
        if s.layer == module_path and s.mean_activations["causal_importance"] > MIN_MEAN_CI
    ]
    components.sort(key=lambda t: t[1], reverse=True)
    return [idx for idx, _ in components]


def _get_component_info(
    component_idx: int,
    module_path: str,
    summary: dict[str, ComponentSummary],
    interp: dict[str, InterpretationResult],
) -> ComponentInfo:
    key = f"{module_path}:{component_idx}"
    ci = summary[key].mean_activations["causal_importance"]
    result = interp.get(key)
    # Escape $ to prevent matplotlib mathtext parsing
    label = result.label.replace("$", r"\$") if result and result.label else None
    reasoning = result.reasoning.replace("$", r"\$") if result and result.reasoning else None
    return ComponentInfo(
        idx=component_idx,
        causal_importance=ci,
        label=label,
        reasoning=reasoning,
    )


def _compute_attention_contributions(
    q_component: LinearComponents,
    k_component: LinearComponents,
    q_alive: list[int],
    k_alive: list[int],
    n_q_heads: int,
    n_kv_heads: int,
    head_dim: int,
    rotary_cos: torch.Tensor,
    rotary_sin: torch.Tensor,
) -> NDArray[np.floating]:
    """Compute (n_offsets, n_q_alive, n_k_alive) mean attention contributions at each offset.

    Each head's contributions are normalized by T^h (total QK dot product for that head),
    then averaged across heads. Pair contributions within a head sum to 1.
    """
    V_q_norms = torch.linalg.norm(q_component.V[:, q_alive], dim=0).float()
    V_k_norms = torch.linalg.norm(k_component.V[:, k_alive], dim=0).float()

    U_q = q_component.U[q_alive].float() * V_q_norms[:, None]
    U_q = U_q.reshape(len(q_alive), n_q_heads, head_dim)

    U_k = k_component.U[k_alive].float() * V_k_norms[:, None]
    U_k = U_k.reshape(len(k_alive), n_kv_heads, head_dim)

    g = n_q_heads // n_kv_heads
    U_k_expanded = U_k.repeat_interleave(g, dim=1)

    head_results = []
    for h in range(n_q_heads):
        A, B = compute_qk_rope_coefficients(U_q[:, h, :], U_k_expanded[:, h, :])
        W_h = evaluate_qk_at_offsets(A, B, rotary_cos, rotary_sin, STORY_OFFSETS)
        head_results.append(W_h)  # (n_offsets, n_q, n_k)

    # (n_heads, n_offsets, n_q, n_k) -> normalize per head, then mean
    # Use T^h at offset 0 so cross-offset structure is preserved
    W = torch.stack(head_results)  # (n_heads, n_offsets, n_q, n_k)
    assert STORY_OFFSETS[0] == 0
    T_h = W[:, 0].sum(dim=(-2, -1), keepdim=True).unsqueeze(1)  # (n_heads, 1, 1, 1)
    W_normalized = W / T_h
    return W_normalized.mean(dim=0).cpu().numpy()


def _compute_cooccurrence_matrix(
    corr: CorrelationStorage,
    k_path: str,
    v_path: str,
    k_alive: list[int],
    v_alive: list[int],
) -> NDArray[np.floating]:
    """Compute (n_v_alive, n_k_alive) CI co-occurrence count matrix."""
    k_corr_idx = [corr.key_to_idx[f"{k_path}:{idx}"] for idx in k_alive]
    v_corr_idx = [corr.key_to_idx[f"{v_path}:{idx}"] for idx in v_alive]

    k_idx = torch.tensor(k_corr_idx)
    v_idx = torch.tensor(v_corr_idx)
    return corr.count_ij[v_idx[:, None], k_idx[None, :]].float().numpy()


# -- Page renderers -----------------------------------------------------------


def _render_overview_page(
    pdf: PdfPages,
    W_by_offset: NDArray[np.floating],
    cooccur: NDArray[np.floating] | None,
    q_alive: list[int],
    k_alive: list[int],
    v_alive: list[int],
    layer_idx: int,
    run_id: str,
) -> None:
    overview_offsets = STORY_OFFSETS[:2]  # Show Δ=0 and Δ=1
    n_qk = len(overview_offsets)
    n_panels = n_qk + (1 if cooccur is not None else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 8.5), squeeze=False)

    # Shared color scale across QK panels
    qk_vmax = float(max(np.abs(W_by_offset[idx]).max() for idx in range(n_qk))) or 1.0

    for panel_idx in range(n_qk):
        ax = axes[0, panel_idx]
        W = W_by_offset[panel_idx]
        im = ax.imshow(W, aspect="auto", cmap="RdBu_r", vmin=-qk_vmax, vmax=qk_vmax)
        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        ax.set_title(f"Q\u2192K attention (\u0394={overview_offsets[panel_idx]})", fontsize=11)
        ax.set_xlabel("K component")
        ax.set_ylabel("Q component")
        ax.set_xticks(range(len(k_alive)))
        ax.set_xticklabels([f"C{idx}" for idx in k_alive], fontsize=5, rotation=90)
        ax.set_yticks(range(len(q_alive)))
        ax.set_yticklabels([f"C{idx}" for idx in q_alive], fontsize=5)

    # K->V CI co-occurrence
    if cooccur is not None:
        ax_kv = axes[0, n_qk]
        im2 = ax_kv.imshow(cooccur, aspect="auto", cmap="Purples", vmin=0)
        fig.colorbar(im2, ax=ax_kv, shrink=0.8, pad=0.02, label="CI co-occurrence count")
        ax_kv.set_title("K\u2192V CI co-occurrence", fontsize=11)
        ax_kv.set_xlabel("K component")
        ax_kv.set_ylabel("V component")
        ax_kv.set_xticks(range(len(k_alive)))
        ax_kv.set_xticklabels([f"C{idx}" for idx in k_alive], fontsize=5, rotation=90)
        ax_kv.set_yticks(range(len(v_alive)))
        ax_kv.set_yticklabels([f"C{idx}" for idx in v_alive], fontsize=5)

    fig.suptitle(
        f"{run_id}  |  Layer {layer_idx} \u2014 Overview  (ci>{MIN_MEAN_CI})\n"
        f"Q: {len(q_alive)} alive  |  K: {len(k_alive)} alive  |  V: {len(v_alive)} alive",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    pdf.savefig(fig)
    plt.close(fig)


def _render_bar_chart(
    ax: plt.Axes,
    partners: list[KPartner],
    color: str,
    title: str,
    labels_on_right: bool,
) -> None:
    """Render a horizontal bar chart of K partners."""
    if not partners:
        ax.set_visible(False)
        return

    y_pos = np.arange(len(partners))
    values = [abs(kp.attention_contribution) for kp in partners]

    ax.barh(y_pos, values, color=color, height=0.6)

    ytick_labels = [f"K C{kp.info.idx}" for kp in partners]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ytick_labels, fontsize=7, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("|attention contribution| (peak)", fontsize=7)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", labelsize=7)

    if labels_on_right:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")


def _render_kv_text(
    ax: plt.Axes,
    partners: list[KPartner],
) -> None:
    """Render K->V text with bold headers and regular reasoning."""
    ax.axis("off")
    if not partners:
        return

    y = 0.98
    for kp in partners[:N_K_TEXT_PER_SIDE]:
        # K component header with multi-offset breakdown
        offset_str = ", ".join(f"\u0394={d}: {v:+.2f}" for d, v in kp.contributions_by_offset)
        k_header = f"K C{kp.info.idx} (ci={kp.info.causal_importance:.3f})  [{offset_str}]"
        if kp.info.label:
            k_header += f'  \u2014  "{kp.info.label}"'
        ax.text(
            0.01,
            y,
            k_header,
            fontsize=7,
            fontweight="bold",
            fontfamily="monospace",
            va="top",
            transform=ax.transAxes,
        )
        y -= LINE_HEIGHT

        # K reasoning (regular)
        if kp.info.reasoning:
            wrapped = textwrap.fill(
                kp.info.reasoning,
                width=TEXT_WRAP_WIDTH,
                initial_indent="  ",
                subsequent_indent="  ",
            )
            n_lines = wrapped.count("\n") + 1
            ax.text(
                0.01,
                y,
                wrapped,
                fontsize=6,
                fontfamily="monospace",
                va="top",
                transform=ax.transAxes,
            )
            y -= LINE_HEIGHT * n_lines

        # V partners
        for v_info, count in kp.v_partners:
            v_header = (
                f"  \u2192 V C{v_info.idx} (co-occ={count:.0f}, ci={v_info.causal_importance:.3f})"
            )
            if v_info.label:
                v_header += f'  \u2014  "{v_info.label}"'
            ax.text(
                0.01,
                y,
                v_header,
                fontsize=7,
                fontweight="bold",
                fontfamily="monospace",
                va="top",
                transform=ax.transAxes,
            )
            y -= LINE_HEIGHT

            if v_info.reasoning:
                wrapped = textwrap.fill(
                    v_info.reasoning,
                    width=TEXT_WRAP_WIDTH,
                    initial_indent="      ",
                    subsequent_indent="      ",
                )
                n_lines = wrapped.count("\n") + 1
                ax.text(
                    0.01,
                    y,
                    wrapped,
                    fontsize=6,
                    fontfamily="monospace",
                    va="top",
                    transform=ax.transAxes,
                )
                y -= LINE_HEIGHT * n_lines

        y -= LINE_HEIGHT * 0.3  # gap between K blocks


def _render_story_page(
    pdf: PdfPages,
    q_info: ComponentInfo,
    pos_partners: list[KPartner],
    neg_partners: list[KPartner],
    layer_idx: int,
    run_id: str,
) -> None:
    fig = plt.figure(figsize=(16, 18))
    gs = GridSpec(
        3,
        2,
        figure=fig,
        height_ratios=[0.08, 0.15, 0.77],
        wspace=0.25,
        hspace=0.15,
    )

    # -- Header (spans both columns) -----------------------------------------
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.axis("off")

    header_line = f"Q Component C{q_info.idx}   |   ci = {q_info.causal_importance:.4f}"
    if q_info.label:
        header_line += f'   |   "{q_info.label}"'
    ax_header.text(
        0.5,
        0.7,
        header_line,
        fontsize=12,
        fontweight="bold",
        ha="center",
        va="center",
        transform=ax_header.transAxes,
    )
    if q_info.reasoning:
        reasoning_wrapped = textwrap.fill(q_info.reasoning, width=130)
        ax_header.text(
            0.5,
            0.15,
            reasoning_wrapped,
            fontsize=9,
            ha="center",
            va="center",
            transform=ax_header.transAxes,
        )

    # -- Positive bar chart (left) --------------------------------------------
    ax_pos_bars = fig.add_subplot(gs[1, 0])
    _render_bar_chart(
        ax_pos_bars,
        pos_partners,
        "#4477AA",
        "Looks for (positive attention)",
        labels_on_right=False,
    )

    # -- Negative bar chart (right) -------------------------------------------
    ax_neg_bars = fig.add_subplot(gs[1, 1])
    _render_bar_chart(
        ax_neg_bars,
        neg_partners,
        "#CC6677",
        "Avoids (negative attention)",
        labels_on_right=True,
    )

    # -- Positive K->V text (left) --------------------------------------------
    ax_pos_text = fig.add_subplot(gs[2, 0])
    ax_pos_text.set_title("K \u2192 V associations", fontsize=9)
    _render_kv_text(ax_pos_text, pos_partners)

    # -- Negative K->V text (right) -------------------------------------------
    ax_neg_text = fig.add_subplot(gs[2, 1])
    ax_neg_text.set_title("K \u2192 V associations", fontsize=9)
    _render_kv_text(ax_neg_text, neg_partners)

    fig.suptitle(
        f"{run_id}  |  Layer {layer_idx}",
        fontsize=10,
        fontstyle="italic",
    )
    fig.subplots_adjust(
        left=0.06,
        right=0.94,
        top=0.97,
        bottom=0.02,
        wspace=0.25,
        hspace=0.12,
    )
    pdf.savefig(fig)
    plt.close(fig)


# -- Markdown output ----------------------------------------------------------


def _md_component(info: ComponentInfo, prefix: str, extra: str = "") -> str:
    line = f"**{prefix} C{info.idx}** (ci={info.causal_importance:.3f})"
    if extra:
        line += f"  {extra}"
    if info.label:
        line += f'  \u2014  **"{info.label}"**'
    if info.reasoning:
        line += f"\n  {info.reasoning}"
    return line


def _md_k_partners(partners: list[KPartner], section_title: str) -> str:
    if not partners:
        return f"### {section_title}\n\n(none)\n"
    lines = [f"### {section_title}\n"]
    for kp in partners:
        offset_str = "  ".join(f"\u0394={d}: {v:+.3f}" for d, v in kp.contributions_by_offset)
        lines.append(_md_component(kp.info, "K", extra=f"[{offset_str}]"))
        if kp.v_partners:
            for v_info, count in kp.v_partners:
                lines.append(
                    "  " + _md_component(v_info, "\u2192 V", extra=f"(co-occ={count:.0f})")
                )
        else:
            lines.append("  (no strong V associations)")
        lines.append("")
    return "\n".join(lines)


def _write_layer_markdown(
    md_path: Path,
    run_id: str,
    layer_idx: int,
    q_alive: list[int],
    k_alive: list[int],
    v_alive: list[int],
    stories: list[tuple[ComponentInfo, list[KPartner], list[KPartner]]],
) -> None:
    lines = [
        f"# {run_id} \u2014 Layer {layer_idx} Attention Stories\n",
        f"Q: {len(q_alive)} alive | K: {len(k_alive)} alive | V: {len(v_alive)} alive"
        f"  (ci>{MIN_MEAN_CI})\n",
        "---\n",
    ]

    for q_info, pos_partners, neg_partners in stories:
        header = f"## Q Component C{q_info.idx} (ci={q_info.causal_importance:.4f})"
        if q_info.label:
            header += f'  \u2014  "{q_info.label}"'
        lines.append(header + "\n")
        if q_info.reasoning:
            lines.append(f"{q_info.reasoning}\n")

        lines.append(_md_k_partners(pos_partners, "Looks for (positive attention)"))
        lines.append(_md_k_partners(neg_partners, "Avoids (negative attention)"))
        lines.append("---\n")

    md_path.write_text("\n".join(lines))
    logger.info(f"Saved {md_path}")


# -- Main ---------------------------------------------------------------------


def _build_k_partners(
    k_ranks: NDArray[np.integer],
    peak_values: NDArray[np.floating],
    q_offset_slice: NDArray[np.floating],
    k_alive: list[int],
    v_alive: list[int],
    k_path: str,
    v_path: str,
    summary: dict[str, ComponentSummary],
    interp: dict[str, InterpretationResult],
    cooccur: NDArray[np.floating] | None,
) -> list[KPartner]:
    partners: list[KPartner] = []
    for k_rank in k_ranks:
        k_idx = k_alive[int(k_rank)]
        k_info = _get_component_info(k_idx, k_path, summary, interp)
        k_contrib = float(peak_values[int(k_rank)])
        contrib_by_offset = [
            (STORY_OFFSETS[i], float(q_offset_slice[i, int(k_rank)]))
            for i in range(len(STORY_OFFSETS))
        ]

        v_partners: list[tuple[ComponentInfo, float]] = []
        if cooccur is not None and v_alive:
            cooccur_col = cooccur[:, int(k_rank)]
            top_v_ranks = np.argsort(-cooccur_col)[:TOP_V_PER_K]
            for v_rank in top_v_ranks:
                count = float(cooccur_col[int(v_rank)])
                if count <= 0:
                    continue
                v_idx = v_alive[int(v_rank)]
                v_info = _get_component_info(v_idx, v_path, summary, interp)
                v_partners.append((v_info, count))

        partners.append(
            KPartner(
                info=k_info,
                attention_contribution=k_contrib,
                contributions_by_offset=contrib_by_offset,
                v_partners=v_partners,
            )
        )
    return partners


def generate_attention_stories(wandb_path: ModelPath) -> None:
    _entity, _project, run_id = parse_wandb_run_path(str(wandb_path))
    run_info = SPDRunInfo.from_path(wandb_path)

    out_dir = SCRIPT_DIR / "out" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    model = ComponentModel.from_run_info(run_info)
    model.eval()

    repo = HarvestRepo.open_most_recent(run_id)
    assert repo is not None, f"No harvest data found for {run_id}"
    summary = repo.get_summary()
    corr = repo.get_correlations()

    # Autointerp data (optional)
    interp: dict[str, InterpretationResult] = {}
    interp_repo = InterpRepo.open(run_id)
    if interp_repo is not None:
        interp = interp_repo.get_all_interpretations()
        logger.info(f"Loaded {len(interp)} autointerp interpretations")
    else:
        logger.info("No autointerp data found (labels will be omitted)")

    if corr is not None:
        logger.info(
            f"Loaded correlations: {len(corr.component_keys)} components, {corr.count_total} tokens"
        )
    else:
        logger.info("No correlation data found (K\u2192V associations will be omitted)")

    target_model = model.target_model
    assert isinstance(target_model, LlamaSimpleMLP)
    blocks = target_model._h
    assert not blocks[0].attn.rotary_adjacent_pairs, "RoPE math assumes non-adjacent pairs layout"
    head_dim = blocks[0].attn.head_dim
    n_q_heads = blocks[0].attn.n_head
    n_kv_heads = blocks[0].attn.n_key_value_heads
    n_layers = len(blocks)
    logger.info(
        f"Model: {n_layers} layers, head_dim={head_dim}, "
        f"n_q_heads={n_q_heads}, n_kv_heads={n_kv_heads}"
    )

    with torch.no_grad():
        for layer_idx in range(n_layers):
            q_path = f"h.{layer_idx}.attn.q_proj"
            k_path = f"h.{layer_idx}.attn.k_proj"
            v_path = f"h.{layer_idx}.attn.v_proj"

            q_alive = _get_alive_indices(summary, q_path)
            k_alive = _get_alive_indices(summary, k_path)
            v_alive = _get_alive_indices(summary, v_path)
            logger.info(f"Layer {layer_idx}: Q={len(q_alive)}, K={len(k_alive)}, V={len(v_alive)}")

            if not q_alive or not k_alive:
                logger.info(f"Layer {layer_idx}: skipping (no alive Q or K)")
                continue

            q_component = model.components[q_path]
            k_component = model.components[k_path]
            assert isinstance(q_component, LinearComponents)
            assert isinstance(k_component, LinearComponents)

            rotary_cos = blocks[layer_idx].attn.rotary_cos
            rotary_sin = blocks[layer_idx].attn.rotary_sin
            assert isinstance(rotary_cos, torch.Tensor)
            assert isinstance(rotary_sin, torch.Tensor)

            W_by_offset = _compute_attention_contributions(
                q_component,
                k_component,
                q_alive,
                k_alive,
                n_q_heads,
                n_kv_heads,
                head_dim,
                rotary_cos,
                rotary_sin,
            )

            cooccur: NDArray[np.floating] | None = None
            if corr is not None and v_alive:
                cooccur = _compute_cooccurrence_matrix(corr, k_path, v_path, k_alive, v_alive)

            # Build all stories for this layer
            stories: list[tuple[ComponentInfo, list[KPartner], list[KPartner]]] = []
            for q_rank, q_idx in enumerate(q_alive[:N_STORIES_PER_LAYER]):
                q_info = _get_component_info(q_idx, q_path, summary, interp)

                # (n_offsets, n_k_alive) for this Q component
                q_offset_slice = W_by_offset[:, q_rank, :]

                # Rank K partners by peak |W(Δ)| across offsets
                peak_offset_idx = np.argmax(np.abs(q_offset_slice), axis=0)  # (n_k_alive,)
                peak_values = q_offset_slice[peak_offset_idx, np.arange(q_offset_slice.shape[1])]

                pos_mask = peak_values > 0
                neg_mask = peak_values < 0

                pos_ranks = np.where(pos_mask)[0]
                pos_ranks = pos_ranks[np.argsort(-peak_values[pos_ranks])][:TOP_K_PER_SIDE]

                neg_ranks = np.where(neg_mask)[0]
                neg_ranks = neg_ranks[np.argsort(peak_values[neg_ranks])][:TOP_K_PER_SIDE]

                pos_partners = _build_k_partners(
                    pos_ranks,
                    peak_values,
                    q_offset_slice,
                    k_alive,
                    v_alive,
                    k_path,
                    v_path,
                    summary,
                    interp,
                    cooccur,
                )
                neg_partners = _build_k_partners(
                    neg_ranks,
                    peak_values,
                    q_offset_slice,
                    k_alive,
                    v_alive,
                    k_path,
                    v_path,
                    summary,
                    interp,
                    cooccur,
                )
                stories.append((q_info, pos_partners, neg_partners))

            # Write PDF
            pdf_path = out_dir / f"layer{layer_idx}.pdf"
            with PdfPages(pdf_path) as pdf:
                _render_overview_page(
                    pdf,
                    W_by_offset,
                    cooccur,
                    q_alive,
                    k_alive,
                    v_alive,
                    layer_idx,
                    run_id,
                )
                for q_info, pos_partners, neg_partners in stories:
                    _render_story_page(
                        pdf,
                        q_info,
                        pos_partners,
                        neg_partners,
                        layer_idx,
                        run_id,
                    )
            logger.info(f"Saved {pdf_path}")

            # Write companion markdown
            md_path = out_dir / f"layer{layer_idx}.md"
            _write_layer_markdown(
                md_path,
                run_id,
                layer_idx,
                q_alive,
                k_alive,
                v_alive,
                stories,
            )

    logger.info(f"All reports saved to {out_dir}")


if __name__ == "__main__":
    fire.Fire(generate_attention_stories)
