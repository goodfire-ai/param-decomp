"""Identify which V and O components align with each head's OV read/write subspaces.

Loads saved W_OV matrices from plot_wv_subspace_overlap and computes alignment scores
between each head's OV circuit and the PD component vectors. Outputs
markdown files listing the top-aligned components per head with their autointerp labels.

Produces variants:
  - Raw: alignment computed on raw W_OV matrices
  - Data-weighted: alignment computed on W_OV @ Z_bar @ S_bar (PCA-weighted)
  - K-filtered: alignment using PCA of activations at K-component-active positions (auto-discovered)

Usage:
    python -m param_decomp.scripts.plot_wv_subspace_overlap.analyze_ov_subspace_semantics \
        wandb:goodfire/spd/runs/<run_id> --layer 1
"""

from pathlib import Path

import fire
import numpy as np
from numpy.typing import NDArray

from param_decomp.autointerp.repo import InterpRepo
from param_decomp.autointerp.schemas import InterpretationResult
from param_decomp.log import logger
from param_decomp.models.component_model import ComponentModel
from param_decomp.param_decomp_types import ModelPath
from param_decomp.utils.wandb_utils import parse_wandb_run_path

SCRIPT_DIR = Path(__file__).parent
TOP_K = 20


def _compute_read_alignment(
    ov_weight_per_head: NDArray[np.floating],
    v_proj_V: NDArray[np.floating],
    v_proj_U: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Read-side alignment: how much each head's W_OV amplifies each V component's read direction.

    Each v_proj component's weight is U_v[c] V_v[c]^T. We scale V_v[:, c] by ||U_v[c, :]||
    to reflect the component's true contribution magnitude.

    Returns: (n_heads, C_v)
    """
    u_norms = np.linalg.norm(v_proj_U, axis=1)  # (C_v,)
    V_scaled = v_proj_V * u_norms[None, :]  # (d_model, C_v)

    # (n_heads, d_out, d_model) @ (d_model, C_v) -> (n_heads, d_out, C_v)
    projected = np.einsum("hdi,ic->hdc", ov_weight_per_head, V_scaled)
    return np.sqrt((projected**2).sum(axis=1))  # (n_heads, C_v)


def _compute_write_alignment(
    ov_weight_per_head: NDArray[np.floating],
    o_proj_U: NDArray[np.floating],
    o_proj_V: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Write-side alignment: how much each head's W_OV^T amplifies each O component's write direction.

    Each o_proj component's weight is U_o[c] V_o[c]^T. We scale U_o[c, :] by ||V_o[:, c]||
    to reflect the component's true contribution magnitude.

    Returns: (n_heads, C_o)
    """
    v_norms = np.linalg.norm(o_proj_V, axis=0)  # (C_o,)
    U_scaled = o_proj_U * v_norms[:, None]  # (C_o, d_model)

    # W_OV^{hT} u_scaled = einsum with transposed OV
    projected = np.einsum("hid,ci->hcd", ov_weight_per_head, U_scaled)
    return np.sqrt((projected**2).sum(axis=2))  # (n_heads, C_o)


def _generate_markdown(
    read_alignment: NDArray[np.floating],
    write_alignment: NDArray[np.floating],
    all_interps: dict[str, InterpretationResult],
    layer: int,
    run_id: str,
    top_k: int,
    variant_label: str,
) -> str:
    """Generate markdown content for one variant (raw or data-weighted)."""
    n_heads = read_alignment.shape[0]

    def get_label(proj: str, idx: int) -> str:
        key = f"h.{layer}.attn.{proj}:{idx}"
        if key in all_interps:
            return all_interps[key].label
        return "(no label)"

    lines: list[str] = []
    lines.append(f"# OV Subspace Semantics ({variant_label}) — Layer {layer} — {run_id}")
    lines.append("")
    lines.append(
        f"Top {top_k} V components (read-aligned) and O components (write-aligned) per head."
    )
    lines.append(
        "Alignment = ||W v_scaled_c|| (read) or ||W^T u_scaled_c|| (write), "
        "where vectors are scaled by the norm of the other factor in the rank-1 decomposition."
    )
    lines.append("")

    for h in range(n_heads):
        lines.append(f"## Head {h}")
        lines.append("")

        lines.append(f"### Read-aligned V components (top {top_k})")
        lines.append("")
        lines.append("| Rank | Comp | Alignment | Label |")
        lines.append("|------|------|-----------|-------|")
        top_v = np.argsort(-read_alignment[h])[:top_k]
        for rank, idx in enumerate(top_v):
            label = get_label("v_proj", idx)
            lines.append(f"| {rank + 1} | v.{idx} | {read_alignment[h, idx]:.4f} | {label} |")
        lines.append("")

        lines.append(f"### Write-aligned O components (top {top_k})")
        lines.append("")
        lines.append("| Rank | Comp | Alignment | Label |")
        lines.append("|------|------|-----------|-------|")
        top_o = np.argsort(-write_alignment[h])[:top_k]
        for rank, idx in enumerate(top_o):
            label = get_label("o_proj", idx)
            lines.append(f"| {rank + 1} | o.{idx} | {write_alignment[h, idx]:.4f} | {label} |")
        lines.append("")

    return "\n".join(lines)


def analyze_ov_subspace_semantics(
    wandb_path: ModelPath,
    layer: int = 1,
    top_k: int = TOP_K,
) -> None:
    _entity, _project, run_id = parse_wandb_run_path(str(wandb_path))

    data_dir = SCRIPT_DIR / "out" / run_id / "ov"
    assert data_dir.exists(), f"Run plot_wv_subspace_overlap first: {data_dir}"

    ov_weight_per_head = np.load(data_dir / f"layer{layer}_ov_weight_per_head.npy")
    var_svectors = np.load(data_dir / f"layer{layer}_var_svectors.npy")
    var_singular_values = np.load(data_dir / f"layer{layer}_var_singular_values.npy")
    n_heads = ov_weight_per_head.shape[0]
    logger.info(f"Loaded W_OV: {ov_weight_per_head.shape}")

    # Load component model
    logger.info("Loading component model...")
    model = ComponentModel.from_pretrained(wandb_path)

    v_comp = model.components[f"h.{layer}.attn.v_proj"]
    v_proj_V = v_comp.V.detach().float().cpu().numpy()  # (d_model, C_v)
    v_proj_U = v_comp.U.detach().float().cpu().numpy()  # (C_v, n_heads*head_dim)

    o_comp = model.components[f"h.{layer}.attn.o_proj"]
    o_proj_U = o_comp.U.detach().float().cpu().numpy()  # (C_o, d_model)
    o_proj_V = o_comp.V.detach().float().cpu().numpy()  # (n_heads*head_dim, C_o)

    logger.info(f"V components: {v_proj_V.shape[1]}, O components: {o_proj_U.shape[0]}")

    # Load autointerp labels
    interp_repo = InterpRepo.open(run_id)
    all_interps = interp_repo.get_all_interpretations() if interp_repo else {}

    def _apply_data_weighting(
        svectors: NDArray[np.floating], singular_values: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        Z_diag_s = svectors.T * singular_values[None, :]
        return np.array([ov_weight_per_head[h] @ Z_diag_s for h in range(n_heads)])

    # Build list of variants to run
    variants: list[tuple[str, NDArray[np.floating], str]] = [
        ("Raw", ov_weight_per_head, "raw"),
        (
            "Data weighted",
            _apply_data_weighting(var_svectors, var_singular_values),
            "data_weighted",
        ),
    ]

    # Discover K-filtered SVD data
    for k_dir in sorted(data_dir.glob("k_*")):
        sv_path = k_dir / f"layer{layer}_var_svectors.npy"
        s_path = k_dir / f"layer{layer}_var_singular_values.npy"
        if sv_path.exists() and s_path.exists():
            k_label = k_dir.name.replace("k_", "K.")
            filt_svectors = np.load(sv_path)
            filt_sv = np.load(s_path)
            variants.append(
                (
                    f"Data weighted: {k_label}",
                    _apply_data_weighting(filt_svectors, filt_sv),
                    k_dir.name,
                )
            )
            logger.info(f"Found K-filtered data: {k_dir.name}")

    for variant_label, ov_matrices, subdir in variants:
        logger.info(f"Computing {variant_label} alignments...")
        read_alignment = _compute_read_alignment(ov_matrices, v_proj_V, v_proj_U)
        write_alignment = _compute_write_alignment(ov_matrices, o_proj_U, o_proj_V)

        md = _generate_markdown(
            read_alignment, write_alignment, all_interps, layer, run_id, top_k, variant_label
        )

        variant_out_dir = data_dir / subdir
        variant_out_dir.mkdir(exist_ok=True)
        out_path = variant_out_dir / f"layer{layer}_subspace_semantics.md"
        out_path.write_text(md)
        logger.info(f"Saved {out_path}")


if __name__ == "__main__":
    fire.Fire(analyze_ov_subspace_semantics)
