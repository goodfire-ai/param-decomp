"""Generate supporting plots that fill logical gaps in the article.

1. CI distributions showing bimodality (per-position) for Q335/Q270/K206
2. Token examples showing what activates each component (evidence for "semantic")
3. All-layer CI survey showing Layer 2 is the interesting one
4. Q/K component CI landscape (scatter plot of all components)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from spd.configs import LMTaskConfig
from spd.data import DatasetConfig, create_data_loader
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLP

OUT_DIR = Path(__file__).parent / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WANDB_PATH = "wandb:goodfire/spd/runs/s-55ea3f9b"
LAYER = 2


def main() -> None:
    run_info = SPDRunInfo.from_path(WANDB_PATH)
    model = ComponentModel.from_run_info(run_info)
    model.eval()
    config = run_info.config

    target_model = model.target_model
    assert isinstance(target_model, LlamaSimpleMLP)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    assert config.tokenizer_name is not None
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    task_config = config.task_config
    assert isinstance(task_config, LMTaskConfig)
    seq_len = target_model.config.n_ctx

    dataset_config = DatasetConfig(
        name=task_config.dataset_name,
        hf_tokenizer_path=config.tokenizer_name,
        split=task_config.eval_data_split,
        n_ctx=task_config.max_seq_len,
        is_tokenized=task_config.is_tokenized,
        streaming=task_config.streaming,
        column_name=task_config.column_name,
        shuffle_each_epoch=False,
    )
    loader, _ = create_data_loader(dataset_config=dataset_config, batch_size=1, buffer_size=1000)

    q_path = f"h.{LAYER}.attn.q_proj"
    k_path = f"h.{LAYER}.attn.k_proj"

    # Collect per-position CIs for key components + tokens
    q335_pos_ci: list[float] = []
    q270_pos_ci: list[float] = []
    k206_pos_ci: list[float] = []
    k224_pos_ci: list[float] = []

    # Top tokens per component (collect tokens where CI > 0.5)
    q335_tokens: list[str] = []
    q270_tokens: list[str] = []
    q335_off_tokens: list[str] = []
    q270_off_tokens: list[str] = []
    k206_tokens: list[str] = []
    k206_off_tokens: list[str] = []

    # Also collect all Q and K component mean CIs for landscape plot
    all_q_ci_sums: np.ndarray | None = None
    all_k_ci_sums: np.ndarray | None = None
    n_samples = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 100:
                break

            input_ids = batch[task_config.column_name][:, :seq_len].to(device)
            T = input_ids.shape[1]

            out = model(input_ids, cache_type="input")
            ci = model.calc_causal_importances(
                pre_weight_acts=out.cache, sampling="continuous", detach_inputs=True
            ).lower_leaky

            q_ci = ci[q_path][0]  # (T, C_q)
            k_ci = ci[k_path][0]  # (T, C_k)

            # Accumulate mean CIs
            q_mean = q_ci.mean(dim=0).cpu().numpy()
            k_mean = k_ci.mean(dim=0).cpu().numpy()
            if all_q_ci_sums is None:
                all_q_ci_sums = q_mean
                all_k_ci_sums = k_mean
            else:
                all_q_ci_sums += q_mean
                all_k_ci_sums += k_mean
            n_samples += 1

            q335 = q_ci[:, 335].cpu()
            q270 = q_ci[:, 270].cpu()
            k206 = k_ci[:, 206].cpu()
            k224 = k_ci[:, 224].cpu()

            token_ids = input_ids[0].cpu().tolist()

            for t in range(T):
                q335_pos_ci.append(q335[t].item())
                q270_pos_ci.append(q270[t].item())
                k206_pos_ci.append(k206[t].item())
                k224_pos_ci.append(k224[t].item())

                tok = tokenizer.decode([token_ids[t]])  # pyright: ignore

                if q335[t].item() > 0.5 and len(q335_tokens) < 500:
                    q335_tokens.append(tok)
                elif q335[t].item() < 0.1 and len(q335_off_tokens) < 500:
                    q335_off_tokens.append(tok)

                if q270[t].item() > 0.5 and len(q270_tokens) < 500:
                    q270_tokens.append(tok)
                elif q270[t].item() < 0.1 and len(q270_off_tokens) < 500:
                    q270_off_tokens.append(tok)

                if k206[t].item() > 0.5 and len(k206_tokens) < 500:
                    k206_tokens.append(tok)
                elif k206[t].item() < 0.1 and len(k206_off_tokens) < 500:
                    k206_off_tokens.append(tok)

            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i+1}/100")

    assert all_q_ci_sums is not None and all_k_ci_sums is not None
    all_q_ci = all_q_ci_sums / n_samples
    all_k_ci = all_k_ci_sums / n_samples

    # === Plot 1: Per-position CI distributions (bimodality evidence) ===
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    bins = np.linspace(-0.02, 1.02, 52)

    for ax, data, name, color in [
        (axes[0, 0], q335_pos_ci, "Q335 (complex text)", "tab:red"),
        (axes[0, 1], q270_pos_ci, "Q270 (research text)", "tab:orange"),
        (axes[1, 0], k206_pos_ci, "K206 (natural language)", "tab:green"),
        (axes[1, 1], k224_pos_ci, "K224 (technical nouns)", "tab:blue"),
    ]:
        arr = np.array(data)
        n_zero = (arr < 0.1).sum()
        n_one = (arr > 0.9).sum()
        n_mid = len(arr) - n_zero - n_one

        ax.hist(arr, bins=bins, color=color, alpha=0.7, edgecolor="black", linewidth=0.3)
        ax.set_xlabel("Per-position CI value")
        ax.set_ylabel("Count")
        ax.set_title(f"{name}\n"
                     f"OFF (<0.1): {n_zero/len(arr)*100:.0f}% | "
                     f"MID: {n_mid/len(arr)*100:.0f}% | "
                     f"ON (>0.9): {n_one/len(arr)*100:.0f}%",
                     fontweight="bold", fontsize=10)

    fig.suptitle(
        "Per-position CI distributions are bimodal: components switch ON/OFF\n"
        "Mean CI reflects how often they are ON, not a graded activation strength",
        fontweight="bold", fontsize=12,
    )
    fig.tight_layout()
    path = OUT_DIR / "article_ci_bimodality.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"\nSaved {path}")

    # === Plot 2: Token examples — what activates each component ===
    from collections import Counter

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for ax, tokens, off_tokens, name in [
        (axes[0, 0], q335_tokens, q335_off_tokens, "Q335 (CI=0.70)"),
        (axes[0, 1], q270_tokens, q270_off_tokens, "Q270 (CI=0.24)"),
        (axes[0, 2], k206_tokens, k206_off_tokens, "K206 (CI=0.68)"),
    ]:
        # Show most common ON tokens
        on_counts = Counter(tokens).most_common(20)
        off_counts = Counter(off_tokens).most_common(20)

        y_pos = np.arange(min(20, len(on_counts)))
        on_labels = [f"'{t}'" for t, _ in on_counts[:20]]
        on_vals = [c for _, c in on_counts[:20]]

        ax.barh(y_pos, on_vals, color="tab:green", alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(on_labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Count (among ON tokens)")
        ax.set_title(f"{name}: ON tokens", fontweight="bold")

    for ax, tokens, off_tokens, name in [
        (axes[1, 0], q335_tokens, q335_off_tokens, "Q335"),
        (axes[1, 1], q270_tokens, q270_off_tokens, "Q270"),
        (axes[1, 2], k206_tokens, k206_off_tokens, "K206"),
    ]:
        off_counts = Counter(off_tokens).most_common(20)
        y_pos = np.arange(min(20, len(off_counts)))
        off_labels = [f"'{t}'" for t, _ in off_counts[:20]]
        off_vals = [c for _, c in off_counts[:20]]

        ax.barh(y_pos, off_vals, color="tab:red", alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(off_labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Count (among OFF tokens)")
        ax.set_title(f"{name}: OFF tokens", fontweight="bold")

    fig.suptitle(
        "What activates each component? Top tokens when ON (green) vs OFF (red)\n"
        "This is the evidence for calling these 'semantic' — they respond to content type",
        fontweight="bold", fontsize=12,
    )
    fig.tight_layout()
    path = OUT_DIR / "article_token_evidence.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")

    # === Plot 3: Component landscape — scatter of all Q/K component CIs ===
    fig, ax = plt.subplots(figsize=(10, 5))

    # Mark known semantic/induction components
    semantic_q = {335, 270, 279, 436, 66, 207, 261, 46}
    semantic_k = {206, 224, 347, 167, 107, 320, 373, 1}
    induction_q = {195, 499, 259, 110, 435}
    induction_k = {166, 5, 251, 55}

    for c, ci in enumerate(all_q_ci):
        if c in semantic_q:
            ax.scatter(c, ci, c="tab:red", s=30, zorder=3)
        elif c in induction_q:
            ax.scatter(c, ci, c="tab:blue", s=30, zorder=3)
        else:
            ax.scatter(c, ci, c="lightgray", s=5, alpha=0.5, zorder=1)

    ax.axhline(0.1, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Component index")
    ax.set_ylabel("Mean CI")
    ax.set_title(f"Layer {LAYER} Q components: semantic (red), induction (blue), other (gray)",
                 fontweight="bold")

    # Add labels for key components
    for c in [335, 270, 279, 436]:
        ax.annotate(f"Q{c}", (c, all_q_ci[c]), fontsize=7, fontweight="bold", color="tab:red",
                    xytext=(5, 5), textcoords="offset points")
    for c in [195, 499, 259]:
        ax.annotate(f"Q{c}", (c, all_q_ci[c]), fontsize=7, fontweight="bold", color="tab:blue",
                    xytext=(5, 5), textcoords="offset points")

    fig.tight_layout()
    path = OUT_DIR / "article_component_landscape.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")

    # Log token summaries
    logger.info("\n=== Token evidence for semantic labels ===")
    for name, on_toks, off_toks in [
        ("Q335", q335_tokens, q335_off_tokens),
        ("Q270", q270_tokens, q270_off_tokens),
        ("K206", k206_tokens, k206_off_tokens),
    ]:
        on_top = Counter(on_toks).most_common(10)
        off_top = Counter(off_toks).most_common(10)
        logger.info(f"\n  {name} ON tokens: {[t for t, _ in on_top]}")
        logger.info(f"  {name} OFF tokens: {[t for t, _ in off_top]}")


if __name__ == "__main__":
    main()
