"""Plot per-head attention heatmaps for specific prompts.

For each prompt, produces a (n_layers x n_heads) grid of attention pattern
heatmaps with token labels. Causally masked positions are left uncolored
(light gray) rather than shown as zero.

Usage:
    python -m param_decomp.scripts.plot_prompt_attention.plot_prompt_attention \
        wandb:goodfire/spd/runs/<run_id> \
        --prompts_file path/to/prompts.json
"""

import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from param_decomp.configs import LMTaskConfig
from param_decomp.log import logger
from param_decomp.models.component_model import ParamDecompRunInfo
from param_decomp.param_decomp_types import ModelPath
from param_decomp.pretrain.models.llama_simple_mlp import LlamaSimpleMLP
from param_decomp.scripts.collect_attention_patterns import collect_attention_patterns
from param_decomp.utils.wandb_utils import parse_wandb_run_path

SCRIPT_DIR = Path(__file__).parent
MASKED_COLOR = "#e0e0e0"


def _plot_attention_heatmaps(
    patterns: list[torch.Tensor],
    tokens: list[str],
    out_path: Path,
) -> None:
    """Plot (n_layers x n_heads) grid of attention heatmaps with causal mask grayed out.

    Args:
        patterns: List of (n_heads, seq_len, seq_len) tensors, one per layer.
        tokens: Decoded token strings for axis labels.
        out_path: Where to save the figure.
    """
    n_layers = len(patterns)
    n_heads = patterns[0].shape[0]
    seq_len = len(tokens)

    # Build causal mask: True where position should be masked (upper triangle, offset=1)
    causal_mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)

    # Escape $ so matplotlib doesn't interpret tokens as math
    safe_tokens = [t.replace("$", r"\$") for t in tokens]

    # Each token gets ~0.25 inches in each subplot, with a minimum subplot size of 2 inches
    cell_size = max(2.0, seq_len * 0.25)
    tick_fontsize = 7

    fig, axes = plt.subplots(
        n_layers,
        n_heads,
        figsize=(n_heads * cell_size, n_layers * cell_size),
        squeeze=False,
    )

    for layer_idx in range(n_layers):
        for h in range(n_heads):
            ax = axes[layer_idx, h]
            pattern = patterns[layer_idx][h].numpy()
            masked = np.ma.masked_array(pattern, mask=causal_mask)

            ax.set_facecolor(MASKED_COLOR)
            ax.imshow(masked, aspect="auto", cmap="viridis", vmin=0, vmax=1)
            ax.set_title(f"L{layer_idx}H{h}", fontsize=9)

            ax.set_xticks(range(seq_len))
            ax.set_xticklabels(safe_tokens, rotation=45, ha="right", fontsize=tick_fontsize)
            ax.set_yticks(range(seq_len))
            ax.set_yticklabels(safe_tokens, fontsize=tick_fontsize)

            if h == 0:
                ax.set_ylabel(f"Layer {layer_idx}", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def plot_prompt_attention(
    wandb_path: ModelPath,
    prompts_file: str,
) -> None:
    _entity, _project, run_id = parse_wandb_run_path(str(wandb_path))
    run_info = ParamDecompRunInfo.from_path(wandb_path)

    out_dir = SCRIPT_DIR / "out" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    config = run_info.config
    assert config.pretrained_model_name is not None
    task_config = config.task_config
    assert isinstance(task_config, LMTaskConfig)

    target_model = LlamaSimpleMLP.from_pretrained(config.pretrained_model_name)
    target_model.eval()
    for block in target_model._h:
        block.attn.flash_attention = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_model = target_model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    max_seq_len = task_config.max_seq_len

    prompts_path = Path(prompts_file)
    assert prompts_path.exists(), f"Prompts file not found: {prompts_path}"

    with open(prompts_path) as f:
        prompts: list[str] = json.load(f)

    logger.info(f"Loaded {len(prompts)} prompts from {prompts_path}")

    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            encoded = tokenizer.encode(prompt, add_special_tokens=False)
            if len(encoded) > max_seq_len:
                encoded = encoded[:max_seq_len]

            input_ids = torch.tensor([encoded], device=device)
            tokens = [tokenizer.decode([t]) for t in encoded]  # pyright: ignore[reportAttributeAccessIssue]

            patterns = collect_attention_patterns(target_model, input_ids)
            # Squeeze batch dim: (1, n_heads, T, T) -> (n_heads, T, T)
            patterns = [p[0].float().cpu() for p in patterns]

            out_path = out_dir / f"prompt_{i}_attention.png"
            _plot_attention_heatmaps(patterns, tokens, out_path)

            logger.info(f"[{i + 1}/{len(prompts)}] {len(tokens)} tokens: {prompt[:60]}...")

    logger.info(f"All plots saved to {out_dir}")


if __name__ == "__main__":
    fire.Fire(plot_prompt_attention)
