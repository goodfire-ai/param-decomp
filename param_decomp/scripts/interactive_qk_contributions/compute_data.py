"""Precompute QK component contribution data for the interactive viewer.

Computes the W tensor (weight-only QK scores at all offsets), component activations,
pair rankings, and ComponentModel attention patterns for a single layer. Outputs
a JSON file shaped {prompts: [{tokens, label, layer_idx, ...layer fields}]}.

Usage:
    python -m param_decomp.scripts.interactive_qk_contributions.compute_data \
        wandb:goodfire/spd/runs/<run_id> --layer 1 \
        --prompts_file path/to/prompts.json

    python -m param_decomp.scripts.interactive_qk_contributions.compute_data \
        wandb:goodfire/spd/runs/<run_id> --layer 1 \
        --dataset_samples 30 --seq_len_min 20 --seq_len_max 40
"""

import json
import math
import random
from pathlib import Path

import fire
import numpy as np
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from param_decomp.app.backend.app_tokenizer import AppTokenizer
from param_decomp.configs import LMTaskConfig
from param_decomp.data import DatasetConfig, create_data_loader
from param_decomp.harvest.repo import HarvestRepo
from param_decomp.harvest.schemas import ComponentSummary
from param_decomp.log import logger
from param_decomp.models.component_model import ComponentModel, ParamDecompRunInfo
from param_decomp.models.components import LinearComponents
from param_decomp.param_decomp_types import ModelPath
from param_decomp.pretrain.models.llama_simple_mlp import LlamaSimpleMLP
from param_decomp.scripts.collect_attention_patterns import collect_attention_patterns_with_logits
from param_decomp.scripts.prompt_utils import load_prompts
from param_decomp.scripts.rope_aware_qk import compute_qk_rope_coefficients, evaluate_qk_at_offsets
from param_decomp.utils.wandb_utils import parse_wandb_run_path

SCRIPT_DIR = Path(__file__).parent
MIN_DENSITY = 0.001


def _get_alive_indices(
    summary: dict[str, ComponentSummary], module_path: str, min_density: float
) -> list[int]:
    """Return component indices sorted by firing density descending, filtered by threshold."""
    components = [
        (s.component_idx, s.firing_density)
        for s in summary.values()
        if s.layer == module_path and s.firing_density > min_density
    ]
    components.sort(key=lambda t: t[1], reverse=True)
    return [idx for idx, _ in components]


def _compute_alive_only_attention(
    model: ComponentModel,
    target_model: LlamaSimpleMLP,
    input_ids: torch.Tensor,
    layer_idx: int,
    q_alive: list[int],
    k_alive: list[int],
) -> np.ndarray:
    """Get attention patterns using only alive components' reconstructed weights.

    Reconstructs Q/K weights from alive components only: W = (V[:, alive] @ U[alive, :]).T
    Returns (n_heads, seq_len, seq_len) softmax attention weights.
    """
    q_path = f"h.{layer_idx}.attn.q_proj"
    k_path = f"h.{layer_idx}.attn.k_proj"

    block = target_model._h[layer_idx]
    orig_q_weight = block.attn.q_proj.weight.data.clone()
    orig_k_weight = block.attn.k_proj.weight.data.clone()

    q_component = model.components[q_path]
    k_component = model.components[k_path]
    assert isinstance(q_component, LinearComponents)
    assert isinstance(k_component, LinearComponents)

    # Reconstruct weight from alive components only: (V[:, alive] @ U[alive, :]).T
    with torch.no_grad():
        q_weight = (q_component.V[:, q_alive] @ q_component.U[q_alive, :]).T
        k_weight = (k_component.V[:, k_alive] @ k_component.U[k_alive, :]).T

    block.attn.q_proj.weight.data = q_weight
    block.attn.k_proj.weight.data = k_weight

    for blk in target_model._h:
        blk.attn.flash_attention = False

    with torch.no_grad():
        results = collect_attention_patterns_with_logits(target_model, input_ids)

    block.attn.q_proj.weight.data = orig_q_weight
    block.attn.k_proj.weight.data = orig_k_weight

    attn_weights, _ = results[layer_idx]
    return attn_weights[0].float().cpu().numpy()  # (n_heads, seq_len, seq_len)


def _compute_qk_tensors(
    model: ComponentModel,
    target_model: LlamaSimpleMLP,
    input_ids: torch.Tensor,
    layer_idx: int,
    q_idxs: list[int],
    k_idxs: list[int],
) -> dict[str, object]:
    """Compute W (n_heads, seq, n_q, n_k), per-component activations, and alive-only attention."""
    q_path = f"h.{layer_idx}.attn.q_proj"
    k_path = f"h.{layer_idx}.attn.k_proj"

    with torch.no_grad():
        out = model(input_ids, cache_type="input")

    component_acts = model.get_all_component_acts(out.cache)
    q_acts = component_acts[q_path][0].float()[:, q_idxs].detach().cpu().numpy()
    k_acts = component_acts[k_path][0].float()[:, k_idxs].detach().cpu().numpy()

    # Per-token causal importance (lower-leaky branch). Reuses the input cache.
    with torch.no_grad():
        ci_outputs = model.calc_causal_importances(
            pre_weight_acts=out.cache, sampling="continuous", detach_inputs=True
        )
    q_ci = ci_outputs.lower_leaky[q_path][0][:, q_idxs].detach().cpu().numpy()
    k_ci = ci_outputs.lower_leaky[k_path][0][:, k_idxs].detach().cpu().numpy()

    q_component = model.components[q_path]
    k_component = model.components[k_path]
    assert isinstance(q_component, LinearComponents)
    assert isinstance(k_component, LinearComponents)

    block = target_model._h[layer_idx]
    n_q_heads = block.attn.n_head
    n_kv_heads = block.attn.n_key_value_heads
    head_dim = block.attn.head_dim
    g = n_q_heads // n_kv_heads
    scale = 1.0 / math.sqrt(head_dim)

    U_q = q_component.U.float()[q_idxs].reshape(len(q_idxs), n_q_heads, head_dim)
    U_k = k_component.U.float()[k_idxs].reshape(len(k_idxs), n_kv_heads, head_dim)
    U_k_expanded = U_k.repeat_interleave(g, dim=1)

    rotary_cos = block.attn.rotary_cos
    rotary_sin = block.attn.rotary_sin
    assert isinstance(rotary_cos, torch.Tensor)
    assert isinstance(rotary_sin, torch.Tensor)

    seq_len = input_ids.shape[1]
    offsets = tuple(range(seq_len))

    W_all_heads = []
    for h in range(n_q_heads):
        A, B = compute_qk_rope_coefficients(U_q[:, h, :], U_k_expanded[:, h, :])
        W_h = evaluate_qk_at_offsets(A, B, rotary_cos, rotary_sin, offsets)
        W_all_heads.append(W_h)

    W = torch.stack(W_all_heads).detach().cpu().numpy()

    component_attn = _compute_alive_only_attention(
        model, target_model, input_ids, layer_idx, q_idxs, k_idxs
    )

    return {
        "W": W,
        "q_acts": q_acts,
        "k_acts": k_acts,
        "q_ci": q_ci,
        "k_ci": k_ci,
        "component_attn": component_attn,
        "n_heads": n_q_heads,
        "head_dim": head_dim,
        "scale": scale,
    }


def _compute_layer_data(
    model: ComponentModel,
    target_model: LlamaSimpleMLP,
    input_ids: torch.Tensor,
    layer_idx: int,
    q_alive: list[int],
    k_alive: list[int],
    top_k: int | None = None,
) -> dict[str, object]:
    """Compute W tensor, activations, and pair rankings for one layer."""
    tensors = _compute_qk_tensors(model, target_model, input_ids, layer_idx, q_alive, k_alive)
    W = tensors["W"]
    q_acts = tensors["q_acts"]
    k_acts = tensors["k_acts"]
    q_ci = tensors["q_ci"]
    k_ci = tensors["k_ci"]
    component_attn = tensors["component_attn"]
    scale = tensors["scale"]
    assert isinstance(W, np.ndarray)
    assert isinstance(q_acts, np.ndarray)
    assert isinstance(k_acts, np.ndarray)
    assert isinstance(q_ci, np.ndarray)
    assert isinstance(k_ci, np.ndarray)
    assert isinstance(component_attn, np.ndarray)
    assert isinstance(scale, float)

    # Rank pairs by peak absolute contribution across all heads and positions
    # Approximate: use W * max(|q_act|) * max(|k_act|) as proxy for peak contribution
    q_act_max = np.abs(q_acts).max(axis=0)
    k_act_max = np.abs(k_acts).max(axis=0)
    W_peak = np.abs(W).max(axis=(0, 1))
    pair_scores = scale * W_peak * q_act_max[:, None] * k_act_max[None, :]
    flat_order = np.argsort(pair_scores.ravel())[::-1]
    top_pairs_full = []
    for flat_idx in flat_order[:100]:
        qi, ki = divmod(int(flat_idx), len(k_alive))
        top_pairs_full.append([qi, ki, float(pair_scores[qi, ki])])

    if top_k is not None:
        top_pairs_full = top_pairs_full[:top_k]
        used_qi = sorted({p[0] for p in top_pairs_full})
        used_ki = sorted({p[1] for p in top_pairs_full})
        qi_remap = {old: new for new, old in enumerate(used_qi)}
        ki_remap = {old: new for new, old in enumerate(used_ki)}
        top_pairs = [[qi_remap[p[0]], ki_remap[p[1]], p[2]] for p in top_pairs_full]
        q_alive = [q_alive[i] for i in used_qi]
        k_alive = [k_alive[i] for i in used_ki]
        w_out = W[:, :, used_qi, :][:, :, :, used_ki]
        q_acts = q_acts[:, used_qi]
        k_acts = k_acts[:, used_ki]
        q_ci = q_ci[:, used_qi]
        k_ci = k_ci[:, used_ki]
    else:
        top_pairs = top_pairs_full
        w_out = W

    return {
        "layer_idx": layer_idx,
        "n_heads": tensors["n_heads"],
        "head_dim": tensors["head_dim"],
        "scale": scale,
        "alive_q": q_alive,
        "alive_k": k_alive,
        "W": w_out.tolist(),
        "q_acts": q_acts.tolist(),
        "k_acts": k_acts.tolist(),
        # CI values are in [0, 1]; quantize to ~3 decimal places to keep JSON small
        # (file sizes for layer 2 can otherwise grow well past 60MB)
        "q_ci": np.round(q_ci, 3).tolist(),
        "k_ci": np.round(k_ci, 3).tolist(),
        "component_model_attn": component_attn.tolist(),
        "top_pairs": top_pairs,
    }


def _make_label(prompt: str) -> str:
    text = prompt.strip().replace("\n", " ")
    if len(text) <= 200:
        return text
    return text[:200].rstrip() + "\u2026"


def _samples_from_prompts_file(
    prompts_file: str,
    tokenizer: PreTrainedTokenizer,
    max_seq_len: int,
) -> list[tuple[list[int], str]]:
    """Load prompts from JSON file, tokenize, return (token_ids, label) pairs."""
    prompts = load_prompts(Path(prompts_file))
    samples = []
    for prompt in prompts:
        encoded = tokenizer.encode(prompt, add_special_tokens=False)
        if len(encoded) > max_seq_len:
            encoded = encoded[:max_seq_len]
        samples.append((encoded, _make_label(prompt)))
    return samples


def _samples_from_dataset(
    task_config: LMTaskConfig,
    tokenizer_name: str,
    n_samples: int,
    seq_len_min: int,
    seq_len_max: int,
    seed: int = 42,
) -> list[tuple[list[int], str]]:
    """Sample random subsequences from the training dataset.

    Each sample's length is drawn uniformly from [seq_len_min, seq_len_max].
    """
    assert seq_len_min <= seq_len_max
    dataset_config = DatasetConfig(
        name=task_config.dataset_name,
        hf_tokenizer_path=tokenizer_name,
        split=task_config.train_data_split,
        n_ctx=task_config.max_seq_len,
        is_tokenized=task_config.is_tokenized,
        streaming=task_config.streaming,
        column_name=task_config.column_name,
        shuffle_each_epoch=False,
    )
    loader, tokenizer = create_data_loader(
        dataset_config=dataset_config,
        batch_size=1,
        buffer_size=1000,
        global_seed=seed,
    )

    rng = random.Random(seed)
    samples: list[tuple[list[int], str]] = []
    for batch in loader:
        if len(samples) >= n_samples:
            break
        ids = batch[task_config.column_name][0].tolist()
        seq_len = rng.randint(seq_len_min, seq_len_max)
        if len(ids) < seq_len:
            continue
        start = rng.randint(0, len(ids) - seq_len)
        subseq = ids[start : start + seq_len]
        label = tokenizer.decode(subseq)  # pyright: ignore[reportAttributeAccessIssue]
        samples.append((subseq, _make_label(label)))

    return samples


def compute_data(
    wandb_path: ModelPath,
    layer: int,
    prompts_file: str | None = None,
    dataset_samples: int | None = None,
    seq_len_min: int = 12,
    seq_len_max: int | None = None,
    min_density: float = MIN_DENSITY,
    top_k: int | None = None,
    output: str | None = None,
    seed: int = 42,
) -> None:
    assert (prompts_file is None) != (dataset_samples is None), (
        "Specify exactly one of --prompts_file or --dataset_samples"
    )

    _entity, _project, run_id = parse_wandb_run_path(str(wandb_path))
    run_info = ParamDecompRunInfo.from_path(wandb_path)
    config = run_info.config

    out_path = Path(output) if output is not None else SCRIPT_DIR / "out" / run_id / "prompts.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = ComponentModel.from_run_info(run_info)
    model.eval()

    target_model = model.target_model
    assert isinstance(target_model, LlamaSimpleMLP)
    for blk in target_model._h:
        blk.attn.flash_attention = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    assert config.tokenizer_name is not None
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    task_config = config.task_config
    assert isinstance(task_config, LMTaskConfig)

    repo = HarvestRepo.open_most_recent(run_id)
    assert repo is not None, f"No harvest data found for {run_id}"
    summary = repo.get_summary()

    assert 0 <= layer < len(target_model._h), f"layer {layer} out of range"

    if prompts_file is not None:
        samples = _samples_from_prompts_file(prompts_file, tokenizer, task_config.max_seq_len)
    else:
        assert dataset_samples is not None
        samples = _samples_from_dataset(
            task_config,
            config.tokenizer_name,
            dataset_samples,
            seq_len_min,
            seq_len_max if seq_len_max is not None else seq_len_min,
            seed,
        )

    logger.info(f"Loaded {len(samples)} samples, computing for layer {layer}")

    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    app_tokenizer = AppTokenizer(tokenizer)

    q_path = f"h.{layer}.attn.q_proj"
    k_path = f"h.{layer}.attn.k_proj"
    q_alive = _get_alive_indices(summary, q_path, min_density)
    k_alive = _get_alive_indices(summary, k_path, min_density)
    assert q_alive and k_alive, f"No alive components at layer {layer}"
    logger.info(f"Layer {layer}: {len(q_alive)} q components, {len(k_alive)} k components")

    prompts_out: list[dict[str, object]] = []
    for p_idx, (encoded, label) in enumerate(samples):
        input_ids = torch.tensor([encoded], device=device)
        tokens = app_tokenizer.get_spans(encoded)

        layer_data = _compute_layer_data(
            model, target_model, input_ids, layer, q_alive, k_alive, top_k
        )
        prompts_out.append({"tokens": tokens, "label": label, **layer_data})
        logger.info(f"[{p_idx + 1}/{len(samples)}] {len(tokens)} tokens: {label}")

    with open(out_path, "w") as f:
        json.dump({"prompts": prompts_out}, f)
    size_mb = out_path.stat().st_size / 1024 / 1024
    logger.info(f"Saved {out_path} ({len(prompts_out)} prompts, {size_mb:.1f} MB)")


if __name__ == "__main__":
    fire.Fire(compute_data)
